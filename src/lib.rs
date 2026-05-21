//! BabbleSim + Zephyr nRF RPC simulation bridge.
//!
//! This crate provides three things:
//!
//! - **Test harness** ([`spawn_zephyr_rpc_server_with_socat`]) — spawn a full
//!   BabbleSim simulation from Rust integration tests.
//! - **xtask CLI** ([`xtask::cli_main`]) — docker, zephyr-setup, and run-bsim
//!   commands that downstream crates can re-export.
//! - **Programmatic setup API** ([`xtask::fetch_prebuilt_binaries`],
//!   [`xtask::zephyr_setup`]) — call from a downstream `build.rs` or any
//!   Rust code without shelling out.
//!
//! # Test harness usage
//!
//! ```no_run
//! use std::collections::HashSet;
//! use std::os::unix::net::UnixStream;
//! use std::path::Path;
//! use std::time::Duration;
//! use babble_bridge::LogOutput;
//!
//! let tests_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sockets"));
//! let (mut processes, socket_path) =
//!     babble_bridge::spawn_zephyr_rpc_server_with_socat(tests_dir, "my_test", LogOutput::Off);
//!
//! // socat is spawned but may not be listening yet — retry until connectable.
//! let start = std::time::Instant::now();
//! let _socket = loop {
//!     match UnixStream::connect(&socket_path) {
//!         Ok(s) => break s,
//!         Err(_) if start.elapsed() < Duration::from_secs(5) => {
//!             std::thread::sleep(Duration::from_millis(50));
//!         }
//!         Err(e) => panic!("socket never became connectable: {e}"),
//!     }
//! };
//!
//! // … write/read via _socket …
//!
//! processes.search_stdout_for_strings(HashSet::from([
//!     "<inf> nrf_ps_server: Initializing RPC server",
//! ]));
//! ```

pub mod xtask;

use std::collections::HashSet;
use std::env;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

/// Controls where the simulation process output (stdout/stderr) is forwarded
/// when [`spawn_zephyr_rpc_server_with_socat`] is called.
///
/// # Variants
///
/// - `Off` — no forwarding; processes write to `/dev/null` or an internal
///   buffer used only for [`TestProcesses::search_stdout_for_strings`].
/// - `Stream` — forward all output to the caller's terminal in real time,
///   labelled per process (e.g. `[rpc-server] …`).  Output goes to
///   `/dev/stderr` directly so it bypasses `cargo test` capture.
/// - `WriteToDir(path)` — write each process's output to a log file under
///   `path` (`rpc-server.log`, `cgm.log`, `phy.log`).  The directory is
///   created if it does not exist, and each log file is **truncated** at the
///   start of every spawn so that stale output from a previous run is cleared.
/// - `Both(path)` — stream to the terminal AND write to files simultaneously.
#[derive(Clone, Debug)]
pub enum LogOutput {
    /// No forwarding (default, silent).
    Off,
    /// Stream all process output to the terminal with `[label]` prefixes.
    Stream,
    /// Write each process's output to `<path>/{rpc-server,cgm,phy}.log`.
    /// Log files are truncated on every spawn.
    WriteToDir(PathBuf),
    /// Stream to terminal AND write to files under `path`.
    Both(PathBuf),
}

/// Owns all child processes spawned for a single simulation run and
/// accumulates their stdout output for later inspection.
///
/// All child processes are killed when this value is dropped.
pub struct TestProcesses {
    children: Vec<Child>,
    /// Combined stdout lines from every process whose stdout was captured.
    stdout_lines: Arc<Mutex<Vec<String>>>,
}

impl TestProcesses {
    /// Block until every string in `expected` appears as a substring of any
    /// accumulated stdout line, or panic after 30 seconds listing missing strings.
    pub fn search_stdout_for_strings(&mut self, expected: HashSet<&str>) {
        self.search_stdout_with_timeout(expected, Duration::from_secs(30));
    }

    /// Like [`search_stdout_for_strings`] but with a caller-supplied timeout.
    /// Useful in tests to avoid 30-second waits.
    pub fn search_stdout_with_timeout(&mut self, expected: HashSet<&str>, timeout: Duration) {
        let start = Instant::now();

        loop {
            let missing: HashSet<&str> = {
                let lines = self.stdout_lines.lock().unwrap();
                expected
                    .iter()
                    .copied()
                    .filter(|needle| !lines.iter().any(|line| line.contains(needle)))
                    .collect()
            };

            if missing.is_empty() {
                return;
            }

            if start.elapsed() >= timeout {
                let lines = self.stdout_lines.lock().unwrap();
                panic!(
                    "search_stdout_for_strings timed out after {:?}.\n\
                     Missing strings:\n{}\n\
                     Captured stdout ({} lines):\n{}",
                    timeout,
                    missing
                        .iter()
                        .map(|s| format!("  - {:?}", s))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    lines.len(),
                    lines
                        .iter()
                        .map(|l| format!("  {l}"))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }

            std::thread::sleep(Duration::from_millis(50));
        }
    }
    
    /// Helper method to dump the current stdout from attached nrf-rpc-server.
    /// Useful when debugging, but will result in search stdout methods no longer
    /// functioning (as this will consume stdout).
    pub fn debug_dump_stdout(&mut self, timeout: Duration) {
        let start = Instant::now();

        loop {
            if start.elapsed() >= timeout {
                return;
            } 
            
            let lines = self.stdout_lines.lock().unwrap();
            println!(
                "Captured stdout:\n{}",
                lines
                    .iter()
                    .map(|l| format!("  {l}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Kill all managed child processes immediately. Called automatically on drop.
    pub fn kill_all(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

impl Drop for TestProcesses {
    fn drop(&mut self) {
        self.kill_all();
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Spawn a background thread that drains `stream` line by line and writes
/// each line to the **real** stderr (fd 2 via `/dev/stderr`) as
/// `[<label>] <line>`.
///
/// We open `/dev/stderr` directly instead of using `eprintln!` so the output
/// reaches the terminal even when `cargo test` has redirected
/// `std::io::stderr()` to its per-test capture buffer (which suppresses
/// passing-test output unless `--nocapture` is passed).
fn pipe_labeled<R>(stream: R, label: &'static str)
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/stderr")
            .expect("open /dev/stderr");
        let reader = BufReader::new(stream);
        for line in reader.lines() {
            if let Ok(line) = line {
                let _ = writeln!(out, "[{label}] {line}");
            }
        }
    });
}

/// Spawn a background thread that drains `stream` line by line and appends
/// each line to the file at `path`.  The file must already exist (caller
/// creates/truncates it before spawning child processes).
fn pipe_to_file<R>(stream: R, path: PathBuf)
where
    R: std::io::Read + Send + 'static,
{
    std::thread::spawn(move || {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("pipe_to_file: could not open {}: {e}", path.display()));
        let reader = BufReader::new(stream);
        for line in reader.lines() {
            if let Ok(line) = line {
                let _ = writeln!(file, "{line}");
            }
        }
    });
}

// ── Public function ───────────────────────────────────────────────────────────

/// Spawns the full BabbleSim simulation stack for a single test:
///
/// 1. `bs_2G4_phy_v1`  — the radio PHY simulator
/// 2. `zephyr_rpc_server_app` — Zephyr nRF RPC server with `-uart0_pty`
/// 3. `cgm_peripheral_sample` — CGM BLE peripheral
///
/// The function waits up to 10 seconds for `zephyr_rpc_server_app` to print
/// its PTY path on stdout (`"UART_0 connected to pseudotty: /dev/pts/N"`),
/// then launches `socat` to bridge that PTY to a UNIX socket at
/// Kills any leftover BabbleSim processes from a previous run with the given
/// `sim_id`. Debugger stops and abnormal exits leave orphaned child processes
/// that hold the sim_id and block the next launch.
pub(crate) fn kill_stale_sim_processes(sim_id: &str) {
    let patterns = [
        format!("bs_2G4_phy_v1.*-s={sim_id}"),
        format!("zephyr_rpc_server_app.*-s={sim_id}"),
        format!("cgm_peripheral_sample.*-s={sim_id}"),
        format!("socat.*{sim_id}.sock"),
    ];
    for pat in &patterns {
        let _ = Command::new("pkill").args(["-9", "-f", pat]).status();
    }
    // Give processes time to fully exit.
    std::thread::sleep(Duration::from_millis(300));

    // BabbleSim stores per-sim IPC files under /tmp/bs_<username>/<sim_id>/.
    // These lock/pipe files must be removed before a new run or the PHY will
    // hang waiting for coordination on stale file descriptors.
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("bs_") {
                let sim_dir = entry.path().join(sim_id);
                if sim_dir.is_dir() {
                    let _ = std::fs::remove_dir_all(&sim_dir);
                }
            }
        }
    }

    // Also clean up any POSIX shared memory objects keyed by sim_id.
    if let Ok(entries) = std::fs::read_dir("/dev/shm") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().contains(sim_id) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}

/// `tests_dir/<test_name>.sock`.
///
/// # Panics
///
/// Panics if any process fails to spawn, if PTY discovery times out, or if
/// `socat` is not found on `PATH`.
pub fn spawn_zephyr_rpc_server_with_socat(
    tests_dir: &Path,
    test_name: &str,
    log: LogOutput,
) -> (TestProcesses, PathBuf) {
    let verbose = matches!(log, LogOutput::Stream | LogOutput::Both(_));
    let log_dir: Option<PathBuf> = match &log {
        LogOutput::WriteToDir(p) | LogOutput::Both(p) => Some(p.clone()),
        _ => None,
    };

    // If a log directory was requested, create it and truncate each log file
    // so output from the previous run is cleared before any process spawns.
    if let Some(ref dir) = log_dir {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("could not create log dir {}: {e}", dir.display()));
        for name in &["phy.log", "rpc-server.log", "cgm.log"] {
            std::fs::File::create(dir.join(name))
                .unwrap_or_else(|e| panic!("could not create log file {name}: {e}"));
        }
    }

    let bsim_bin = Path::new("external/tools/bsim/bin");
    let bsim_out = "external/tools/bsim";
    let bsim_comp = "external/tools/bsim/components";
    let ld_path = match env::var("LD_LIBRARY_PATH") {
        Ok(existing) => format!("external/tools/bsim/lib:{existing}"),
        Err(_) => "external/tools/bsim/lib".to_string(),
    };

    let sim_id = test_name;

    std::fs::create_dir_all(tests_dir)
        .unwrap_or_else(|e| panic!("could not create tests dir {}: {e}", tests_dir.display()));
    let socket_path = tests_dir.join(format!("{test_name}.sock"));

    // Kill orphaned processes FIRST so socat releases its fd on the socket
    // file before we unlink it.  Without this ordering, remove_file succeeds
    // on the directory entry but socat keeps an open fd on the inode, and the
    // new socat fails to bind if the socket is still in use.
    kill_stale_sim_processes(sim_id);
    let _ = std::fs::remove_file(&socket_path);

    // ── 1. PHY ──────────────────────────────────────────────────────────────
    let needs_phy_pipe = verbose || log_dir.is_some();
    let mut phy = Command::new("./bs_2G4_phy_v1")
        .args([
            &format!("-s={sim_id}"),
            "-D=2", // 2 radio devices: zephyr_rpc_server_app (d=0) + cgm_peripheral_sample (d=1)
            "-sim_length=86400e6",
        ])
        .current_dir(bsim_bin)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(if needs_phy_pipe { Stdio::piped() } else { Stdio::null() })
        .env("BSIM_OUT_PATH", bsim_out)
        .env("BSIM_COMPONENTS_PATH", bsim_comp)
        .env("LD_LIBRARY_PATH", &ld_path)
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn bs_2G4_phy_v1: {e}"));
    if let Some(stderr) = phy.stderr.take() {
        if verbose { pipe_labeled(stderr, "babblesim-phy"); }
        else if let Some(ref dir) = log_dir { pipe_to_file(stderr, dir.join("phy.log")); }
    }

    // ── 2. Zephyr RPC server (stdout always piped for PTY discovery + log capture) ──
    //
    // stdout must stay piped regardless of log mode so the PTY path can
    // be extracted.  When verbose, the reader thread additionally forwards
    // every line to stderr with a "[rpc-server]" prefix.  When writing to a
    // dir, the reader thread also writes every line to rpc-server.log.
    let stdout_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (pty_tx, pty_rx) = std::sync::mpsc::channel::<PathBuf>();

    // -force-color tells the Zephyr native-sim tracing layer to emit ANSI
    // escape codes even when stdout/stderr are pipes rather than a real TTY.
    // Without it, isatty() returns 0 on a pipe and colors are stripped.
    let zephyr_color_arg: &[&str] = if verbose { &["-force-color"] } else { &[] };

    let needs_zephyr_stderr = verbose || log_dir.is_some();
    let mut zephyr_proc = Command::new("./zephyr_rpc_server_app")
        .args([
            &format!("-s={sim_id}"),
            "-d=0",
            "-uart0_pty",
            "-uart_pty_pollT=1000",
        ])
        .args(zephyr_color_arg)
        .current_dir(bsim_bin)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(if needs_zephyr_stderr { Stdio::piped() } else { Stdio::null() })
        .env("BSIM_OUT_PATH", bsim_out)
        .env("BSIM_COMPONENTS_PATH", bsim_comp)
        .env("LD_LIBRARY_PATH", &ld_path)
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn zephyr_rpc_server_app: {e}"));

    // Drain Zephyr stderr (kernel/driver logs).
    if let Some(stderr) = zephyr_proc.stderr.take() {
        if verbose { pipe_labeled(stderr, "rpc-server"); }
        else if let Some(ref dir) = log_dir { pipe_to_file(stderr, dir.join("rpc-server.log")); }
    }

    // Drain Zephyr stdout in a background thread:
    // - send the PTY path once via `pty_tx` when the "pseudotty" line appears
    // - append every line to the shared `stdout_lines` buffer
    // - when verbose, also forward each line to stderr with a "[rpc-server]" prefix
    // - when writing to dir, also write every line to rpc-server.log
    let zephyr_stdout = zephyr_proc.stdout.take().expect("stdout was piped");
    let stdout_lines_clone = Arc::clone(&stdout_lines);
    let rpc_log_path = log_dir.as_ref().map(|d| d.join("rpc-server.log"));
    std::thread::spawn(move || {
        use std::io::Write;
        // Same /dev/stderr trick as pipe_labeled — bypasses cargo test capture.
        let mut real_stderr = verbose.then(|| {
            std::fs::OpenOptions::new()
                .write(true)
                .open("/dev/stderr")
                .expect("open /dev/stderr")
        });
        let mut log_file = rpc_log_path.as_ref().map(|p| {
            std::fs::OpenOptions::new()
                .append(true)
                .open(p)
                .unwrap_or_else(|e| panic!("could not open rpc-server.log: {e}"))
        });
        let reader = BufReader::new(zephyr_stdout);
        let mut pty_sent = false;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            // PTY discovery: nsi_print_trace writes to stdout
            // format: "<uart_name> connected to pseudotty: <slave_path>"
            if !pty_sent {
                if let Some(idx) = line.find("connected to pseudotty: ") {
                    let pty_path_str = line[idx + "connected to pseudotty: ".len()..].trim();
                    let pty_path = PathBuf::from(pty_path_str);
                    let _ = pty_tx.send(pty_path);
                    pty_sent = true;
                }
            }
            if let Some(ref mut out) = real_stderr {
                let _ = writeln!(out, "[rpc-server] {line}");
            }
            if let Some(ref mut f) = log_file {
                let _ = writeln!(f, "{line}");
            }
            stdout_lines_clone.lock().unwrap().push(line);
        }
    });

    // ── 3. CGM peripheral ────────────────────────────────────────────────────
    // When verbose or writing to a log dir, pipe stdout/stderr so we can
    // forward them.  Otherwise redirect to a local fallback log file (old
    // behaviour) so the process doesn't block writing to a closed pipe.
    let mut cgm = if verbose || log_dir.is_some() {
        Command::new("./cgm_peripheral_sample")
            .args([&format!("-s={sim_id}"), "-d=1"])
            .current_dir(bsim_bin)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("BSIM_OUT_PATH", bsim_out)
            .env("BSIM_COMPONENTS_PATH", bsim_comp)
            .env("LD_LIBRARY_PATH", &ld_path)
            .process_group(0)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn cgm_peripheral_sample: {e}"))
    } else {
        let cgm_log_path = bsim_bin.join("cgm_peripheral_sample.log");
        let cgm_log_file = std::fs::File::create(&cgm_log_path)
            .unwrap_or_else(|e| panic!("could not create cgm log file: {e}"));
        let cgm_log_clone = cgm_log_file
            .try_clone()
            .expect("could not clone cgm log file handle");
        Command::new("./cgm_peripheral_sample")
            .args([&format!("-s={sim_id}"), "-d=1"])
            .current_dir(bsim_bin)
            .stdin(Stdio::null())
            .stdout(cgm_log_file)
            .stderr(cgm_log_clone)
            .env("BSIM_OUT_PATH", bsim_out)
            .env("BSIM_COMPONENTS_PATH", bsim_comp)
            .env("LD_LIBRARY_PATH", &ld_path)
            .process_group(0)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn cgm_peripheral_sample: {e}"))
    };
    if let (Some(stdout), Some(stderr)) = (cgm.stdout.take(), cgm.stderr.take()) {
        if verbose {
            pipe_labeled(stdout, "cgm");
            pipe_labeled(stderr, "cgm");
        } else if let Some(ref dir) = log_dir {
            pipe_to_file(stdout, dir.join("cgm.log"));
            pipe_to_file(stderr, dir.join("cgm.log"));
        }
    }

    // ── 4. Wait for PTY path ─────────────────────────────────────────────────
    let pty_path = pty_rx
        .recv_timeout(Duration::from_secs(30))
        .unwrap_or_else(|_| {
            panic!(
                "timed out waiting for zephyr_rpc_server_app to announce UART PTY path \
                 (expected a stdout line containing \"connected to pseudotty: \")"
            )
        });

    // ── 5. socat bridge: PTY → UNIX socket ───────────────────────────────────
    let socket_path_str = socket_path
        .to_str()
        .expect("socket path must be valid UTF-8");
    let pty_path_str = pty_path
        .to_str()
        .expect("PTY path must be valid UTF-8");

    let socat = Command::new("socat")
        .arg(format!("UNIX-LISTEN:{socket_path_str},fork"))
        .arg(format!("{pty_path_str},raw,echo=0"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| {
            panic!(
                "failed to spawn socat (is it installed?): {e}\n\
                 socat bridges the Zephyr UART PTY ({pty_path_str}) to the test UNIX socket \
                 ({socket_path_str})"
            )
        });

    let processes = TestProcesses {
        children: vec![phy, zephyr_proc, cgm, socat],
        stdout_lines,
    };

    (processes, socket_path)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a TestProcesses with a pre-filled stdout buffer and no
    // real child processes.
    fn make_tp(lines: Vec<&str>) -> TestProcesses {
        let buf = Arc::new(Mutex::new(
            lines.into_iter().map(str::to_owned).collect(),
        ));
        TestProcesses {
            children: vec![],
            stdout_lines: buf,
        }
    }

    // ── PTY path parsing ──────────────────────────────────────────────────────

    #[test]
    fn parses_pty_path_from_typical_stdout_line() {
        let line = "UART_0 connected to pseudotty: /dev/pts/5";
        let needle = "connected to pseudotty: ";
        let idx = line.find(needle).expect("needle present");
        let path = line[idx + needle.len()..].trim();
        assert_eq!(path, "/dev/pts/5");
    }

    #[test]
    fn parses_pty_path_ignores_leading_whitespace() {
        let line = "  UARTE_1 connected to pseudotty:  /dev/pts/12  ";
        let needle = "connected to pseudotty:";
        let idx = line.find(needle).expect("needle present");
        let path = line[idx + needle.len()..].trim();
        assert_eq!(path, "/dev/pts/12");
    }

    // ── search_stdout_with_timeout ────────────────────────────────────────────

    #[test]
    fn search_finds_exact_line_match() {
        let mut tp = make_tp(vec!["<inf> nrf_ps_server: Initializing RPC server"]);
        // Must not panic.
        tp.search_stdout_with_timeout(
            HashSet::from(["Initializing RPC server"]),
            Duration::from_millis(500),
        );
    }

    #[test]
    fn search_finds_multiple_strings_across_different_lines() {
        let mut tp = make_tp(vec![
            "<inf> nrf_ps_server: Initializing RPC server",
            "<dbg> NRF_RPC: Done initializing nRF RPC module",
            "some other log line",
        ]);
        tp.search_stdout_with_timeout(
            HashSet::from([
                "Initializing RPC server",
                "Done initializing nRF RPC module",
            ]),
            Duration::from_millis(500),
        );
    }

    #[test]
    fn search_succeeds_on_empty_expected_set() {
        let mut tp = make_tp(vec![]);
        // Empty set → nothing to wait for → should return immediately.
        tp.search_stdout_with_timeout(HashSet::new(), Duration::from_millis(100));
    }

    #[test]
    #[should_panic(expected = "timed out")]
    fn search_panics_when_string_is_absent() {
        let mut tp = make_tp(vec!["something irrelevant"]);
        tp.search_stdout_with_timeout(
            HashSet::from(["this string is not present"]),
            Duration::from_millis(200),
        );
    }

    #[test]
    #[should_panic(expected = "timed out")]
    fn search_panics_when_only_some_strings_are_found() {
        let mut tp = make_tp(vec!["line A present"]);
        tp.search_stdout_with_timeout(
            HashSet::from(["line A present", "line B missing"]),
            Duration::from_millis(200),
        );
    }

    // ── kill_all is a no-op on an empty children list ─────────────────────────

    #[test]
    fn kill_all_on_empty_children_does_not_panic() {
        let mut tp = make_tp(vec![]);
        tp.kill_all(); // should be a silent no-op
    }

    // ── LogOutput variant helpers ─────────────────────────────────────────────

    #[test]
    fn log_output_off_is_not_verbose() {
        let verbose = matches!(LogOutput::Off, LogOutput::Stream | LogOutput::Both(_));
        assert!(!verbose);
    }

    #[test]
    fn log_output_write_to_dir_is_not_verbose() {
        let verbose = matches!(
            LogOutput::WriteToDir(PathBuf::from("/tmp")),
            LogOutput::Stream | LogOutput::Both(_)
        );
        assert!(!verbose);
    }

    #[test]
    fn log_output_stream_is_verbose() {
        let verbose = matches!(LogOutput::Stream, LogOutput::Stream | LogOutput::Both(_));
        assert!(verbose);
    }

    #[test]
    fn log_output_both_is_verbose() {
        let verbose = matches!(
            LogOutput::Both(PathBuf::from("/tmp")),
            LogOutput::Stream | LogOutput::Both(_)
        );
        assert!(verbose);
    }

    #[test]
    fn log_output_off_has_no_log_dir() {
        let log_dir: Option<PathBuf> = match &LogOutput::Off {
            LogOutput::WriteToDir(p) | LogOutput::Both(p) => Some(p.clone()),
            _ => None,
        };
        assert!(log_dir.is_none());
    }

    #[test]
    fn log_output_write_to_dir_extracts_path() {
        let expected = PathBuf::from("/tmp/sim-logs");
        let log_dir: Option<PathBuf> = match &LogOutput::WriteToDir(expected.clone()) {
            LogOutput::WriteToDir(p) | LogOutput::Both(p) => Some(p.clone()),
            _ => None,
        };
        assert_eq!(log_dir, Some(expected));
    }

    #[test]
    fn log_output_both_extracts_path() {
        let expected = PathBuf::from("/tmp/sim-logs");
        let log_dir: Option<PathBuf> = match &LogOutput::Both(expected.clone()) {
            LogOutput::WriteToDir(p) | LogOutput::Both(p) => Some(p.clone()),
            _ => None,
        };
        assert_eq!(log_dir, Some(expected));
    }

    // ── pipe_to_file ──────────────────────────────────────────────────────────

    #[test]
    fn pipe_to_file_writes_lines_to_file() {
        use std::io::Cursor;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.log");
        // Pre-create so pipe_to_file's append open succeeds.
        std::fs::File::create(&path).unwrap();

        let content = b"line one\nline two\nline three\n";
        pipe_to_file(Cursor::new(content), path.clone());

        // Give the background thread time to finish.
        std::thread::sleep(Duration::from_millis(200));

        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.contains("line one"), "missing 'line one' in {written:?}");
        assert!(written.contains("line two"), "missing 'line two' in {written:?}");
        assert!(written.contains("line three"), "missing 'line three' in {written:?}");
    }

    #[test]
    fn file_create_truncates_existing_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale.log");
        std::fs::write(&path, "old sentinel content\n").unwrap();

        // This is exactly what spawn_zephyr_rpc_server_with_socat does to clear logs.
        std::fs::File::create(&path).unwrap();

        let after = std::fs::read_to_string(&path).unwrap();
        assert!(after.is_empty(), "file should be empty after File::create, got {after:?}");
    }
}
