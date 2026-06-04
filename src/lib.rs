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
//! let (mut processes, socket_path) = babble_bridge::spawn_zephyr_rpc_server_with_socat(
//!     tests_dir,
//!     "my_test",
//!     LogOutput::Off,
//!     babble_bridge::SimConfig::default(),
//! );
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

/// Read the real-time speed ratio written by `cargo xtask start-sim`.
///
/// `cargo xtask start-sim` writes `<sim_dir>/<sim_id>.speed` containing the
/// configured `--speed` value (or `0.0` when no handbrake was used).
///
/// Use this from host-side library code to obtain the conversion factor:
///
/// ```no_run
/// use std::path::Path;
/// use std::time::Duration;
///
/// let speed = babble_bridge::read_sim_speed(
///     Path::new("tests/sockets"),
///     "sim",
/// );
/// if let Some(ratio) = speed {
///     // ratio > 0: sleep sim_duration / ratio wall-clock seconds
///     // ratio == 0: no handbrake — measure experimentally with run-bsim
///     let wall = Duration::from_secs(1).div_f64(ratio.max(1.0));
///     println!("1 simulated second ≈ {wall:?} wall time");
/// }
/// ```
pub fn read_sim_speed(sim_dir: &std::path::Path, sim_id: &str) -> Option<f64> {
    let path = sim_dir.join(format!("{sim_id}.speed"));
    let contents = std::fs::read_to_string(path).ok()?;
    contents.trim().parse::<f64>().ok()
}

use std::collections::HashSet;
use std::env;
use std::io::{BufRead, BufReader};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Public types ─────────────────────────────────────────────────────────────

/// Timing configuration for a BabbleSim simulation run.
///
/// Controls how long the simulation runs in simulated time and, optionally,
/// how fast the simulation clock advances relative to wall-clock time via
/// [`bs_device_handbrake`](https://babblesim.github.io/).
///
/// Use [`SimConfig::default`] to reproduce the original behaviour: 24 hours
/// of simulated time at unlimited speed (~2200× real time on typical hardware).
///
/// # BabbleSim time model
///
/// BabbleSim drives a **virtual clock** completely independent of the host's
/// wall clock. By default, this clock advances as fast as the CPU allows
/// (typically ~2200× faster than real time). The `speed` field engages
/// `bs_device_handbrake` to throttle the simulation to any desired ratio:
///
/// | `speed`       | behaviour                                              |
/// |---------------|--------------------------------------------------------|
/// | `None`        | unlimited (default, ~2200× real time)                 |
/// | `Some(1.0)`   | real-time: 1 simulated second ≈ 1 wall-clock second   |
/// | `Some(100.0)` | 100× real time                                        |
/// | `Some(1000.0)`| 1000× real time                                       |
#[derive(Clone, Debug)]
pub struct SimConfig {
    /// Total simulated duration in seconds (default: 86400 = 24 h).
    ///
    /// Converted to microseconds for BabbleSim's `-sim_length=<us>` argument.
    /// The PHY terminates the simulation when this virtual time is reached,
    /// causing all device processes to exit cleanly.
    pub sim_length_secs: f64,
    /// Optional real-time speed ratio via `bs_device_handbrake`.
    ///
    /// - `None` — no throttle; runs as fast as the host allows (~2200× real time)
    /// - `Some(1.0)` — wall-clock speed (1 sim-second ≈ 1 real second)
    /// - `Some(N)` — `N` simulated seconds per real second
    pub speed: Option<f64>,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            sim_length_secs: 86400.0,
            speed: None,
        }
    }
}

impl SimConfig {
    /// Convert a simulated duration to its expected wall-clock duration.
    ///
    /// Returns `Some(wall)` when [`speed`](SimConfig::speed) is configured
    /// (`wall = sim / speed`), or `None` when the simulation runs at
    /// unlimited speed (no handbrake).  The caller can use the returned
    /// duration directly as the argument to `std::thread::sleep` or
    /// an equivalent async sleep.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use babble_bridge::SimConfig;
    ///
    /// let cfg = SimConfig { sim_length_secs: 60.0, speed: Some(40.0) };
    /// // To wait for 1 simulated second at 40× speed:
    /// let wall = cfg.wall_duration_for(Duration::from_secs(1));
    /// assert_eq!(wall, Some(Duration::from_millis(25)));
    /// ```
    pub fn wall_duration_for(&self, sim_duration: std::time::Duration) -> Option<std::time::Duration> {
        self.speed
            .map(|s| sim_duration.div_f64(s))
    }
}

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
    /// PID of the `bs_2G4_phy_v1` process.
    ///
    /// The PHY is the simulation clock master and exits when the configured
    /// [`SimConfig::sim_length_secs`] of simulated time has elapsed.  After
    /// calling [`std::mem::forget`] on this struct (as `cargo xtask start-sim`
    /// does to keep processes alive after the command exits), you can monitor
    /// `/proc/<phy_pid>` to detect when the simulation has finished.
    pub phy_pid: u32,
    /// The real-time speed ratio this simulation was started with.
    ///
    /// Mirrors [`SimConfig::speed`]:
    /// - `Some(s)` — `bs_device_handbrake -r={s}` is running; the simulation
    ///   advances `s` simulated seconds per wall-clock second.
    ///   Use [`SimConfig::wall_duration_for`] to convert a simulated duration
    ///   to the wall-clock sleep your host code should use.
    /// - `None` — no handbrake; the simulation runs as fast as the CPU allows
    ///   (~2200× on typical hardware for simple workloads; measure with
    ///   `cargo xtask run-bsim --cgm-peripheral --sim-length 60`).
    pub configured_speed: Option<f64>,
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
#[cfg(test)]
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

// ── Public functions ──────────────────────────────────────────────────────────

/// Kills any leftover BabbleSim processes from a previous run with the given
/// `sim_id`. Debugger stops and abnormal exits leave orphaned child processes
/// that hold the sim_id and block the next launch.
pub(crate) fn kill_stale_sim_processes(sim_id: &str) {
    let patterns = [
        format!("bs_2G4_phy_v1.*-s={sim_id}"),
        format!("zephyr_rpc_server_app.*-s={sim_id}"),
        format!("cgm_peripheral_sample.*-s={sim_id}"),
        format!("bs_device_handbrake.*-s={sim_id}"),
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

/// Spawns the full BabbleSim simulation stack for a single test:
///
/// 1. `bs_2G4_phy_v1`  — the radio PHY simulator
/// 2. `zephyr_rpc_server_app` — Zephyr nRF RPC server with `-uart0_pty`
/// 3. `cgm_peripheral_sample` — CGM BLE peripheral
/// 4. `bs_device_handbrake` — optional real-time throttle (when `sim.speed` is `Some`)
///
/// The function waits up to 30 seconds for `zephyr_rpc_server_app` to print
/// its PTY path on stdout (`"UART_0 connected to pseudotty: /dev/pts/N"`),
/// then launches `socat` to bridge that PTY to a UNIX socket at
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
    sim: SimConfig,
) -> (TestProcesses, PathBuf) {
    let verbose = matches!(log, LogOutput::Stream | LogOutput::Both(_));
    // In persistent mode the child processes write directly to open file
    // descriptors that they inherit from us.  Because the FD lives in the
    // child process (not in a parent-side thread), logging continues even
    // after `start-sim` exits and the parent process is gone.  This is the
    // correct mode for `cargo xtask start-sim --log-dir …`.
    // In non-persistent (Stream) mode we use pipes + background threads,
    // which is correct for in-process test usage where the parent stays
    // alive for the whole test.
    let persistent = matches!(log, LogOutput::WriteToDir(_) | LogOutput::Both(_));
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

    // Compute PHY arguments from SimConfig.
    // BabbleSim uses microseconds for sim_length; multiply seconds by 1e6.
    let sim_length_arg = format!("-sim_length={}", (sim.sim_length_secs * 1_000_000.0) as u64);
    // Handbrake occupies one extra device slot when speed throttling is requested.
    let device_count = if sim.speed.is_some() { 3 } else { 2 };

    // ── 1. PHY ──────────────────────────────────────────────────────────────
    // Persistent mode: pass an open file FD directly to the child so it
    // keeps writing after the parent exits.  Non-persistent: use a pipe so
    // the parent thread can label and forward lines to /dev/stderr.
    let phy_stderr: Stdio = if persistent {
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(log_dir.as_ref().unwrap().join("phy.log"))
            .unwrap_or_else(|e| panic!("could not open phy.log: {e}"));
        Stdio::from(f)
    } else if verbose {
        Stdio::piped()
    } else {
        Stdio::null()
    };
    let mut phy = Command::new("./bs_2G4_phy_v1")
        .args([
            &format!("-s={sim_id}"),
            &format!("-D={device_count}"),
            &sim_length_arg,
        ])
        .current_dir(bsim_bin)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(phy_stderr)
        .env("BSIM_OUT_PATH", bsim_out)
        .env("BSIM_COMPONENTS_PATH", bsim_comp)
        .env("LD_LIBRARY_PATH", &ld_path)
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn bs_2G4_phy_v1: {e}"));
    // Only reached when verbose && !persistent (pipe was opened above).
    if let Some(stderr) = phy.stderr.take() {
        pipe_labeled(stderr, "babblesim-phy");
    }

    // ── 2. Zephyr RPC server ─────────────────────────────────────────────────
    //
    // Stdout must always be readable by the parent for PTY discovery, but the
    // mechanism differs by mode:
    //
    //  * Non-persistent (Off / Stream): pipe stdout → background thread
    //    (PTY discovery + stdout_lines + optional /dev/stderr labelling).
    //    Dies when the parent exits.
    //
    //  * Persistent (WriteToDir / Both): redirect stdout directly to
    //    rpc-server.log via an inherited FD so Zephyr keeps writing after
    //    the parent exits.  A background thread reads the *growing file*
    //    (tail-f style) for PTY discovery + stdout_lines + optional stderr
    //    labelling.  Stderr is also redirected to the same file so all Zephyr
    //    output is in one place.
    let stdout_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (pty_tx, pty_rx) = std::sync::mpsc::channel::<PathBuf>();

    // -force-color tells the Zephyr native-sim tracing layer to emit ANSI
    // escape codes even when stdout/stderr are pipes rather than a real TTY.
    // Without it, isatty() returns 0 on a pipe and colors are stripped.
    let zephyr_color_arg: &[&str] = if verbose { &["-force-color"] } else { &[] };

    let (zephyr_stdout_stdio, zephyr_rpc_log_path): (Stdio, Option<PathBuf>) = if persistent {
        let log_path = log_dir.as_ref().unwrap().join("rpc-server.log");
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .unwrap_or_else(|e| panic!("could not open rpc-server.log for writing: {e}"));
        (Stdio::from(f), Some(log_path))
    } else {
        (Stdio::piped(), None)
    };

    // Zephyr stderr: in persistent mode redirect to the same rpc-server.log
    // so all output is in one file.  In Stream mode pipe it for labelling.
    let zephyr_stderr_stdio: Stdio = if persistent {
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(log_dir.as_ref().unwrap().join("rpc-server.log"))
            .unwrap_or_else(|e| panic!("could not open rpc-server.log for stderr: {e}"));
        Stdio::from(f)
    } else if verbose {
        Stdio::piped()
    } else {
        Stdio::null()
    };

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
        .stdout(zephyr_stdout_stdio)
        .stderr(zephyr_stderr_stdio)
        .env("BSIM_OUT_PATH", bsim_out)
        .env("BSIM_COMPONENTS_PATH", bsim_comp)
        .env("LD_LIBRARY_PATH", &ld_path)
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn zephyr_rpc_server_app: {e}"));

    // Drain Zephyr stderr via pipe only in Stream mode (non-persistent).
    // In persistent mode the child writes directly to rpc-server.log.
    if let Some(stderr) = zephyr_proc.stderr.take() {
        // Only reached when verbose && !persistent.
        pipe_labeled(stderr, "rpc-server");
    }

    let stdout_lines_clone = Arc::clone(&stdout_lines);
    if let Some(rpc_log_path) = zephyr_rpc_log_path {
        // ── Persistent path: read the growing log file (tail-f style) ────────
        //
        // Zephyr writes to rpc-server.log via its inherited stdout FD.  We
        // open the same file for reading in a thread.  When read_line() hits
        // EOF it means no new data has arrived yet — we sleep briefly and
        // retry.  Because the FD is inherited by Zephyr (not held by this
        // thread), it stays open and Zephyr continues writing even after this
        // thread (and the parent process) exits.
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader, Write};
            let mut real_stderr = verbose.then(|| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open("/dev/stderr")
                    .expect("open /dev/stderr")
            });
            // Wait for the file to be openable (it was just created above, so
            // this should succeed immediately, but be defensive).
            let file = loop {
                match std::fs::File::open(&rpc_log_path) {
                    Ok(f) => break f,
                    Err(_) => std::thread::sleep(Duration::from_millis(10)),
                }
            };
            let mut reader = BufReader::new(file);
            let mut pty_sent = false;
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        // No new data yet — wait and retry.
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Ok(_) => {
                        let line = line
                            .trim_end_matches('\n')
                            .trim_end_matches('\r')
                            .to_string();
                        if !pty_sent {
                            if let Some(idx) = line.find("connected to pseudotty: ") {
                                let pty_str =
                                    line[idx + "connected to pseudotty: ".len()..].trim();
                                let _ = pty_tx.send(PathBuf::from(pty_str));
                                pty_sent = true;
                            }
                        }
                        if let Some(ref mut out) = real_stderr {
                            let _ = writeln!(out, "[rpc-server] {line}");
                        }
                        stdout_lines_clone.lock().unwrap().push(line);
                    }
                    Err(_) => break,
                }
            }
        });
    } else {
        // ── Non-persistent path: drain piped stdout in a background thread ───
        let zephyr_stdout = zephyr_proc.stdout.take().expect("stdout was piped");
        std::thread::spawn(move || {
            use std::io::Write;
            let mut real_stderr = verbose.then(|| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open("/dev/stderr")
                    .expect("open /dev/stderr")
            });
            let reader = BufReader::new(zephyr_stdout);
            let mut pty_sent = false;
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                // PTY discovery: nsi_print_trace writes to stdout.
                // format: "<uart_name> connected to pseudotty: <slave_path>"
                if !pty_sent {
                    if let Some(idx) = line.find("connected to pseudotty: ") {
                        let pty_path_str =
                            line[idx + "connected to pseudotty: ".len()..].trim();
                        let pty_path = PathBuf::from(pty_path_str);
                        let _ = pty_tx.send(pty_path);
                        pty_sent = true;
                    }
                }
                if let Some(ref mut out) = real_stderr {
                    let _ = writeln!(out, "[rpc-server] {line}");
                }
                stdout_lines_clone.lock().unwrap().push(line);
            }
        });
    }

    // ── 3. CGM peripheral ────────────────────────────────────────────────────
    //
    // Persistent mode: redirect stdout and stderr directly to cgm.log via
    // inherited FDs — no parent-side thread required, writes survive parent
    // exit.  Non-persistent verbose: pipe + label.  Off: redirect to a
    // local fallback file so the process doesn't block on a broken pipe.
    let mut cgm = if persistent {
        let cgm_out = std::fs::OpenOptions::new()
            .append(true)
            .open(log_dir.as_ref().unwrap().join("cgm.log"))
            .unwrap_or_else(|e| panic!("could not open cgm.log: {e}"));
        let cgm_err = cgm_out
            .try_clone()
            .expect("could not clone cgm.log file handle");
        Command::new("./cgm_peripheral_sample")
            .args([&format!("-s={sim_id}"), "-d=1"])
            .current_dir(bsim_bin)
            .stdin(Stdio::null())
            .stdout(Stdio::from(cgm_out))
            .stderr(Stdio::from(cgm_err))
            .env("BSIM_OUT_PATH", bsim_out)
            .env("BSIM_COMPONENTS_PATH", bsim_comp)
            .env("LD_LIBRARY_PATH", &ld_path)
            .process_group(0)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn cgm_peripheral_sample: {e}"))
    } else if verbose {
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
    // Only reachable when verbose && !persistent.
    if let (Some(stdout), Some(stderr)) = (cgm.stdout.take(), cgm.stderr.take()) {
        pipe_labeled(stdout, "cgm");
        pipe_labeled(stderr, "cgm");
    }

    // ── 3.5. Handbrake (optional speed throttle) ─────────────────────────────
    //
    // `bs_device_handbrake -r=<N>` stalls the simulation every poke_period
    // to keep the ratio of simulated time to wall-clock time at N:1.
    //   N=1    → real-time  (1 simulated second ≈ 1 wall-clock second)
    //   N=1000 → 1000× faster than real time
    //
    // Without the handbrake, BabbleSim runs as fast as the CPU allows
    // (~2200× real time on typical hardware).
    let handbrake: Option<Child> = if let Some(speed) = sim.speed {
        let hb = Command::new("./bs_device_handbrake")
            .args([
                &format!("-s={sim_id}"),
                "-d=2",
                &format!("-r={speed}"),
            ])
            .current_dir(bsim_bin)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .env("BSIM_OUT_PATH", bsim_out)
            .env("BSIM_COMPONENTS_PATH", bsim_comp)
            .env("LD_LIBRARY_PATH", &ld_path)
            .process_group(0)
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn bs_device_handbrake: {e}"));
        Some(hb)
    } else {
        None
    };

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

    let phy_pid = phy.id();
    let configured_speed = sim.speed;
    let mut children = vec![phy, zephyr_proc, cgm, socat];
    if let Some(hb) = handbrake {
        children.push(hb);
    }
    let processes = TestProcesses {
        children,
        phy_pid,
        configured_speed,
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
            phy_pid: 0,
            configured_speed: None,
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

    // ── persistent flag (WriteToDir / Both → inherited FD mode) ──────────────

    #[test]
    fn log_output_off_is_not_persistent() {
        let persistent = matches!(LogOutput::Off, LogOutput::WriteToDir(_) | LogOutput::Both(_));
        assert!(!persistent);
    }

    #[test]
    fn log_output_stream_is_not_persistent() {
        let persistent =
            matches!(LogOutput::Stream, LogOutput::WriteToDir(_) | LogOutput::Both(_));
        assert!(!persistent);
    }

    #[test]
    fn log_output_write_to_dir_is_persistent() {
        let persistent = matches!(
            LogOutput::WriteToDir(PathBuf::from("/tmp")),
            LogOutput::WriteToDir(_) | LogOutput::Both(_)
        );
        assert!(persistent);
    }

    #[test]
    fn log_output_both_is_persistent() {
        let persistent = matches!(
            LogOutput::Both(PathBuf::from("/tmp")),
            LogOutput::WriteToDir(_) | LogOutput::Both(_)
        );
        assert!(persistent);
    }

    // ── tail-f style file reader (persistent PTY discovery mechanism) ─────────

    /// The persistent path uses `BufReader::read_line` on a regular file.
    /// At EOF it returns `Ok(0)`, and after the file grows it returns new
    /// content on the next call.  This test verifies that behaviour — it is
    /// the foundation of the tail-f reader thread in
    /// `spawn_zephyr_rpc_server_with_socat`.
    #[test]
    fn tail_f_reader_sees_content_appended_after_eof() {
        use std::io::{BufRead, BufReader, Write};
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("grow.log");

        // Write the first line.
        let mut writer = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(writer, "first line").unwrap();
        drop(writer); // flush to disk

        // Open for reading and consume the first line.
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "expected to read first line");
        assert_eq!(line.trim_end(), "first line");

        // At EOF: read_line must return Ok(0).
        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        assert_eq!(n, 0, "expected Ok(0) at EOF");

        // Append a second line; reader must return it on the next call.
        let mut writer = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(writer, "second line").unwrap();
        drop(writer);

        line.clear();
        let n = reader.read_line(&mut line).unwrap();
        assert!(n > 0, "expected new content after file grew");
        assert_eq!(line.trim_end(), "second line");
    }

    /// End-to-end test of the persistent PTY-discovery mechanism: a writer
    /// thread appends lines (including a PTY announcement) to a file, while
    /// a reader thread polls the same file tail-f style.  Mirrors exactly
    /// what the background thread in `spawn_zephyr_rpc_server_with_socat`
    /// does in persistent mode.
    #[test]
    fn tail_f_reader_discovers_pty_line_written_by_concurrent_writer() {
        use std::io::{BufRead, BufReader, Write};
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rpc-server.log");

        // Pre-create to match spawn_zephyr_rpc_server_with_socat behaviour.
        std::fs::File::create(&path).unwrap();

        // Writer: wait briefly then append preamble + PTY line.
        let write_path = path.clone();
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&write_path)
                .unwrap();
            writeln!(f, "some preamble line").unwrap();
            writeln!(f, "UART_0 connected to pseudotty: /dev/pts/42").unwrap();
            writeln!(f, "line after pty announcement").unwrap();
        });

        // Reader: tail-f loop identical to the persistent-mode thread.
        let (tx, rx) = std::sync::mpsc::channel::<PathBuf>();
        let read_path = path.clone();
        let reader = std::thread::spawn(move || {
            let file = std::fs::File::open(&read_path).unwrap();
            let mut reader = BufReader::new(file);
            let start = std::time::Instant::now();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        assert!(
                            start.elapsed() < Duration::from_secs(5),
                            "timed out waiting for PTY line"
                        );
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Ok(_) => {
                        let line =
                            line.trim_end_matches('\n').trim_end_matches('\r').to_string();
                        if let Some(idx) = line.find("connected to pseudotty: ") {
                            let pty_str = line[idx + "connected to pseudotty: ".len()..].trim();
                            tx.send(PathBuf::from(pty_str)).unwrap();
                            break;
                        }
                    }
                    Err(e) => panic!("read_line error: {e}"),
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        let pty = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("PTY path should have been sent");
        assert_eq!(pty, PathBuf::from("/dev/pts/42"));
    }

    // ── SimConfig::wall_duration_for ──────────────────────────────────────────

    #[test]
    fn sim_config_default_is_24h_unlimited() {
        let cfg = SimConfig::default();
        assert_eq!(cfg.sim_length_secs, 86400.0);
        assert_eq!(cfg.speed, None);
    }

    #[test]
    fn wall_duration_no_speed_returns_none() {
        let cfg = SimConfig { sim_length_secs: 60.0, speed: None };
        assert_eq!(cfg.wall_duration_for(Duration::from_secs(1)), None);
    }

    #[test]
    fn wall_duration_speed_40x() {
        // 1 simulated second / 40× = 25 ms wall time
        let cfg = SimConfig { sim_length_secs: 60.0, speed: Some(40.0) };
        assert_eq!(
            cfg.wall_duration_for(Duration::from_secs(1)),
            Some(Duration::from_millis(25)),
        );
    }

    #[test]
    fn wall_duration_realtime_speed_1x() {
        // 1:1 — wall time equals sim time
        let cfg = SimConfig { sim_length_secs: 60.0, speed: Some(1.0) };
        assert_eq!(
            cfg.wall_duration_for(Duration::from_secs(10)),
            Some(Duration::from_secs(10)),
        );
    }

    #[test]
    fn wall_duration_speed_1000x() {
        // 500 ms sim / 1000× = 500 µs wall
        let cfg = SimConfig { sim_length_secs: 60.0, speed: Some(1000.0) };
        assert_eq!(
            cfg.wall_duration_for(Duration::from_millis(500)),
            Some(Duration::from_micros(500)),
        );
    }

    #[test]
    fn wall_duration_scales_with_magnitude() {
        let cfg = SimConfig { sim_length_secs: 60.0, speed: Some(100.0) };
        // 1 min sim / 100× = 600 ms wall
        assert_eq!(
            cfg.wall_duration_for(Duration::from_secs(60)),
            Some(Duration::from_millis(600)),
        );
    }

    // ── read_sim_speed ────────────────────────────────────────────────────────

    #[test]
    fn read_sim_speed_reads_float() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("test.speed"), "40.0\n").unwrap();
        assert_eq!(read_sim_speed(dir.path(), "test"), Some(40.0));
    }

    #[test]
    fn read_sim_speed_reads_integer() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("sim.speed"), "1000\n").unwrap();
        assert_eq!(read_sim_speed(dir.path(), "sim"), Some(1000.0));
    }

    #[test]
    fn read_sim_speed_zero_means_unlimited() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("s.speed"), "0\n").unwrap();
        assert_eq!(read_sim_speed(dir.path(), "s"), Some(0.0));
    }

    #[test]
    fn read_sim_speed_missing_file_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read_sim_speed(dir.path(), "nonexistent"), None);
    }

    #[test]
    fn read_sim_speed_invalid_content_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("bad.speed"), "not_a_number\n").unwrap();
        assert_eq!(read_sim_speed(dir.path(), "bad"), None);
    }

    #[test]
    fn read_sim_speed_trims_whitespace() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("ws.speed"), "  97.6  \n").unwrap();
        assert_eq!(read_sim_speed(dir.path(), "ws"), Some(97.6));
    }

    #[test]
    fn sim_length_to_microseconds_conversion() {
        // The PHY -sim_length= argument is in microseconds.
        // 300 s → 300_000_000 µs (fits in u64, no overflow)
        assert_eq!((300.0_f64 * 1_000_000.0) as u64, 300_000_000);
        // Original hard-coded value 86400e6 matches our formula.
        assert_eq!((86400.0_f64 * 1_000_000.0) as u64, 86_400_000_000);
    }
}
