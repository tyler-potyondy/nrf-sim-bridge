//! Xtask commands (docker, zephyr-setup, run-bsim) and programmatic API.
//!
//! ## CLI usage
//!
//! Downstream crates re-export [`cli_main`] via a thin `xtask` binary so
//! that `cargo xtask <command>` works from their workspace root.
//!
//! ## Library / build-script usage
//!
//! The heavy-lifting functions are also exposed as a public API so that
//! another crate's `build.rs` (or any Rust code) can call them directly
//! without shelling out:
//!
//! ```no_run
//! use std::path::Path;
//! use nrf_sim_bridge::xtask;
//!
//! let root = Path::new("/path/to/workspace");
//! let external = root.join("external");
//! xtask::fetch_prebuilt_binaries(&root, &external)
//!     .expect("failed to fetch BabbleSim binaries");
//! ```

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Boxed error type used by all public functions in this module.
pub type DynError = Box<dyn std::error::Error>;
/// Result alias used by all public functions in this module.
pub type Result<T> = std::result::Result<T, DynError>;

const DOCKER_IMAGE_TAG: &str = "nrf-sim-bridge:latest";

/// How BabbleSim/Zephyr binaries should be installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMode {
    /// Build everything from source (~30 min, requires full Zephyr toolchain).
    BuildFromSource,
    /// Download a prebuilt release archive from GitHub (~30 s).
    FetchPrebuilt,
}

/// Top-level entry point. Call this from your `xtask` binary's `main()`.
///
/// Parses `std::env::args()`, dispatches to the matching subcommand, and
/// calls `std::process::exit(1)` on failure.
pub fn cli_main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(cmd) = args.next() else {
        print_usage();
        return Ok(());
    };

    match cmd.as_str() {
        "zephyr-setup" => {
            require_linux("zephyr-setup")?;
            let args: Vec<String> = args.collect();
            let clean = args.iter().any(|a| a == "--clean");
            let mode = if args.iter().any(|a| a == "--prebuilt") {
                InstallMode::FetchPrebuilt
            } else if args.iter().any(|a| a == "--build-from-source") {
                InstallMode::BuildFromSource
            } else {
                prompt_install_mode()?
            };
            let root = workspace_root()?;
            zephyr_setup(&root, clean, mode)
        }
        "run-bsim" => {
            require_linux("run-bsim")?;
            let args: Vec<String> = args.collect();
            let nrf_rpc_server = args.iter().any(|a| a == "--nrf-rpc-server");
            let cgm_peripheral = args.iter().any(|a| a == "--cgm-peripheral");
            run_bsim(nrf_rpc_server, cgm_peripheral)
        }
        "start-sim" => {
            require_linux("start-sim")?;
            let args: Vec<String> = args.collect();
            let sim_id = parse_sim_flag(&args, "--sim-id").unwrap_or("insulin_pump");
            let sim_dir = parse_sim_flag(&args, "--sim-dir")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/tmp/nrf-sim-bridge"));
            cmd_start_sim(sim_id, &sim_dir)
        }
        "stop-sim" => {
            require_linux("stop-sim")?;
            let args: Vec<String> = args.collect();
            let sim_id = parse_sim_flag(&args, "--sim-id").unwrap_or("insulin_pump");
            cmd_stop_sim(sim_id)
        }
        "docker-build" => docker_build(),
        "docker-attach" => docker_attach(),
        "docker-run" => {
            let rest: Vec<String> = args.collect();
            // Allow an optional `--` separator before the command.
            let cmd_args: Vec<&str> = rest
                .iter()
                .skip_while(|a| a.as_str() == "--")
                .map(String::as_str)
                .collect();
            if cmd_args.is_empty() {
                return Err("docker-run requires a command to run inside the container".into());
            }
            docker_run(&cmd_args)
        }
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        _ => Err(format!("Unknown command: {cmd}").into()),
    }
}

fn print_usage() {
    println!("Usage: cargo xtask <command> [options]");
    println!();
    println!("Commands:");
    println!("  docker-build                      Build the dev-container image");
    println!("  docker-attach                     Open an interactive shell in the container");
    println!("  docker-run [--] <cmd> [args...]   Run a command non-interactively in the container (for CI)");
    println!();
    println!("  zephyr-setup [--clean]            Set up Zephyr/BabbleSim (prompts for install mode)");
    println!("    --prebuilt                      Fetch prebuilt binaries from GitHub Releases");
    println!("    --build-from-source             Build from source (non-interactive, for CI)");
    println!();
    println!("  run-bsim                          Run BabbleSim simulation (Linux only)");
    println!("    --nrf-rpc-server                Launch the nRF RPC server (default: on)");
    println!("    --cgm-peripheral                Launch the CGM peripheral sample (default: on)");
    println!();
    println!("  start-sim                         Start simulation stack in the background (Linux only)");
    println!("    --sim-id <id>                   Simulation identifier (default: insulin_pump)");
    println!("    --sim-dir <path>                Directory for the socket file (default: /tmp/nrf-sim-bridge)");
    println!("    Prints the socket path on success.");
    println!();
    println!("  stop-sim                          Stop a running simulation (Linux only)");
    println!("    --sim-id <id>                   Simulation identifier to stop (default: insulin_pump)");
}

fn require_linux(cmd: &str) -> Result<()> {
    if !cfg!(target_os = "linux") {
        return Err(format!(
            "`xtask {cmd}` requires Linux. \
             Use `cargo xtask docker-build` to build the dev-container image, \
             then work inside it."
        )
        .into());
    }
    Ok(())
}

// ── Install-mode prompt ──────────────────────────────────────────────────────

fn prompt_install_mode() -> Result<InstallMode> {
    println!();
    println!("How would you like to set up the Zephyr/BabbleSim environment?");
    println!("  [1] Build from source   (slow, ~30 min; requires a full Zephyr toolchain)");
    println!("  [2] Fetch prebuilt binaries  (fast; downloads a release archive from GitHub)");
    print!("Enter choice [1/2] (default: 2): ");
    io::stdout().flush()?;

    let line = io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or("No input received — stdin was empty")??;

    match line.trim() {
        "1" => {
            println!("Selected: build from source.");
            Ok(InstallMode::BuildFromSource)
        }
        "2" | "" => {
            println!("Selected: fetch prebuilt binaries.");
            Ok(InstallMode::FetchPrebuilt)
        }
        other => Err(format!("Invalid choice '{other}'. Please enter 1 or 2.").into()),
    }
}

// ── Docker helpers ──────────────────────────────────────────────────────────

fn docker_build() -> Result<()> {
    let root = workspace_root()?;
    let dockerfile = root.join(".devcontainer/Dockerfile");
    if !dockerfile.exists() {
        return Err(format!(
            "Dockerfile not found at {}",
            dockerfile.display()
        )
        .into());
    }

    let uid = std::env::var("UID").unwrap_or_else(|_| "1000".into());
    let gid = std::env::var("GID").unwrap_or_else(|_| "1000".into());

    println!("Building Docker image {DOCKER_IMAGE_TAG} …");
    run_cmd(
        "docker",
        &[
            "build",
            "--platform", "linux/amd64",
            "-f", ".devcontainer/Dockerfile",
            "--build-arg", &format!("USER_UID={uid}"),
            "--build-arg", &format!("USER_GID={gid}"),
            "-t", DOCKER_IMAGE_TAG,
            ".",
        ],
        Some(&root),
    )?;
    println!("Image built: {DOCKER_IMAGE_TAG}");
    Ok(())
}

fn docker_attach() -> Result<()> {
    let root = workspace_root()?;
    let workspace = root
        .to_str()
        .ok_or("Workspace path contains non-UTF-8 characters")?;

    run_cmd(
        "docker",
        &[
            "run",
            "--rm",
            "--interactive",
            "--tty",
            "--platform", "linux/amd64",
            "-v", &format!("{workspace}:/workspace"),
            "-w", "/workspace",
            DOCKER_IMAGE_TAG,
            "bash",
        ],
        Some(&root),
    )
}

/// Run a command non-interactively inside a one-shot container.
///
/// Unlike `docker-attach`, this does not allocate a TTY, so it works in CI
/// environments where stdin is not a terminal. The workspace is bind-mounted
/// at `/workspace`, so state written by the command (e.g. `external/` setup
/// output) persists on the host across invocations.
fn docker_run(cmd_args: &[&str]) -> Result<()> {
    let root = workspace_root()?;
    let workspace = root
        .to_str()
        .ok_or("Workspace path contains non-UTF-8 characters")?;

    let mount = format!("{workspace}:/workspace");
    let mut docker_args: Vec<&str> = vec![
        "run",
        "--rm",
        "--platform", "linux/amd64",
        "-v", &mount,
        "-w", "/workspace",
        DOCKER_IMAGE_TAG,
    ];
    docker_args.extend_from_slice(cmd_args);

    run_cmd("docker", &docker_args, Some(&root))
}

// ── Workspace / utility helpers ──────────────────────────────────────────────

/// Walk up from the current directory until a `Cargo.toml` is found.
///
/// This is the same heuristic the CLI uses; exposed publicly so that
/// downstream `build.rs` scripts can locate the workspace root without
/// duplicating the logic.
pub fn workspace_root() -> Result<PathBuf> {
    let mut dir = env::current_dir()?;
    loop {
        if dir.join("Cargo.toml").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Err("Could not find workspace root (no Cargo.toml found in any parent directory)".into());
        }
    }
}

fn run_cmd(cmd: &str, args: &[&str], cwd: Option<&Path>) -> Result<()> {
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.stdin(Stdio::inherit());
    command.stdout(Stdio::inherit());
    command.stderr(Stdio::inherit());

    let status = command.status()?;
    if !status.success() {
        return Err(format!("Command failed: {cmd} {}", args.join(" ")).into());
    }
    Ok(())
}

fn clean_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name() == ".gitignore" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            fs::remove_dir_all(path)?;
        } else {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

/// Hardcoded "latest release" URL for the prebuilt BabbleSim bundle.
///
/// The asset filenames are FIXED (no SHA in the name), so this URL pattern
/// resolves to the most recently published release without any GitHub API
/// calls or authentication. See `.github/workflows/publish.yml`.
const PREBUILT_RELEASE_URL_BASE: &str =
    "https://github.com/tyler-potyondy/nrf-sim-bridge/releases/latest/download";
const PREBUILT_TARBALL_NAME: &str = "bsim-prebuilt.tar.gz";
const PREBUILT_SHA256_NAME: &str = "bsim-prebuilt.tar.gz.sha256";

/// Download the latest published prebuilt BabbleSim bundle, verify its
/// SHA-256, and extract it into `<external_dir>/tools/bsim/`.
///
/// The tarball produced by the publish workflow contains `bin/`, `lib/`, and
/// `components/` at its root, so extracting into `external/tools/bsim/`
/// reproduces the exact layout that [`zephyr_setup`] with
/// [`InstallMode::BuildFromSource`] would create — without spending ~30
/// minutes rebuilding Zephyr/BabbleSim from source.
///
/// `root` is the workspace root (currently unused but reserved for future
/// path resolution).  `external_dir` is typically `root.join("external")`.
///
/// Requires `curl`, `sha256sum`, and `tar` on `PATH` (all present in the
/// devcontainer).
pub fn fetch_prebuilt_binaries(root: &Path, external_dir: &Path) -> Result<()> {
    let _ = root;
    let bsim_dir = external_dir.join("tools/bsim");
    fs::create_dir_all(&bsim_dir)?;

    let download_dir = external_dir.join(".prebuilt-download");
    if download_dir.exists() {
        fs::remove_dir_all(&download_dir)?;
    }
    fs::create_dir_all(&download_dir)?;

    let tarball = download_dir.join(PREBUILT_TARBALL_NAME);
    let sha_file = download_dir.join(PREBUILT_SHA256_NAME);
    let tarball_url = format!("{PREBUILT_RELEASE_URL_BASE}/{PREBUILT_TARBALL_NAME}");
    let sha_url = format!("{PREBUILT_RELEASE_URL_BASE}/{PREBUILT_SHA256_NAME}");

    let tarball_str = tarball.to_str().ok_or("Invalid UTF-8 path for tarball")?;
    let sha_file_str = sha_file.to_str().ok_or("Invalid UTF-8 path for sha file")?;
    let bsim_dir_str = bsim_dir.to_str().ok_or("Invalid UTF-8 path for bsim dir")?;

    println!("Downloading {tarball_url} ...");
    run_cmd(
        "curl",
        &["--fail", "--location", "--show-error", "--silent",
          "--output", tarball_str, &tarball_url],
        None,
    )?;

    println!("Downloading {sha_url} ...");
    run_cmd(
        "curl",
        &["--fail", "--location", "--show-error", "--silent",
          "--output", sha_file_str, &sha_url],
        None,
    )?;

    println!("Verifying SHA-256 ...");
    // `sha256sum -c` matches filenames as written in the .sha256 file
    // (relative to cwd), so run it from the directory containing both files.
    run_cmd(
        "sha256sum",
        &["--check", "--strict", PREBUILT_SHA256_NAME],
        Some(&download_dir),
    )?;

    // Wipe any stale bin/lib/components from a previous setup so we don't
    // mix files from a partially-extracted older bundle.
    for sub in ["bin", "lib", "components"] {
        let p = bsim_dir.join(sub);
        if p.exists() {
            fs::remove_dir_all(&p)?;
        }
    }

    println!("Extracting into {} ...", bsim_dir.display());
    run_cmd(
        "tar",
        &["-xzf", tarball_str, "-C", bsim_dir_str],
        None,
    )?;

    fs::remove_dir_all(&download_dir)?;

    println!("Prebuilt binaries installed to {}", bsim_dir.display());
    println!("  bin/        — BabbleSim + Zephyr app binaries");
    println!("  lib/        — shared libraries (LD_LIBRARY_PATH)");
    println!("  components/ — BabbleSim runtime components");
    Ok(())
}

// ── Zephyr setup ─────────────────────────────────────────────────────────────

/// Set up the Zephyr / BabbleSim environment under `root`.
///
/// When `clean` is true, everything under `<root>/external/` (except
/// `.gitignore`) is removed first.  `mode` selects between a full
/// source build and a fast prebuilt-binary download.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use nrf_sim_bridge::xtask::{self, InstallMode};
///
/// let root = xtask::workspace_root().unwrap();
/// xtask::zephyr_setup(&root, false, InstallMode::FetchPrebuilt).unwrap();
/// ```
pub fn zephyr_setup(root: &Path, clean: bool, mode: InstallMode) -> Result<()> {
    let external_dir = root.join("external");

    if clean {
        println!("Cleaning up {}...", external_dir.display());
        clean_dir(&external_dir)?;
    }

    fs::create_dir_all(&external_dir)?;

    if let InstallMode::FetchPrebuilt = mode {
        return fetch_prebuilt_binaries(root, &external_dir);
    }

    // Initialize the nrf submodule. This is idempotent: a no-op if
    // already initialized, otherwise clones it. In CI, `actions/checkout`
    // with `submodules: recursive` already populates it, in which case
    // this call just verifies the SHA matches.
    println!("Setting up nrf submodule...");
    run_cmd(
        "git",
        &["submodule", "update", "--init", "external/nrf"],
        Some(root),
    )?;

    let venv_dir = external_dir.join(".venv");
    let venv_python = venv_dir.join("bin/python3");
    let venv_stamp = venv_dir.join(".requirements_installed");

    let python_ok = venv_python.exists()
        && Command::new(&venv_python)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
    let venv_valid = python_ok && venv_stamp.exists();

    if !venv_valid {
        if venv_dir.exists() {
            if !python_ok {
                println!("Existing venv is stale or from a different Python, recreating...");
            } else {
                println!("Existing venv has incomplete requirements, recreating...");
            }
            fs::remove_dir_all(&venv_dir)?;
        } else {
            println!("Creating venv...");
        }
        run_cmd("python3", &["-m", "venv", ".venv"], Some(&external_dir))?;
    }

    let pip = external_dir.join(".venv/bin/pip");
    let west = external_dir.join(".venv/bin/west");
    let pip_str = pip.to_str().ok_or("Invalid UTF-8 path for pip")?;
    let west_str = west.to_str().ok_or("Invalid UTF-8 path for west")?;

    let venv_bin = venv_dir.join("bin");
    let venv_bin_str = venv_bin.to_str().ok_or("Invalid UTF-8 path for venv bin")?;
    let path = env::var("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{venv_bin_str}:{path}"));
    std::env::set_var("VIRTUAL_ENV", &venv_dir);

    run_cmd(pip_str, &["install", "west"], Some(&external_dir))?;

    let west_state = external_dir.join(".west");
    if west_state.exists() {
        println!("Previous west workspace found, resetting...");
        fs::remove_dir_all(west_state)?;
    }

    run_cmd(west_str, &["init", "-l", "nrf"], Some(&external_dir))?;
    println!("Fetching west dependencies (BabbleSim + Zephyr)...");
    run_cmd(
        west_str,
        &["config", "manifest.group-filter", "--", "+babblesim"],
        Some(&external_dir),
    )?;
    run_cmd(west_str, &["update"], Some(&external_dir))?;

    run_cmd(
        pip_str,
        &["install", "-r", "nrf/scripts/requirements.txt"],
        Some(&external_dir),
    )?;
    run_cmd(
        pip_str,
        &["install", "-r", "zephyr/scripts/requirements.txt"],
        Some(&external_dir),
    )?;

    println!("Verifying all requirements are installed...");
    let dry_run = Command::new(pip_str)
        .args([
            "install",
            "-r", "nrf/scripts/requirements.txt",
            "-r", "zephyr/scripts/requirements.txt",
            "--dry-run",
        ])
        .current_dir(&external_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()?;

    let dry_run_out = String::from_utf8_lossy(&dry_run.stdout);
    let dry_run_err = String::from_utf8_lossy(&dry_run.stderr);
    let combined = format!("{dry_run_out}{dry_run_err}");
    if combined.contains("Would install") {
        return Err(format!(
            "Requirements are not fully installed after pip install — \
             the following packages are still missing or out of range:\n{combined}\n\
             Re-run with --clean to start fresh."
        )
        .into());
    }

    fs::write(&venv_stamp, "")?;

    println!("Building BabbleSim...");
    run_cmd(
        "make",
        &["-C", "tools/bsim", "everything", "-j", "4"],
        Some(&external_dir),
    )?;

    println!("Checking out cgm-bsim branch in nrf...");
    let current_branch = Command::new("git")
        .args(["-C", "nrf", "rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(&external_dir)
        .output()?;
    if !current_branch.status.success() {
        return Err("Failed to read current nrf branch".into());
    }
    if String::from_utf8(current_branch.stdout)?.trim() != "cgm-bsim" {
        run_cmd(
            "git",
            &["-C", "nrf", "fetch", "origin", "cgm-bsim"],
            Some(&external_dir),
        )?;
        run_cmd(
            "git",
            &["-C", "nrf", "checkout", "-B", "cgm-bsim", "FETCH_HEAD"],
            Some(&external_dir),
        )?;
    }

    println!("Building Zephyr server app...");
    run_cmd(
        west_str,
        &[
            "build", "-b", "nrf52_bsim", "-p", "always",
            "--build-dir", "build/zephyr_server_app",
            "nrf/samples/nrf_rpc/protocols_serialization/server",
            "-S", "ble",
        ],
        Some(&external_dir),
    )?;

    println!("Building CGM peripheral sample...");
    run_cmd(
        west_str,
        &[
            "build", "-b", "nrf52_bsim", "-p", "always",
            "--build-dir", "build/cgm_peripheral_sample",
            "nrf/samples/bluetooth/peripheral_cgms",
        ],
        Some(&external_dir),
    )?;

    fs::copy(
        external_dir.join("build/zephyr_server_app/server/zephyr/zephyr.exe"),
        external_dir.join("tools/bsim/bin/zephyr_rpc_server_app"),
    )?;
    fs::copy(
        external_dir.join("build/cgm_peripheral_sample/peripheral_cgms/zephyr/zephyr.exe"),
        external_dir.join("tools/bsim/bin/cgm_peripheral_sample"),
    )?;

    println!("Done. Build artifacts copied to external/tools/bsim/bin/");
    Ok(())
}

// ── Sim-management subcommands ────────────────────────────────────────────────

/// Return the value that follows `flag` in `args`, if present.
fn parse_sim_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

/// `cargo xtask start-sim` — launch the full simulation stack in the background.
///
/// Spawns PHY + `zephyr_rpc_server_app` + `cgm_peripheral_sample` + `socat`,
/// waits until the UNIX socket at `<sim_dir>/<sim_id>.sock` is connectable,
/// then prints the socket path and exits, leaving all child processes running.
fn cmd_start_sim(sim_id: &str, sim_dir: &Path) -> Result<()> {
    // Delegate all process spawning + PTY discovery + socat bridging to the
    // shared library function.  It panics on hard failures (binary not found,
    // PTY timeout), which is acceptable for a CLI command.
    let (processes, socket_path) =
        crate::spawn_zephyr_rpc_server_with_socat(sim_dir, sim_id);

    // Wait until socat is actually listening on the socket before we exit.
    // socat needs a moment after spawning before it accepts connections.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if UnixStream::connect(&socket_path).is_ok() {
            break;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out waiting for socket {} to become connectable",
                socket_path.display()
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // IMPORTANT: prevent TestProcesses::drop() from calling kill_all().
    // Without this the Drop impl would kill every child process the instant
    // cmd_start_sim returns, tearing down the sim before the caller can use it.
    std::mem::forget(processes);

    println!("{}", socket_path.display());
    Ok(())
}

/// `cargo xtask stop-sim` — kill all processes belonging to a running simulation.
fn cmd_stop_sim(sim_id: &str) -> Result<()> {
    crate::kill_stale_sim_processes(sim_id);
    println!("Stopped simulation '{sim_id}'");
    Ok(())
}

// ── BabbleSim runner ─────────────────────────────────────────────────────────

fn bsim_ld_library_path() -> String {
    match env::var("LD_LIBRARY_PATH") {
        Ok(existing) => format!("external/tools/bsim/lib:{existing}"),
        Err(_) => "external/tools/bsim/lib".to_string(),
    }
}

fn spawn_in_bsim_bin(sim_id: &str, exe: &str, args: &[&str]) -> Result<Child> {
    let bsim_bin = Path::new("external/tools/bsim/bin");
    Command::new(exe)
        .args(args)
        .current_dir(bsim_bin)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("BSIM_OUT_PATH", "external/tools/bsim")
        .env("BSIM_COMPONENTS_PATH", "external/tools/bsim/components")
        .env("LD_LIBRARY_PATH", bsim_ld_library_path())
        .spawn()
        .map_err(|e| format!("Failed to spawn '{exe}' for sim '{sim_id}': {e}").into())
}

fn pkill_sim(sim_id: &str) {
    for process in ["bs_2G4_phy_v1", "zephyr_rpc_server_app", "cgm_peripheral_sample"] {
        let pattern = format!("{process} -s={sim_id}");
        let _ = Command::new("pkill").args(["-f", &pattern]).status();
        let _ = Command::new("pkill").args(["-9", "-f", &pattern]).status();
    }
}

fn generate_sim_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let pid = std::process::id();
    format!("sim_{:08x}", nanos ^ (pid << 16))
}

fn run_bsim(nrf_rpc_server: bool, cgm_peripheral: bool) -> Result<()> {
    let (run_nrf, run_cgm) = if !nrf_rpc_server && !cgm_peripheral {
        (true, true)
    } else {
        (nrf_rpc_server, cgm_peripheral)
    };

    let sim_id = generate_sim_id();
    pkill_sim(&sim_id);

    let _ = fs::remove_dir_all(format!(
        "/tmp/bs_{}/{}",
        env::var("USER").unwrap_or_default(),
        &sim_id
    ));

    let device_count = (run_nrf as u32) + (run_cgm as u32);

    const SEP: &str = "────────────────────────────────────────────────────────────";

    println!("  Starting PHY simulator...");
    let _phy = spawn_in_bsim_bin(
        &sim_id,
        "./bs_2G4_phy_v1",
        &[
            &format!("-s={sim_id}"),
            &format!("-D={device_count}"),
            "-sim_length=86400e6",
        ],
    )?;

    let nrf_device_idx: u32 = 0;
    let cgm_device_idx: u32 = run_nrf as u32;

    let mut nrf_proc = if run_nrf {
        println!("  Starting nRF RPC server (device {nrf_device_idx})...");
        Some(spawn_in_bsim_bin(
            &sim_id,
            "./zephyr_rpc_server_app",
            &[
                &format!("-s={sim_id}"),
                &format!("-d={nrf_device_idx}"),
                "-uart0_pty",
                "-uart_pty_pollT=1000",
            ],
        )?)
    } else {
        None
    };

    let mut cgm_proc = if run_cgm {
        println!("  Starting CGM peripheral (device {cgm_device_idx})...");
        let cgm_log = fs::File::create("external/tools/bsim/bin/cgm_peripheral_sample.log")?;
        Some(
            Command::new("./cgm_peripheral_sample")
                .args([&format!("-s={sim_id}"), &format!("-d={cgm_device_idx}")])
                .current_dir("external/tools/bsim/bin")
                .stdin(Stdio::null())
                .stdout(cgm_log.try_clone()?)
                .stderr(cgm_log)
                .env("BSIM_OUT_PATH", "external/tools/bsim")
                .env("BSIM_COMPONENTS_PATH", "external/tools/bsim/components")
                .env("LD_LIBRARY_PATH", bsim_ld_library_path())
                .spawn()?,
        )
    } else {
        None
    };

    let mut device_list = Vec::new();
    if run_nrf { device_list.push(format!("nrf-rpc-server [d={nrf_device_idx}]")); }
    if run_cgm { device_list.push(format!("cgm-peripheral [d={cgm_device_idx}]")); }
    let device_str = device_list.join(", ");

    println!();
    println!("{SEP}");
    println!("  Simulation ID : {sim_id}");
    println!("  Devices       : {device_str}");
    println!("  Duration      : 86400 s  (~24 h simulated, ~39 s real time)");
    println!("{SEP}");

    if run_nrf {
        println!();
        println!("  To test RX, run in another terminal:");
        println!();
        println!("    socat UNIX-LISTEN:/tmp/nrf_rpc_server.sock,fork /dev/pts/XX,raw,echo=0");
        println!("    printf '\\x04\\x00\\xff\\x00\\xff\\x00\\x62\\x74\\x5f\\x72\\x70\\x63' \\");
        println!("      | socat - UNIX-CONNECT:/tmp/nrf_rpc_server.sock");
    }

    println!();
    println!("  Press Ctrl+C to stop.");
    println!();

    if let Some(ref mut proc) = nrf_proc {
        let status = proc.wait()?;
        if let Some(ref mut cgm) = cgm_proc {
            let _ = cgm.kill();
        }
        if !status.success() {
            return Err(format!("zephyr_rpc_server_app exited with status: {status}").into());
        }
    } else if let Some(ref mut proc) = cgm_proc {
        let status = proc.wait()?;
        if !status.success() {
            return Err(format!("cgm_peripheral_sample exited with status: {status}").into());
        }
    }

    Ok(())
}
