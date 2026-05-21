# babble-bridge

BabbleSim + Zephyr nRF RPC simulation bridge. Provides:

- **Test harness** — spawn a full BabbleSim simulation from Rust integration tests
- **xtask CLI** — setup, sim lifecycle, and Docker management commands
- **Programmatic API** — call setup and spawn functions directly from `build.rs` or code

---

## Quickstart for downstream crate authors

### 1. Add the dependency

Root `Cargo.toml`:

```toml
[workspace]
members = ["your-app", "xtask"]

[workspace.dependencies]
babble-bridge = "0.1.3"
```

Your crate's `Cargo.toml`:

```toml
[dev-dependencies]
babble-bridge.workspace = true
```

### 2. Create an xtask crate

```bash
cargo init xtask
```

`xtask/Cargo.toml`:

```toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"

[dependencies]
babble-bridge.workspace = true
```

`xtask/src/main.rs`:

```rust
fn main() {
    babble_bridge::xtask::cli_main();
}
```

`.cargo/config.toml`:

```toml
[alias]
xtask = "run -p xtask --"
```

---

## Platform setup

### Linux (native or inside a container)

```bash
cargo xtask zephyr-setup --prebuilt   # download Linux BabbleSim + Zephyr binaries (~30 s)
```

Also creates `tests/sockets/` with restricted permissions (`0700`).

### macOS

BabbleSim only runs on Linux. Use the `--container` flag — it manages a
persistent Docker container for you:

```bash
cargo xtask start-sim --container     # builds image if needed, starts sim in container
```

On first run per workspace this also runs `zephyr-setup --prebuilt` inside the container
to fetch Linux binaries.

---

## Simulation lifecycle

| Command | Description |
|---------|-------------|
| `cargo xtask start-sim` | Start PHY + Zephyr RPC server + CGM peripheral + socat bridge (Linux only) |
| `cargo xtask start-sim --container` | Same, but runs inside a managed container (macOS) |
| `cargo xtask stop-sim` | Kill simulation processes and clean up BabbleSim IPC |
| `cargo xtask tail-logs --log-dir <path>` | Follow log files live (Ctrl+C to stop); add `--container` on macOS |
| `cargo xtask clean-sockets` | Remove all `*.sock` files from `tests/sockets/` |

Options for `start-sim`:

```
--sim-id <id>       Socket name and BabbleSim identifier (default: sim)
--sim-dir <path>    Directory for the socket file (default: <workspace>/tests/sockets)
--container         Build image if needed and run inside a container (macOS)
--log-stream        Also stream all process output to the terminal with [label] prefixes
--log-dir <path>    Directory for log files (default: <sim-dir>/logs);
                    writes rpc-server.log, cgm.log, phy.log; truncated each run
```

`start-sim` **exits immediately** after spawning the simulation processes — it does not block.
The child processes (PHY, Zephyr RPC server, CGM peripheral) keep running in the background
and write their logs directly to files. Logging is on by default; pass `--log-dir` only to
override the location.

### Watching logs

Because `start-sim` exits right away, `--log-stream` only shows the brief startup window.
For sustained log viewing, use `tail-logs` after starting the sim.

**Linux:**

```bash
# Terminal 1 — start the sim (returns immediately)
cargo xtask start-sim --sim-id sim
# Logs are now being written to tests/sockets/logs/

# Terminal 2 — follow the logs live (Ctrl+C to stop)
cargo xtask tail-logs --log-dir tests/sockets/logs
```

Or chain them in one terminal:

```bash
cargo xtask start-sim --sim-id sim && \
  cargo xtask tail-logs --log-dir tests/sockets/logs
```

**macOS (container):**

```bash
# Terminal 1 — start the sim inside Docker (returns immediately)
cargo xtask start-sim --container --sim-id sim
# Logs are written to /workspace/tests/sockets/logs/ inside the container

# Terminal 2 — follow the logs from macOS
cargo xtask tail-logs --log-dir /workspace/tests/sockets/logs --container
```

The three log files written are:

| File | Process |
|------|---------|
| `rpc-server.log` | Zephyr RPC server (stdout + stderr — Zephyr `<inf>` log lines, PTY announcements) |
| `cgm.log` | CGM peripheral sample (stdout + stderr) |
| `phy.log` | BabbleSim PHY (`bs_2G4_phy_v1` stderr) |

All three are **truncated at the start of each `start-sim` run** and written via inherited file
descriptors, so they continue growing even after the `start-sim` process exits.

The socket is created at `tests/sockets/<sim-id>.sock`.

---

## Connecting to the simulation

### From Linux (or inside the container)

```rust
use std::os::unix::net::UnixStream;
let socket = UnixStream::connect("tests/sockets/sim.sock")?;
```

### From macOS

Unix sockets don't cross the OS boundary. `start-sim --container` automatically
starts a TCP bridge inside the container and publishes it to `127.0.0.1` on your Mac:

```
TCP bridge ready: connect from macOS at 127.0.0.1:<port>
```

The port is stable and derived from your workspace path:

```rust
use std::net::TcpStream;
let stream = TcpStream::connect("127.0.0.1:<port>")?;
```

To run code that needs to reach the socket from macOS, use `exec` to run it
inside the container instead:

```bash
cargo xtask exec -- cargo test --test my_integration_test
cargo xtask exec -- cargo run --example my_example
```

---

## Integration tests

```rust
use std::collections::HashSet;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;
use babble_bridge::LogOutput;

let tests_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sockets"));
let (mut processes, socket_path) =
    babble_bridge::spawn_zephyr_rpc_server_with_socat(tests_dir, "my_test", LogOutput::Off);

// socat needs a moment to start listening after spawn — retry until connectable.
let start = std::time::Instant::now();
let socket = loop {
    match UnixStream::connect(&socket_path) {
        Ok(s) => break s,
        Err(_) if start.elapsed() < Duration::from_secs(5) => {
            std::thread::sleep(Duration::from_millis(50));
        }
        Err(e) => panic!("socket never became connectable: {e}"),
    }
};

// Run test logic via `socket`, then assert on Zephyr-side log output:
processes.search_stdout_for_strings(HashSet::from([
    "<inf> nrf_ps_server: Initializing RPC server",
]));
```

> **Note:** `spawn_zephyr_rpc_server_with_socat` returns as soon as socat is
> *spawned*, not when the socket is *connectable*. Always use a retry loop (as
> above) before calling `UnixStream::connect`. The `sim_harness.rs` integration
> tests use a `SimUart::connect` helper that does exactly this.

### LogOutput — controlling where process logs go

| Variant | Behaviour |
|---|---|
| `LogOutput::Off` | Silent — no forwarding (default) |
| `LogOutput::Stream` | Real-time `[label]` prefixed output to terminal (bypasses `cargo test` capture) |
| `LogOutput::WriteToDir(path)` | Writes `rpc-server.log`, `cgm.log`, `phy.log` into `path`; **truncated on every spawn**; child processes hold the file descriptors directly so **logs persist after the parent process exits** |
| `LogOutput::Both(path)` | Same persistent file logging as `WriteToDir`, plus real-time terminal streaming while the parent is alive |

```rust
// Verbose during interactive debugging:
babble_bridge::spawn_zephyr_rpc_server_with_socat(tests_dir, "my_test", LogOutput::Stream);

// Persistent log files, cleared on each run:
babble_bridge::spawn_zephyr_rpc_server_with_socat(
    tests_dir, "my_test",
    LogOutput::WriteToDir("tests/sockets/logs".into()),
);
```

> **Note:** `features = ["sim-log"]` is now a no-op. Replace it with `LogOutput::Stream` at the call site.

Tests require Linux — run them inside the container on macOS:

```bash
cargo xtask exec -- cargo test --test my_integration_test
```

---

## Docker commands

| Command | Description |
|---------|-------------|
| `cargo xtask docker-build` | Build the dev-container image |
| `cargo xtask docker-attach` | Open an interactive shell in the container |
| `cargo xtask docker-run -- <cmd>` | Run a one-off command in a fresh container |
| `cargo xtask exec -- <cmd>` | Run a command in the persistent sim container |

`docker-run` creates a fresh container each time (no running sim).
`exec` targets the persistent container started by `start-sim --container`,
where the simulation socket is reachable.
