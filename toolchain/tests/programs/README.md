To see list of all available built-in targets:

```bash
rustc --print target-list
```

We will currently use the risc0 target until we fork Rust to provide our own RISC-V target.

WARNING: to prevent from building for your host machine, make sure you do not have `rustflags = ["-Ctarget-cpu=native"]` in your `~/.cargo/config.toml`.

Build example with full command:

```bash
cargo +nightly build -Z build-std=alloc,core,proc_macro,panic_abort --target riscv32im-risc0-zkvm-elf --example fibonacci
```

Also works with just `cargo +nightly build` because we have a `.cargo/config.toml` that specifies the target and unstable build features (if uncommented).

After this the ELF will be found via

```bash
file target/riscv32im-risc0-zkvm-elf/debug/examples/axvm-fibonacci-program
target/riscv32im-risc0-zkvm-elf/debug/examples/axvm-fibonacci-program: ELF 32-bit LSB executable, UCB RISC-V, soft-float ABI, version 1 (SYSV), statically linked, with debug_info, not stripped
```

To disassemble the ELF to read the instructions, [install cargo-binutils](https://github.com/rust-embedded/cargo-binutils) and run

```bash
rust-objdump -d target/riscv32im-risc0-zkvm-elf/debug/examples/axvm-fibonacci-program
```

where `-d` is short for `--disassemble`.
