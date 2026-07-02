To see list of all available built-in targets:

```bash
rustc --print target-list
```

We currently use the risc0 target (`riscv32im-risc0-zkvm-elf`) until we contribute our own RISC-V target to Rust.

WARNING: to prevent from building for your host machine, make sure you do not have `rustflags = ["-Ctarget-cpu=native"]` in your `~/.cargo/config.toml`.

Guest programs live in the `programs/examples` directory of each extension test crate (e.g. [`extensions/rv32im/tests/programs/examples`](../../../../extensions/rv32im/tests/programs/examples)). They are compiled to a RISC-V ELF for the risc0 target by the `build_example_program_at_path` utility, which reads back the output ELF file. See the crate's top-level [README.md](../README.md) for how to add and build guest programs.

To disassemble a compiled ELF to read the instructions, [install cargo-binutils](https://github.com/rust-embedded/cargo-binutils) and run

```bash
rust-objdump -d <path-to-elf>
```

where `-d` is short for `--disassemble`.
