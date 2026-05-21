Integration tests for the OpenVM toolchain: builds guest programs and exercises the transpiler/ELF decoder on the resulting binaries.

Guest builds in these tests go through `openvm-build`'s `build_guest_package`, which invokes `cargo build` under the prebuilt `openvm-1.94.0` toolchain for the `riscv64im-unknown-openvm-elf` target. Install the toolchain first:

```bash
cargo openvm toolchain install
```

Then run the tests as usual:

```bash
cargo nextest run --cargo-profile=fast -p openvm-toolchain-tests
```

To inspect a built guest ELF directly, the binary lives under `target/riscv64im-unknown-openvm-elf/release/`. Disassemble with [cargo-binutils](https://github.com/rust-embedded/cargo-binutils):

```bash
rust-objdump -d target/riscv64im-unknown-openvm-elf/release/<example-name>
```
