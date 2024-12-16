# Cross-Compilation

First let's define some key terms used in cross-compilation:
- **host** - the machine you're compiling and/or proving on. Note that one can compile and prove on different machines, but they are both called *host* as they are traditional machine architectures.
- **guest** - the executable to be run in a different VM architecture (e.g. the OpenVM runtime, or Android app).

There are multiple things happening in the `cargo openvm build` command as in the section [here](./write-program.md). In short, this command compiles on host to an executable for guest target.
It first compiles the program normally on your *host* platform with RISC-V and then transpiles it to a different target. See here for some explanation of [cross-compilation](https://rust-lang.github.io/rustup/cross-compilation.html).
Right now we use `riscv32im-risc0-zkvm-elf` target which is available in the [Rust toolchain](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html), but we will contribute an OpenVM target to Rust in the future.
