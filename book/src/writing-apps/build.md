# Compiling a Program

First let's define some key terms used in cross-compilation:

- **host** - the machine you're compiling and/or proving on. Note that one can compile and prove on different machines, but they are both called _host_ as they are traditional machine architectures.
- **guest** - the executable to be run in a different VM architecture (e.g. the OpenVM runtime, or Android app).

The command `cargo openvm build` compiles the program on host to an executable for guest target.
It first compiles the program normally on your _host_ platform with RISC-V and then transpiles it to a different target. See here for some explanation of [cross-compilation](https://rust-lang.github.io/rustup/cross-compilation.html).
Right now we use `riscv32im-risc0-zkvm-elf` target which is available in the [Rust toolchain](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html), but we will contribute an OpenVM target to Rust in the future.

## Build flags

The following flags are available for the `cargo openvm build` command:

- `--manifest-dir <MANIFEST_DIR>`

  **Description**: Specifies the directory containing the `Cargo.toml` file for the guest code.

  **Default**: The current directory (`.`).

  **Usage Example**: If your `Cargo.toml` is located in `my_project/`, you can run:

  ```bash
  cargo openvm build --manifest-dir my_project
  ```

  This ensures the build command is executed in that directory.

- `--target-dir <TARGET_DIR>`

  **Description**: Specifies the directory where the guest binary will be built. If not specified, the default target directory is used.

  **Default**: The `target` directory in the package root directory.

  **Usage Example**: To build the guest binary in the `my_target` directory:

  ```bash
  cargo openvm build --target-dir my_target
  ```

- `--features <FEATURES>`

  **Description**: Passes a list of feature flags to the Cargo build process. These flags enable or disable conditional compilation features defined in your `Cargo.toml`.

  **Usage Example**: To enable the `my_feature` feature:

  ```bash
  cargo openvm build --features my_feature
  ```

- `--bin <NAME>`

  **Description**: Restricts the build to the binary target with the given name, similar to `cargo build --bin <NAME>`. If your project has multiple target types (binaries, libraries, examples, etc.), using `--bin <NAME>` narrows down the build to the binary target with the given name.

  **Usage Example**:

  ```bash
  cargo openvm build --bin my_bin
  ```

- `--example <NAME>`

  **Description**: Restricts the build to the example target with the given name, similar to `cargo build --example <NAME>`. Projects often include code samples or demos under the examples directory, and this flag focuses on compiling a specific example.

  **Usage Example**:

  ```bash
  cargo openvm build --example my_example
  ```

- `--no-transpile`

  **Description**: After building the guest code, doesn't transpile the target ELF into an OpenVM-compatible executable (by default it does).

  **Usage Example**:

  ```bash
  cargo openvm build --no-transpile
  ```

- `--config <CONFIG>`

  **Description**: Specifies the path to a .toml configuration file that defines which VM extensions to use.

  **Default**: `./openvm.toml` if `--config` flag is not provided.

  **Usage Example**:

  ```bash
  cargo openvm build --config path/to/openvm.toml
  ```

  This allows you to customize the extensions. Currently the CLI only supports known extensions listed in the [Using Existing Extensions](../custom-extensions/overview.md) section. To use other extensions, use the [SDK](../advanced-usage/sdk.md).

- `--exe-output <EXE_OUTPUT>`

  **Description**: Sets the output path for the transpiled program.

  **Default**: `./openvm/app.vmexe` if `--exe-output` flag is not provided.

  **Usage Example**: To specify a custom output filename:

  ```bash
  cargo openvm build --exe-output ./output/custom_name.vmexe
  ```

- `--profile <PROFILE>`

  **Description**: Determines the build profile used by Cargo. Common profiles are dev (faster builds, less optimization) and release (slower builds, more optimization).

  **Default**: release

  **Usage Example**:

  ```bash
  cargo openvm build --profile dev
  ```

- `--help`

  **Description**: Prints a help message describing the available options and their usage.

  **Usage Example**:

  ```bash
  cargo openvm build --help
  ```

## Running a Program

After building and transpiling a program, you can execute it using the `run` command. The `run` command has the following arguments:

```bash
cargo openvm run
    --exe <path_to_transpiled_program>
    --config <path_to_app_config>
    --input <path_to_input>
```

If `--exe` and/or `--config` are not provided, the command will search for these files in `./openvm/app.vmexe` and `./openvm.toml` respectively. If `./openvm.toml` is not present, a default configuration will be used.

If your program doesn't require inputs, you can (and should) omit the `--input` flag.
