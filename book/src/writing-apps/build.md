# Compiling a Program

First let's define some key terms used in cross-compilation:

- **host** - the machine you're compiling and/or proving on. Note that one can compile and prove on different machines, but they are both called _host_ as they are traditional machine architectures.
- **guest** - the executable to be run in a different VM architecture (e.g. the OpenVM runtime, or Android app).

The command `cargo openvm build` compiles the program on host to an executable for guest target.
It first compiles the program normally on your _host_ platform with RISC-V and then transpiles it to a different target. See here for some explanation of [cross-compilation](https://rust-lang.github.io/rustup/cross-compilation.html).
Right now we use `riscv32im-risc0-zkvm-elf` target which is available in the [Rust toolchain](https://doc.rust-lang.org/rustc/platform-support/riscv32im-risc0-zkvm-elf.html), but we will contribute an OpenVM target to Rust in the future.

## Build Flags

The following flags are available for the `cargo openvm build` command. You can run `cargo openvm build --help` for this list within the command line.

Generally, outputs will always be built to the **target directory**, which will either be determined by the manifest path or explicitly set using the `--target-dir` option. By default Cargo sets this to be `<workspace_or_package_root>/target/`. 

OpenVM-specific artifacts will be placed in `${target_dir}/openvm/`, but if `--output-dir` is specified they will be copied to `${output-dir}/` as well.

### OpenVM Options

- `--no-transpile`

  **Description**: Skips transpilation into an OpenVM-compatible `.vmexe` executable when set.

- `--config <CONFIG>`

  **Description**: Path to the OpenVM config `.toml` file that specifies the VM extensions. By default will search the manifest directory for `openvm.toml`. If no file is found, OpenVM will use a default configuration. Currently the CLI only supports known extensions listed in the [Using Existing Extensions](../custom-extensions/overview.md) section. To use other extensions, use the [SDK](../advanced-usage/sdk.md).

- `--output_dir <OUTPUT_DIR>`

  **Description**: Output directory for OpenVM artifacts to be copied to. Keys will be placed in `${output-dir}/`, while all other artifacts will be in `${output-dir}/${profile}`.

- `--init-file-name <INIT_FILE_NAME>`

  **Description**: Name of the generated initialization file, which will be written into the manifest directory.

  **Default**: `openvm_init.rs`

### Package Selection

As with `cargo build`, default package selection depends on the working directory. If the working directory is a subdirectory of a specific package, then only that package will be built. Else, all packages in the workspace will be built by default.

- `--package <PACKAGES>`

  **Description**: Builds only the specified packages. This flag may be specified multiple times or as a comma-separated list.

- `--workspace`

  **Description**: Builds all members of the workspace (alias `--all`).

- `--exclude <PACKAGES>`

  **Description**: Excludes the specified packages. Must be used in conjunction with `--workspace`. This flag may be specified multiple times or as a comma-separated list.

### Target Selection

By default all package libraries and binaries will be built. To build samples or demos under the `examples` directory, use either the `--example` or `--examples` option.

- `--lib`

  **Description**: Builds the package's library.

- `--bin <BIN>`

  **Description**: Builds the specified binary. This flag may be specified multiple times or as a comma-separated list.

- `--bins`

  **Description**: Builds all binary targets.

- `--example <EXAMPLE>`

  **Description**: Builds the specified example. This flag may be specified multiple times or as a comma-separated list.

- `--examples`

  **Description**: Builds all example targets.

- `--all-targets`

  **Description**: Builds all package targets. Equivalent to specifying `--lib` `--bins` `--examples`.

### Feature Selection

The following options enable or disable conditional compilation features defined in your `Cargo.toml`.

- `-F`, `--features <FEATURES>`

  **Description**: Space or comma separated list of features to activate. Features of workspace members may be enabled with `package-name/feature-name` syntax. This flag may also be specified multiple times.

- `--all-features`

  **Description**: Activates all available features of all selected packages.

- `--no-default-features`

  **Description**: Do not activate the `default` feature of the selected packages.

### Compilation Options

- `--profile <NAME>`

  **Description**: Builds with the given profile. Common profiles are `dev` (faster builds, less optimization) and `release` (slower builds, more optimization). For more information on profiles, see [Cargo's reference page](https://doc.rust-lang.org/cargo/reference/profiles.html).

  **Default**: `release`

### Output Options

- `--target_dir <TARGET_DIR>`

  **Description**: Directory for all generated artifacts and intermediate files. Defaults to directory `target/` at the root of the workspace.

### Display Options

- `-v`, `--verbose`

  **Description**: Use verbose output.

- `-q`, `--quiet`

  **Description**: Do not print Cargo log messages.

- `--color <WHEN>`

  **Description**: Controls when colored output is used.

  **Default**: `always`

### Manifest Options

- `--manifest-path <PATH>`

  **Description**: Path to the guest code Cargo.toml file. By default, `build` searches for the file in the current or any parent directory. The `build` command will be executed in that directory.

- `--ignore-rust-version`

  **Description**: Ignores rust-version specification in packages.

- `--locked`

  **Description**: Asserts the same dependencies and versions are used as when the existing Cargo.lock file was originally generated.

- `--offline`

  **Description**: Prevents Cargo from accessing the network for any reason.

- `--frozen`

  **Description**: Equivalent to specifying both `--locked` and `--offline`.
