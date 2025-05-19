# Running a Program

After building and transpiling a program, you can execute it using the `run` command. For example, you can call:

```bash
cargo openvm run
    --exe <path_to_transpiled_program>
    --config <path_to_app_config>
    --input <path_to_input>
```

If `--exe` is not provided, OpenVM will call `build` prior to attempting to run the executable. Note that only one executable may be run, so if your project contains multiple targets you will have to specify which one to run using the `--bin` or `--example` flag.

If your program doesn't require inputs, you can (and should) omit the `--input` flag.

## Run Flags

Many of the options for `cargo openvm run` will be passed to `cargo openvm build` if `--exe` is not specified. For more information on `build`, see [Compiling](./writing-apps/build.md).

### OpenVM Options

- `--exe <EXE>`

  **Description**: Path to the OpenVM executable, if specified `build` will be skipped.

- `--config <CONFIG>`

  **Description**: Path to the OpenVM config `.toml` file that specifies the VM extensions. By default will search the manifest directory for `openvm.toml`. If no file is found, OpenVM will use a default configuration. Currently the CLI only supports known extensions listed in the [Using Existing Extensions](../custom-extensions/overview.md) section. To use other extensions, use the [SDK](../advanced-usage/sdk.md).

- `--output_dir <OUTPUT_DIR>`

  **Description**: Output directory for OpenVM artifacts to be copied to. Keys will be placed in `${output-dir}/`, while all other artifacts will be in `${output-dir}/${profile}`.

- `--input <INPUT>`

  **Description**: Input to the OpenVM program, or a hex string.

- `--init-file-name <INIT_FILE_NAME>`

  **Description**: Name of the generated initialization file, which will be written into the manifest directory.

  **Default**: `openvm_init.rs`

### Package Selection

- `--package <PACKAGES>`

  **Description**: The package to run, by default the package in the current workspace.

### Target Selection

Only one target may be built and run. 

- `--bin <BIN>`

  **Description**: Runs the specified binary.

- `--example <EXAMPLE>`

  **Description**: Runs the specified example.

### Feature Selection

- `-F`, `--features <FEATURES>`

  **Description**: Space or comma separated list of features to activate. Features of workspace members may be enabled with `package-name/feature-name` syntax. This flag may also be specified multiple times.

- `--all-features`

  **Description**: Activates all available features of all selected packages.

- `--no-default-features`

  **Description**: Do not activate the `default` feature of the selected packages.

### Compilation Options

- `--profile <NAME>`

  **Description**: Runs with the given profile. Common profiles are `dev` (faster builds, less optimization) and `release` (slower builds, more optimization). For more information on profiles, see [Cargo's reference page](https://doc.rust-lang.org/cargo/reference/profiles.html).

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

  **Description**: Path to the guest code Cargo.toml file. By default, `run` searches for the file in the current or any parent directory. The `run` command will be executed in that directory.

- `--ignore-rust-version`

  **Description**: Ignores rust-version specification in packages.

- `--locked`

  **Description**: Asserts the same dependencies and versions are used as when the existing Cargo.lock file was originally generated.

- `--offline`

  **Description**: Prevents Cargo from accessing the network for any reason.

- `--ignore-rust-version`

  **Description**: Ignores rust-version specification in packages.

## Examples

### Running a Specific Binary

```bash
cargo openvm run --bin bin_name
```

### Skipping Build Using `--exe`

```bash
cargo openvm build --output-dir ./my_output_dir
cargo openvm run --exe ./my_output_dir/bin_name.vmexe
```
