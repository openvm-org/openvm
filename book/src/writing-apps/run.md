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

Many of the options for `cargo openvm run` will be passed to `cargo openvm build` if `--exe` is not specified. For more information on `build` (or `run`'s **Feature Selection**, **Compilation**, **Output**, **Display**, and/or **Manifest** options) see [Compiling](./writing-apps/build.md).

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
