# Contributor Setup

To get `rustfmt` to work with our nightly options, add the following to your IDE settings file (e.g., `.vscode/settings.json`):

```json
{  
  "rust-analyzer.rustfmt.extraArgs": [
    "+nightly"
  ],
}
```

## Development without CUDA

No additional settings need to be set to develop without CUDA, as the `cuda` feature is disabled by default throughout OpenVM. Please feature gate code that is **not** compatible with CUDA using the `#[cfg(not(feature = "cuda"))]` Rust attribute.

## Development with CUDA

### Machine Setup

The CUDA crates in this repository should build via `cargo` on machines with [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation) 12.8 or later installed. To check the CUDA toolkit version you have installed, run
```bash
nvcc --version
```
Additionally, ensure that your shell profile or startup script sets the proper [path variables for CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#environment-setup):
```bash
PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Note that `/usr/local/cuda` is a symlink for the latest version of CUDA installed.

### IDE Setup

To enable the `cuda` feature for `rust-analyzer`, add the following to your IDE settings file (e.g., `.vscode/settings.json`):

```json
{
  "rust-analyzer.cargo.features": [
    "cuda"
  ],
}
```

> [!NOTE]
> Note that to build the project from the CLI, you still need to add `--features cuda` to your `cargo` command to enable CUDA.

In addition to `rust-analyzer` for linting Rust code, we recommend installing a `clangd` server for linting CUDA code (note there is a `clangd` VS Code extension that does this). For the `clangd` server to work properly, run

```bash
scripts/generate_clangd.sh
```

to generate a local `.clangd` config file. This file cannot be committed to the repository as it includes local paths.

> [!NOTE]
> Several lints and analyzers are run on every pull request, including `cargo fmt` and `cargo clippy` for Rust code and `clang-tidy` for CUDA code. It is probably worthwhile installing these tools on your device to save development time.
