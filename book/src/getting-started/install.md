# Install

To use OpenVM for generating proofs, you must install the OpenVM command line tool `cargo-openvm`.

`cargo-openvm` can be installed in two different ways. You can either install via git URL or build from source.

## Option 1: Install Via Git URL (Recommended)

Begin the installation:

```bash
cargo +1.85 install --locked --git http://github.com/openvm-org/openvm.git --tag v1.3.0 cargo-openvm
```

This will globally install `cargo-openvm`. You can validate a successful installation with:

```bash
cargo openvm --version
```

## Option 2: Build from source

To build from source, clone the repository and begin the installation.

```bash
git clone --branch v1.3.0 --single-branch https://github.com/openvm-org/openvm.git
cd openvm
cargo install --locked --force --path crates/cli
```

This will globally install `cargo-openvm`. You can validate a successful installation with:

```bash
cargo openvm --version
```

## Install Rust Toolchain

In order for the `cargo-openvm` build command to work, you must install certain Rust nightly components:

```bash
rustup install nightly-2025-02-14
rustup component add rust-src --toolchain nightly-2025-02-14
```
