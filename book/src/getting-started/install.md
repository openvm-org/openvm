# Install

To use OpenVM for generating proofs, you must install the OpenVM cli tool `cargo-openvm`.

`cargo-openvm` can be installed in two different ways. You can either install via git url or build from source.

## Install Via Git Url (Recommended)

```bash
cargo install --git ssh://git@github.com/openvm-org/openvm.git cargo-openvm --branch main --locked --force
```

This will globally install `cargo-openvm`. You can validate a successful installation with:

```bash
cargo openvm --version
```

## Build from source

To build from source, you will need the nightly toolchain. You can install it with:

```bash
rustup toolchain install nightly
```

Then, clone the repository and begin the installation.

```bash
git clone https://github.com/openvm-org/openvm.git
cd openvm
cargo install --force --path crates/cli
```

This will globally install `cargo-openvm`. You can validate a successful installation with:

```bash
cargo openvm --version
```
