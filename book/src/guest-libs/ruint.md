# Ruint 

The Ruint guest library is a fork of [ruint](https://github.com/recmo/uint) that allows for patching of U256 operations with logic from [openvm-bigint-guest](../custom-extensions/bigint.md).

## Example matrix multiplication using `U256`

See the full example [here](https://github.com/openvm-org/openvm/blob/main/examples/u256/src/main.rs).

```rust,no_run,noplayground
{{ #include ../../../examples/u256/src/main.rs }}
```

To be able to import the `U256` struct, add the following to your `Cargo.toml` file:

```toml
openvm-ruint = { git = "https://github.com/openvm-org/openvm.git", package = "ruint" }
```

### Example matrix multiplication using `I256`

See the full example [here](https://github.com/openvm-org/openvm/blob/main/examples/i256/src/main.rs).

```rust,no_run,noplayground
{{ #include ../../../examples/i256/src/main.rs }}
```

To be able to import the `I256` struct, add the following to your `Cargo.toml` file:

```toml
openvm-ruint = { git = "https://github.com/openvm-org/openvm.git", package = "ruint" }
```

### Config parameters

For the guest program to build successfully add the following to your `.toml` file:

```toml
[app_vm_config.bigint]
```
