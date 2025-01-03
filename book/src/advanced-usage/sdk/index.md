# Using the SDK

While the CLI provides a convenient way to build, prove, and verify programs, you may want more fine-grained control over the process. The OpenVM Rust SDK allows you to customize various aspects of the workflow programmatically.

For more information on the basic CLI flow, see [Overview of Basic Usage](../../writing-apps/overview.md). Writing a guest program is the same as in the CLI.

## Imports and Setup

If you have a guest program and would like to try running the **host program** specified in the next section, you can do so by adding the following imports and setup at the top of the file. You may need to modify the imports and/or the `SomeStruct` struct to match your program.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:dependencies }}
```

## Types of Proofs

The SDK can generate two types of proofs, depending on your specific use case.

- [App Proof](./app-proof.md): Generates a STARK proof of the guest program
- [EVM Proof](./evm-proof.md): Generates a halo2 proof that can be posted on-chain
