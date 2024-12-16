## Testing the program

### Running on the host machine

To test the program on the host machine, one can use the `std` feature: `cargo run --features std`. So for example to run the [fibonacci program](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs/fibonacci):

```bash
printf '\xA0\x86\x01\x00\x00\x00\x00\x00' | cargo run --features std
```

### Running with the OpenVM runtime

For more information on building, transpiling, running, generating proofs, and verifying proofs with the CLI, see the [CLI](../writing-apps/overview.md)section. To do the same with the SDK, see the [SDK](sdk.md) section.

## Troubleshooting

todo

## FAQ

todo