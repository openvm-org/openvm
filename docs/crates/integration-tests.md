# How to write integration tests for an extension

Make a `test` crate inside the extension folder. As an example, here is the structure of the `rv32im-extension-test` crate:

```
extensions/rv32im/tests/
├── Cargo.toml
├── openvm.toml
├── src
│   └── lib.rs
├── programs
│   └── Cargo.toml
│   └── examples
│       └── example1.rs
│       └── example2.rs
│       └── ...
```

The `examples` folder contains the test programs in `rust`. 

`fibonacci.rs` example:
```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

pub fn main() {
    let n = core::hint::black_box(1 << 10);
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    if a == 0 {
        panic!();
    }
}
```


And then to `transpile`, `run`, and `prove` the above program, in the `src/lib.rs` file, you can do:

```rust
#[test]
fn test_fibonacci_prove() -> Result<()> {
    let elf = build_example_program_at_path(get_programs_dir!(), "fibonacci")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let config = Rv32IConfig::default();
    air_test(config, exe, vec![]);
    Ok(())
}
```

Note: If the crate with example is not in `./programs`, specify the relative path with `get_programs_dir!("path to the programs crate")`. 

To build the program with CLI, you can go to the `extensions/rv32im/tests/programs` folder and do:

```bash
cargo openvm build --example --name example_name
```
Refer to the [overview of extensions](../../book/src/custom-extensions/overview.md) to see what needs to go into the `openvm.toml` file.
