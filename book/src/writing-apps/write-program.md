# Writing a Program

## Writing a guest program

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `openvm::entry!` macro. You should still create a `main` function, and then add `openvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with testing when `std` feature is enabled (see below), you should still name it `main`.

To support host machine execution, the top of your guest program should have:

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

Some examples of guest programs are in the [benchmarks/programs](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs) directory.

### no-std

Although it's ususally ok to use std (like in quickstart), not all std functionalities are supported (e.g., randomness). There might be unexpected runtime errors if one uses std, so it is recommended you develop no_std libraries if possible to reduce surprises.

### reading input

`openvm::io::read_vec` and `openvm::io::read` will read from stdin. `read` takes the next vec and deserialize it into a generic type `T`, so one should specify the type when calling it:
```rust
let n: u64 = read();
```
`read_vec` reads the size of the vec first (`size: u32`) and then `size` bytes into a vector.

## Testing the program

### Running on the host machine

To test the program on the host machine, one can use the `std` feature: `cargo run --features std`. So for example to run the [fibonacci program](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs/fibonacci):

```bash
printf '\xA0\x86\x01\x00\x00\x00\x00\x00' | cargo run --features std
```

### Running with the OpenVM runtime

*TODO*: point to how to install CLI

First to build the guest program:
```
cargo axiom build
```

This compiles the guest program into an [ELF](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format) that can be found at `target/riscv32im-risc0-zkvm-elf` directory.
Next, a host program is needed to run the ELF with openvm runtime. This is where one can configure the openvm with different parameters. There are a few steps:

```rust
use openvm::transpiler::{openvm_platform::memory::MEM_SIZE, elf::Elf};
use openvm_circuit::arch::instructions::exe::OpenVmExe
use openvm_circuit::arch::VmExecutor;
use openvm_sdk::{config::SdkVmConfig, Sdk, StdIn};

let sdk = Sdk;
// 1. Build the vm config with the extensions needed.
// TODO: link to extension
let vm_config = SdkVmConfig::builder()
    .system(Default::default())
    .rv32i(Default::default())
    .io(Default::default())
    .build();

// 2. Load the ELF
let elf = Elf::decode("your_path_to_elf", MEM_SIZE as u32)?;
let exe = OpenVmExe::from_elf(elf, vm_config.transpiler()).unwrap();

// 3. Prepare the input data
let my_input = SomeStruct; // anything that can be serialized
let mut stdin = StdIn::default();
stdin.write(StdIn::from_bytes(my_input.as_bytes()));

// 4. Run the program
let executor = VmExecutor::<_, _>::new(vm_config);
executor.execute(exe, stdin)?;
```
Some example host programs can be found [here](https://github.com/openvm-org/openvm/tree/main/benchmarks/src/bin).

### Generating to prove

To generate a proof besides executing the program, instead of using `executor` above (step 4), do the following:
```rust
// Some additional configuration.
let app_log_blowup = 2;
let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
let app_config = AppConfig { ... };

// Keygen and prove
let app_pk = sdk.app_keygen(app_config)?;
let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;
let mut app_prover =
    AppProver::new(app_pk.app_vm_pk.clone(), app_committed_exe)
        .with_program_name(program_name);
let proof = app_prover.generate_app_proof(stdin);
let app_vk = app_pk.get_vk();
sdk.verify_app_proof(&app_vk, &proof)?;
```

## Troubleshooting

todo

## FAQ

todo