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