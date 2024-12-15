# Generating Proofs
## SDK
After building and transpiling a program, you can generate a proof using the SDK. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.
```rust
// 0. VmConfig and VmExe from building and transpiling
let vm_confing = ...
let exe = ...

// 1. Set app configuration
let app_log_blowup = 2;
let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
let app_config = AppConfig { ... };

// 2. Commit the exe
let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;

// 3. Generate an AppProvingKey
let app_pk = sdk.app_keygen(app_config)?;

// 4. Format your input into StdIn
let my_input = SomeStruct; // anything that can be serialized
let mut stdin = StdIn::default();
stdin.write(&my_input);

// 5. Generate a proof
let proof = sdk.generate_app_proof(app_pk, app_committed_exe, stdin)?;
```
## CLI
Generating a proof using the CLI is simple - given an app configuration TOML file, you first need to generate a proving and verifying key:
```bash
cargo openvvm keygen --config <path_to_app_config> --output <path_to_app_pk> --vk_output <path_to_app_vk>
```
If `--output` and `--vk_output` are not provided, the keys will be written to default locations. This would look like:
```bash
cargo openvvm keygen --config <path_to_app_config>
```

After generating the keys, you can generate a proof by running:
```bash
cargo openvvm prove app --app_pk <path_to_app_pk> --exe <path_to_compiled_program> --input <path_to_input> --output <path_to_output>
```
Again, if `--output` is not provided, the proof will be written to a default location. If you ommitted `--output` in the keygen command, we recommend you omit `app_pk` in the prove command.
```bash
cargo openvvm prove app --exe <path_to_compiled_program> --input <path_to_input>
```
