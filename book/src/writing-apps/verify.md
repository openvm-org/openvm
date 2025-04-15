# Verifying Proofs

## Application Level

Verifying a proof at the application level requires both the proof and application verifying key.

```bash
cargo openvm verify app
    --app_vk <path_to_app_vk>
    --proof <path_to_proof>
```

If you omit `--app_vk` and/or `--proof`, the command will search for those files at `./openvm/app.vk` and `./openvm/app.proof` respectively.

Once again, if you omitted `--output` and `--vk_output` in the `keygen` and `prove` commands, you can omit `--app_vk` and `--proof` in the `verify` command.

## EVM Level

EVM level proof setup requires large amounts of computation and memory (~200GB). It is recommended to run this process on a server.

### Install Solc

Install  `solc` `0.8.19` using `svm`

```bash
# Install svm
cargo install --version 0.5.7 svm-rs
# Add the binary to your path
export PATH="$HOME/.cargo/bin:$PATH"

# Install solc 0.8.19
svm install 0.8.19
svm use 0.8.19
```

### Generating the Aggregation Proving Key and EVM Verifier Contract

The workflow for generating an end-to-end EVM proof requires first generating an aggregation proving key and EVM verifier contract. This can be done by running the following command:

```bash
cargo openvm setup
```

Note that `cargo openvm setup` may attempt to download other files (i.e. KZG parameters) from an AWS S3 bucket into `~/.openvm/`.

This command can take ~20mins on a `m6a.16xlarge` instance due to the keygen time.

Upon a successful run, the command will write the files

- `agg.pk`
- `halo2/Halo2Verifier.sol`
- `halo2/OpenVmHalo2Verifier.sol`
- `halo2/interfaces/IOpenVmHalo2Verifier.sol`
- `halo2/verifier.bytecode.json`

to `~/.openvm/`, where `~` is the directory specified by environment variable `$HOME`. Every command that requires these files will look for them in this directory.

The `agg.pk` contains all aggregation proving keys necessary for aggregating to a final EVM proof.
The `OpenVmHalo2Verifier.sol` file contains a Solidity contract to verify the final EVM proof. The contract is named `OpenVmHalo2Verifier` and it implements the `IOpenVmHalo2Verifier` interface.

```solidity
interface IOpenVmHalo2Verifier {
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit, bytes32 appVmCommit)
        external
        view;
}
```

In addition, the command outputs a JSON file `verifier.bytecode.json` of the form

```json
{
    "sol_compiler_version": "0.8.19",
    "sol_compiler_options": "",
    "bytecode": "0x..."
}
```

where `sol_compiler_version` is the Solidity compiler version used to compile the contract (currently fixed to `0.8.19`),
`sol_compiler_options` are additional compiler options used, and
`bytecode` is the compiled EVM bytecode as a hex string.

> ⚠️ **WARNING**
>
> If the `$HOME` environment variable is not set, this command may fail.
>
> This command requires very large amounts of computation and memory (~200 GB).

## Generating and Verifying an EVM Proof

To generate and verify an EVM proof, you need to run the following commands:

```bash
cargo openvm prove evm --input <path_to_input>
cargo openvm verify evm --proof <path_to_proof>
```

If `proof` is omitted, the `verify` command will search for the proof at `./openvm/evm.proof`.

### EVM Proof: JSON Format

The EVM proof is written to `evm.proof` as a JSON of the following format:

```json
{
  "app_exe_commit": "0x..",
  "app_vm_commit": "0x..",
  "user_public_values": "0x..",
  "proof_data": {
    "accumulator": "0x..",
    "proof": "0x.."
  },
}
```

where each field is a hex string. We explain what each field represents:

- `app_exe_commit`: `32` bytes for the commitment of the app executable.
- `app_vm_commit`: `32` bytes for the commitment of the app VM configuration.
- `user_public_values`: concatenation of 32 byte chunks for user public values. The number of user public values is a configuration parameter.
- `accumulator`: `12 * 32` bytes representing the KZG accumulator of the proof, where the proof is from a SNARK using the KZG commitment scheme.
- `proof`: The rest of the proof required by the SNARK as a hex string of `43 * 32` bytes.

### EVM Proof: Calldata Format

The `cargo openvm verify evm` command reads the EVM proof from JSON file and then simulates the call to the verifier contract using [Revm](https://github.com/bluealloy/revm/tree/main). This function should only be used for testing and development purposes but not for production.

To verify the EVM proof in an EVM execution environment, the entries of the JSON can be passed as function arguments for the `verify` function, where the `proofData` argument is constructed by `proofData = abi.encodePacked(accumulator, proof)`.
