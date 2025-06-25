# Introduction
A library to facilitate the recursive verification of OpenVM STARK proofs within a guest program.

# Guest
## Function definition Macro
Macro `define_verify_stark_proof` can define a function to verify a stark proof. Users need to specify:
- The function name.
- An ASM file, which contains instructions to verify proofs of **a specific aggregation config**. The ASM file could be generated during the aggregation keygen(`generate_root_verifier_asm` function in SDK).
 
## Function call
Required arguments to call the verify function in a guest program:
- App exe commit
- App VM commit
- Public values
The proof to verify is read from the VM stream. See the host section for more details.

## Restrictions
The verify function overwrites the native address space(address space 4). User should persist any data in the native address space before calling the verify function.

# Host
## Input
The function can verify an internal proof or a leaf proof **of a specific aggregation config**(see [docs](../../docs/specs/continuations.md)).
The proof to verify should prove a successful whole execution, which starts from the initial state and terminates with exit code 0.
Because the proof to verify is an aggregation proof, the underlying App VM and program could be arbitrary, as long as the guest program verifies the commit of the App VM and program.

For usability, the proof to verify is read from keystore in the VM stream instead stdin. 
The key is the concatenation of `ASM filename`/`exe commit in u32`/`vm commit in u32`/`public values of the proof`, which is encoded by the host function`compute_hint_key_for_verify_openvm_stark`. 
The value is a proof encoded by host function `encode_proof_to_kv_store_value`. Check the doc comments of `define_verify_stark_proof` macro for more details. 

# Examples
Guest program: `examples/verify_openvm_stark` folder.
Host integration test: `tests/integration_tests.rs`.