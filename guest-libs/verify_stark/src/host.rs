use openvm_native_recursion::hints::Hintable;
use openvm_rv32im_guest::hint_load_by_key_encode;
use openvm_sdk::SC;
use openvm_stark_sdk::{openvm_stark_backend::proof::Proof, p3_baby_bear::BabyBear};

/// Compute the hint key for `verify_openvm_stark` function, which reads a stark proof from stream
/// `kv_store`.
pub fn compute_hint_key_for_verify_openvm_stark(
    asm_filename: &str,
    exe_commit_u32: &[u32; 8],
    vm_commit_u32: &[u32; 8],
    pvs: &[u8],
) -> Vec<u8> {
    asm_filename
        .as_bytes()
        .iter()
        .cloned()
        .chain(exe_commit_u32.iter().flat_map(|x| x.to_le_bytes()))
        .chain(vm_commit_u32.iter().flat_map(|x| x.to_le_bytes()))
        .chain(pvs.iter().cloned())
        .collect()
}

/// Encode a proof into a KV store value so `verify_openvm_stark` can hint it.
pub fn encode_proof_to_kv_store_value(proof: &Proof<SC>) -> Vec<u8> {
    let to_encode: Vec<Vec<BabyBear>> = proof.write();
    hint_load_by_key_encode(&to_encode)
}
