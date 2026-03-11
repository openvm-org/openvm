use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use recursion_circuit::utils::poseidon2_hash_slice_with_states;

/// Given N element digests, compute the N−1 intermediate full permutation states
/// and the final digest, matching the layout of [`HashSliceCtx`].
///
/// If provided, the pre-permutation inputs for the first N−1 elements are appended
/// to `poseidon2_permute_inputs`, and the pre-permutation input for the last element
/// is appended to `poseidon2_compress_inputs`.
pub fn hash_slice_trace(
    elements: &[[F; DIGEST_SIZE]],
    poseidon2_permute_inputs: Option<&mut Vec<[F; POSEIDON2_WIDTH]>>,
    poseidon2_compress_inputs: Option<&mut Vec<[F; POSEIDON2_WIDTH]>>,
) -> (Vec<[F; POSEIDON2_WIDTH]>, [F; DIGEST_SIZE]) {
    let n = elements.len();
    assert!(n > 1);

    let flat: Vec<F> = elements.iter().flatten().copied().collect();
    let (result, pre_states, post_states) = poseidon2_hash_slice_with_states(&flat);

    debug_assert_eq!(post_states.len(), n);
    debug_assert_eq!(pre_states.len(), n);

    if let Some(permute) = poseidon2_permute_inputs {
        permute.extend_from_slice(&pre_states[..n - 1]);
    }
    if let Some(compress) = poseidon2_compress_inputs {
        compress.push(pre_states[n - 1]);
    }

    let intermediate = post_states[..n - 1].to_vec();
    (intermediate, result)
}
