use openvm_poseidon2_guest::POSEIDON2_WIDTH;
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear};
use p3_field::PrimeField32;
use p3_symmetric::Permutation;

/// Applies the Poseidon2 permutation in place using the canonical plonky3 BabyBear
/// implementation (`default_babybear_poseidon2_16`), which is the same permutation the
/// `PERMUTE` circuit constrains.
///
/// # Panics
///
/// Panics if any state word is not a canonical field element (i.e. is not less than
/// `0x78000001`).
pub fn permute(state: &mut [u32; POSEIDON2_WIDTH]) {
    assert!(
        state.iter().all(|&word| word < crate::BABY_BEAR_ORDER),
        "poseidon2 state words must be canonical field elements"
    );
    let mut input: [BabyBear; POSEIDON2_WIDTH] = core::array::from_fn(|i| BabyBear::new(state[i]));
    default_babybear_poseidon2_16().permute_mut(&mut input);
    for (word, elem) in state.iter_mut().zip(input.iter()) {
        *word = elem.as_canonical_u32();
    }
}
