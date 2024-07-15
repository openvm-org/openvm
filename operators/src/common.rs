use afs_test_utils::config::baby_bear_poseidon2::random_perm;
use p3_field::AbstractField;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use p3_uni_stark::StarkGenericConfig;

#[derive(PartialEq, Clone, Debug)]
pub struct Commitment<const LEN: usize> {
    commit: [u32; LEN],
}

impl<const LEN: usize> Default for Commitment<LEN> {
    fn default() -> Self {
        Self { commit: [0; LEN] }
    }
}

pub fn poseidon2_hash<SC: StarkGenericConfig, F: AbstractField>(input: &[F]) -> [&F; 8] {
    let hash = PaddingFreeSponge::<_, 16, 8, 8>::new(random_perm());
    hash.hash_iter(input)
}
