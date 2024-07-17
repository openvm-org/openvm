use afs_test_utils::config::baby_bear_poseidon2::random_perm;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};

#[derive(PartialEq, Clone, Debug, Eq, Hash, derive_new::new)]
pub struct Commitment<const LEN: usize> {
    commit: [BabyBear; LEN],
}

impl<const LEN: usize> Default for Commitment<LEN> {
    fn default() -> Self {
        Self {
            commit: [BabyBear::zero(); LEN],
        }
    }
}

pub fn poseidon2_hash(input: Vec<BabyBear>) -> [BabyBear; 8] {
    let hash = PaddingFreeSponge::<_, 16, 8, 8>::new(random_perm());
    hash.hash_iter(input)
}
