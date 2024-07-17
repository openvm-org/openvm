use afs_test_utils::config::baby_bear_poseidon2::random_perm;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};

#[derive(PartialEq, Clone, Debug, Eq, Hash, derive_new::new)]
pub struct Commitment<const LEN: usize> {
    commit: [u32; LEN],
}

impl<const LEN: usize> Default for Commitment<LEN> {
    fn default() -> Self {
        Self { commit: [0; LEN] }
    }
}

impl<const LEN: usize> From<[BabyBear; LEN]> for Commitment<LEN> {
    fn from(commit: [BabyBear; LEN]) -> Self {
        Self {
            commit: commit.map(|b| b.as_canonical_u32()),
        }
    }
}

pub fn poseidon2_hash(input: Vec<BabyBear>) -> [BabyBear; 8] {
    let hash = PaddingFreeSponge::<_, 16, 8, 8>::new(random_perm());
    hash.hash_iter(input)
}
