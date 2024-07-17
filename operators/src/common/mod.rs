use afs_test_utils::config::baby_bear_poseidon2::random_perm;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

pub mod provider;

#[derive(PartialEq, Clone, Debug, Eq, Hash, PartialOrd, Ord, derive_new::new)]
pub struct Commitment<const LEN: usize> {
    commit: [u32; LEN],
}

impl<const LEN: usize> Serialize for Commitment<LEN> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Commitment", 1)?;
        state.serialize_field("commit", &self.commit.to_vec())?;
        state.end()
    }
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

pub fn hash_struct<T: Serialize, const COMMIT_LEN: usize>(input: &T) -> Commitment<COMMIT_LEN> {
    let u8s = bincode::serialize(input).unwrap();
    let baby_bears = u8s
        .iter()
        .map(|u8| BabyBear::from_canonical_u8(*u8))
        .collect();

    let commitment: Commitment<8> = poseidon2_hash(baby_bears).into();

    if COMMIT_LEN != 8 {
        panic!("COMMIT_LEN must be 8");
    }

    let mut commit_array = [0u32; COMMIT_LEN];
    commit_array.copy_from_slice(&commitment.commit);
    Commitment {
        commit: commit_array,
    }
}

pub fn poseidon2_hash(input: Vec<BabyBear>) -> [BabyBear; 8] {
    let hash = PaddingFreeSponge::<_, 16, 8, 8>::new(random_perm());
    hash.hash_iter(input)
}
