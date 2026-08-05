use std::io::{self, Write};

use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::codec::{DecodableConfig, Decode, EncodableConfig, Encode};
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use super::{public_values_cells, public_values_commit, PUBLIC_VALUE_LIMBS};
use crate::arch::{hasher::Hasher, PublicValuesState};

/// Terminal opening of the append-only public-output commitment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicValuesOpening<F> {
    /// Four little-endian `u16` cells per configured output, with a zero-padded suffix.
    pub public_values: Vec<F>,
    /// Number of outputs in the committed prefix.
    pub num_values: usize,
}

#[derive(Error, Debug)]
pub enum PublicValuesOpeningError {
    #[error(
        "unexpected public-values shape: {public_values_len} u16 cells for {num_values} values"
    )]
    UnexpectedShape {
        public_values_len: usize,
        num_values: usize,
    },
    #[error("public-value cell {index} is not a u16")]
    NonCanonicalCell { index: usize },
    #[error("nonzero public-value padding at cell {index}")]
    NonzeroPadding { index: usize },
    #[error("final public-values commitment mismatch")]
    FinalPublicValuesCommitMismatch,
}

impl<F: PrimeField32> PublicValuesOpening<F> {
    /// Constructs the fixed-capacity terminal opening from VM state.
    #[instrument(name = "public_values_opening_from_state", skip_all)]
    pub fn from_state(state: &PublicValuesState) -> Self {
        Self {
            public_values: public_values_cells(state),
            num_values: state.len(),
        }
    }

    pub fn verify(
        &self,
        hasher: &impl Hasher<VM_DIGEST_WIDTH, F>,
        final_commit: [F; VM_DIGEST_WIDTH],
    ) -> Result<(), PublicValuesOpeningError> {
        if !self.public_values.len().is_multiple_of(PUBLIC_VALUE_LIMBS)
            || !(self.public_values.len() / PUBLIC_VALUE_LIMBS).is_power_of_two()
            || self.num_values > self.public_values.len() / PUBLIC_VALUE_LIMBS
        {
            return Err(PublicValuesOpeningError::UnexpectedShape {
                public_values_len: self.public_values.len(),
                num_values: self.num_values,
            });
        }
        let published_cells = self.num_values * PUBLIC_VALUE_LIMBS;
        for (index, cell) in self.public_values[..published_cells].iter().enumerate() {
            if cell.as_canonical_u32() > u16::MAX as u32 {
                return Err(PublicValuesOpeningError::NonCanonicalCell { index });
            }
        }
        for (offset, cell) in self.public_values[published_cells..].iter().enumerate() {
            if *cell != F::ZERO {
                return Err(PublicValuesOpeningError::NonzeroPadding {
                    index: published_cells + offset,
                });
            }
        }

        let values = self.public_values[..published_cells]
            .chunks_exact(PUBLIC_VALUE_LIMBS)
            .map(|limbs| {
                limbs.iter().enumerate().fold(0u64, |value, (i, limb)| {
                    value | ((limb.as_canonical_u32() as u64) << (16 * i))
                })
            })
            .collect::<Vec<_>>();
        let commitment = public_values_commit(
            &values,
            self.public_values.len() / PUBLIC_VALUE_LIMBS,
            hasher,
        );
        if commitment != final_commit {
            return Err(PublicValuesOpeningError::FinalPublicValuesCommitMismatch);
        }
        Ok(())
    }
}

impl<F> PublicValuesOpening<F> {
    pub fn encode<SC: EncodableConfig<F = F>, W: Write>(&self, writer: &mut W) -> io::Result<()> {
        SC::encode_base_field_slice(&self.public_values, writer)?;
        u32::try_from(self.num_values)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "public-values count exceeds u32",
                )
            })?
            .encode(writer)
    }

    pub fn decode<SC: DecodableConfig<F = F>, R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Self {
            public_values: SC::decode_base_field_vec(reader)?,
            num_values: u32::decode(reader)? as usize,
        })
    }
}

/// Returns the published prefix as packed little-endian `u64` bytes.
pub fn extract_public_values(state: &PublicValuesState) -> Vec<u8> {
    state
        .values()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use openvm_stark_sdk::{
        config::baby_bear_poseidon2::BabyBearPoseidon2Config, p3_baby_bear::BabyBear,
    };
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::arch::hasher::poseidon2::vm_poseidon2_hasher;

    type TestSC = BabyBearPoseidon2Config;

    fn commit(state: &PublicValuesState) -> [BabyBear; VM_DIGEST_WIDTH] {
        public_values_commit(
            state.values(),
            state.max_public_values(),
            &vm_poseidon2_hasher(),
        )
    }

    #[test]
    fn empty_and_full_openings_verify() {
        let hasher = vm_poseidon2_hasher();
        let empty = PublicValuesState::new(2);
        let empty_opening = PublicValuesOpening::<BabyBear>::from_state(&empty);
        assert_eq!(empty_opening.num_values, 0);
        assert_eq!(empty_opening.public_values, vec![BabyBear::ZERO; 8]);
        empty_opening.verify(&hasher, commit(&empty)).unwrap();

        let mut full = PublicValuesState::new(2);
        full.try_push(0).unwrap();
        full.try_push(0x8877_6655_4433_2211).unwrap();
        let full_opening = PublicValuesOpening::<BabyBear>::from_state(&full);
        assert_eq!(full_opening.num_values, 2);
        assert_eq!(
            extract_public_values(&full)[8..],
            0x8877_6655_4433_2211u64.to_le_bytes()
        );
        full_opening.verify(&hasher, commit(&full)).unwrap();
    }

    #[test]
    fn opening_binds_order_and_commitment() {
        let hasher = vm_poseidon2_hasher();
        let mut state = PublicValuesState::new(2);
        state.try_push(1).unwrap();
        state.try_push(2).unwrap();
        let final_commit = commit(&state);
        let opening = PublicValuesOpening::<BabyBear>::from_state(&state);

        let mut reordered = opening.clone();
        reordered.public_values[..8].rotate_left(4);
        assert!(matches!(
            reordered.verify(&hasher, final_commit),
            Err(PublicValuesOpeningError::FinalPublicValuesCommitMismatch)
        ));

        let mut wrong_commit = final_commit;
        wrong_commit[0] += BabyBear::ONE;
        assert!(matches!(
            opening.verify(&hasher, wrong_commit),
            Err(PublicValuesOpeningError::FinalPublicValuesCommitMismatch)
        ));
    }

    #[test]
    fn revealed_zero_is_distinct_from_padding() {
        let hasher = vm_poseidon2_hasher();
        let mut one_value = PublicValuesState::new(2);
        one_value.try_push(7).unwrap();
        let mut two_values = one_value.clone();
        two_values.try_push(0).unwrap();

        let one_opening = PublicValuesOpening::<BabyBear>::from_state(&one_value);
        let two_opening = PublicValuesOpening::<BabyBear>::from_state(&two_values);
        assert_eq!(one_opening.public_values, two_opening.public_values);
        assert_eq!(one_opening.num_values, 1);
        assert_eq!(two_opening.num_values, 2);
        assert_ne!(commit(&one_value), commit(&two_values));
        one_opening.verify(&hasher, commit(&one_value)).unwrap();
        two_opening.verify(&hasher, commit(&two_values)).unwrap();
    }

    #[test]
    fn malformed_shapes_are_rejected() {
        let hasher = vm_poseidon2_hasher();
        let final_commit = [BabyBear::ZERO; VM_DIGEST_WIDTH];
        for opening in [
            PublicValuesOpening {
                public_values: vec![BabyBear::ZERO; 3],
                num_values: 0,
            },
            PublicValuesOpening {
                public_values: vec![BabyBear::ZERO; 12],
                num_values: 0,
            },
            PublicValuesOpening {
                public_values: vec![BabyBear::ZERO; 4],
                num_values: 2,
            },
        ] {
            assert!(matches!(
                opening.verify(&hasher, final_commit),
                Err(PublicValuesOpeningError::UnexpectedShape { .. })
            ));
        }
    }

    #[test]
    fn noncanonical_cells_and_nonzero_padding_are_rejected() {
        let hasher = vm_poseidon2_hasher();
        let final_commit = [BabyBear::ZERO; VM_DIGEST_WIDTH];
        let noncanonical = PublicValuesOpening {
            public_values: vec![
                BabyBear::from_u32(u16::MAX as u32 + 1),
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
            ],
            num_values: 1,
        };
        assert!(matches!(
            noncanonical.verify(&hasher, final_commit),
            Err(PublicValuesOpeningError::NonCanonicalCell { index: 0 })
        ));

        let nonzero_padding = PublicValuesOpening {
            public_values: vec![
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ONE,
                BabyBear::ZERO,
                BabyBear::ZERO,
                BabyBear::ZERO,
            ],
            num_values: 1,
        };
        assert!(matches!(
            nonzero_padding.verify(&hasher, final_commit),
            Err(PublicValuesOpeningError::NonzeroPadding { index: 4 })
        ));
    }

    #[test]
    fn opening_codec_roundtrip() {
        let mut state = PublicValuesState::new(2);
        state.try_push(0x8877_6655_4433_2211).unwrap();
        let opening = PublicValuesOpening::<BabyBear>::from_state(&state);
        let mut encoded = Vec::new();
        opening.encode::<TestSC, _>(&mut encoded).unwrap();

        let mut reader = Cursor::new(&encoded);
        let decoded = PublicValuesOpening::<BabyBear>::decode::<TestSC, _>(&mut reader).unwrap();
        assert_eq!(decoded, opening);
        assert_eq!(reader.position() as usize, encoded.len());
    }
}
