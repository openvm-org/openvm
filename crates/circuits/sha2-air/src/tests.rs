use std::{borrow::BorrowMut, cmp::max, sync::Arc};

use openvm_circuit::arch::{
    instructions::riscv::RV32_CELL_BITS,
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    SubAir,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;

use crate::{
    compose, small_sig0_field, Sha256Config, Sha2Air, Sha2Config, Sha384Config, Sha512Config,
    ShaDigestColsRefMut, ShaRoundColsRef, ShaRoundColsRefMut,
};

// A wrapper AIR purely for testing purposes
#[derive(Clone, Debug)]
pub struct ShaTestAir<C: Sha2Config> {
    pub sub_air: Sha2Air<C>,
}

impl<F: Field, C: Sha2Config> BaseAirWithPublicValues<F> for ShaTestAir<C> {}
impl<F: Field, C: Sha2Config> PartitionedBaseAir<F> for ShaTestAir<C> {}
impl<F: Field, C: Sha2Config> BaseAir<F> for ShaTestAir<C> {
    fn width(&self) -> usize {
        <Sha2Air<C> as BaseAir<F>>::width(&self.sub_air)
    }
}

impl<AB: InteractionBuilder, C: Sha2Config> Air<AB> for ShaTestAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

// A wrapper Chip purely for testing purposes
pub struct ShaTestChip<C: Sha2Config> {
    pub air: ShaTestAir<C>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub records: Vec<(Vec<u8>, bool)>, // length of inner vec is BLOCK_U8S
}

impl<SC: StarkGenericConfig, C: Sha2Config + 'static> Chip<SC> for ShaTestChip<C>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = crate::generate_trace::<Val<SC>, C>(
            &self.air.sub_air,
            self.bitwise_lookup_chip.clone(),
            self.records,
        );
        AirProofInput::simple_no_pis(trace)
    }
}

impl<C: Sha2Config> ChipUsageGetter for ShaTestChip<C> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() * C::ROWS_PER_BLOCK
    }

    fn trace_width(&self) -> usize {
        max(C::ROUND_WIDTH, C::DIGEST_WIDTH)
    }
}

const SELF_BUS_IDX: BusIndex = 28;
fn rand_sha_test<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|i| {
            (
                (0..C::BLOCK_U8S)
                    .map(|_| rng.gen::<u8>())
                    .collect::<Vec<_>>(),
                rng.gen::<bool>() || i == len - 1,
            )
        })
        .collect();
    let chip = ShaTestChip {
        air: ShaTestAir {
            sub_air: Sha2Air::<C>::new(bitwise_bus, SELF_BUS_IDX),
        },
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha_test::<Sha256Config>();
}

#[test]
fn rand_sha512_test() {
    rand_sha_test::<Sha512Config>();
}

#[test]
fn rand_sha384_test() {
    rand_sha_test::<Sha384Config>();
}

// A wrapper Chip to test that the final_hash is properly constrained.
// This chip implements a malicious trace gen that violates the final_hash constraints.
pub struct ShaTestBadFinalHashChip<C: Sha2Config> {
    pub air: ShaTestAir<C>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub records: Vec<(Vec<u8>, bool)>, // length of inner vec should be C::BLOCK_U8S
}

impl<SC: StarkGenericConfig, C: Sha2Config + 'static> Chip<SC> for ShaTestBadFinalHashChip<C>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let mut trace = crate::generate_trace::<Val<SC>, C>(
            &self.air.sub_air,
            self.bitwise_lookup_chip.clone(),
            self.records.clone(),
        );

        // Set the final_hash in the digest row of the last block of each hash to zero.
        // That is, every hash that this chip does will result in a final_hash of zero.
        for (i, row) in self.records.iter().enumerate() {
            if row.1 {
                let last_digest_row_idx = (i + 1) * C::ROWS_PER_BLOCK - 1;
                let mut last_digest_row: crate::ShaDigestColsRefMut<Val<SC>> =
                    ShaDigestColsRefMut::from::<C>(
                        trace.row_mut(last_digest_row_idx)[..C::DIGEST_WIDTH].borrow_mut(),
                    );
                // Set the final_hash to all zeros
                for i in 0..C::HASH_WORDS {
                    for j in 0..C::WORD_U8S {
                        last_digest_row.final_hash[[i, j]] = Val::<SC>::ZERO;
                    }
                }

                let (last_round_row, last_digest_row) =
                    trace.row_pair_mut(last_digest_row_idx - 1, last_digest_row_idx);
                let last_round_row: crate::ShaRoundColsRefMut<Val<SC>> =
                    ShaRoundColsRefMut::from::<C>(last_round_row.borrow_mut());
                let mut last_digest_row: crate::ShaRoundColsRefMut<Val<SC>> =
                    ShaRoundColsRefMut::from::<C>(last_digest_row.borrow_mut());
                // fix the intermed_4 for the digest row
                generate_intermed_4::<Val<SC>, C>(
                    &ShaRoundColsRef::from_mut::<C>(&last_round_row),
                    &mut last_digest_row,
                );
            }
        }

        let non_padded_height = self.records.len() * C::ROWS_PER_BLOCK;
        let width = <Sha2Air<C> as BaseAir<Val<SC>>>::width(&self.air.sub_air);
        // recalculate the missing cells (second pass of generate_trace)
        trace.values[width..]
            .par_chunks_mut(width * C::ROWS_PER_BLOCK)
            .take(non_padded_height / C::ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.air.sub_air.generate_missing_cells(chunk, width, 0);
            });

        AirProofInput::simple_no_pis(trace)
    }
}

// Copy of private method in Sha256Air used for testing
/// Puts the correct intermed_4 in the `next_row`
fn generate_intermed_4<F: PrimeField32, C: Sha2Config>(
    local_cols: &ShaRoundColsRef<F>,
    next_cols: &mut ShaRoundColsRefMut<F>,
) {
    let w = [
        local_cols
            .message_schedule
            .w
            .rows()
            .into_iter()
            .collect::<Vec<_>>(),
        next_cols
            .message_schedule
            .w
            .rows()
            .into_iter()
            .collect::<Vec<_>>(),
    ]
    .concat();

    // length of inner vec is C::WORD_U16S
    let w_limbs: Vec<Vec<F>> = w
        .iter()
        .map(|x| {
            (0..C::WORD_U16S)
                .map(|i| compose::<F>(&x.as_slice().unwrap()[i * 16..(i + 1) * 16], 1))
                .collect::<Vec<F>>()
        })
        .collect();
    for i in 0..C::ROUNDS_PER_ROW {
        let sig_w = small_sig0_field::<F, C>(w[i + 1].as_slice().unwrap());
        let sig_w_limbs: Vec<F> = (0..C::WORD_U16S)
            .map(|j| compose::<F>(&sig_w[j * 16..(j + 1) * 16], 1))
            .collect();
        for (j, sig_w_limb) in sig_w_limbs.iter().enumerate() {
            next_cols.schedule_helper.intermed_4[[i, j]] = w_limbs[i][j] + *sig_w_limb;
        }
    }
}

impl<C: Sha2Config> ChipUsageGetter for ShaTestBadFinalHashChip<C> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() * C::ROWS_PER_BLOCK
    }

    fn trace_width(&self) -> usize {
        max(C::ROUND_WIDTH, C::DIGEST_WIDTH)
    }
}

fn test_sha_final_hash_constraints<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|_| {
            (
                (0..C::BLOCK_U8S)
                    .map(|_| rng.gen::<u8>())
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect();
    let chip = ShaTestBadFinalHashChip {
        air: ShaTestAir {
            sub_air: Sha2Air::<C>::new(bitwise_bus, SELF_BUS_IDX),
        },
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
#[should_panic]
fn test_sha256_final_hash_constraints() {
    test_sha_final_hash_constraints::<Sha256Config>();
}

#[test]
#[should_panic]
fn test_sha512_final_hash_constraints() {
    test_sha_final_hash_constraints::<Sha512Config>();
}

#[test]
#[should_panic]
fn test_sha384_final_hash_constraints() {
    test_sha_final_hash_constraints::<Sha384Config>();
}
