use std::{cmp::max, sync::Arc};

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
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
    AirRef, Chip, ChipUsageGetter,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use crate::{
    Sha256Config, Sha2Air, Sha2Config, Sha2StepHelper, Sha384Config, Sha512Config,
    ShaDigestColsRefMut,
};

// A wrapper AIR purely for testing purposes
#[derive(Clone, Debug)]
pub struct Sha2TestAir<C: Sha2Config> {
    pub sub_air: Sha2Air<C>,
}

impl<F: Field, C: Sha2Config> BaseAirWithPublicValues<F> for Sha2TestAir<C> {}
impl<F: Field, C: Sha2Config> PartitionedBaseAir<F> for Sha2TestAir<C> {}
impl<F: Field, C: Sha2Config> BaseAir<F> for Sha2TestAir<C> {
    fn width(&self) -> usize {
        <Sha2Air<C> as BaseAir<F>>::width(&self.sub_air)
    }
}

impl<AB: InteractionBuilder, C: Sha2Config> Air<AB> for Sha2TestAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

// A wrapper Chip purely for testing purposes
pub struct Sha2TestChip<C: Sha2Config> {
    pub air: Sha2TestAir<C>,
    pub step: Sha2StepHelper<C>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub records: Vec<(Vec<u8>, bool)>, // length of inner vec is C::BLOCK_U8S
}

impl<SC: StarkGenericConfig, C: Sha2Config + 'static> Chip<SC> for Sha2TestChip<C>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = crate::generate_trace::<Val<SC>, C>(
            &self.step,
            self.bitwise_lookup_chip.clone(),
            <Sha2Air<C> as BaseAir<Val<SC>>>::width(&self.air.sub_air),
            self.records,
        );
        AirProofInput::simple_no_pis(trace)
    }
}

impl<C: Sha2Config> ChipUsageGetter for Sha2TestChip<C> {
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
type F = BabyBear;

fn create_chip_with_rand_records<C: Sha2Config + 'static>(
) -> (Sha2TestChip<C>, SharedBitwiseOperationLookupChip<8>) {
    let mut rng = create_seeded_rng();
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
    let chip = Sha2TestChip {
        air: Sha2TestAir {
            sub_air: Sha2Air::<C>::new(bitwise_bus, SELF_BUS_IDX),
        },
        step: Sha2StepHelper::<C>::new(),
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    (chip, bitwise_chip)
}

fn rand_sha2_test<C: Sha2Config + 'static>() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records::<C>();
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha2_test::<Sha256Config>();
}

#[test]
fn rand_sha512_test() {
    rand_sha2_test::<Sha512Config>();
}

#[test]
fn rand_sha384_test() {
    rand_sha2_test::<Sha384Config>();
}

fn negative_sha2_test_bad_final_hash<C: Sha2Config + 'static>() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records::<C>();

    // Set the final_hash to all zeros
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        trace.row_chunks_exact_mut(1).for_each(|row| {
            let mut row_slice = row.row_slice(0).to_vec();
            let mut cols: ShaDigestColsRefMut<F> =
                ShaDigestColsRefMut::from::<C>(&mut row_slice[..C::DIGEST_WIDTH]);
            if cols.flags.is_last_block.is_one() && cols.flags.is_digest_row.is_one() {
                for i in 0..C::HASH_WORDS {
                    for j in 0..C::WORD_U8S {
                        cols.final_hash[[i, j]] = F::ZERO;
                    }
                }
                row.values.copy_from_slice(&row_slice);
            }
        });
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
}

#[test]
#[should_panic]
fn negative_sha256_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha256Config>();
}

#[test]
#[should_panic]
fn negative_sha512_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha512Config>();
}

#[test]
#[should_panic]
fn negative_sha384_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha384Config>();
}
