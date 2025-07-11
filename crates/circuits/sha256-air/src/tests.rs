use std::{array, borrow::BorrowMut, cmp::max, sync::Arc};

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
    Sha256Air, Sha256DigestCols, Sha256StepHelper, SHA256_BLOCK_U8S, SHA256_DIGEST_WIDTH,
    SHA256_HASH_WORDS, SHA256_ROUND_WIDTH, SHA256_ROWS_PER_BLOCK, SHA256_WORD_U8S,
};

// A wrapper AIR purely for testing purposes
#[derive(Clone, Debug)]
pub struct Sha256TestAir {
    pub sub_air: Sha256Air,
}

impl<F: Field> BaseAirWithPublicValues<F> for Sha256TestAir {}
impl<F: Field> PartitionedBaseAir<F> for Sha256TestAir {}
impl<F: Field> BaseAir<F> for Sha256TestAir {
    fn width(&self) -> usize {
        <Sha256Air as BaseAir<F>>::width(&self.sub_air)
    }
}

impl<AB: InteractionBuilder> Air<AB> for Sha256TestAir {
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

// A wrapper Chip purely for testing purposes
pub struct Sha256TestChip {
    pub air: Sha256TestAir,
    pub step: Sha256StepHelper,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub records: Vec<([u8; SHA256_BLOCK_U8S], bool)>,
}

impl<SC: StarkGenericConfig> Chip<SC> for Sha256TestChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = crate::generate_trace::<Val<SC>>(
            &self.step,
            self.bitwise_lookup_chip.as_ref(),
            <Sha256Air as BaseAir<Val<SC>>>::width(&self.air.sub_air),
            self.records,
        );
        AirProofInput::simple_no_pis(trace)
    }
}

impl ChipUsageGetter for Sha256TestChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() * SHA256_ROWS_PER_BLOCK
    }

    fn trace_width(&self) -> usize {
        max(SHA256_ROUND_WIDTH, SHA256_DIGEST_WIDTH)
    }
}

const SELF_BUS_IDX: BusIndex = 28;
type F = BabyBear;

fn create_chip_with_rand_records() -> (Sha256TestChip, SharedBitwiseOperationLookupChip<8>) {
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|i| {
            (
                array::from_fn(|_| rng.gen::<u8>()),
                rng.gen::<bool>() || i == len - 1,
            )
        })
        .collect();
    let chip = Sha256TestChip {
        air: Sha256TestAir {
            sub_air: Sha256Air::new(bitwise_bus, SELF_BUS_IDX),
        },
        step: Sha256StepHelper::new(),
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };
    (chip, bitwise_chip)
}

#[test]
fn rand_sha256_test() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records();
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_sha256_test_bad_final_hash() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records();

    // Set the final_hash to all zeros
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        trace.row_chunks_exact_mut(1).for_each(|row| {
            let mut row_slice = row.row_slice(0).to_vec();
            let cols: &mut Sha256DigestCols<F> = row_slice[..SHA256_DIGEST_WIDTH].borrow_mut();
            if cols.flags.is_last_block.is_one() && cols.flags.is_digest_row.is_one() {
                for i in 0..SHA256_HASH_WORDS {
                    for j in 0..SHA256_WORD_U8S {
                        cols.final_hash[i][j] = F::ZERO;
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
