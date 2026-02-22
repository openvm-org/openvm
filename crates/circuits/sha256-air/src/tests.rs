use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::arch::{
    instructions::riscv::RV32_CELL_BITS,
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    Chip, SubAir,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend, MatrixDimensions},
    utils::disable_debug_builder,
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, Val,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use crate::{
    Sha256Air, Sha256DigestCols, Sha256FillerHelper, SHA256_BLOCK_U8S, SHA256_DIGEST_WIDTH,
    SHA256_HASH_WORDS, SHA256_WIDTH, SHA256_WORD_U8S,
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

const SELF_BUS_IDX: BusIndex = 28;
type F = BabyBear;
type RecordType = Vec<([u8; SHA256_BLOCK_U8S], bool)>;

// A wrapper Chip purely for testing purposes
pub struct Sha256TestChip {
    pub step: Sha256FillerHelper,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
}

impl<SC: StarkProtocolConfig> Chip<RecordType, CpuBackend<SC>> for Sha256TestChip
where
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, records: RecordType) -> AirProvingContext<CpuBackend<SC>> {
        let trace = crate::generate_trace::<Val<SC>>(
            &self.step,
            self.bitwise_lookup_chip.as_ref(),
            SHA256_WIDTH,
            records,
        );
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
    }
}

#[allow(clippy::type_complexity)]
fn create_air_with_air_ctx<SC: StarkProtocolConfig>() -> (
    (AirRef<SC>, AirProvingContext<CpuBackend<SC>>),
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
)
where
    Val<SC>: PrimeField32,
{
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let len = rng.random_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|i| {
            (
                array::from_fn(|_| rng.random::<u8>()),
                rng.random::<bool>() || i == len - 1,
            )
        })
        .collect();

    let air = Sha256TestAir {
        sub_air: Sha256Air::new(bitwise_bus, SELF_BUS_IDX),
    };
    let chip = Sha256TestChip {
        step: Sha256FillerHelper::new(),
        bitwise_lookup_chip: bitwise_chip.clone(),
    };
    let air_ctx = chip.generate_proving_ctx(random_records);

    ((Arc::new(air), air_ctx), (bitwise_chip.air, bitwise_chip))
}

#[test]
fn rand_sha256_test() {
    let tester = VmChipTestBuilder::default();
    let (air_ctx, bitwise) = create_air_with_air_ctx();
    let tester = tester
        .build()
        .load_air_proving_ctx(air_ctx)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
#[should_panic]
fn negative_sha256_test_bad_final_hash() {
    use openvm_stark_backend::SystemParams;

    let tester = VmChipTestBuilder::default();
    let ((air, mut air_ctx), bitwise) = create_air_with_air_ctx();

    // Set the final_hash to all zeros
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        trace.row_chunks_exact_mut(1).for_each(|row| {
            let mut row_slice = row.row_slice(0).expect("row exists").to_vec();
            let cols: &mut Sha256DigestCols<F> = row_slice[..SHA256_DIGEST_WIDTH].borrow_mut();
            if cols.flags.is_last_block_and_digest_row.is_one() && cols.flags.is_digest_row.is_one()
            {
                for i in 0..SHA256_HASH_WORDS {
                    for j in 0..SHA256_WORD_U8S {
                        cols.final_hash[i][j] = F::ZERO;
                    }
                }
                row.values.copy_from_slice(&row_slice);
            }
        });
    };

    // Modify the air_ctx: convert ColMajorMatrix to RowMajorMatrix, modify, convert back
    let w = air_ctx.common_main.width();
    let h = air_ctx.common_main.height();
    let mut rm_values = F::zero_vec(w * h);
    for r in 0..h {
        for c in 0..w {
            rm_values[r * w + c] = air_ctx.common_main.values[c * h + r];
        }
    }
    let mut trace = RowMajorMatrix::new(rm_values, w);
    modify_trace(&mut trace);
    air_ctx.common_main = ColMajorMatrix::from_row_major(&trace);

    disable_debug_builder();
    let mut params = SystemParams::new_for_testing(20);
    params.max_constraint_degree = 4;
    let tester = tester
        .build()
        .load_air_proving_ctx((air, air_ctx))
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test_with_params(params)
        .expect("Verification failed");
}
