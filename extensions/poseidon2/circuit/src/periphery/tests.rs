use std::sync::Arc;

use openvm_circuit::arch::testing::{TestSC, VmChipTestBuilder, POSEIDON2_DIRECT_BUS};
use openvm_circuit_primitives::Chip;
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::{Poseidon2Config, POSEIDON2_WIDTH};
use openvm_stark_backend::{
    interaction::LookupBus,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    prover::{AirProvingContext, MatrixDimensions},
    test_utils::dummy_airs::interaction::dummy_interaction_air::{
        DummyInteractionChip, DummyInteractionData,
    },
    AirRef,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::RngCore;

use crate::periphery::{Poseidon2PeripheryAir, Poseidon2PeripheryChip};

fn create_test_chip() -> (AirRef<TestSC>, Poseidon2PeripheryChip<BabyBear>) {
    let chip = Poseidon2PeripheryChip::<BabyBear>::new(Poseidon2Config::default());
    let air = Arc::new(Poseidon2PeripheryAir::<BabyBear>::new(
        Poseidon2Config::default(),
        LookupBus::new(POSEIDON2_DIRECT_BUS),
    ));
    (air, chip)
}

#[test]
fn poseidon2_periphery_empty_trace() {
    let chip = Poseidon2PeripheryChip::<BabyBear>::new(Poseidon2Config::default());
    for _ in 0..2 {
        let ctx: AirProvingContext<CpuBackend<TestSC>> = chip.generate_proving_ctx(());
        assert_eq!(
            ctx.common_main.height(),
            0,
            "Poseidon2PeripheryChip with no records should return an empty trace",
        );
    }
}

/// Test that the direct bus interactions work.
#[test]
fn poseidon2_periphery_direct_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    let states: [[BabyBear; POSEIDON2_WIDTH]; NUM_OPS] = std::array::from_fn(|_| {
        std::array::from_fn(|_| BabyBear::from_u32(rng.next_u32() % (1 << 30)))
    });
    let (air, chip) = create_test_chip();

    let outs: [[BabyBear; POSEIDON2_WIDTH]; NUM_OPS] =
        std::array::from_fn(|i| chip.perm_and_record(states[i]));

    let mut dummy_interaction_chip: DummyInteractionChip<TestSC> =
        DummyInteractionChip::new_without_partition(
            2 * POSEIDON2_WIDTH,
            true,
            POSEIDON2_DIRECT_BUS,
        );
    let count = vec![1; NUM_OPS];
    let fields = states
        .iter()
        .zip(outs)
        .map(|(input, output)| {
            input
                .iter()
                .chain(output.iter())
                .map(|y| y.as_canonical_u32())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    dummy_interaction_chip.load_data(DummyInteractionData { count, fields });

    let tester = VmChipTestBuilder::default();
    let dummy_ref_ctx = dummy_interaction_chip.generate_proving_ctx();
    let dummy_ctx = AirProvingContext {
        cached_mains: vec![],
        common_main: {
            let view: openvm_stark_backend::prover::StridedColMajorMatrixView<'_, _> =
                dummy_ref_ctx.common_main.as_view().into();
            view.to_row_major_matrix()
        },
        public_values: dummy_ref_ctx.public_values,
    };
    let dummy_air = Arc::new(dummy_interaction_chip.air) as AirRef<TestSC>;
    let mut tester = tester.build().load_periphery_ref((air, chip)).finalize();
    tester.air_ctxs.push((dummy_air, dummy_ctx));
    tester.simple_test().expect("Verification failed");
}

#[test]
fn poseidon2_periphery_duplicate_test() {
    let mut rng = create_seeded_rng();
    const NUM_OPS: usize = 50;
    let states: [[BabyBear; POSEIDON2_WIDTH]; NUM_OPS] = std::array::from_fn(|_| {
        std::array::from_fn(|_| BabyBear::from_u32(rng.next_u32() % (1 << 30)))
    });
    let counts: [u32; NUM_OPS] = std::array::from_fn(|_| rng.next_u32() % 20);

    let (air, chip) = create_test_chip();

    let outs: [[BabyBear; POSEIDON2_WIDTH]; NUM_OPS] = std::array::from_fn(|i| {
        for _ in 0..counts[i] {
            let _ = chip.perm_and_record(states[i]);
        }
        chip.permute(states[i])
    });

    let mut dummy_interaction_chip: DummyInteractionChip<TestSC> =
        DummyInteractionChip::new_without_partition(
            2 * POSEIDON2_WIDTH,
            true,
            POSEIDON2_DIRECT_BUS,
        );
    let count = counts.to_vec();
    let fields = states
        .iter()
        .zip(outs)
        .map(|(input, output)| {
            input
                .iter()
                .chain(output.iter())
                .map(|y| y.as_canonical_u32())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    dummy_interaction_chip.load_data(DummyInteractionData { count, fields });

    let tester = VmChipTestBuilder::default();
    let dummy_ref_ctx = dummy_interaction_chip.generate_proving_ctx();
    let dummy_ctx = AirProvingContext {
        cached_mains: vec![],
        common_main: {
            let view: openvm_stark_backend::prover::StridedColMajorMatrixView<'_, _> =
                dummy_ref_ctx.common_main.as_view().into();
            view.to_row_major_matrix()
        },
        public_values: dummy_ref_ctx.public_values,
    };
    let dummy_air = Arc::new(dummy_interaction_chip.air) as AirRef<TestSC>;
    let mut tester = tester.build().load_periphery_ref((air, chip)).finalize();
    tester.air_ctxs.push((dummy_air, dummy_ctx));
    tester.simple_test().expect("Verification failed");
}
