use std::{collections::HashSet, iter, sync::Arc};

use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::Rng;
use test_log::test;

use crate::system::memory::{
    offline_checker::MemoryBus, volatile::VolatileBoundaryChip, TimestampedEquipartition,
    TimestampedValues,
};

type Val = BabyBear;

#[test]
fn boundary_air_test() {
    let mut rng = create_seeded_rng();

    const MEMORY_BUS: usize = 1;
    const RANGE_CHECKER_BUS: usize = 3;
    const MAX_ADDRESS_SPACE: usize = 4;
    const LIMB_BITS: usize = 15;
    const MAX_VAL: usize = 1 << LIMB_BITS;
    const DECOMP: usize = 8;
    let memory_bus = MemoryBus(MEMORY_BUS);

    let num_addresses = 10;
    let mut distinct_addresses = HashSet::new();
    while distinct_addresses.len() < num_addresses {
        let addr_space = Val::from_canonical_usize(rng.gen_range(0..MAX_ADDRESS_SPACE));
        let pointer = Val::from_canonical_usize(rng.gen_range(0..MAX_VAL));
        distinct_addresses.insert((addr_space, pointer));
    }

    let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, DECOMP);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
    let boundary_chip = VolatileBoundaryChip::new(memory_bus, 2, LIMB_BITS, range_checker.clone());

    let mut final_memory = TimestampedEquipartition::new();

    for (addr_space, pointer) in distinct_addresses.iter().cloned() {
        let final_data = Val::from_canonical_usize(rng.gen_range(0..MAX_VAL));
        let final_clk = rng.gen_range(1..MAX_VAL) as u32;

        final_memory.insert(
            (addr_space, pointer.as_canonical_u32() as usize),
            TimestampedValues {
                values: [final_data],
                timestamp: final_clk,
            },
        );
    }

    let diff_height = num_addresses.next_power_of_two() - num_addresses;

    let init_memory_dummy_air = DummyInteractionAir::new(5, false, MEMORY_BUS);
    let final_memory_dummy_air = DummyInteractionAir::new(5, true, MEMORY_BUS);

    let init_memory_trace = RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                vec![
                    Val::ONE,
                    *addr_space,
                    *pointer,
                    Val::ZERO,
                    Val::ZERO,
                    Val::ONE,
                ]
            })
            .chain(iter::repeat(Val::ZERO).take(6 * diff_height))
            .collect(),
        6,
    );

    let final_memory_trace = RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                let timestamped_value = final_memory
                    .get(&(*addr_space, pointer.as_canonical_u32() as usize))
                    .unwrap();

                vec![
                    Val::ONE,
                    *addr_space,
                    *pointer,
                    timestamped_value.values[0],
                    Val::from_canonical_u32(timestamped_value.timestamp),
                    Val::ONE,
                ]
            })
            .chain(iter::repeat(Val::ZERO).take(6 * diff_height))
            .collect(),
        6,
    );

    let boundary_trace = boundary_chip.generate_trace(&final_memory, None);
    // test trace height override
    {
        let overridden_height = boundary_trace.height() * 2;
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let boundary_chip =
            VolatileBoundaryChip::new(memory_bus, 2, LIMB_BITS, range_checker.clone());
        let boundary_trace = boundary_chip.generate_trace(&final_memory, Some(overridden_height));
        assert_eq!(
            boundary_trace.height(),
            overridden_height.next_power_of_two()
        );
    }

    let range_checker_trace = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![
            boundary_chip.air,
            range_checker.air,
            init_memory_dummy_air,
            final_memory_dummy_air
        ],
        vec![
            boundary_trace,
            range_checker_trace,
            init_memory_trace,
            final_memory_trace,
        ],
    )
    .expect("Verification failed");
}
