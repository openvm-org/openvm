use std::{
    collections::{BTreeMap, HashSet},
    iter,
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis,
    interaction::dummy_interaction_air::DummyInteractionAir, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{
        audit::MemoryAuditChip, manager::access_cell::AccessCell, offline_checker::bus::MemoryBus,
    },
};

type Val = BabyBear;

#[test]
fn audit_air_test() {
    let mut rng = create_seeded_rng();

    const MEMORY_BUS: usize = 1;
    const MAX_ADDRESS_SPACE: usize = 4;
    const LIMB_BITS: usize = 29;
    const MAX_VAL: usize = 1 << LIMB_BITS;
    const WORD_SIZE: usize = 2;
    const DECOMP: usize = 8;
    let memory_bus = MemoryBus(1);

    let mut random_f = |range: usize| Val::from_canonical_usize(rng.gen_range(0..range));

    let num_addresses = 10;
    let mut distinct_addresses = HashSet::new();
    while distinct_addresses.len() < num_addresses {
        let addr_space = random_f(MAX_ADDRESS_SPACE);
        let pointer = random_f(MAX_VAL);
        distinct_addresses.insert((addr_space, pointer));
    }

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << DECOMP));
    let mut audit_chip = MemoryAuditChip::<WORD_SIZE, Val>::new(
        memory_bus,
        2,
        LIMB_BITS,
        DECOMP,
        range_checker.clone(),
    );

    let mut final_memory: BTreeMap<_, _> = BTreeMap::new();

    for (addr_space, pointer) in distinct_addresses.iter().cloned() {
        let final_data = [random_f(MAX_VAL); WORD_SIZE];
        let final_clk = random_f(MAX_VAL) + Val::one();

        audit_chip.touch_address(addr_space, pointer, [Val::zero(); WORD_SIZE]);
        final_memory.insert(
            (addr_space, pointer),
            AccessCell {
                data: final_data,
                clk: final_clk,
            },
        );
    }

    let diff_height = num_addresses.next_power_of_two() - num_addresses;

    let init_memory_dummy_air = DummyInteractionAir::new(3 + WORD_SIZE, false, MEMORY_BUS);
    let final_memory_dummy_air = DummyInteractionAir::new(3 + WORD_SIZE, true, MEMORY_BUS);

    let init_memory_trace = RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                vec![Val::one(), *addr_space, *pointer]
                    .into_iter()
                    .chain(vec![Val::zero(); WORD_SIZE])
                    .chain(iter::once(Val::zero()))
                    .collect::<Vec<_>>()
            })
            .chain(iter::repeat(Val::zero()).take((4 + WORD_SIZE) * (diff_height)))
            .collect(),
        4 + WORD_SIZE,
    );

    let final_memory_trace = RowMajorMatrix::new(
        distinct_addresses
            .iter()
            .flat_map(|(addr_space, pointer)| {
                let final_cell = final_memory.get(&(*addr_space, *pointer)).unwrap();

                vec![Val::one(), *addr_space, *pointer]
                    .into_iter()
                    .chain(final_cell.data.iter().copied())
                    .chain(iter::once(final_cell.clk))
                    .collect::<Vec<_>>()
            })
            .chain(iter::repeat(Val::zero()).take((4 + WORD_SIZE) * (diff_height)))
            .collect(),
        4 + WORD_SIZE,
    );

    let audit_trace = audit_chip.generate_trace(&final_memory);
    let range_checker_trace = range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![
            &audit_chip.air,
            &range_checker.air,
            &init_memory_dummy_air,
            &final_memory_dummy_air,
        ],
        vec![
            audit_trace,
            range_checker_trace,
            init_memory_trace,
            final_memory_trace,
        ],
    )
    .expect("Verification failed");
}
