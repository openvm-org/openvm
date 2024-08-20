use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::{
    cpu::RANGE_CHECKER_BUS,
    memory::{audit::MemoryAuditChip, manager::access_cell::AccessCell},
};

type Val = BabyBear;

#[test]
fn audit_air_test() {
    let mut rng = create_seeded_rng();

    const MAX_ADDRESS_SPACE: usize = 4;
    const LIMB_BITS: usize = 29;
    const MAX_VAL: usize = 1 << LIMB_BITS;
    const WORD_SIZE: usize = 2;
    const DECOMP: usize = 8;

    let mut random_f = |range: usize| Val::from_canonical_usize(rng.gen_range(0..range));

    let num_addresses = 10;
    let mut distinct_addresses = HashSet::new();
    while distinct_addresses.len() < num_addresses {
        let addr_space = random_f(2);
        let pointer = random_f(MAX_VAL);
        distinct_addresses.insert((addr_space, pointer));
    }

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << DECOMP));
    let mut audit_chips: Vec<MemoryAuditChip<WORD_SIZE, Val>> = (0..2)
        .map(|_| {
            MemoryAuditChip::<WORD_SIZE, Val>::new(2, LIMB_BITS, DECOMP, range_checker.clone())
        })
        .collect();

    let mut memory: Vec<BTreeMap<_, _>> = (0..2).map(|_| BTreeMap::new()).collect();

    for _ in 0..num_addresses {
        let addr_space = random_f(MAX_ADDRESS_SPACE);
        let pointer = random_f(MAX_VAL);

        let data_read = [random_f(MAX_VAL); WORD_SIZE];
        let clk_read = random_f(MAX_VAL);

        let data_write = [random_f(MAX_VAL); WORD_SIZE];
        let clk_write = random_f(MAX_VAL);

        audit_chips[0].touch_address(addr_space, pointer, data_read, clk_read);
        audit_chips[1].touch_address(addr_space, pointer, data_write, clk_write);

        memory[0].insert(
            (addr_space, pointer),
            AccessCell {
                data: data_write,
                clk: clk_write,
            },
        );
        memory[1].insert(
            (addr_space, pointer),
            AccessCell {
                data: data_read,
                clk: clk_read,
            },
        );
    }

    let traces = vec![
        audit_chips[0].generate_trace(&memory[0].clone()),
        audit_chips[1].generate_trace(&memory[1].clone()),
        range_checker.generate_trace(),
    ];

    run_simple_test_no_pis(
        vec![&audit_chips[0].air, &audit_chips[1].air, &range_checker.air],
        traces,
    )
    .expect("Verification failed");
}
