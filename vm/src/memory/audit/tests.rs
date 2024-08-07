use std::{collections::BTreeMap, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::{cpu::RANGE_CHECKER_BUS, memory::audit::air::AuditAir};

type Val = BabyBear;

#[test]
fn audit_air_test() {
    let mut rng = create_seeded_rng();

    const MAX_ADDRESS_SPACE: usize = 4;
    const LIMB_BITS: usize = 29;
    const MAX_VAL: usize = 1 << LIMB_BITS;
    const WORD_SIZE: usize = 2;
    const DECOMP: usize = 8;
    const TRACE_HEIGHT: usize = 16;

    let mut random_f = |range: usize| Val::from_canonical_usize(rng.gen_range(0..range));

    let num_addresses = 10;

    let mut first_access = BTreeMap::new();
    let mut last_access = BTreeMap::new();

    for _ in 0..num_addresses {
        let address_space = random_f(MAX_ADDRESS_SPACE);
        let address = random_f(MAX_VAL);

        let data_read = [random_f(MAX_VAL); WORD_SIZE];
        let clk_read = random_f(MAX_VAL);

        let data_write = [random_f(MAX_VAL); WORD_SIZE];
        let clk_write = random_f(MAX_VAL);

        first_access.insert((address_space, address), (clk_read, data_read));
        last_access.insert((address_space, address), (clk_write, data_write));
    }

    let audit_airs: Vec<AuditAir<WORD_SIZE>> = (0..2)
        .map(|_| AuditAir::<WORD_SIZE>::new(2, LIMB_BITS, DECOMP))
        .collect();

    let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << DECOMP));

    let traces = vec![
        audit_airs[0].generate_trace(
            first_access.clone(),
            last_access.clone(),
            TRACE_HEIGHT,
            range_checker.clone(),
        ),
        audit_airs[1].generate_trace(
            last_access,
            first_access,
            TRACE_HEIGHT,
            range_checker.clone(),
        ),
        range_checker.generate_trace(),
    ];

    run_simple_test_no_pis(
        vec![&audit_airs[0], &audit_airs[1], &range_checker.air],
        traces,
    )
    .expect("Verification failed");
}
