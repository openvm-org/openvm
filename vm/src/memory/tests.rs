use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{offline_checker::OfflineChecker, MemoryAccess, OpType};

#[test]
fn test_offline_checker() {
    let range_bus_index: usize = 0;
    let ops_bus_index: usize = 1;
    let data_len: usize = 3;
    let addr_space_limb_bits: usize = 8;
    let pointer_limb_bits: usize = 8;
    let clk_limb_bits: usize = 8;
    let decomp: usize = 4;
    let range_max: u32 = 1 << decomp;

    let trace_degree = 16;

    let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus_index, range_max));
    let offline_checker = OfflineChecker::new(
        range_bus_index,
        ops_bus_index,
        data_len,
        addr_space_limb_bits,
        pointer_limb_bits,
        clk_limb_bits,
        decomp,
    );

    let ops: Vec<MemoryAccess<BabyBear>> = vec![
        MemoryAccess {
            clock: 0,
            op_type: OpType::Write,
            address_space: BabyBear::zero(),
            address: BabyBear::zero(),
            data: vec![
                BabyBear::from_canonical_usize(2324),
                BabyBear::from_canonical_usize(433),
                BabyBear::from_canonical_usize(1778),
            ],
        },
        MemoryAccess {
            clock: 1,
            op_type: OpType::Write,
            address_space: BabyBear::zero(),
            address: BabyBear::one(),
            data: vec![
                BabyBear::from_canonical_usize(232),
                BabyBear::from_canonical_usize(888),
                BabyBear::from_canonical_usize(5954),
            ],
        },
        MemoryAccess {
            clock: 2,
            op_type: OpType::Read,
            address_space: BabyBear::zero(),
            address: BabyBear::one(),
            data: vec![
                BabyBear::from_canonical_usize(232),
                BabyBear::from_canonical_usize(888),
                BabyBear::from_canonical_usize(5954),
            ],
        },
        MemoryAccess {
            clock: 3,
            op_type: OpType::Write,
            address_space: BabyBear::zero(),
            address: BabyBear::one(),
            data: vec![
                BabyBear::from_canonical_usize(3243),
                BabyBear::from_canonical_usize(3214),
                BabyBear::from_canonical_usize(6639),
            ],
        },
        MemoryAccess {
            clock: 4,
            op_type: OpType::Write,
            address_space: BabyBear::one(),
            address: BabyBear::zero(),
            data: vec![
                BabyBear::from_canonical_usize(231),
                BabyBear::from_canonical_usize(3883),
                BabyBear::from_canonical_usize(17),
            ],
        },
        MemoryAccess {
            clock: 5,
            op_type: OpType::Write,
            address_space: BabyBear::two(),
            address: BabyBear::zero(),
            data: vec![
                BabyBear::from_canonical_usize(4382),
                BabyBear::from_canonical_usize(8837),
                BabyBear::from_canonical_usize(192),
            ],
        },
        MemoryAccess {
            clock: 6,
            op_type: OpType::Read,
            address_space: BabyBear::two(),
            address: BabyBear::zero(),
            data: vec![
                BabyBear::from_canonical_usize(4382),
                BabyBear::from_canonical_usize(8837),
                BabyBear::from_canonical_usize(192),
            ],
        },
    ];

    let offline_checker_trace =
        offline_checker.generate_trace(ops, range_checker.clone(), trace_degree);
    let range_checker_trace = range_checker.generate_trace();

    run_simple_test_no_pis(
        vec![&offline_checker, &range_checker.air],
        vec![offline_checker_trace, range_checker_trace],
    )
    .expect("Verification failed");
}
