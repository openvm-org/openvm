use std::sync::Arc;

use afs_chips::range_gate::RangeCheckerGateChip;
use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;

use super::{offline_checker::OfflineChecker, OpType, Operation};

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

    let ops: Vec<Operation> = vec![
        Operation {
            clk: 0,
            addr_space: 0,
            pointer: 0,
            data: vec![22783, 433, 1778],
            op_type: OpType::Write,
        },
        Operation {
            clk: 1,
            addr_space: 0,
            pointer: 1,
            data: vec![232, 888, 5954],
            op_type: OpType::Write,
        },
        Operation {
            clk: 2,
            addr_space: 0,
            pointer: 1,
            data: vec![232, 888, 5954],
            op_type: OpType::Read,
        },
        Operation {
            clk: 3,
            addr_space: 0,
            pointer: 1,
            data: vec![3243, 3214, 6639],
            op_type: OpType::Write,
        },
        Operation {
            clk: 4,
            addr_space: 1,
            pointer: 0,
            data: vec![231, 3883, 17],
            op_type: OpType::Write,
        },
        Operation {
            clk: 5,
            addr_space: 2,
            pointer: 0,
            data: vec![4382, 8837, 192],
            op_type: OpType::Write,
        },
        Operation {
            clk: 6,
            addr_space: 2,
            pointer: 0,
            data: vec![4382, 8837, 192],
            op_type: OpType::Read,
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
