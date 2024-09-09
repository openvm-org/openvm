// use std::{collections::HashMap, iter, sync::Arc};

// use afs_primitives::var_range::VariableRangeCheckerChip;
// use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
// use ax_sdk::{
//     config::baby_bear_poseidon2::run_simple_test_no_pis,
//     interaction::dummy_interaction_air::DummyInteractionAir,
// };
// use p3_baby_bear::BabyBear;
// use p3_field::AbstractField;
// use p3_matrix::dense::RowMajorMatrix;

// use super::{offline_checker::MemoryChip, MemoryAccess, OpType};
// use crate::{
//     cpu::{MEMORY_BUS, RANGE_CHECKER_BUS},
//     vm::config::MemoryConfig,
// };

// const WORD_SIZE: usize = 3;
// const ADDR_SPACE_LIMB_BITS: usize = 8;
// const POINTER_LIMB_BITS: usize = 8;
// const CLK_LIMB_BITS: usize = 8;
// const DECOMP: usize = 4;
// const RANGE_MAX: u32 = 1 << DECOMP;

// const TRACE_DEGREE: usize = 16;

// #[test]
// fn test_offline_checker() {
//     let mem_config = MemoryConfig {
//         addr_space_max_bits: ADDR_SPACE_LIMB_BITS,
//         pointer_max_bits: POINTER_LIMB_BITS,
//         clk_max_bits: CLK_LIMB_BITS,
//         decomp: DECOMP,
//     };

//     let range_checker = Arc::new(VariableRangeCheckerChip::new(RANGE_CHECKER_BUS, RANGE_MAX));
//     let mut memory_chip = MemoryChip::new(mem_config, HashMap::new());
//     let requester = DummyInteractionAir::new(
//         2 + memory_chip.air.offline_checker.idx_data_width(),
//         true,
//         MEMORY_BUS,
//     );

//     let ops: Vec<MemoryAccess<WORD_SIZE, BabyBear>> = vec![
//         MemoryAccess {
//             timestamp: 1,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::one(),
//             data: [
//                 BabyBear::from_canonical_usize(232),
//                 BabyBear::from_canonical_usize(888),
//                 BabyBear::from_canonical_usize(5954),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 0,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::zero(),
//             data: [
//                 BabyBear::from_canonical_usize(2324),
//                 BabyBear::from_canonical_usize(433),
//                 BabyBear::from_canonical_usize(1778),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 4,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::zero(),
//             data: [
//                 BabyBear::from_canonical_usize(231),
//                 BabyBear::from_canonical_usize(3883),
//                 BabyBear::from_canonical_usize(17),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 2,
//             op_type: OpType::Read,
//             address_space: BabyBear::one(),
//             address: BabyBear::one(),
//             data: [
//                 BabyBear::from_canonical_usize(232),
//                 BabyBear::from_canonical_usize(888),
//                 BabyBear::from_canonical_usize(5954),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 6,
//             op_type: OpType::Read,
//             address_space: BabyBear::two(),
//             address: BabyBear::zero(),
//             data: [
//                 BabyBear::from_canonical_usize(4382),
//                 BabyBear::from_canonical_usize(8837),
//                 BabyBear::from_canonical_usize(192),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 5,
//             op_type: OpType::Write,
//             address_space: BabyBear::two(),
//             address: BabyBear::zero(),
//             data: [
//                 BabyBear::from_canonical_usize(4382),
//                 BabyBear::from_canonical_usize(8837),
//                 BabyBear::from_canonical_usize(192),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 3,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::one(),
//             data: [
//                 BabyBear::from_canonical_usize(3243),
//                 BabyBear::from_canonical_usize(3214),
//                 BabyBear::from_canonical_usize(6639),
//             ],
//         },
//     ];
//     let mut ops_sorted = ops.clone();
//     ops_sorted.sort_by_key(|op| op.timestamp);

//     for op in ops_sorted.iter() {
//         match op.op_type {
//             OpType::Read => {
//                 assert_eq!(
//                     memory_chip.read_word(op.timestamp, op.address_space, op.address),
//                     op.data
//                 );
//             }
//             OpType::Write => {
//                 memory_chip.write_word(op.timestamp, op.address_space, op.address, op.data);
//             }
//         }
//     }

//     let trace = memory_chip.generate_trace(range_checker.clone());
//     let range_checker_trace = range_checker.generate_trace();
//     let requester_trace = RowMajorMatrix::new(
//         ops.iter()
//             .flat_map(|op: &MemoryAccess<WORD_SIZE, BabyBear>| {
//                 [
//                     BabyBear::one(),
//                     BabyBear::from_canonical_usize(op.timestamp),
//                     BabyBear::from_canonical_u8(op.op_type as u8),
//                     op.address_space,
//                     op.address,
//                 ]
//                 .into_iter()
//                 .chain(op.data.iter().cloned())
//             })
//             .chain(
//                 iter::repeat_with(|| {
//                     iter::repeat(BabyBear::zero()).take(1 + requester.field_width())
//                 })
//                 .take(TRACE_DEGREE - ops.len())
//                 .flatten(),
//             )
//             .collect(),
//         1 + requester.field_width(),
//     );

//     run_simple_test_no_pis(
//         vec![&memory_chip.air, &range_checker.air, &requester],
//         vec![trace, range_checker_trace, requester_trace],
//     )
//     .expect("Verification failed");
// }

// #[test]
// fn test_offline_checker_valid_first_read() {
//     let mem_config = MemoryConfig {
//         addr_space_max_bits: ADDR_SPACE_LIMB_BITS,
//         pointer_max_bits: POINTER_LIMB_BITS,
//         clk_max_bits: CLK_LIMB_BITS,
//         decomp: DECOMP,
//     };

//     let range_checker = Arc::new(VariableRangeCheckerChip::new(RANGE_CHECKER_BUS, RANGE_MAX));
//     let mut memory_chip = MemoryChip::new(
//         ADDR_SPACE_LIMB_BITS,
//         POINTER_LIMB_BITS,
//         CLK_LIMB_BITS,
//         DECOMP,
//         HashMap::new(),
//     );
//     let requester = DummyInteractionAir::new(
//         2 + memory_chip.air.offline_checker.idx_data_width(),
//         true,
//         MEMORY_BUS,
//     );

//     memory_chip.write_word(
//         0,
//         BabyBear::one(),
//         BabyBear::zero(),
//         [BabyBear::zero(), BabyBear::zero(), BabyBear::zero()],
//     );
//     // read before writing, but first operation in block so should pass
//     memory_chip.accesses[0].op_type = OpType::Read;

//     let memory_trace = memory_chip.generate_trace(range_checker.clone());
//     let range_checker_trace = range_checker.generate_trace();
//     let requester_trace = RowMajorMatrix::new(
//         memory_chip
//             .accesses
//             .iter()
//             .flat_map(|op: &MemoryAccess<WORD_SIZE, BabyBear>| {
//                 iter::once(BabyBear::one())
//                     .chain(iter::once(BabyBear::from_canonical_usize(op.timestamp)))
//                     .chain(iter::once(BabyBear::from_canonical_u8(op.op_type as u8)))
//                     .chain(iter::once(op.address_space))
//                     .chain(iter::once(op.address))
//                     .chain(op.data.iter().cloned())
//             })
//             .chain(
//                 iter::repeat_with(|| {
//                     iter::repeat(BabyBear::zero()).take(1 + requester.field_width())
//                 })
//                 .take(TRACE_DEGREE - memory_chip.accesses.len())
//                 .flatten(),
//             )
//             .collect(),
//         1 + requester.field_width(),
//     );

//     run_simple_test_no_pis(
//         vec![&memory_chip.air, &range_checker.air, &requester],
//         vec![memory_trace, range_checker_trace, requester_trace],
//     )
//     .expect("Verification failed");
// }

// #[test]
// fn test_offline_checker_negative_data_mismatch() {
//     let range_checker = Arc::new(VariableRangeCheckerChip::new(RANGE_CHECKER_BUS, RANGE_MAX));
//     let mut memory_chip = MemoryChip::new(
//         ADDR_SPACE_LIMB_BITS,
//         POINTER_LIMB_BITS,
//         CLK_LIMB_BITS,
//         DECOMP,
//         HashMap::new(),
//     );
//     let requester = DummyInteractionAir::new(
//         2 + memory_chip.air.offline_checker.idx_data_width(),
//         true,
//         MEMORY_BUS,
//     );

//     let ops: Vec<MemoryAccess<WORD_SIZE, BabyBear>> = vec![
//         MemoryAccess {
//             timestamp: 0,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::zero(),
//             data: [
//                 BabyBear::from_canonical_usize(2324),
//                 BabyBear::from_canonical_usize(433),
//                 BabyBear::from_canonical_usize(1778),
//             ],
//         },
//         MemoryAccess {
//             timestamp: 1,
//             op_type: OpType::Write,
//             address_space: BabyBear::one(),
//             address: BabyBear::one(),
//             data: [
//                 BabyBear::from_canonical_usize(232),
//                 BabyBear::from_canonical_usize(888),
//                 BabyBear::from_canonical_usize(5954),
//             ],
//         },
//         // data read does not match write from previous operation
//         MemoryAccess {
//             timestamp: 2,
//             op_type: OpType::Read,
//             address_space: BabyBear::one(),
//             address: BabyBear::one(),
//             data: [
//                 BabyBear::from_canonical_usize(233),
//                 BabyBear::from_canonical_usize(888),
//                 BabyBear::from_canonical_usize(5954),
//             ],
//         },
//     ];

//     memory_chip.accesses.clone_from(&ops);

//     let trace = memory_chip.generate_trace(range_checker.clone());

//     let range_checker_trace = range_checker.generate_trace();
//     let requester_trace = RowMajorMatrix::new(
//         ops.iter()
//             .flat_map(|op: &MemoryAccess<WORD_SIZE, BabyBear>| {
//                 iter::once(BabyBear::one())
//                     .chain(iter::once(BabyBear::from_canonical_usize(op.timestamp)))
//                     .chain(iter::once(BabyBear::from_canonical_u8(op.op_type as u8)))
//                     .chain(iter::once(op.address_space))
//                     .chain(iter::once(op.address))
//                     .chain(op.data.iter().cloned())
//             })
//             .chain(
//                 iter::repeat_with(|| {
//                     iter::repeat(BabyBear::zero()).take(1 + requester.field_width())
//                 })
//                 .take(TRACE_DEGREE - ops.len())
//                 .flatten(),
//             )
//             .collect(),
//         1 + requester.field_width(),
//     );

//     USE_DEBUG_BUILDER.with(|debug| {
//         *debug.lock().unwrap() = false;
//     });
//     assert_eq!(
//         run_simple_test_no_pis(
//             vec![&memory_chip.air, &range_checker.air, &requester,],
//             vec![trace, range_checker_trace, requester_trace],
//         ),
//         Err(VerificationError::OodEvaluationMismatch),
//         "Expected verification to fail, but it passed"
//     );
// }
