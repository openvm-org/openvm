use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::{core::run_add_sub, AddSubCoreAir, Rv64AddSubChip, Rv64AddSubExecutor};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    add_sub::AddSubCoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    AddSubFiller, Rv64AddSubAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddSubExecutor, Rv64AddSubAir, Rv64AddSubChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddSubAir, Rv64AddSubExecutor, Rv64AddSubChip<F>) {
    let air = Rv64AddSubAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        AddSubCoreAir::new(bitwise_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddSubExecutor::new(
        Rv64BaseAluAdapterExecutor::new(),
        BaseAluOpcode::CLASS_OFFSET,
    );
    let chip = Rv64AddSubChip::new(
        AddSubFiller::new(
            Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            BaseAluOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BaseAluOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    is_imm: Option<bool>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (c_imm, c) = if is_imm.unwrap_or(rng.random_bool(0.5)) {
        let (imm, c) = if let Some(c) = c {
            ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
        } else {
            generate_rv64_is_type_immediate(rng)
        };
        (Some(imm), c)
    } else {
        (
            None,
            c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX))),
        )
    };

    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        c_imm,
        opcode.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let a = run_add_sub::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(opcode, &b, &c).map(F::from_u8);
    assert_eq!(a, tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(ADD, 100)]
#[test_case(SUB, 100)]
fn rand_rv64_add_sub_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write(2, 1024, [F::ONE; 8]);
    tester.write(2, 1032, [F::ONE; 8]);
    let sm_lo: [F; 8] = tester.read(2, 1024);
    let sm_hi: [F; 8] = tester.read(2, 1032);
    assert_eq!(sm_lo, [F::ONE; 8]);
    assert_eq!(sm_hi, [F::ONE; 8]);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(ADD, 100)]
#[test_case(SUB, 100)]
fn rand_rv64_add_sub_test_persistent(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write(2, 1024, [F::ONE; 8]);
    tester.write(2, 1032, [F::ONE; 8]);
    let sm_lo: [F; 8] = tester.read(2, 1024);
    let sm_hi: [F; 8] = tester.read(2, 1032);
    assert_eq!(sm_lo, [F::ONE; 8]);
    assert_eq!(sm_hi, [F::ONE; 8]);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_alu_test(
    opcode: BaseAluOpcode,
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_c: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    prank_opcode_flags: Option<[bool; 2]>,
    is_imm: Option<bool>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        is_imm,
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddSubCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_u32);
        }
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_add_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_sub_flag = F::from_bool(prank_opcode_flags[1]);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_alu_add_wrong_negative_test() {
    run_negative_alu_test(
        ADD,
        [246, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv64_alu_add_out_of_range_negative_test() {
    run_negative_alu_test(
        ADD,
        [500, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv64_alu_sub_wrong_negative_test() {
    run_negative_alu_test(
        SUB,
        [255, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv64_alu_sub_out_of_range_negative_test() {
    run_negative_alu_test(
        SUB,
        [F::NEG_ONE.as_canonical_u32(), 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv64_alu_adapter_unconstrained_imm_limb_test() {
    run_negative_alu_test(
        ADD,
        [255, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [255, 7, 0, 0, 0, 0, 0, 0],
        Some([511, 6, 0, 0, 0, 0, 0, 0]),
        None,
        Some(true),
        true,
    );
}

#[test]
fn rv64_alu_adapter_unconstrained_rs2_read_test() {
    run_negative_alu_test(
        ADD,
        [2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        None,
        Some([false, false]),
        Some(false),
        false,
    );
}

#[test]
fn rv64_alu_adapter_imm_sign_extension_negative_test() {
    // Prank c[4] = 1 while sign byte (c[2]) = 0. The adapter must catch that
    // limbs 4-7 don't match the sign byte. Also prank a[4] = 1 so the ADD core
    // constraint (a = b + c) still holds.
    run_negative_alu_test(
        ADD,
        [5, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        Some([5, 0, 0, 0, 1, 0, 0, 0]),
        None,
        Some(true),
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [23, 205, 73, 49, 219, 69, 50, 155];
    let result = run_add_sub::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(ADD, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_sub_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [179, 118, 240, 172, 71, 255, 255, 254];
    let result = run_add_sub::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(SUB, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
