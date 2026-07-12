use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluWOpcode::{self, *};
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
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64BaseAluWRegU16AdapterRecord, AddSubCoreRecord, Rv64AddSubWChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{AddSubWCoreAir, AddSubWFiller, Rv64AddSubWChip, Rv64AddSubWExecutor};
use crate::{
    adapters::{
        Rv64BaseAluWRegU16AdapterAir, Rv64BaseAluWRegU16AdapterCols,
        Rv64BaseAluWRegU16AdapterExecutor, Rv64BaseAluWRegU16AdapterFiller,
        RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS, RV64_WORD_U16_LIMBS, U16_BITS,
    },
    add_sub::AddSubCoreCols,
    test_utils::rv64_rand_write_register_or_imm,
    Rv64AddSubWAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddSubWExecutor, Rv64AddSubWAir, Rv64AddSubWChip<F>>;
type AddSubWCoreCols<T> = AddSubCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

#[inline(always)]
fn run_alu_w(
    opcode: BaseAluWOpcode,
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let x = u32::from_le_bytes(*x);
    let y = u32::from_le_bytes(*y);
    let rd_word = match opcode {
        ADDW => x.wrapping_add(y),
        SUBW => x.wrapping_sub(y),
    };
    (rd_word as i32 as i64 as u64).to_le_bytes()
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddSubWAir, Rv64AddSubWExecutor, Rv64AddSubWChip<F>) {
    let air = Rv64AddSubWAir::new(
        Rv64BaseAluWRegU16AdapterAir::new(
            execution_bridge,
            memory_bridge,
            range_checker_chip.bus(),
        ),
        AddSubWCoreAir::new(range_checker_chip.bus(), BaseAluWOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddSubWExecutor::new(
        Rv64BaseAluWRegU16AdapterExecutor,
        BaseAluWOpcode::CLASS_OFFSET,
    );
    let chip = Rv64AddSubWChip::new(
        AddSubWFiller::new(
            Rv64BaseAluWRegU16AdapterFiller::new(range_checker_chip.clone()),
            range_checker_chip,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BaseAluWOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));

    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let b_word: [u8; RV64_WORD_NUM_LIMBS] = b[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let c_word: [u8; RV64_WORD_NUM_LIMBS] = c[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let expected = run_alu_w(opcode, &b_word, &c_word);
    assert_eq!(
        expected.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
    expected
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(ADDW, 100)]
#[test_case(SUBW, 100)]
fn rand_rv64_add_sub_w_test(opcode: BaseAluWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write_bytes(2, 1024, [F::ONE; RV64_REGISTER_NUM_LIMBS]);
    tester.write_bytes(2, 1032, [F::ONE; RV64_REGISTER_NUM_LIMBS]);
    let sm_lo: [F; RV64_REGISTER_NUM_LIMBS] = tester.read_bytes(2, 1024);
    let sm_hi: [F; RV64_REGISTER_NUM_LIMBS] = tester.read_bytes(2, 1032);
    assert_eq!(sm_lo, [F::ONE; RV64_REGISTER_NUM_LIMBS]);
    assert_eq!(sm_hi, [F::ONE; RV64_REGISTER_NUM_LIMBS]);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_alu_w_test(
    opcode: BaseAluWOpcode,
    prank_a: [u32; RV64_WORD_U16_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_b: Option<[u32; RV64_WORD_U16_LIMBS]>,
    prank_c: Option<[u32; RV64_WORD_U16_LIMBS]>,
    prank_opcode_flags: Option<[bool; 2]>,
    prank_result_sign: Option<u32>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64BaseAluWRegU16AdapterCols<F> = adapter_row.borrow_mut();
        let cols: &mut AddSubWCoreCols<F> = core_row.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_b) = prank_b {
            cols.b = prank_b.map(F::from_u32);
        }
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_u32);
        }
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_add_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_sub_flag = F::from_bool(prank_opcode_flags[1]);
        }
        if let Some(prank_result_sign) = prank_result_sign {
            adapter_cols.result_sign = F::from_u32(prank_result_sign);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_alu_addw_wrong_negative_test() {
    // Real low-word result is [500, 0]; pranking a[0] breaks the ADD carry constraint.
    run_negative_alu_w_test(
        ADDW,
        [246, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv64_alu_addw_out_of_range_negative_test() {
    // b[0] = c[0] = 65535; the correct low word is [65534, 1]. Pranking a = [131070, 0]
    // satisfies every carry constraint, so only the 16-bit range check on a[0] (or the
    // memory write interaction) can catch it.
    run_negative_alu_w_test(
        ADDW,
        [131070, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv64_alu_subw_wrong_negative_test() {
    // Real low-word result is [65535, 65535]; pranking a[1] breaks the SUB carry constraint.
    run_negative_alu_w_test(
        SUBW,
        [65535, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv64_alu_subw_out_of_range_negative_test() {
    // a[0] = -1 in the field satisfies the SUB carry constraints for 1 - 2, but is not a
    // canonical u16, so only the range check on a[0] can catch it.
    run_negative_alu_w_test(
        SUBW,
        [F::NEG_ONE.as_canonical_u32(), 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv64_aluw_adapter_unconstrained_rs2_read_test() {
    // Both opcode flags false => is_valid = 0, so the rs2 register read must be forced off.
    run_negative_alu_w_test(
        ADDW,
        [514, 514],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        None,
        None,
        Some([false, false]),
        None,
        false,
    );
}

#[test]
fn rv64_aluw_wrong_upper_sign_extension_negative_test() {
    // Positive result (sign should be 0); pranking result_sign = 1 forces a 0xFFFF upper write
    // that mismatches memory and breaks the sign decomposition range check.
    run_negative_alu_w_test(
        ADDW,
        [5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        Some(1),
        true,
    );
}

#[test]
fn rv64_aluw_wrong_upper_sign_extension_negative_to_zero_test() {
    // Result 0x80000000 (low word [0, 0x8000]) is negative; pranking result_sign = 0 forces a
    // 0x0000 upper write and pushes a[1] - sign*2^15 = 0x8000 out of the 15-bit range.
    run_negative_alu_w_test(
        ADDW,
        [0, 0x8000],
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        Some(0),
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_addw_noncanonical_upper_bytes_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    let rs1 = [170u8, 16, 32, 48, 13, 14, 15, 16];
    let rs2 = [1u8, 2, 3, 4, 21, 22, 23, 24];
    // 0x302010AA + 0x04030201 = 0x342312AB (positive, sign-extended to 0x00000000_342312AB)
    let expected: [u8; RV64_REGISTER_NUM_LIMBS] = 0x00000000_342312ABu64.to_le_bytes();
    let result = set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        ADDW,
        Some(rs1),
        Some(rs2),
    );
    assert_eq!(result, expected);

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn run_subw_noncanonical_upper_bytes_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    let rs1 = [0u8, 0, 0, 128, 13, 14, 15, 16];
    let rs2 = [1u8, 0, 0, 0, 21, 22, 23, 24];
    // 0x80000000 - 0x00000001 = 0x7FFFFFFF (positive, sign-extended to 0x00000000_7FFFFFFF)
    let expected: [u8; RV64_REGISTER_NUM_LIMBS] = 0x00000000_7FFFFFFFu64.to_le_bytes();
    let result = set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        SUBW,
        Some(rs1),
        Some(rs2),
    );
    assert_eq!(result, expected);

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64AddSubWExecutor,
    Rv64AddSubWAir,
    Rv64AddSubWChipGpu,
    Rv64AddSubWChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddSubWChipGpu::new(
        tester.range_checker(),
        tester.timestamp_max_bits(),
        Default::default(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BaseAluWOpcode::ADDW, 100)]
#[test_case(BaseAluWOpcode::SUBW, 100)]
fn test_cuda_rand_add_sub_w_tracegen(opcode: BaseAluWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

    let mut harness = create_cuda_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluWRegU16AdapterRecord,
        &'a mut AddSubCoreRecord<RV64_WORD_U16_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluWRegU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
