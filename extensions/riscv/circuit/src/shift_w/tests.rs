use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftWOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::FieldAlgebra,
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
    crate::{adapters::Rv32BaseAluAdapterRecord, Rv32ShiftChipGpu, ShiftCoreRecord},
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::{core::run_shift_w, Rv64ShiftWChip, ShiftWCoreAir, ShiftWCoreCols, ShiftWFiller};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
    },
    test_utils::{
        generate_rv64_is_type_immediate, get_verification_error, rv64_rand_write_register_or_imm,
    },
    Rv64ShiftWAir, Rv64ShiftWExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<F, Rv64ShiftWExecutor, Rv64ShiftWAir, Rv64ShiftWChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_checker: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64ShiftWAir, Rv64ShiftWExecutor, Rv64ShiftWChip<F>) {
    let air = Rv64ShiftWAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        ShiftWCoreAir::new(
            bitwise_chip.bus(),
            range_checker.bus(),
            ShiftWOpcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64ShiftWExecutor::new(Rv64BaseAluAdapterExecutor, ShiftWOpcode::CLASS_OFFSET);
    let chip = Rv64ShiftWChip::<F>::new(
        ShiftWFiller::new(
            Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            range_checker,
            ShiftWOpcode::CLASS_OFFSET,
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
    let range_checker = tester.range_checker().clone();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_checker,
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
    opcode: ShiftWOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    is_imm: Option<bool>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let (c_imm, c) = if is_imm.unwrap_or(rng.gen_bool(0.5)) {
        let (imm, c) = if let Some(c) = c {
            ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
        } else {
            generate_rv64_is_type_immediate(rng)
        };
        (Some(imm), c)
    } else {
        (
            None,
            c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX))),
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

    let (a, _, _) = run_shift_w(opcode, &b, &c);
    assert_eq!(
        a.map(F::from_canonical_u8),
        tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(SLLW, 100)]
#[test_case(SRLW, 100)]
#[test_case(SRAW, 100)]
fn run_rv64w_shift_rand_test(opcode: ShiftWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise_chip) = create_harness(&tester);

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
        .load_periphery(bitwise_chip)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct ShiftPrankValues {
    pub bit_shift: Option<u32>,
    pub bit_multiplier_left: Option<u32>,
    pub bit_multiplier_right: Option<u32>,
    pub b_sign: Option<u32>,
    pub word_sign: Option<u32>,
    pub bit_shift_marker: Option<[u32; RV64_CELL_BITS]>,
    pub limb_shift_marker: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_shift_test(
    opcode: ShiftWOpcode,
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: ShiftPrankValues,
    interaction_error: bool,
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
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut ShiftWCoreCols<F> = values.split_at_mut(adapter_width).1.borrow_mut();

        cols.a = prank_a.map(F::from_canonical_u32);
        if let Some(bit_multiplier_left) = prank_vals.bit_multiplier_left {
            cols.bit_multiplier_left = F::from_canonical_u32(bit_multiplier_left);
        }
        if let Some(bit_multiplier_right) = prank_vals.bit_multiplier_right {
            cols.bit_multiplier_right = F::from_canonical_u32(bit_multiplier_right);
        }
        if let Some(b_sign) = prank_vals.b_sign {
            cols.b_sign = F::from_canonical_u32(b_sign);
        }
        if let Some(word_sign) = prank_vals.word_sign {
            cols.word_sign = F::from_canonical_u32(word_sign);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_canonical_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_canonical_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_canonical_u32);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv64_shiftw_wrong_negative_test() {
    let a = [1, 0, 0, 0, 0, 0, 0, 0];
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = Default::default();
    run_negative_shift_test(SLLW, a, b, c, prank_vals, false);
    run_negative_shift_test(SRLW, a, b, c, prank_vals, false);
    run_negative_shift_test(SRAW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_sllw_wrong_bit_shift_negative_test() {
    let a = [0, 4, 4, 4, 0, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 10, 100, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLLW, a, b, c, prank_vals, true);
}

#[test]
fn rv64_sllw_wrong_limb_shift_negative_test() {
    let a = [0, 0, 2, 2, 0, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLLW, a, b, c, prank_vals, true);
}

#[test]
fn rv64_sllw_wrong_bit_carry_negative_test() {
    let a = [0, 510, 510, 510, 0, 0, 0, 0];
    let b = [255, 255, 255, 255, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_carry: Some([0, 0, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLLW, a, b, c, prank_vals, true);
}

#[test]
fn rv64_sllw_wrong_bit_mult_side_negative_test() {
    let a = [128, 128, 128, 0, 0, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(0),
        bit_multiplier_right: Some(1),
        ..Default::default()
    };
    run_negative_shift_test(SLLW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_srlw_wrong_bit_shift_negative_test() {
    let a = [0, 0, 32, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRLW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_srlw_wrong_limb_shift_negative_test() {
    let a = [0, 64, 0, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 1, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRLW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_srxw_wrong_bit_mult_side_negative_test() {
    let a = [0, 0, 0, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(1),
        bit_multiplier_right: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SRLW, a, b, c, prank_vals, false);
    run_negative_shift_test(SRAW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_sraw_wrong_bit_shift_negative_test() {
    let a = [0, 0, 224, 255, 255, 255, 255, 255];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRAW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_sraw_wrong_limb_shift_negative_test() {
    let a = [0, 192, 255, 255, 255, 255, 255, 255];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 1, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRAW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_sraw_wrong_sign_negative_test() {
    let a = [0, 0, 64, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        b_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SRAW, a, b, c, prank_vals, true);
}

#[test]
fn rv64_shiftw_wrong_upper_sign_extension_negative_test() {
    let a = [2, 0, 0, 0, 255, 0, 0, 0];
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        word_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SLLW, a, b, c, prank_vals, false);
}

#[test]
fn rv64_shiftw_adapter_imm_sign_extension_negative_test() {
    // Execute SLLW with an immediate (shift by 1), then prank c[4] = 1 while sign byte
    // (c[2]) = 0. The shift core only uses c[0] so core constraints still hold, but
    // the adapter must catch that limbs 4-7 don't match the sign byte.
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        SLLW,
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
        Some(true),
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut ShiftWCoreCols<F> = values.split_at_mut(adapter_width).1.borrow_mut();
        cols.c[4] = F::ONE;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(false));
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sllw_sanity_test() {
    // Upper bytes are ignored for SLLW. Result upper bytes sign-extend low 32-bit result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [45, 7, 61, 186, 31, 190, 221, 200];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [91, 0, 100, 0, 49, 190, 190, 113];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [0, 0, 0, 104, 0, 0, 0, 0];
    let (result, limb_shift, bit_shift) = run_shift_w(SLLW, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS);
    assert_eq!(shift / RV64_CELL_BITS, limb_shift);
    assert_eq!(shift % RV64_CELL_BITS, bit_shift);
}

#[test]
fn run_srlw_sanity_test() {
    // Upper bytes are ignored for SRLW. Result upper bytes sign-extend low 32-bit result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 45, 7, 61, 186];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [49, 190, 190, 190, 113, 20, 50, 80];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 100, 0, 0, 0, 0, 0, 0];
    let (result, limb_shift, bit_shift) = run_shift_w(SRLW, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS);
    assert_eq!(shift / RV64_CELL_BITS, limb_shift);
    assert_eq!(shift % RV64_CELL_BITS, bit_shift);
}

#[test]
fn run_sraw_sanity_test() {
    // Upper bytes are ignored for SRAW. Result upper bytes sign-extend low 32-bit result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 45, 7, 61, 186];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [113, 20, 50, 80, 49, 190, 190, 113];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 228, 255, 255, 255, 255, 255, 255];
    let (result, limb_shift, bit_shift) = run_shift_w(SRAW, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS);
    assert_eq!(shift / RV64_CELL_BITS, limb_shift);
    assert_eq!(shift % RV64_CELL_BITS, bit_shift);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv32ShiftExecutor, Rv32ShiftAir, Rv32ShiftChipGpu, Rv32ShiftChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_bus = default_var_range_checker_bus();

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32ShiftChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(ShiftOpcode::SLL, 100)]
#[test_case(ShiftOpcode::SRL, 100)]
#[test_case(ShiftOpcode::SRA, 100)]
fn test_cuda_rand_shift_tracegen(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

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
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv32BaseAluAdapterRecord,
        &'a mut ShiftCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32BaseAluAdapterExecutor<RV32_CELL_BITS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
