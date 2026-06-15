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
use openvm_riscv_transpiler::ShiftOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64BaseAluAdapterRecord, Rv64ShiftLeftChipGpu, ShiftLeftCoreRecord},
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::{
    core::run_shift_left, Rv64ShiftLeftAir, Rv64ShiftLeftChip, Rv64ShiftLeftExecutor,
    ShiftLeftCoreAir, ShiftLeftCoreCols, ShiftLeftFiller,
};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type LeftHarness =
    TestChipHarness<F, Rv64ShiftLeftExecutor, Rv64ShiftLeftAir, Rv64ShiftLeftChip<F>>;

fn create_left_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    range_checker: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64ShiftLeftAir,
    Rv64ShiftLeftExecutor,
    Rv64ShiftLeftChip<F>,
) {
    let air = Rv64ShiftLeftAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        ShiftLeftCoreAir::new(
            bitwise_chip.bus(),
            range_checker.bus(),
            ShiftOpcode::CLASS_OFFSET,
        ),
    );
    let executor =
        Rv64ShiftLeftExecutor::new(Rv64BaseAluAdapterExecutor, ShiftOpcode::CLASS_OFFSET);
    let chip = Rv64ShiftLeftChip::<F>::new(
        ShiftLeftFiller::new(
            Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            range_checker,
            ShiftOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_left_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    LeftHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker().clone();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_left_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_checker,
        tester.memory_helper(),
    );
    let harness = LeftHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
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
    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, c_imm, SLL.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let (a, _, _) = run_shift_left::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(&b, &c);
    assert_eq!(
        a.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////
#[test]
fn run_rv64_shift_left_rand_test() {
    let num_ops = 100;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, bitwise_chip) = create_left_harness(&tester);
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
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
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct ShiftLeftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bit_multiplier: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
}

fn run_negative_shift_left_test(
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: ShiftLeftPrankValues<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_left_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftLeftCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        cols.a = prank_a.map(F::from_u32);
        if let Some(bit_multiplier) = prank_vals.bit_multiplier {
            cols.bit_multiplier = F::from_u32(bit_multiplier);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_u32);
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
fn rv64_shift_left_wrong_negative_test() {
    let a = [1, 0, 0, 0, 0, 0, 0, 0];
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = Default::default();
    run_negative_shift_left_test(a, b, c, prank_vals);
}

#[test]
fn rv64_sll_wrong_bit_shift_negative_test() {
    let a = [0, 4, 4, 4, 4, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 10, 100, 0, 0, 0, 0, 0];
    let prank_vals = ShiftLeftPrankValues {
        bit_multiplier: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_left_test(a, b, c, prank_vals);
}

#[test]
fn rv64_sll_wrong_limb_shift_negative_test() {
    let a = [0, 0, 2, 2, 2, 2, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftLeftPrankValues {
        limb_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_left_test(a, b, c, prank_vals);
}

#[test]
fn rv64_sll_wrong_bit_carry_negative_test() {
    let a = [0, 510, 510, 510, 510, 510, 510, 510];
    let b = [255, 255, 255, 255, 255, 255, 255, 255];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftLeftPrankValues {
        bit_shift_carry: Some([0, 0, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_left_test(a, b, c, prank_vals);
}

#[test]
fn rv64_sll_wrong_bit_mult_side_negative_test() {
    let a = [128, 128, 128, 0, 0, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftLeftPrankValues {
        bit_multiplier: Some(0),
        ..Default::default()
    };
    run_negative_shift_left_test(a, b, c, prank_vals);
}

#[test]
fn rv64_shift_adapter_imm_sign_extension_negative_test() {
    // Execute SLL with an immediate (shift by 1), then prank c[4] = 1 while sign byte
    // (c[2]) = 0. The shift core only uses c[0] so core constraints still hold, but
    // the adapter must catch that limbs 4-7 don't match the sign byte.
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_left_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
        Some(true),
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftLeftCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.c[4] = F::ONE;
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

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sll_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [45, 7, 61, 186, 31, 190, 221, 200];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [91, 0, 100, 0, 49, 190, 190, 113];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [0, 0, 0, 104, 57, 232, 209, 253];
    let (result, limb_shift, bit_shift) =
        run_shift_left::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(&x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_REGISTER_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuLeftHarness = GpuTestChipHarness<
    F,
    Rv64ShiftLeftExecutor,
    Rv64ShiftLeftAir,
    Rv64ShiftLeftChipGpu,
    Rv64ShiftLeftChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_left_harness(tester: &GpuChipTestBuilder) -> GpuLeftHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_bus = default_var_range_checker_bus();

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_left_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64ShiftLeftChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_shift_left_tracegen() {
    let num_ops = 100;
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_left_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            None,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluAdapterRecord,
        &'a mut ShiftLeftCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
