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
    crate::{
        adapters::Rv64BaseAluAdapterRecord, Rv64SllChipGpu, Rv64SraChipGpu, Rv64SrlChipGpu,
        ShiftSplitRecord,
    },
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::core::run_shift;
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    ArithShiftFiller, LogicalShiftCoreAir, LogicalShiftFiller, Rv64SllAir, Rv64SllChip,
    Rv64SllExecutor, Rv64SraAir, Rv64SraChip, Rv64SraExecutor, Rv64SrlAir, Rv64SrlChip,
    Rv64SrlExecutor, ShiftCols, ShiftSraCols, Sll, SraCoreAir, Srl,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
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

    let (a, _, _) = run_shift::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(opcode, &b, &c);
    assert_eq!(
        a.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace passes all
// constraints. One harness is built per split chip.
//////////////////////////////////////////////////////////////////////////////////////

/// Generates a per-opcode test module with a harness constructor and a random positive test.
macro_rules! shift_chip_tests {
    ($md:ident, $op:expr, $Air:ty, $Exec:ty, $Chip:ty, $CoreAir:ty, $Filler:ident) => {
        mod $md {
            use super::*;

            pub(super) type Harness = TestChipHarness<F, $Exec, $Air, $Chip>;

            pub(super) fn create_harness_fields(
                memory_bridge: MemoryBridge,
                execution_bridge: ExecutionBridge,
                bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
                range_checker: Arc<VariableRangeCheckerChip>,
                memory_helper: SharedMemoryHelper<F>,
            ) -> ($Air, $Exec, $Chip) {
                let air = <$Air>::new(
                    Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
                    <$CoreAir>::new(
                        bitwise_chip.bus(),
                        range_checker.bus(),
                        ShiftOpcode::CLASS_OFFSET,
                    ),
                );
                let executor = <$Exec>::new(Rv64BaseAluAdapterExecutor, ShiftOpcode::CLASS_OFFSET);
                let chip = <$Chip>::new(
                    $Filler::new(
                        Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
                        bitwise_chip,
                        range_checker,
                        ShiftOpcode::CLASS_OFFSET,
                    ),
                    memory_helper,
                );
                (air, executor, chip)
            }

            pub(super) fn create_harness(
                tester: &VmChipTestBuilder<F>,
            ) -> (
                Harness,
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

            #[test]
            fn rand_test() {
                let mut rng = create_seeded_rng();
                let mut tester = VmChipTestBuilder::default();
                let (mut harness, bitwise_chip) = create_harness(&tester);

                for _ in 0..100 {
                    set_and_execute(
                        &mut tester,
                        &mut harness.executor,
                        &mut harness.arena,
                        &mut rng,
                        $op,
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
        }
    };
}

shift_chip_tests!(
    sll,
    SLL,
    Rv64SllAir,
    Rv64SllExecutor,
    Rv64SllChip<F>,
    LogicalShiftCoreAir<Sll, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    LogicalShiftFiller
);
shift_chip_tests!(
    srl,
    SRL,
    Rv64SrlAir,
    Rv64SrlExecutor,
    Rv64SrlChip<F>,
    LogicalShiftCoreAir<Srl, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    LogicalShiftFiller
);
shift_chip_tests!(
    sra,
    SRA,
    Rv64SraAir,
    Rv64SraExecutor,
    Rv64SraChip<F>,
    SraCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    ArithShiftFiller
);

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, set up a chip and run the test. We replace part of the
// trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
struct ShiftPrank {
    bit_multiplier: Option<u32>,
    b_sign: Option<u32>,
    bit_shift_marker: Option<[u32; RV64_BYTE_BITS]>,
    limb_shift_marker: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    bit_shift_carry: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
}

/// Negative test for the logical shifts (SLL/SRL), which use [ShiftCols].
macro_rules! run_negative_logical_test {
    ($create:path, $op:expr, $prank_a:expr, $b:expr, $c:expr, $prank:expr) => {{
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
        let (mut harness, bitwise) = $create(&tester);

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            $op,
            Some($b),
            Some(false),
            Some($c),
        );

        let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
        let prank: ShiftPrank = $prank;
        let prank_a: [u32; RV64_REGISTER_NUM_LIMBS] = $prank_a;
        let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
            let mut values = trace.row_slice(0).unwrap().to_vec();
            let cols: &mut ShiftCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
                values.split_at_mut(adapter_width).1.borrow_mut();
            cols.a = prank_a.map(F::from_u32);
            if let Some(v) = prank.bit_multiplier {
                cols.bit_multiplier = F::from_u32(v);
            }
            if let Some(v) = prank.bit_shift_marker {
                cols.bit_shift_marker = v.map(F::from_u32);
            }
            if let Some(v) = prank.limb_shift_marker {
                cols.limb_shift_marker = v.map(F::from_u32);
            }
            if let Some(v) = prank.bit_shift_carry {
                cols.bit_shift_carry = v.map(F::from_u32);
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
    }};
}

/// Negative test for the arithmetic right shift (SRA), which uses [ShiftSraCols].
macro_rules! run_negative_sra_test {
    ($prank_a:expr, $b:expr, $c:expr, $prank:expr) => {{
        let mut rng = create_seeded_rng();
        let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
        let (mut harness, bitwise) = sra::create_harness(&tester);

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            SRA,
            Some($b),
            Some(false),
            Some($c),
        );

        let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
        let prank: ShiftPrank = $prank;
        let prank_a: [u32; RV64_REGISTER_NUM_LIMBS] = $prank_a;
        let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
            let mut values = trace.row_slice(0).unwrap().to_vec();
            let cols: &mut ShiftSraCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
                values.split_at_mut(adapter_width).1.borrow_mut();
            cols.a = prank_a.map(F::from_u32);
            if let Some(v) = prank.bit_multiplier {
                cols.bit_multiplier = F::from_u32(v);
            }
            if let Some(v) = prank.b_sign {
                cols.b_sign = F::from_u32(v);
            }
            if let Some(v) = prank.bit_shift_marker {
                cols.bit_shift_marker = v.map(F::from_u32);
            }
            if let Some(v) = prank.limb_shift_marker {
                cols.limb_shift_marker = v.map(F::from_u32);
            }
            if let Some(v) = prank.bit_shift_carry {
                cols.bit_shift_carry = v.map(F::from_u32);
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
    }};
}

#[test]
fn rv64_shift_wrong_result_negative_test() {
    // Leave `a` equal to the input even though the result should differ.
    let a = [1, 0, 0, 0, 0, 0, 0, 0];
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank::default();
    run_negative_logical_test!(sll::create_harness, SLL, a, b, c, prank);
    run_negative_logical_test!(srl::create_harness, SRL, a, b, c, prank);
    run_negative_sra_test!(a, b, c, prank);
}

#[test]
fn rv64_sll_wrong_bit_shift_negative_test() {
    let a = [0, 4, 4, 4, 4, 0, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 10, 100, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        bit_multiplier: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_logical_test!(sll::create_harness, SLL, a, b, c, prank);
}

#[test]
fn rv64_sll_wrong_limb_shift_negative_test() {
    let a = [0, 0, 2, 2, 2, 2, 0, 0];
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        limb_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_logical_test!(sll::create_harness, SLL, a, b, c, prank);
}

#[test]
fn rv64_sll_wrong_bit_carry_negative_test() {
    let a = [0, 510, 510, 510, 510, 510, 510, 510];
    let b = [255, 255, 255, 255, 255, 255, 255, 255];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        bit_shift_carry: Some([0, 0, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_logical_test!(sll::create_harness, SLL, a, b, c, prank);
}

#[test]
fn rv64_srl_wrong_bit_shift_negative_test() {
    // SRL([0,0,0,128,0,...], 9) correct result: [0,0,64,0,...].
    // Prank bit shift to 2 (shift=10): the c[0] decomposition no longer matches.
    let a = [0, 0, 32, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        bit_multiplier: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_logical_test!(srl::create_harness, SRL, a, b, c, prank);
}

#[test]
fn rv64_srl_wrong_limb_shift_negative_test() {
    let a = [0, 64, 0, 0, 0, 0, 0, 0];
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        limb_shift_marker: Some([0, 1, 0, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_logical_test!(srl::create_harness, SRL, a, b, c, prank);
}

#[test]
fn rv64_sra_wrong_bit_shift_negative_test() {
    // SRA([0,...,0,128], 9) correct: [0,...,192,255] (sign-extended).
    // Prank bit shift to 2 (shift=10): the c[0] decomposition no longer matches.
    let a = [0, 0, 0, 0, 0, 0, 224, 255];
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        bit_multiplier: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_sra_test!(a, b, c, prank);
}

#[test]
fn rv64_sra_wrong_sign_negative_test() {
    // `a` is the SRL result for this input (SRA would sign-extend instead).
    // `b` is negative (byte 7 has the sign bit set), so `b_sign` should be 1.
    // Pranking `b_sign` to 0 should cause a constraint error.
    let a = [0, 0, 0, 0, 0, 0, 64, 0];
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank = ShiftPrank {
        b_sign: Some(0),
        ..Default::default()
    };
    run_negative_sra_test!(a, b, c, prank);
}

#[test]
fn rv64_shift_adapter_imm_sign_extension_negative_test() {
    // Execute SLL with an immediate (shift by 1), then prank c[4] = 1 while sign byte
    // (c[2]) = 0. The shift core only uses c[0] so core constraints still hold, but
    // the adapter must catch that limbs 4-7 don't match the sign byte.
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = sll::create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        SLL,
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
        Some(true),
        Some([1, 0, 0, 0, 0, 0, 0, 0]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
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
        run_shift::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(SLL, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_REGISTER_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

#[test]
fn run_srl_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 45, 7, 61, 186];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [81, 190, 190, 190, 113, 20, 50, 80];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 228, 150, 131, 30, 93, 0, 0];
    let (result, limb_shift, bit_shift) =
        run_shift::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(SRL, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_REGISTER_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

#[test]
fn run_sra_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 45, 7, 61, 186];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [81, 20, 50, 80, 49, 190, 190, 113];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 228, 150, 131, 30, 221, 255, 255];
    let (result, limb_shift, bit_shift) =
        run_shift::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(SRA, &x, &y);
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
macro_rules! shift_cuda_test {
    ($test_name:ident, $op:expr, $create:path, $ChipGpu:ty) => {
        #[test]
        fn $test_name() {
            let mut rng = create_seeded_rng();
            let mut tester =
                GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

            let bitwise_bus = default_bitwise_lookup_bus();
            let range_bus = default_var_range_checker_bus();
            let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
                bitwise_bus,
            ));
            let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

            let (air, executor, cpu_chip) = $create(
                tester.memory_bridge(),
                tester.execution_bridge(),
                dummy_bitwise_chip,
                dummy_range_checker,
                tester.dummy_memory_helper(),
            );
            let gpu_chip = <$ChipGpu>::new(
                tester.range_checker(),
                tester.bitwise_op_lookup(),
                tester.timestamp_max_bits(),
            );
            let mut harness = GpuTestChipHarness::with_capacity(
                executor,
                air,
                gpu_chip,
                cpu_chip,
                MAX_INS_CAPACITY,
            );

            for _ in 0..100 {
                set_and_execute(
                    &mut tester,
                    &mut harness.executor,
                    &mut harness.dense_arena,
                    &mut rng,
                    $op,
                    None,
                    None,
                    None,
                );
            }

            type Record<'a> = (
                &'a mut Rv64BaseAluAdapterRecord,
                &'a mut ShiftSplitRecord<RV64_REGISTER_NUM_LIMBS>,
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
    };
}

#[cfg(feature = "cuda")]
shift_cuda_test!(
    test_cuda_rand_sll_tracegen,
    SLL,
    sll::create_harness_fields,
    Rv64SllChipGpu
);
#[cfg(feature = "cuda")]
shift_cuda_test!(
    test_cuda_rand_srl_tracegen,
    SRL,
    srl::create_harness_fields,
    Rv64SrlChipGpu
);
#[cfg(feature = "cuda")]
shift_cuda_test!(
    test_cuda_rand_sra_tracegen,
    SRA,
    sra::create_harness_fields,
    Rv64SraChipGpu
);
