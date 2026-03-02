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
use openvm_riscv_transpiler::BaseAluWOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
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
    crate::{adapters::Rv32BaseAluAdapterRecord, BaseAluCoreRecord, Rv32BaseAluChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{core::run_alu, BaseAluCoreAir, BaseAluFiller, Rv64BaseAluWChip, Rv64BaseAluWExecutor};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    base_alu_w::BaseAluCoreCols,
    test_utils::{
        generate_rv64_is_type_immediate, get_verification_error, rv64_rand_write_register_or_imm,
    },
    Rv64BaseAluWAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64BaseAluWExecutor, Rv64BaseAluWAir, Rv64BaseAluWChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64BaseAluWAir, Rv64BaseAluWExecutor, Rv64BaseAluWChip<F>) {
    let air = Rv64BaseAluWAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        BaseAluCoreAir::new(bitwise_chip.bus(), BaseAluWOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BaseAluWExecutor::new(
        Rv64BaseAluAdapterExecutor::new(),
        BaseAluWOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BaseAluWChip::new(
        BaseAluFiller::new(
            Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            BaseAluWOpcode::CLASS_OFFSET,
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
    opcode: BaseAluWOpcode,
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

    let a = run_alu::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(opcode, &b, &c)
        .map(F::from_canonical_u8);
    assert_eq!(a, tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(ADDW, 100)]
#[test_case(SUBW, 100)]
fn rand_rv64w_alu_test(opcode: BaseAluWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write(2, 1024, [F::ONE; 8]);
    tester.write(2, 1032, [F::ONE; 8]);
    let sm: [F; 16] = tester.read(2, 1024);
    assert_eq!(sm, [F::ONE; 16]);

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

#[test_case(ADDW, 100)]
#[test_case(SUBW, 100)]
fn rand_rv64w_alu_test_persistent(opcode: BaseAluWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default_persistent();
    let (mut harness, bitwise) = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write(2, 1024, [F::ONE; 8]);
    tester.write(2, 1032, [F::ONE; 8]);
    let sm: [F; 16] = tester.read(2, 1024);
    assert_eq!(sm, [F::ONE; 16]);

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
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_alu_test(
    opcode: BaseAluWOpcode,
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_c: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    prank_opcode_flags: Option<[bool; 2]>,
    prank_word_sign: Option<u32>,
    is_imm: Option<bool>,
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
        is_imm,
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BaseAluCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_canonical_u32);
        }
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_addw_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_subw_flag = F::from_bool(prank_opcode_flags[1]);
        }
        if let Some(prank_word_sign) = prank_word_sign {
            cols.word_sign = F::from_canonical_u32(prank_word_sign);
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
fn rv64_alu_addw_wrong_negative_test() {
    run_negative_alu_test(
        ADDW,
        [246, 0, 0, 0, 0, 0, 0, 0],
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
    run_negative_alu_test(
        ADDW,
        [500, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv64_alu_subw_wrong_negative_test() {
    run_negative_alu_test(
        SUBW,
        [255, 0, 0, 0, 255, 255, 255, 255],
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
    run_negative_alu_test(
        SUBW,
        [F::NEG_ONE.as_canonical_u32(), 0, 0, 0, 255, 255, 255, 255],
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
fn rv64_aluw_adapter_unconstrained_imm_limb_test() {
    run_negative_alu_test(
        ADDW,
        [5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        Some([5, 0, 0, 0, 1, 0, 0, 0]),
        None,
        None,
        Some(true),
        false,
    );
}

#[test]
fn rv64_aluw_adapter_unconstrained_rs2_read_test() {
    run_negative_alu_test(
        ADDW,
        [2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        None,
        Some([false, false]),
        None,
        Some(false),
        false,
    );
}

#[test]
fn rv64_aluw_wrong_upper_sign_extension_negative_test() {
    run_negative_alu_test(
        ADDW,
        [5, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        Some(0),
        Some(false),
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_addw_sanity_test() {
    // Upper bytes are ignored by ADDW. Result upper bytes are sign-extension of byte 3.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [23, 205, 73, 49, 0, 0, 0, 0];
    let result = run_alu::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(ADDW, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_subw_sanity_test() {
    // Upper bytes are ignored by SUBW. Result upper bytes are sign-extension of byte 3.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [179, 118, 240, 172, 255, 255, 255, 255];
    let result = run_alu::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(SUBW, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv32BaseAluExecutor,
    Rv32BaseAluAir,
    Rv32BaseAluChipGpu,
    Rv32BaseAluChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32BaseAluChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BaseAluOpcode::ADD, 100)]
#[test_case(BaseAluOpcode::SUB, 100)]
#[test_case(BaseAluOpcode::XOR, 100)]
#[test_case(BaseAluOpcode::OR, 100)]
#[test_case(BaseAluOpcode::AND, 100)]
fn test_cuda_rand_alu_tracegen(opcode: BaseAluOpcode, num_ops: usize) {
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
        &'a mut BaseAluCoreRecord<RV32_REGISTER_NUM_LIMBS>,
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
