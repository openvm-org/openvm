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
use openvm_riscv_transpiler::BaseAluOpcode::{self, ADD};
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
use test_case::test_case;

use super::{
    core::run_add, AddCoreAir, AddCoreCols, AddFiller, Rv64AddAir, Rv64AddChip, Rv64AddExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddExecutor, Rv64AddAir, Rv64AddChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddAir, Rv64AddExecutor, Rv64AddChip<F>) {
    let air = Rv64AddAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        AddCoreAir::new(bitwise_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddExecutor::new(
        Rv64BaseAluAdapterExecutor::new(),
        BaseAluOpcode::CLASS_OFFSET,
    );
    let chip = Rv64AddChip::new(
        AddFiller::new(
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
        rv64_rand_write_register_or_imm(tester, b, c, c_imm, ADD.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let a = run_add::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(&b, &c).map(F::from_u8);
    assert_eq!(a, tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(100)]
fn rand_rv64_add_test(num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

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
fn run_negative_add_test(
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_c: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    prank_is_valid: Option<bool>,
    is_imm: Option<bool>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(b),
        is_imm,
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_u32);
        }
        if let Some(prank_is_valid) = prank_is_valid {
            cols.is_valid = F::from_bool(prank_is_valid);
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
fn rv64_add_wrong_negative_test() {
    run_negative_add_test(
        [246, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_out_of_range_negative_test() {
    run_negative_add_test(
        [500, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_adapter_unconstrained_imm_limb_test() {
    run_negative_add_test(
        [255, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [255, 7, 0, 0, 0, 0, 0, 0],
        Some([511, 6, 0, 0, 0, 0, 0, 0]),
        None,
        Some(true),
    );
}

#[test]
fn rv64_add_adapter_unconstrained_rs2_read_test() {
    run_negative_add_test(
        [2, 2, 2, 2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        None,
        Some(false),
        Some(false),
    );
}

#[test]
fn rv64_add_adapter_imm_sign_extension_negative_test() {
    run_negative_add_test(
        [5, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0],
        Some([5, 0, 0, 0, 1, 0, 0, 0]),
        None,
        Some(true),
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [23, 205, 73, 49, 219, 69, 50, 155];
    let result = run_add::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(&x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
