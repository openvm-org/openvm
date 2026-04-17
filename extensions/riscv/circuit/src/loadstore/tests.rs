use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, MemoryConfig, PreflightExecutor,
    },
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS, offline_checker::MemoryBridge, SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
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
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32LoadStoreAdapterRecord, LoadStoreCoreRecord, Rv32LoadStoreChipGpu},
    openvm_circuit::arch::{
        testing::{
            default_var_range_checker_bus, dummy_range_checker, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::{
    run_write_data, selector_point_for_opcode_shift, LoadStoreCoreAir, LoadStoreCoreCols,
    Rv64LoadStoreChip, LOADSTORE_SELECTOR_WIDTH,
};
use crate::{
    adapters::{
        Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterCols, Rv64LoadStoreAdapterExecutor,
        Rv64LoadStoreAdapterFiller, RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    test_utils::get_verification_error,
    LoadStoreFiller, Rv64LoadStoreAir, Rv64LoadStoreExecutor,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;
type Harness = TestChipHarness<F, Rv64LoadStoreExecutor, Rv64LoadStoreAir, Rv64LoadStoreChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64LoadStoreAir,
    Rv64LoadStoreExecutor,
    Rv64LoadStoreChip<F>,
) {
    let air = Rv64LoadStoreAir::new(
        Rv64LoadStoreAdapterAir::new(
            memory_bridge,
            execution_bridge,
            range_checker_chip.bus(),
            address_bits,
        ),
        LoadStoreCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LoadStoreExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(address_bits),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreChip::<F>::new(
        LoadStoreFiller::new(
            Rv64LoadStoreAdapterFiller::new(address_bits, range_checker_chip),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
    rs1: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    mem_as: Option<usize>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or(rng.gen_range(0..2));
    let imm_ext = imm + imm_sign * 0xffff0000;

    let alignment = match opcode {
        LOADD | STORED => 3,
        LOADWU | STOREW => 2,
        LOADHU | STOREH => 1,
        LOADBU | STOREB => 0,
        _ => unreachable!("loadstore tests should not handle sign-extension load opcodes"),
    };

    let ptr_val: u32 = rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    let ptr = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
    let rs1 = rs1.unwrap_or([ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0]);
    let rs1_low = u32::from_le_bytes(rs1[..4].try_into().unwrap());
    let ptr_val = imm_ext.wrapping_add(rs1_low);
    let shift_amount = (ptr_val as usize) & (RV64_REGISTER_NUM_LIMBS - 1);

    let a = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let b = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    let is_load = [LOADD, LOADWU, LOADHU, LOADBU].contains(&opcode);
    let mem_as = mem_as.unwrap_or(if is_load {
        2
    } else {
        *[2, 3, 4].choose(rng).unwrap()
    });

    tester.write(1, b, rs1.map(F::from_canonical_u8));

    let mut prev_data: [F; RV64_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV64_CELL_BITS))));
    let mut read_data: [F; RV64_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV64_CELL_BITS))));

    if is_load {
        if a == 0 {
            prev_data = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
        }
        tester.write(1, a, prev_data);
        tester.write(mem_as, (ptr_val as usize) - shift_amount, read_data);
    } else {
        if mem_as == 4 {
            prev_data = array::from_fn(|_| rng.gen());
        }
        if a == 0 {
            read_data = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
        }
        tester.write(mem_as, (ptr_val as usize) - shift_amount, prev_data);
        tester.write(1, a, read_data);
    }

    let enabled_write = !(is_load & (a == 0));

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                1,
                mem_as,
                enabled_write as usize,
                imm_sign as usize,
            ],
        ),
    );

    let write_data = run_write_data(
        opcode,
        read_data.map(|x| x.as_canonical_u32() as u8),
        prev_data.map(|x| x.as_canonical_u32()),
        shift_amount,
    )
    .map(F::from_canonical_u32);
    if is_load {
        if enabled_write {
            assert_eq!(write_data, tester.read::<RV64_REGISTER_NUM_LIMBS>(1, a));
        } else {
            assert_eq!(
                [F::ZERO; RV64_REGISTER_NUM_LIMBS],
                tester.read::<RV64_REGISTER_NUM_LIMBS>(1, a)
            );
        }
    } else {
        assert_eq!(
            write_data,
            tester.read::<RV64_REGISTER_NUM_LIMBS>(mem_as, (ptr_val as usize) - shift_amount)
        );
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test_case(LOADBU, 100)]
#[test_case(LOADHU, 100)]
#[test_case(LOADD, 100)]
#[test_case(LOADWU, 100)]
#[test_case(STOREB, 100)]
#[test_case(STOREH, 100)]
#[test_case(STORED, 100)]
#[test_case(STOREW, 100)]
fn rand_loadstore_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    if [STORED, STOREW, STOREB, STOREH].contains(&opcode) {
        mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    }
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

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
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_loadwu_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(2),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_loadhu_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(2),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_storew_public_values_test() {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(3),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_stored_native_test() {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STORED,
        Some([0, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        Some(4),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct LoadStorePrankValues {
    rs1_data: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    read_data: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    prev_data: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    write_data: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    flags: Option<[u32; LOADSTORE_SELECTOR_WIDTH]>,
    is_load: Option<bool>,
    mem_as: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_loadstore_test(
    opcode: Rv64LoadStoreOpcode,
    rs1: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    prank_vals: LoadStorePrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        rs1,
        imm,
        imm_sign,
        None,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64LoadStoreAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut LoadStoreCoreCols<F, RV64_REGISTER_NUM_LIMBS> = core_row.borrow_mut();

        if let Some(rs1_data) = prank_vals.rs1_data {
            adapter_cols.rs1_data = rs1_data.map(F::from_canonical_u32);
        }
        if let Some(read_data) = prank_vals.read_data {
            core_cols.read_data = read_data.map(F::from_canonical_u32);
        }
        if let Some(prev_data) = prank_vals.prev_data {
            core_cols.prev_data = prev_data.map(F::from_canonical_u32);
        }
        if let Some(write_data) = prank_vals.write_data {
            core_cols.write_data = write_data.map(F::from_canonical_u32);
        }
        if let Some(flags) = prank_vals.flags {
            core_cols.selector = flags.map(F::from_canonical_u32);
        }
        if let Some(is_load) = prank_vals.is_load {
            core_cols.is_load = F::from_bool(is_load);
        }
        if let Some(mem_as) = prank_vals.mem_as {
            adapter_cols.mem_as = F::from_canonical_u32(mem_as);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn negative_wrong_opcode_tests() {
    run_negative_loadstore_test(
        LOADD,
        None,
        None,
        None,
        LoadStorePrankValues {
            is_load: Some(false),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        LOADBU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(1),
        None,
        LoadStorePrankValues {
            flags: Some(selector_point_for_opcode_shift(LOADBU, 0)),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STOREH,
        Some([11, 169, 76, 28, 0, 0, 0, 0]),
        Some(37121),
        None,
        LoadStorePrankValues {
            flags: Some(selector_point_for_opcode_shift(STOREH, 0)),
            is_load: Some(true),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadStorePrankValues {
            flags: Some(selector_point_for_opcode_shift(LOADWU, 0)),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn negative_invalid_rs1_tests() {
    run_negative_loadstore_test(
        LOADD,
        None,
        None,
        None,
        LoadStorePrankValues {
            rs1_data: Some([0, 0, 0, 0, 1, 0, 0, 0]),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn negative_write_data_tests() {
    run_negative_loadstore_test(
        LOADHU,
        Some([13, 11, 156, 23, 0, 0, 0, 0]),
        Some(43641),
        None,
        LoadStorePrankValues {
            rs1_data: None,
            read_data: Some([175, 33, 198, 250, 131, 74, 186, 29]),
            prev_data: Some([90, 121, 64, 205, 159, 213, 89, 34]),
            write_data: Some([175, 33, 0, 0, 0, 0, 0, 0]),
            flags: Some(selector_point_for_opcode_shift(LOADHU, 0)),
            is_load: Some(true),
            mem_as: None,
        },
        true,
    );

    run_negative_loadstore_test(
        STOREB,
        Some([45, 123, 87, 24, 0, 0, 0, 0]),
        Some(28122),
        Some(0),
        LoadStorePrankValues {
            rs1_data: None,
            read_data: Some([175, 33, 198, 250, 131, 74, 186, 29]),
            prev_data: Some([90, 121, 64, 205, 159, 213, 89, 34]),
            write_data: Some([175, 121, 64, 205, 159, 213, 89, 34]),
            flags: Some(selector_point_for_opcode_shift(STOREB, 3)),
            is_load: None,
            mem_as: None,
        },
        false,
    );

    run_negative_loadstore_test(
        LOADWU,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadStorePrankValues {
            rs1_data: None,
            read_data: Some([138, 45, 202, 76, 131, 74, 186, 29]),
            prev_data: Some([159, 213, 89, 34, 142, 67, 210, 88]),
            write_data: Some([138, 45, 202, 76, 0, 0, 0, 0]),
            flags: Some(selector_point_for_opcode_shift(LOADWU, 4)),
            is_load: Some(true),
            mem_as: None,
        },
        false,
    );
}

#[test]
fn negative_wrong_address_space_tests() {
    run_negative_loadstore_test(
        LOADD,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(3),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        LOADWU,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(4),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STOREW,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(1),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STORED,
        None,
        None,
        None,
        LoadStorePrankValues {
            mem_as: Some(1),
            ..Default::default()
        },
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn run_loadd_stored_sanity_test() {
    let read_data = [138, 45, 202, 76, 131, 74, 186, 29];
    let prev_data = [159, 213, 89, 34, 142, 67, 210, 88];
    assert_eq!(
        run_write_data(LOADD, read_data, prev_data, 0),
        read_data.map(u32::from)
    );
    assert_eq!(
        run_write_data(STORED, read_data, prev_data, 0),
        read_data.map(u32::from)
    );
}

#[test]
fn run_loadwu_storew_sanity_test() {
    let read_data = [138, 45, 202, 76, 131, 74, 186, 29];
    let prev_data = [159, 213, 89, 34, 142, 67, 210, 88];

    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 0),
        [138, 45, 202, 76, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADWU, read_data, prev_data, 4),
        [131, 74, 186, 29, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 0),
        [138, 45, 202, 76, 142, 67, 210, 88]
    );
    assert_eq!(
        run_write_data(STOREW, read_data, prev_data, 4),
        [159, 213, 89, 34, 138, 45, 202, 76]
    );
}

#[test]
fn run_storeh_sanity_test() {
    let read_data = [250, 123, 67, 198, 175, 33, 198, 250];
    let prev_data = [144, 56, 175, 92, 90, 121, 64, 205];

    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 0),
        [250, 123, 175, 92, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 2),
        [144, 56, 250, 123, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 4),
        [144, 56, 175, 92, 250, 123, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREH, read_data, prev_data, 6),
        [144, 56, 175, 92, 90, 121, 250, 123]
    );
}

#[test]
fn run_storeb_sanity_test() {
    let read_data = [221, 104, 58, 147, 175, 33, 198, 250];
    let prev_data = [199, 83, 243, 12, 90, 121, 64, 205];

    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 0),
        [221, 83, 243, 12, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 1),
        [199, 221, 243, 12, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 2),
        [199, 83, 221, 12, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 3),
        [199, 83, 243, 221, 90, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 4),
        [199, 83, 243, 12, 221, 121, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 5),
        [199, 83, 243, 12, 90, 221, 64, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 6),
        [199, 83, 243, 12, 90, 121, 221, 205]
    );
    assert_eq!(
        run_write_data(STOREB, read_data, prev_data, 7),
        [199, 83, 243, 12, 90, 121, 64, 221]
    );
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = [175, 33, 198, 250, 131, 74, 186, 29];
    let prev_data = [90, 121, 64, 205, 142, 67, 210, 88];

    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 0),
        [175, 33, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 2),
        [198, 250, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 4),
        [131, 74, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADHU, read_data, prev_data, 6),
        [186, 29, 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = [131, 74, 186, 29, 138, 45, 202, 76];
    let prev_data = [142, 67, 210, 88, 159, 213, 89, 34];

    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 0),
        [131, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 1),
        [74, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 2),
        [186, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 3),
        [29, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 4),
        [138, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 5),
        [45, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 6),
        [202, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        run_write_data(LOADBU, read_data, prev_data, 7),
        [76, 0, 0, 0, 0, 0, 0, 0]
    );
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv32LoadStoreExecutor,
    Rv32LoadStoreAir,
    Rv32LoadStoreChipGpu,
    Rv32LoadStoreChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_bus = default_var_range_checker_bus();
    let dummy_range_checker_chip = dummy_range_checker(range_bus);

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv32LoadStoreChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADW, 100)]
#[test_case(LOADBU, 100)]
#[test_case(LOADHU, 100)]
#[test_case(STOREW, 100)]
#[test_case(STOREB, 100)]
#[test_case(STOREH, 100)]
fn test_cuda_rand_load_store_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
    if [STOREW, STOREB, STOREH].contains(&opcode) {
        mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    }
    let mut tester = GpuChipTestBuilder::volatile(mem_config, default_var_range_checker_bus());

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
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv32LoadStoreAdapterRecord,
        &'a mut LoadStoreCoreRecord<RV32_REGISTER_NUM_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32LoadStoreAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
