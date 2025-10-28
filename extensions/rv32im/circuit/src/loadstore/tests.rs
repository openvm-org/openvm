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
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_AS, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
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

use super::{run_write_data, LoadStoreCoreAir, Rv32LoadStoreChip};
use crate::{
    adapters::{
        Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterCols, Rv32LoadStoreAdapterExecutor,
        Rv32LoadStoreAdapterFiller, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    loadstore::LoadStoreCoreCols,
    test_utils::get_verification_error,
    LoadStoreFiller, Rv32LoadStoreAir, Rv32LoadStoreExecutor,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;
type Harness = TestChipHarness<F, Rv32LoadStoreExecutor, Rv32LoadStoreAir, Rv32LoadStoreChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv32LoadStoreAir,
    Rv32LoadStoreExecutor,
    Rv32LoadStoreChip<F>,
) {
    let air = Rv32LoadStoreAir::new(
        Rv32LoadStoreAdapterAir::new(
            memory_bridge,
            execution_bridge,
            range_checker_chip.bus(),
            address_bits,
        ),
        LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = Rv32LoadStoreExecutor::new(
        Rv32LoadStoreAdapterExecutor::new(address_bits),
        Rv32LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv32LoadStoreChip::<F>::new(
        LoadStoreFiller::new(
            Rv32LoadStoreAdapterFiller::new(address_bits, range_checker_chip),
            Rv32LoadStoreOpcode::CLASS_OFFSET,
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
    opcode: Rv32LoadStoreOpcode,
    rs1: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    mem_as: Option<usize>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or(rng.gen_range(0..2));
    let imm_ext = imm + imm_sign * 0xffff0000;

    let alignment = match opcode {
        LOADW | STOREW => 2,
        LOADHU | STOREH => 1,
        LOADBU | STOREB => 0,
        _ => unreachable!(),
    };

    let ptr_val: u32 = rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    let rs1 = rs1.unwrap_or(ptr_val.wrapping_sub(imm_ext).to_le_bytes());
    let ptr_val = imm_ext.wrapping_add(u32::from_le_bytes(rs1));
    let a = gen_pointer(rng, 4);
    let b = gen_pointer(rng, 4);

    let is_load = [LOADW, LOADHU, LOADBU].contains(&opcode);
    let mem_as = mem_as.unwrap_or(if is_load {
        2
    } else {
        *[2, 3, 4].choose(rng).unwrap()
    });

    let shift_amount = ptr_val % 4;
    tester.write(1, b, rs1.map(F::from_canonical_u8));

    let mut some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
    let mut read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));

    if is_load {
        if a == 0 {
            some_prev_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        }
        tester.write(1, a, some_prev_data);
        tester.write(mem_as, (ptr_val - shift_amount) as usize, read_data);
    } else {
        if mem_as == 4 {
            some_prev_data = array::from_fn(|_| rng.gen());
        }
        if a == 0 {
            read_data = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
        }
        tester.write(mem_as, (ptr_val - shift_amount) as usize, some_prev_data);
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
        some_prev_data.map(|x| x.as_canonical_u32()),
        shift_amount as usize,
    )
    .map(F::from_canonical_u32);
    if is_load {
        if enabled_write {
            assert_eq!(write_data, tester.read::<4>(1, a));
        } else {
            assert_eq!([F::ZERO; RV32_REGISTER_NUM_LIMBS], tester.read::<4>(1, a));
        }
    } else {
        assert_eq!(
            write_data,
            tester.read::<4>(mem_as, (ptr_val - shift_amount) as usize)
        );
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test_case(LOADW, 100)]
#[test_case(LOADBU, 100)]
#[test_case(LOADHU, 100)]
#[test_case(STOREW, 100)]
#[test_case(STOREB, 100)]
#[test_case(STOREH, 100)]
fn rand_loadstore_test(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
    if [STOREW, STOREB, STOREH].contains(&opcode) {
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

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct LoadStorePrankValues {
    read_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    prev_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    write_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    flags: Option<[u32; 4]>,
    is_load: Option<bool>,
    mem_as: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_loadstore_test(
    opcode: Rv32LoadStoreOpcode,
    rs1: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    prank_vals: LoadStorePrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 1 << 29;
    if [STOREW, STOREB, STOREH].contains(&opcode) {
        mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    }
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
        let adapter_cols: &mut Rv32LoadStoreAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut LoadStoreCoreCols<F, RV32_REGISTER_NUM_LIMBS> = core_row.borrow_mut();

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
            core_cols.flags = flags.map(F::from_canonical_u32);
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
        LOADW,
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
        Some([4, 0, 0, 0]),
        Some(1),
        None,
        LoadStorePrankValues {
            flags: Some([0, 0, 0, 2]),
            ..Default::default()
        },
        false,
    );

    run_negative_loadstore_test(
        STOREH,
        Some([11, 169, 76, 28]),
        Some(37121),
        None,
        LoadStorePrankValues {
            flags: Some([1, 0, 1, 0]),
            is_load: Some(true),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn negative_write_data_tests() {
    run_negative_loadstore_test(
        LOADHU,
        Some([13, 11, 156, 23]),
        Some(43641),
        None,
        LoadStorePrankValues {
            read_data: Some([175, 33, 198, 250]),
            prev_data: Some([90, 121, 64, 205]),
            write_data: Some([175, 33, 0, 0]),
            flags: Some([0, 2, 0, 0]),
            is_load: Some(true),
            mem_as: None,
        },
        true,
    );

    run_negative_loadstore_test(
        STOREB,
        Some([45, 123, 87, 24]),
        Some(28122),
        Some(0),
        LoadStorePrankValues {
            read_data: Some([175, 33, 198, 250]),
            prev_data: Some([90, 121, 64, 205]),
            write_data: Some([175, 121, 64, 205]),
            flags: Some([0, 0, 1, 1]),
            is_load: None,
            mem_as: None,
        },
        false,
    );
}

#[test]
fn negative_wrong_address_space_tests() {
    run_negative_loadstore_test(
        LOADW,
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
        LOADW,
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
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn run_loadw_storew_sanity_test() {
    let read_data = [138, 45, 202, 76];
    let prev_data = [159, 213, 89, 34];
    let store_write_data = run_write_data(STOREW, read_data, prev_data, 0);
    let load_write_data = run_write_data(LOADW, read_data, prev_data, 0);
    assert_eq!(store_write_data, read_data.map(u32::from));
    assert_eq!(load_write_data, read_data.map(u32::from));
}

#[test]
fn run_storeh_sanity_test() {
    let read_data = [250, 123, 67, 198];
    let prev_data = [144, 56, 175, 92];
    let write_data = run_write_data(STOREH, read_data, prev_data, 0);
    let write_data2 = run_write_data(STOREH, read_data, prev_data, 2);
    assert_eq!(write_data, [250, 123, 175, 92]);
    assert_eq!(write_data2, [144, 56, 250, 123]);
}

#[test]
fn run_storeb_sanity_test() {
    let read_data = [221, 104, 58, 147];
    let prev_data = [199, 83, 243, 12];
    let write_data = run_write_data(STOREB, read_data, prev_data, 0);
    let write_data1 = run_write_data(STOREB, read_data, prev_data, 1);
    let write_data2 = run_write_data(STOREB, read_data, prev_data, 2);
    let write_data3 = run_write_data(STOREB, read_data, prev_data, 3);
    assert_eq!(write_data, [221, 83, 243, 12]);
    assert_eq!(write_data1, [199, 221, 243, 12]);
    assert_eq!(write_data2, [199, 83, 221, 12]);
    assert_eq!(write_data3, [199, 83, 243, 221]);
}

#[test]
fn run_loadhu_sanity_test() {
    let read_data = [175, 33, 198, 250];
    let prev_data = [90, 121, 64, 205];
    let write_data = run_write_data(LOADHU, read_data, prev_data, 0);
    let write_data2 = run_write_data(LOADHU, read_data, prev_data, 2);
    assert_eq!(write_data, [175, 33, 0, 0]);
    assert_eq!(write_data2, [198, 250, 0, 0]);
}

#[test]
fn run_loadbu_sanity_test() {
    let read_data = [131, 74, 186, 29];
    let prev_data = [142, 67, 210, 88];
    let write_data = run_write_data(LOADBU, read_data, prev_data, 0);
    let write_data1 = run_write_data(LOADBU, read_data, prev_data, 1);
    let write_data2 = run_write_data(LOADBU, read_data, prev_data, 2);
    let write_data3 = run_write_data(LOADBU, read_data, prev_data, 3);
    assert_eq!(write_data, [131, 0, 0, 0]);
    assert_eq!(write_data1, [74, 0, 0, 0]);
    assert_eq!(write_data2, [186, 0, 0, 0]);
    assert_eq!(write_data3, [29, 0, 0, 0]);
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
fn test_cuda_rand_load_store_tracegen(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
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
