use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
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
    super::{
        Rv64LoadSignExtendByteChipGpu, Rv64LoadSignExtendHalfwordChipGpu,
        Rv64LoadSignExtendWordChipGpu,
    },
    crate::{adapters::Rv64LoadStoreAdapterRecord, loadstore::LoadStoreRecord},
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use super::{
    byte::{
        LoadSignExtendByteCoreAir, LoadSignExtendByteCoreCols, LoadSignExtendByteFiller,
        Rv64LoadSignExtendByteAir, Rv64LoadSignExtendByteChip, Rv64LoadSignExtendByteExecutor,
    },
    halfword::{
        LoadSignExtendHalfwordCoreAir, LoadSignExtendHalfwordFiller, Rv64LoadSignExtendHalfwordAir,
        Rv64LoadSignExtendHalfwordChip, Rv64LoadSignExtendHalfwordExecutor,
        LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH,
    },
    word::{
        LoadSignExtendWordCoreAir, LoadSignExtendWordFiller, Rv64LoadSignExtendWordAir,
        Rv64LoadSignExtendWordChip, Rv64LoadSignExtendWordExecutor,
        LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH,
    },
};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16,
        Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor, Rv64LoadStoreAdapterFiller,
        RV64_BYTE_BITS,
    },
    load_sign_extend::aligned::core::LoadSignExtendAlignedCoreCols,
    loadstore::common::run_write_data,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

mod byte;
mod halfword;
mod word;

type ByteHarness = TestChipHarness<
    F,
    Rv64LoadSignExtendByteExecutor,
    Rv64LoadSignExtendByteAir,
    Rv64LoadSignExtendByteChip<F>,
>;
type HalfwordHarness = TestChipHarness<
    F,
    Rv64LoadSignExtendHalfwordExecutor,
    Rv64LoadSignExtendHalfwordAir,
    Rv64LoadSignExtendHalfwordChip<F>,
>;
type WordHarness = TestChipHarness<
    F,
    Rv64LoadSignExtendWordExecutor,
    Rv64LoadSignExtendWordAir,
    Rv64LoadSignExtendWordChip<F>,
>;

fn create_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    ByteHarness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_checker = tester.range_checker();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let air = Rv64LoadSignExtendByteAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadSignExtendByteExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadSignExtendByteChip::<F>::new(
        LoadSignExtendByteFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        ByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

fn create_halfword_harness(tester: &mut VmChipTestBuilder<F>) -> HalfwordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadSignExtendHalfwordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadSignExtendHalfwordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadSignExtendHalfwordChip::<F>::new(
        LoadSignExtendHalfwordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_word_harness(tester: &mut VmChipTestBuilder<F>) -> WordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadSignExtendWordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadSignExtendWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadSignExtendWordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadSignExtendWordChip::<F>::new(
        LoadSignExtendWordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    WordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
    rs1: Option<[u8; 8]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
) {
    let imm = imm.unwrap_or_else(|| rng.random_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or_else(|| rng.random_range(0..2));
    let imm_ext = sign_extend_imm16(imm, imm_sign);
    let alignment = match opcode {
        LOADW => 2,
        LOADH => 1,
        LOADB => 0,
        _ => unreachable!("signed load test only supports LOADB/LOADH/LOADW"),
    };
    let ptr_val: u32 =
        rng.random_range(0..(1u32 << (tester.address_bits() - alignment))) << alignment;
    let rs1 = rs1.unwrap_or_else(|| {
        let low4 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
        [low4[0], low4[1], low4[2], low4[3], 0, 0, 0, 0]
    });
    let ptr_val = imm_ext.wrapping_add(rv64_bytes_to_u32(rs1));
    let shift_amount = ptr_val % 8;
    let a = gen_pointer(rng, 8);
    let b = gen_pointer(rng, 8);
    let read_data: [u8; 8] = array::from_fn(|_| rng.random());
    let prev_data: [F; 8] = if a != 0 {
        array::from_fn(|_| F::from_u8(rng.random()))
    } else {
        [F::ZERO; 8]
    };

    tester.write_bytes(RV64_REGISTER_AS as usize, b, rs1.map(F::from_u8));
    tester.write_bytes(RV64_REGISTER_AS as usize, a, prev_data);
    tester.write_bytes(
        2,
        (ptr_val - shift_amount) as usize,
        read_data.map(F::from_u8),
    );

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                RV64_REGISTER_AS as usize,
                2,
                (a != 0) as usize,
                imm_sign as usize,
            ],
        ),
    );

    let expected = run_write_data(
        opcode,
        rv64_bytes_to_u16_block(read_data),
        [0; BLOCK_FE_WIDTH],
        shift_amount as usize,
    );
    if a != 0 {
        assert_eq!(
            rv64_u16_block_to_bytes(expected).map(F::from_u8),
            tester.read_bytes::<8>(RV64_REGISTER_AS as usize, a)
        );
    } else {
        assert_eq!(
            [F::ZERO; 8],
            tester.read_bytes::<8>(RV64_REGISTER_AS as usize, a)
        );
    }
}

fn memory_config_for() -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    mem_config
}
fn assert_pranked_byte_fails(prank: impl Fn(&mut LoadSignExtendByteCoreCols<F>)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADB,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked signed byte load trace should fail");
}

fn assert_pranked_halfword_fails(
    prank: impl Fn(&mut LoadSignExtendAlignedCoreCols<F, LOAD_SIGN_EXTEND_HALFWORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_halfword_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADH,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked signed halfword load trace should fail");
}

fn assert_pranked_word_fails(
    prank: impl Fn(&mut LoadSignExtendAlignedCoreCols<F, LOAD_SIGN_EXTEND_WORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for());
    let mut harness = create_word_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADW,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        prank(core_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("pranked signed word load trace should fail");
}

#[test]
fn negative_split_signed_load_tests() {
    assert_pranked_byte_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_halfword_fails(|core| core.data_most_sig_bit += F::ONE);
    assert_pranked_word_fails(|core| core.data_most_sig_bit += F::ONE);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen.
// ////////////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "cuda")]
fn dummy_range_checker() -> Arc<VariableRangeCheckerChip> {
    Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ))
}
#[cfg(feature = "cuda")]
fn transfer_load_sign_extend_records<G, C, A, E>(harness: &mut GpuTestChipHarness<F, E, A, G, C>) {
    type Record<'a> = (&'a mut Rv64LoadStoreAdapterRecord, &'a mut LoadStoreRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new(),
        );
}
