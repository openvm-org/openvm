pub(crate) use std::{array, borrow::BorrowMut, sync::Arc};

pub(crate) use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};
pub(crate) use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
pub(crate) use openvm_instructions::{
    instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode,
};
pub(crate) use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
pub(crate) use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
pub(crate) use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
pub(crate) use rand::{rngs::StdRng, seq::IndexedRandom, Rng};
#[cfg(feature = "cuda")]
pub(crate) use {
    super::{
        LoadStoreRecord, Rv64LoadStoreByteChipGpu, Rv64LoadStoreDoublewordChipGpu,
        Rv64LoadStoreHalfwordChipGpu, Rv64LoadStoreWordChipGpu,
    },
    crate::adapters::Rv64LoadStoreAdapterRecord,
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

pub(crate) use super::{
    aligned::LoadStoreAlignedCoreCols,
    byte::{LoadStoreByteCoreAir, LoadStoreByteCoreCols, LoadStoreByteFiller},
    common::run_write_data,
    doubleword::{
        LoadStoreDoublewordCoreAir, LoadStoreDoublewordCoreCols, LoadStoreDoublewordFiller,
    },
    halfword::{LoadStoreHalfwordCoreAir, LoadStoreHalfwordFiller, HALFWORD_SELECTOR_WIDTH},
    word::{LoadStoreWordCoreAir, LoadStoreWordFiller, WORD_SELECTOR_WIDTH},
    Rv64LoadStoreByteAir, Rv64LoadStoreByteChip, Rv64LoadStoreByteExecutor,
    Rv64LoadStoreDoublewordAir, Rv64LoadStoreDoublewordChip, Rv64LoadStoreDoublewordExecutor,
    Rv64LoadStoreHalfwordAir, Rv64LoadStoreHalfwordChip, Rv64LoadStoreHalfwordExecutor,
    Rv64LoadStoreWordAir, Rv64LoadStoreWordChip, Rv64LoadStoreWordExecutor,
};
pub(crate) use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16,
    Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor, Rv64LoadStoreAdapterFiller,
    RV64_BYTE_BITS,
};

pub(crate) const IMM_BITS: usize = 16;
pub(crate) const MAX_INS_CAPACITY: usize = 128;
pub(crate) type F = BabyBear;

pub(crate) type ByteHarness =
    TestChipHarness<F, Rv64LoadStoreByteExecutor, Rv64LoadStoreByteAir, Rv64LoadStoreByteChip<F>>;
pub(crate) type HalfwordHarness = TestChipHarness<
    F,
    Rv64LoadStoreHalfwordExecutor,
    Rv64LoadStoreHalfwordAir,
    Rv64LoadStoreHalfwordChip<F>,
>;
pub(crate) type WordHarness =
    TestChipHarness<F, Rv64LoadStoreWordExecutor, Rv64LoadStoreWordAir, Rv64LoadStoreWordChip<F>>;
pub(crate) type DoublewordHarness = TestChipHarness<
    F,
    Rv64LoadStoreDoublewordExecutor,
    Rv64LoadStoreDoublewordAir,
    Rv64LoadStoreDoublewordChip<F>,
>;

pub(crate) fn u16_block_to_f_bytes(block: [u16; BLOCK_FE_WIDTH]) -> [F; 8] {
    rv64_u16_block_to_bytes(block).map(F::from_u8)
}

pub(crate) fn create_byte_harness(
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
    let air = Rv64LoadStoreByteAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreByteCoreAir::new(
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.bus(),
            range_checker.bus(),
        ),
    );
    let executor = Rv64LoadStoreByteExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreByteChip::<F>::new(
        LoadStoreByteFiller::new(
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

pub(crate) fn create_halfword_harness(tester: &mut VmChipTestBuilder<F>) -> HalfwordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreHalfwordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreHalfwordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreHalfwordChip::<F>::new(
        LoadStoreHalfwordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

pub(crate) fn create_word_harness(tester: &mut VmChipTestBuilder<F>) -> WordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreWordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreWordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreWordChip::<F>::new(
        LoadStoreWordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    WordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

pub(crate) fn create_doubleword_harness(tester: &mut VmChipTestBuilder<F>) -> DoublewordHarness {
    let range_checker = tester.range_checker();
    let air = Rv64LoadStoreDoublewordAir::new(
        Rv64LoadStoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadStoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, range_checker.bus()),
    );
    let executor = Rv64LoadStoreDoublewordExecutor::new(
        Rv64LoadStoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadStoreDoublewordChip::<F>::new(
        LoadStoreDoublewordFiller::new(
            Rv64LoadStoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            range_checker,
        ),
        tester.memory_helper(),
    );
    DoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
) {
    set_and_execute_with(tester, executor, arena, rng, opcode, None, None, None, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_and_execute_with<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
    rs1: Option<[u8; 8]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    mem_as: Option<usize>,
) {
    let imm = imm.unwrap_or_else(|| rng.random_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or_else(|| rng.random_range(0..2));
    let imm_ext = sign_extend_imm16(imm, imm_sign);
    let alignment = match opcode {
        LOADD | STORED => 3,
        LOADW | LOADWU | STOREW => 2,
        LOADH | LOADHU | STOREH => 1,
        LOADB | LOADBU | STOREB => 0,
    };

    let ptr_val: u32 = rng.random_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    let ptr = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
    let rs1 = rs1.unwrap_or([ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0]);
    let rs1_low = rv64_bytes_to_u32(rs1);
    let ptr_val = imm_ext.wrapping_add(rs1_low);
    let shift_amount = (ptr_val as usize) & 7;
    let base_ptr = (ptr_val as usize) - shift_amount;

    let max_addr = 1usize << tester.address_bits();
    let a = rng.random_range(0..(max_addr - 8)) / 8 * 8;
    let b = rng.random_range(0..(max_addr - 8)) / 8 * 8;
    let is_load = matches!(
        opcode,
        LOADD | LOADW | LOADH | LOADB | LOADWU | LOADHU | LOADBU
    );
    let mem_as: usize = mem_as.unwrap_or_else(|| {
        if is_load {
            2
        } else {
            *[2usize, PUBLIC_VALUES_AS as usize].choose(rng).unwrap()
        }
    });

    tester.write_bytes(RV64_REGISTER_AS as usize, b, rs1.map(F::from_u8));

    let mut prev_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    let mut read_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    if is_load {
        if a == 0 {
            prev_data = [0; BLOCK_FE_WIDTH];
        }
        tester.write_bytes(
            RV64_REGISTER_AS as usize,
            a,
            u16_block_to_f_bytes(prev_data),
        );
        tester.write_bytes(mem_as, base_ptr, u16_block_to_f_bytes(read_data));
    } else {
        if a == 0 {
            read_data = [0; BLOCK_FE_WIDTH];
        }
        tester.write_bytes(mem_as, base_ptr, u16_block_to_f_bytes(prev_data));
        tester.write_bytes(
            RV64_REGISTER_AS as usize,
            a,
            u16_block_to_f_bytes(read_data),
        );
    }

    let enabled_write = !(is_load && a == 0);
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
                mem_as,
                enabled_write as usize,
                imm_sign as usize,
            ],
        ),
    );

    let write_data = run_write_data(opcode, read_data, prev_data, shift_amount);
    if is_load {
        let expected = if enabled_write {
            u16_block_to_f_bytes(write_data)
        } else {
            [F::ZERO; 8]
        };
        assert_eq!(
            expected,
            tester.read_bytes::<8>(RV64_REGISTER_AS as usize, a)
        );
    } else {
        assert_eq!(
            u16_block_to_f_bytes(write_data),
            tester.read_bytes::<8>(mem_as, base_ptr)
        );
    }
}

pub(crate) fn memory_config_for(opcodes: &[Rv64LoadStoreOpcode]) -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    if opcodes
        .iter()
        .any(|opcode| matches!(opcode, STORED | STOREW | STOREH | STOREB))
    {
        mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    }
    mem_config
}
pub(crate) fn b(bytes: [u8; 8]) -> [u16; BLOCK_FE_WIDTH] {
    rv64_bytes_to_u16_block(bytes)
}
pub(crate) fn assert_pranked_byte_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreByteCoreCols<F>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
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
        .expect_err("pranked byte loadstore trace should fail");
}

pub(crate) fn assert_pranked_halfword_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreAlignedCoreCols<F, HALFWORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_halfword_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
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
        .expect_err("pranked halfword loadstore trace should fail");
}

pub(crate) fn assert_pranked_word_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreAlignedCoreCols<F, WORD_SELECTOR_WIDTH>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_word_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
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
        .expect_err("pranked word loadstore trace should fail");
}

pub(crate) fn assert_pranked_doubleword_fails(
    opcode: Rv64LoadStoreOpcode,
    prank: impl Fn(&mut LoadStoreDoublewordCoreCols<F>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(memory_config_for(&[opcode]));
    let mut harness = create_doubleword_harness(&mut tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
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
        .expect_err("pranked doubleword loadstore trace should fail");
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen.
// ////////////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "cuda")]
pub(crate) fn dummy_range_checker() -> Arc<VariableRangeCheckerChip> {
    Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ))
}
#[cfg(feature = "cuda")]
pub(crate) fn transfer_loadstore_records<G, C, A, E>(
    harness: &mut GpuTestChipHarness<F, E, A, G, C>,
) {
    type Record<'a> = (&'a mut Rv64LoadStoreAdapterRecord, &'a mut LoadStoreRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new(),
        );
}
