pub(crate) use std::{array, borrow::BorrowMut, sync::Arc};

pub(crate) use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};
pub(crate) use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
pub(crate) use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
pub(crate) use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{
    self, LOADB, LOADBU, LOADD, LOADH, LOADHU, LOADW, LOADWU, STOREB, STORED, STOREH, STOREW,
};
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
    crate::adapters::{
        Rv64LoadAdapterRecord, Rv64StoreAdapterRecord, LOAD_WIDTH_WORD, STORE_WIDTH_WORD,
    },
    crate::load::{
        LoadRecord, Rv64LoadByteChipGpu, Rv64LoadDoublewordChipGpu, Rv64LoadHalfwordChipGpu,
        Rv64LoadWordChipGpu,
    },
    crate::store::{
        Rv64StoreByteChipGpu, Rv64StoreDoublewordChipGpu, Rv64StoreHalfwordChipGpu,
        Rv64StoreWordChipGpu, StoreRecord,
    },
    openvm_circuit::arch::{
        testing::{
            default_bitwise_lookup_bus, default_var_range_checker_bus, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

pub(crate) use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16,
        Rv64LoadAdapterAir, Rv64LoadAdapterExecutor, Rv64LoadAdapterFiller, Rv64StoreAdapterAir,
        Rv64StoreAdapterCols, Rv64StoreAdapterExecutor, Rv64StoreAdapterFiller, RV64_BYTE_BITS,
    },
    load::{
        common::load_write_data, core::LoadCoreCols, LoadByteCoreAir, LoadByteCoreCols,
        LoadByteFiller, LoadDoublewordCoreAir, LoadDoublewordFiller, LoadHalfwordCoreAir,
        LoadHalfwordFiller, LoadWordCoreAir, LoadWordFiller, Rv64LoadByteAir, Rv64LoadByteChip,
        Rv64LoadByteExecutor, Rv64LoadDoublewordAir, Rv64LoadDoublewordChip,
        Rv64LoadDoublewordExecutor, Rv64LoadHalfwordAir, Rv64LoadHalfwordChip,
        Rv64LoadHalfwordExecutor, Rv64LoadWordAir, Rv64LoadWordChip, Rv64LoadWordExecutor,
        LOAD_DOUBLEWORD_SELECTOR_WIDTH, LOAD_DOUBLEWORD_TOUCHED_CELLS,
        LOAD_HALFWORD_SELECTOR_WIDTH, LOAD_HALFWORD_TOUCHED_CELLS, LOAD_WORD_SELECTOR_WIDTH,
        LOAD_WORD_TOUCHED_CELLS,
    },
    load_sign_extend::common::load_sign_extend_write_data,
    store::{
        common::store_write_data, core::StoreCoreCols, Rv64StoreByteAir, Rv64StoreByteChip,
        Rv64StoreByteExecutor, Rv64StoreDoublewordAir, Rv64StoreDoublewordChip,
        Rv64StoreDoublewordExecutor, Rv64StoreHalfwordAir, Rv64StoreHalfwordChip,
        Rv64StoreHalfwordExecutor, Rv64StoreWordAir, Rv64StoreWordChip, Rv64StoreWordExecutor,
        StoreByteCoreAir, StoreByteCoreCols, StoreByteFiller, StoreDoublewordCoreAir,
        StoreDoublewordFiller, StoreHalfwordCoreAir, StoreHalfwordFiller, StoreWordCoreAir,
        StoreWordFiller, STORE_HALFWORD_SELECTOR_WIDTH, STORE_HALFWORD_VALUE_CELLS,
    },
};

pub(crate) const IMM_BITS: usize = 16;
pub(crate) const MAX_INS_CAPACITY: usize = 128;
pub(crate) type F = BabyBear;

pub(crate) type ByteHarness =
    TestChipHarness<F, Rv64LoadByteExecutor, Rv64LoadByteAir, Rv64LoadByteChip<F>>;
pub(crate) type HalfwordHarness =
    TestChipHarness<F, Rv64LoadHalfwordExecutor, Rv64LoadHalfwordAir, Rv64LoadHalfwordChip<F>>;
pub(crate) type WordHarness =
    TestChipHarness<F, Rv64LoadWordExecutor, Rv64LoadWordAir, Rv64LoadWordChip<F>>;
pub(crate) type DoublewordHarness = TestChipHarness<
    F,
    Rv64LoadDoublewordExecutor,
    Rv64LoadDoublewordAir,
    Rv64LoadDoublewordChip<F>,
>;
pub(crate) type StoreByteHarness =
    TestChipHarness<F, Rv64StoreByteExecutor, Rv64StoreByteAir, Rv64StoreByteChip<F>>;
pub(crate) type StoreHalfwordHarness =
    TestChipHarness<F, Rv64StoreHalfwordExecutor, Rv64StoreHalfwordAir, Rv64StoreHalfwordChip<F>>;
pub(crate) type StoreWordHarness =
    TestChipHarness<F, Rv64StoreWordExecutor, Rv64StoreWordAir, Rv64StoreWordChip<F>>;
pub(crate) type StoreDoublewordHarness = TestChipHarness<
    F,
    Rv64StoreDoublewordExecutor,
    Rv64StoreDoublewordAir,
    Rv64StoreDoublewordChip<F>,
>;

struct MemoryAccess {
    a: usize,
    b: usize,
    base_ptr: usize,
    imm: u32,
    imm_sign: u32,
    rs1: [u8; 8],
    shift_amount: usize,
}

fn random_memory_access(
    tester: &impl TestBuilder<F>,
    rng: &mut StdRng,
    alignment: usize,
    rs1: Option<[u8; 8]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
) -> MemoryAccess {
    let imm = imm.unwrap_or_else(|| rng.random_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or_else(|| rng.random_range(0..2));
    let imm_ext = sign_extend_imm16(imm, imm_sign);

    let max_addr = 1usize << tester.address_bits();
    let imm_signed = if imm_sign == 0 {
        imm as i64
    } else {
        imm as i64 - (1 << IMM_BITS)
    };
    let min_ptr = imm_signed.max(0) as usize;
    let alignment_mask = (1usize << alignment) - 1;
    let min_aligned_ptr = (min_ptr + alignment_mask) >> alignment;
    // Stay 16 bytes clear of the top of the address space so a block-crossing access always
    // has a valid second block.
    let ptr_val = rng.random_range(min_aligned_ptr..((max_addr - 8) >> alignment)) << alignment;
    let rs1_low = (ptr_val as i64 - imm_signed) as u32;
    let ptr = rs1_low.to_le_bytes();
    let rs1 = rs1.unwrap_or([ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0]);
    let rs1_low = rv64_bytes_to_u32(rs1);
    let ptr_val = imm_ext.wrapping_add(rs1_low);
    let shift_amount = (ptr_val as usize) & 7;
    let base_ptr = (ptr_val as usize) - shift_amount;

    let a = gen_pointer(rng, 8);
    let b = gen_pointer(rng, 8);

    MemoryAccess {
        a,
        b,
        base_ptr,
        imm,
        imm_sign,
        rs1,
        shift_amount,
    }
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
    let air = Rv64LoadByteAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadByteExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadByteChip::<F>::new(
        LoadByteFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
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

pub(crate) fn create_store_byte_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreByteHarness,
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
    let air = Rv64StoreByteAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreByteCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreByteExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreByteChip::<F>::new(
        StoreByteFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        StoreByteHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_halfword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    HalfwordHarness,
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
    let air = Rv64LoadHalfwordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadHalfwordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadHalfwordChip::<F>::new(
        LoadHalfwordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        HalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_store_halfword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreHalfwordHarness,
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
    let air = Rv64StoreHalfwordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreHalfwordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreHalfwordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreHalfwordChip::<F>::new(
        StoreHalfwordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        StoreHalfwordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_word_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    WordHarness,
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
    let air = Rv64LoadWordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadWordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadWordChip::<F>::new(
        LoadWordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        WordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_store_word_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreWordHarness,
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
    let air = Rv64StoreWordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreWordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreWordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreWordChip::<F>::new(
        StoreWordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        StoreWordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_doubleword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    DoublewordHarness,
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
    let air = Rv64LoadDoublewordAir::new(
        Rv64LoadAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        LoadDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64LoadDoublewordExecutor::new(
        Rv64LoadAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LoadDoublewordChip::<F>::new(
        LoadDoublewordFiller::new(
            Rv64LoadAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        DoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

pub(crate) fn create_store_doubleword_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    StoreDoublewordHarness,
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
    let air = Rv64StoreDoublewordAir::new(
        Rv64StoreAdapterAir::new(
            tester.memory_bridge(),
            tester.execution_bridge(),
            range_checker.bus(),
            tester.address_bits(),
        ),
        StoreDoublewordCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET, bitwise_chip.bus()),
    );
    let executor = Rv64StoreDoublewordExecutor::new(
        Rv64StoreAdapterExecutor::new(tester.address_bits()),
        Rv64LoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = Rv64StoreDoublewordChip::<F>::new(
        StoreDoublewordFiller::new(
            Rv64StoreAdapterFiller::new(tester.address_bits(), range_checker.clone()),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
            bitwise_chip.clone(),
            range_checker,
        ),
        tester.memory_helper(),
    );
    (
        StoreDoublewordHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_and_execute_load<RA: Arena, E: PreflightExecutor<F, RA>>(
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
    match opcode {
        LOADD | LOADWU | LOADHU | LOADBU => {}
        _ => unreachable!("unsupported unsigned load opcode: {opcode:?}"),
    }
    // Unsigned loads support any byte shift, so sample fully misaligned pointers.
    let access = random_memory_access(tester, rng, 0, rs1, imm, imm_sign);
    let mem_as = mem_as.unwrap_or(RV64_MEMORY_AS as usize);

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.b,
        access.rs1.map(F::from_u8),
    );

    let mut prev_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    let read_data: [[u16; BLOCK_FE_WIDTH]; 2] =
        array::from_fn(|_| array::from_fn(|_| rng.random()));
    if access.a == 0 {
        prev_data = [0; BLOCK_FE_WIDTH];
    }
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.a,
        rv64_u16_block_to_bytes(prev_data).map(F::from_u8),
    );
    tester.write_bytes(
        mem_as,
        access.base_ptr,
        rv64_u16_block_to_bytes(read_data[0]).map(F::from_u8),
    );
    tester.write_bytes(
        mem_as,
        access.base_ptr + 8,
        rv64_u16_block_to_bytes(read_data[1]).map(F::from_u8),
    );

    let enabled_write = access.a != 0;
    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                access.a,
                access.b,
                access.imm as usize,
                RV64_REGISTER_AS as usize,
                mem_as,
                enabled_write as usize,
                access.imm_sign as usize,
            ],
        ),
    );

    let write_data = load_write_data(opcode, read_data, access.shift_amount);
    let expected = if enabled_write {
        rv64_u16_block_to_bytes(write_data).map(F::from_u8)
    } else {
        [F::ZERO; 8]
    };
    assert_eq!(
        expected,
        tester.read_bytes::<8>(RV64_REGISTER_AS as usize, access.a)
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_and_execute_store<RA: Arena, E: PreflightExecutor<F, RA>>(
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
    match opcode {
        STORED | STOREW | STOREH | STOREB => {}
        _ => unreachable!("unsupported store opcode: {opcode:?}"),
    }
    // Stores support any byte shift, so sample fully misaligned pointers.
    let access = random_memory_access(tester, rng, 0, rs1, imm, imm_sign);
    let mem_as = mem_as.unwrap_or_else(|| {
        *[RV64_MEMORY_AS as usize, PUBLIC_VALUES_AS as usize]
            .choose(rng)
            .unwrap()
    });

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.b,
        access.rs1.map(F::from_u8),
    );

    let prev_data: [[u16; BLOCK_FE_WIDTH]; 2] =
        array::from_fn(|_| array::from_fn(|_| rng.random()));
    let mut read_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    if access.a == 0 {
        read_data = [0; BLOCK_FE_WIDTH];
    }
    tester.write_bytes(
        mem_as,
        access.base_ptr,
        rv64_u16_block_to_bytes(prev_data[0]).map(F::from_u8),
    );
    tester.write_bytes(
        mem_as,
        access.base_ptr + 8,
        rv64_u16_block_to_bytes(prev_data[1]).map(F::from_u8),
    );
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.a,
        rv64_u16_block_to_bytes(read_data).map(F::from_u8),
    );

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                access.a,
                access.b,
                access.imm as usize,
                RV64_REGISTER_AS as usize,
                mem_as,
                true as usize,
                access.imm_sign as usize,
            ],
        ),
    );

    let write_data = store_write_data(opcode, read_data, prev_data, access.shift_amount);
    assert_eq!(
        rv64_u16_block_to_bytes(write_data[0]).map(F::from_u8),
        tester.read_bytes::<8>(mem_as, access.base_ptr)
    );
    // The second block is either rewritten by the crossing store or untouched; both must match
    // the model.
    assert_eq!(
        rv64_u16_block_to_bytes(write_data[1]).map(F::from_u8),
        tester.read_bytes::<8>(mem_as, access.base_ptr + 8)
    );
}

pub(crate) fn load_memory_config() -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    mem_config
}

pub(crate) fn store_memory_config() -> MemoryConfig {
    let mut mem_config = load_memory_config();
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    mem_config
}

#[cfg(feature = "cuda")]
pub(crate) fn load_gpu_memory_config() -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << mem_config.pointer_max_bits;
    mem_config
}

#[cfg(feature = "cuda")]
pub(crate) fn store_gpu_memory_config() -> MemoryConfig {
    let mut mem_config = load_gpu_memory_config();
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << mem_config.pointer_max_bits;
    mem_config
}
pub(crate) fn assert_pranked_load_byte_fails(prank: impl Fn(&mut LoadByteCoreCols<F>)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_byte_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADBU,
        None,
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
        .expect_err("pranked byte memory access trace should fail");
}

pub(crate) fn assert_pranked_store_byte_fails(prank: impl Fn(&mut StoreByteCoreCols<F>)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_byte_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREB,
        None,
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
        .expect_err("pranked byte store trace should fail");
}

pub(crate) fn assert_pranked_load_halfword_fails(
    prank: impl Fn(
        &mut LoadCoreCols<F, { LOAD_HALFWORD_SELECTOR_WIDTH }, { LOAD_HALFWORD_TOUCHED_CELLS }>,
    ),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_halfword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADHU,
        None,
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
        .expect_err("pranked halfword memory access trace should fail");
}

pub(crate) fn assert_pranked_store_halfword_fails(
    prank: impl Fn(
        &mut StoreCoreCols<F, { STORE_HALFWORD_SELECTOR_WIDTH }, { STORE_HALFWORD_VALUE_CELLS }>,
    ),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_halfword_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREH,
        None,
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
        .expect_err("pranked halfword store trace should fail");
}

pub(crate) fn assert_pranked_load_word_fails(
    prank: impl Fn(&mut LoadCoreCols<F, { LOAD_WORD_SELECTOR_WIDTH }, { LOAD_WORD_TOUCHED_CELLS }>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_word_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADWU,
        None,
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
        .expect_err("pranked word memory access trace should fail");
}

pub(crate) fn assert_pranked_load_doubleword_fails(
    prank: impl Fn(
        &mut LoadCoreCols<F, { LOAD_DOUBLEWORD_SELECTOR_WIDTH }, { LOAD_DOUBLEWORD_TOUCHED_CELLS }>,
    ),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(load_memory_config());
    let (mut harness, bitwise) = create_doubleword_harness(&mut tester);
    set_and_execute_load(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADD,
        None,
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
        .expect_err("pranked doubleword memory access trace should fail");
}

pub(crate) fn assert_pranked_store_word_adapter_fails(
    prank: impl Fn(&mut Rv64StoreAdapterCols<F>),
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::from_config(store_memory_config());
    let (mut harness, bitwise) = create_store_word_harness(&mut tester);
    set_and_execute_store(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        STOREW,
        None,
        None,
        None,
        None,
    );
    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        prank(adapter_row.borrow_mut());
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };
    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize()
        .simple_test()
        .expect_err("pranked store adapter trace should fail");
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
pub(crate) fn transfer_load_records<G, C, A, E>(harness: &mut GpuTestChipHarness<F, E, A, G, C>) {
    type Record<'a> = (&'a mut Rv64LoadAdapterRecord, &'a mut LoadRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadAdapterExecutor<LOAD_WIDTH_WORD>>::new(),
        );
}

#[cfg(feature = "cuda")]
pub(crate) fn transfer_store_records<G, C, A, E>(harness: &mut GpuTestChipHarness<F, E, A, G, C>) {
    type Record<'a> = (&'a mut Rv64StoreAdapterRecord, &'a mut StoreRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64StoreAdapterExecutor<STORE_WIDTH_WORD>>::new(),
        );
}

#[cfg(test)]
mod tests {
    use openvm_instructions::DEFERRAL_AS;
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

    use super::{
        assert_pranked_load_byte_fails, assert_pranked_load_doubleword_fails,
        assert_pranked_load_halfword_fails, assert_pranked_load_word_fails,
        assert_pranked_store_byte_fails, assert_pranked_store_halfword_fails,
        assert_pranked_store_word_adapter_fails, F,
    };

    #[test]
    fn negative_split_write_data_tests() {
        assert_pranked_store_byte_fails(|core| core.read_data[0] += F::ONE);
        assert_pranked_load_halfword_fails(|core| core.read_data[0][0] += F::ONE);
        assert_pranked_load_word_fails(|core| core.read_data[0][0] += F::ONE);
        assert_pranked_load_doubleword_fails(|core| core.read_data[0][0] += F::ONE);
    }

    #[test]
    fn negative_split_opcode_role_tests() {
        assert_pranked_load_byte_fails(|core| core.selector[0] += F::ONE);
        assert_pranked_store_halfword_fails(|core| core.selector[0] += F::ONE);
        assert_pranked_load_word_fails(|core| core.selector[0] += F::ONE);
        assert_pranked_load_doubleword_fails(|core| core.selector[0] += F::ONE);
    }

    #[test]
    fn negative_split_store_deferral_as_test() {
        assert_pranked_store_word_adapter_fails(|adapter| {
            adapter.mem_as = F::from_u32(DEFERRAL_AS)
        });
    }
}
