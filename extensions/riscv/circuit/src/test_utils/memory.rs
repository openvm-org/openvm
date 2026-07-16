use std::array;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use openvm_circuit::arch::{
    testing::TestBuilder, Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    NUM_RV64_REGISTERS,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{
    self, LOADBU, LOADD, LOADHU, LOADWU, STOREB, STORED, STOREH, STOREW,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, seq::IndexedRandom, Rng};
#[cfg(feature = "cuda")]
use {
    crate::adapters::{
        Rv64LoadByteAdapterRecord, Rv64LoadMultiByteAdapterRecord, Rv64StoreByteAdapterRecord,
        Rv64StoreMultiByteAdapterRecord, LOAD_WIDTH_WORD, STORE_WIDTH_WORD,
    },
    crate::load::{LoadByteRecord, LoadRecord},
    crate::store::{StoreByteRecord, StoreRecord},
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

#[cfg(feature = "cuda")]
use crate::adapters::{
    Rv64LoadByteAdapterExecutor, Rv64LoadMultiByteAdapterExecutor, Rv64StoreByteAdapterExecutor,
    Rv64StoreMultiByteAdapterExecutor,
};
use crate::{
    adapters::{rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16},
    load::common::load_write_data,
    store::common::store_write_data,
};

pub(crate) const IMM_BITS: usize = 16;
pub(crate) const MAX_INS_CAPACITY: usize = 128;
pub(crate) type F = BabyBear;

pub(crate) fn random_register_pointer(rng: &mut StdRng) -> usize {
    rng.random_range(0..NUM_RV64_REGISTERS) * RV64_REGISTER_NUM_LIMBS
}

pub(crate) fn random_nonzero_register_pointer(rng: &mut StdRng) -> usize {
    rng.random_range(1..NUM_RV64_REGISTERS) * RV64_REGISTER_NUM_LIMBS
}

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
    // Leave room for a second block when the access crosses the first one.
    let ptr_val = rng
        .random_range(min_aligned_ptr..((max_addr - RV64_REGISTER_NUM_LIMBS) >> alignment))
        << alignment;
    let rs1_low = (ptr_val as i64 - imm_signed) as u32;
    let ptr = rs1_low.to_le_bytes();
    let rs1 = rs1.unwrap_or([ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0]);
    let rs1_low = rv64_bytes_to_u32(rs1);
    let ptr_val = imm_ext.wrapping_add(rs1_low);
    let shift_amount = (ptr_val as usize) & 7;
    let base_ptr = (ptr_val as usize) - shift_amount;

    let a = random_register_pointer(rng);
    // Keep rs1 nonzero because this helper chooses its contents to produce the sampled address.
    let b = random_nonzero_register_pointer(rng);

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
    // Sample every byte offset within a memory block.
    let access = random_memory_access(tester, rng, 0, rs1, imm, imm_sign);
    let mem_as = mem_as.unwrap_or(RV64_MEMORY_AS as usize);

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.b,
        access.rs1.map(F::from_u8),
    );

    let mut prev_data: [u16; BLOCK_FE_WIDTH] = if access.a == access.b {
        crate::adapters::rv64_bytes_to_u16_block(access.rs1)
    } else {
        array::from_fn(|_| rng.random())
    };
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
        access.base_ptr + RV64_REGISTER_NUM_LIMBS,
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
    // Sample every byte offset within a memory block.
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
    let mut read_data: [u16; BLOCK_FE_WIDTH] = if access.a == access.b {
        crate::adapters::rv64_bytes_to_u16_block(access.rs1)
    } else {
        array::from_fn(|_| rng.random())
    };
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
        access.base_ptr + RV64_REGISTER_NUM_LIMBS,
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
        tester.read_bytes::<8>(mem_as, access.base_ptr + RV64_REGISTER_NUM_LIMBS)
    );
}

pub(crate) fn load_memory_config() -> MemoryConfig {
    MemoryConfig::default()
}

pub(crate) fn store_memory_config() -> MemoryConfig {
    let mut mem_config = load_memory_config();
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
    mem_config
}

#[cfg(feature = "cuda")]
pub(crate) fn load_gpu_memory_config() -> MemoryConfig {
    MemoryConfig::default()
}

#[cfg(feature = "cuda")]
pub(crate) fn store_gpu_memory_config() -> MemoryConfig {
    let mut mem_config = load_gpu_memory_config();
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << mem_config.pointer_max_bits;
    mem_config
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
    type Record<'a> = (&'a mut Rv64LoadMultiByteAdapterRecord, &'a mut LoadRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadMultiByteAdapterExecutor<LOAD_WIDTH_WORD>>::new(),
        );
}

#[cfg(feature = "cuda")]
pub(crate) fn transfer_store_records<G, C, A, E>(harness: &mut GpuTestChipHarness<F, E, A, G, C>) {
    type Record<'a> = (&'a mut Rv64StoreMultiByteAdapterRecord, &'a mut StoreRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64StoreMultiByteAdapterExecutor<STORE_WIDTH_WORD>>::new(),
        );
}

// Byte and multi-byte adapters have different row widths, so record transfer must use the
// matching layout.
#[cfg(feature = "cuda")]
pub(crate) fn transfer_load_byte_records<G, C, A, E>(
    harness: &mut GpuTestChipHarness<F, E, A, G, C>,
) {
    type Record<'a> = (&'a mut Rv64LoadByteAdapterRecord, &'a mut LoadByteRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadByteAdapterExecutor>::new(),
        );
}

#[cfg(feature = "cuda")]
pub(crate) fn transfer_store_byte_records<G, C, A, E>(
    harness: &mut GpuTestChipHarness<F, E, A, G, C>,
) {
    type Record<'a> = (&'a mut Rv64StoreByteAdapterRecord, &'a mut StoreByteRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64StoreByteAdapterExecutor>::new(),
        );
}
