use std::array;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder},
        Arena, MemoryConfig, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::merkle::public_values::PUBLIC_VALUES_AS,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{
    self, LOADBU, LOADD, LOADHU, LOADWU, STOREB, STORED, STOREH, STOREW,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, seq::IndexedRandom, Rng};
#[cfg(feature = "cuda")]
use {
    crate::adapters::{Rv64LoadAdapterRecord, Rv64StoreAdapterRecord},
    crate::load::LoadRecord,
    crate::store::StoreRecord,
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

#[cfg(feature = "cuda")]
use crate::adapters::{Rv64LoadAdapterExecutor, Rv64StoreAdapterExecutor};
use crate::{
    adapters::{rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16},
    load::common::load_write_data,
    store::common::store_write_data,
};

pub(crate) const IMM_BITS: usize = 16;
pub(crate) const MAX_INS_CAPACITY: usize = 128;
pub(crate) type F = BabyBear;

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
    let ptr_val = rng.random_range(min_aligned_ptr..(max_addr >> alignment)) << alignment;
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
    let alignment = match opcode {
        LOADD => 3,
        LOADWU => 2,
        LOADHU => 1,
        LOADBU => 0,
        _ => unreachable!("unsupported unsigned load opcode: {opcode:?}"),
    };
    let access = random_memory_access(tester, rng, alignment, rs1, imm, imm_sign);
    let mem_as = mem_as.unwrap_or(RV64_MEMORY_AS as usize);

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        access.b,
        access.rs1.map(F::from_u8),
    );

    let mut prev_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    let read_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
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
        rv64_u16_block_to_bytes(read_data).map(F::from_u8),
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
    let alignment = match opcode {
        STORED => 3,
        STOREW => 2,
        STOREH => 1,
        STOREB => 0,
        _ => unreachable!("unsupported store opcode: {opcode:?}"),
    };
    let access = random_memory_access(tester, rng, alignment, rs1, imm, imm_sign);
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

    let prev_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    let mut read_data: [u16; BLOCK_FE_WIDTH] = array::from_fn(|_| rng.random());
    if access.a == 0 {
        read_data = [0; BLOCK_FE_WIDTH];
    }
    tester.write_bytes(
        mem_as,
        access.base_ptr,
        rv64_u16_block_to_bytes(prev_data).map(F::from_u8),
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
        rv64_u16_block_to_bytes(write_data).map(F::from_u8),
        tester.read_bytes::<8>(mem_as, access.base_ptr)
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
            EmptyAdapterCoreLayout::<F, Rv64LoadAdapterExecutor>::new(),
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
            EmptyAdapterCoreLayout::<F, Rv64StoreAdapterExecutor>::new(),
        );
}
