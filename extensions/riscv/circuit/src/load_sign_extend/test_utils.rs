use std::array;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder},
    Arena, MemoryConfig, PreflightExecutor,
};
use openvm_instructions::{
    instruction::Instruction, riscv::RV64_REGISTER_AS, LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB, LOADH, LOADW};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::{Rv64LoadAdapterRecord, LOAD_WIDTH_WORD},
        load::LoadRecord,
    },
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

#[cfg(feature = "cuda")]
use crate::adapters::Rv64LoadAdapterExecutor;
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_bytes_to_u32, rv64_u16_block_to_bytes, sign_extend_imm16,
    },
    load_sign_extend::common::load_sign_extend_write_data,
};

pub(crate) const IMM_BITS: usize = 16;
pub(crate) const MAX_INS_CAPACITY: usize = 128;
pub(crate) type F = BabyBear;

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
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
    match opcode {
        LOADB | LOADH | LOADW => {}
        _ => unreachable!("signed load test only supports LOADB/LOADH/LOADW"),
    }
    let max_addr = 1usize << tester.address_bits();
    let imm_signed = if imm_sign == 0 {
        imm as i64
    } else {
        imm as i64 - (1 << IMM_BITS)
    };
    let min_ptr = imm_signed.max(0) as usize;
    // Signed loads support any byte shift, so sample fully misaligned pointers, staying 16
    // bytes clear of the top of the address space so a block-crossing access always has a
    // valid second block.
    let ptr_val = rng.random_range(min_ptr..max_addr - 8);
    let rs1 = rs1.unwrap_or_else(|| {
        let low4 = (ptr_val as i64 - imm_signed).to_le_bytes();
        [low4[0], low4[1], low4[2], low4[3], 0, 0, 0, 0]
    });
    let ptr_val = imm_ext.wrapping_add(rv64_bytes_to_u32(rs1));
    let shift_amount = ptr_val % 8;
    let a = gen_pointer(rng, 8);
    let b = gen_pointer(rng, 8);
    let read_data: [[u8; 8]; 2] = array::from_fn(|_| array::from_fn(|_| rng.random()));
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
        read_data[0].map(F::from_u8),
    );
    tester.write_bytes(
        2,
        (ptr_val - shift_amount) as usize + 8,
        read_data[1].map(F::from_u8),
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

    let expected = load_sign_extend_write_data(
        opcode,
        read_data.map(rv64_bytes_to_u16_block),
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

pub(crate) fn memory_config_for() -> MemoryConfig {
    let mut mem_config = MemoryConfig::default();
    mem_config.addr_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 29;
    mem_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 29;
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
pub(crate) fn transfer_load_sign_extend_records<G, C, A, E>(
    harness: &mut GpuTestChipHarness<F, E, A, G, C>,
) {
    type Record<'a> = (&'a mut Rv64LoadAdapterRecord, &'a mut LoadRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadAdapterExecutor<LOAD_WIDTH_WORD>>::new(),
        );
}

// The byte chip uses the lean byte adapter, whose trace rows are narrower than the width
// (misaligned) adapter's; the layout must match or the core record lands at the wrong offset.
#[cfg(feature = "cuda")]
pub(crate) fn transfer_load_sign_extend_byte_records<G, C, A, E>(
    harness: &mut GpuTestChipHarness<F, E, A, G, C>,
) {
    type Record<'a> = (&'a mut Rv64LoadAdapterRecord, &'a mut LoadRecord);
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadByteAdapterExecutor>::new(),
        );
}
