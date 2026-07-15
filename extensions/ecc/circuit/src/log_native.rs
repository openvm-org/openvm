//! Log-native record assembly for short-Weierstrass operations.

use openvm_algebra_circuit::log_native::{
    assemble_rv64_vec_heap_field_expression, assemble_vec_heap_inline, vec_heap_geometry,
    ModularRecordArena, VecHeapLayout, VecHeapRecordMut,
};
use openvm_circuit::arch::{
    rvr::{
        LogNativeAccessView, LogNativeAssemblerRegistry, VmRvrLogNativeExtension,
        PREFLIGHT_MEMORY_KIND_READ,
    },
    Arena, ExecutionError, RecordArena, MEMORY_BLOCK_BYTES,
};
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ext_algebra::VecHeapRecordDescriptor;
use strum::EnumCount;

use crate::{
    Rv64WeierstrassConfig, WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_48,
};

/// Record-arena requirements contributed by 32- and 48-byte Weierstrass curves.
///
/// One point occupies two coordinates, so the supported layouts use eight
/// memory blocks for 32-byte coordinates and twelve for 48-byte coordinates.
/// The bounds remain arena-agnostic for CPU and GPU reuse.
pub trait WeierstrassRecordArena<F>:
    Arena
    + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, ECC_BLOCKS_32, ECC_BLOCKS_32>,
        VecHeapRecordMut<'a, 2, ECC_BLOCKS_32, ECC_BLOCKS_32>,
    > + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 1, ECC_BLOCKS_32, ECC_BLOCKS_32>,
        VecHeapRecordMut<'a, 1, ECC_BLOCKS_32, ECC_BLOCKS_32>,
    > + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 2, ECC_BLOCKS_48, ECC_BLOCKS_48>,
        VecHeapRecordMut<'a, 2, ECC_BLOCKS_48, ECC_BLOCKS_48>,
    > + for<'a> RecordArena<
        'a,
        VecHeapLayout<F, 1, ECC_BLOCKS_48, ECC_BLOCKS_48>,
        VecHeapRecordMut<'a, 1, ECC_BLOCKS_48, ECC_BLOCKS_48>,
    >
{
}

impl<F, RA> WeierstrassRecordArena<F> for RA where
    RA: Arena
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, ECC_BLOCKS_32, ECC_BLOCKS_32>,
            VecHeapRecordMut<'a, 2, ECC_BLOCKS_32, ECC_BLOCKS_32>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 1, ECC_BLOCKS_32, ECC_BLOCKS_32>,
            VecHeapRecordMut<'a, 1, ECC_BLOCKS_32, ECC_BLOCKS_32>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, ECC_BLOCKS_48, ECC_BLOCKS_48>,
            VecHeapRecordMut<'a, 2, ECC_BLOCKS_48, ECC_BLOCKS_48>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 1, ECC_BLOCKS_48, ECC_BLOCKS_48>,
            VecHeapRecordMut<'a, 1, ECC_BLOCKS_48, ECC_BLOCKS_48>,
        >
{
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for WeierstrassExtension
where
    F: PrimeField32,
    RA: WeierstrassRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        for (curve_idx, curve) in self.supported_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;
            let offset =
                Rv64WeierstrassOpcode::CLASS_OFFSET + curve_idx * Rv64WeierstrassOpcode::COUNT;
            if bytes <= NUM_LIMBS_32 {
                register_curve::<F, RA, ECC_BLOCKS_32>(registry, offset);
            } else if bytes <= NUM_LIMBS_48 {
                register_curve::<F, RA, ECC_BLOCKS_48>(registry, offset);
            } else {
                panic!("Weierstrass modulus exceeds maximum supported size of 384 bits");
            }
        }
    }
}

impl<F, RA> VmRvrLogNativeExtension<F, RA> for Rv64WeierstrassConfig
where
    F: PrimeField32,
    RA: Rv64IRecordArena<F>
        + Rv64MRecordArena<F>
        + Rv64IoRecordArena<F>
        + ModularRecordArena<F>
        + WeierstrassRecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        self.modular.extend_rvr_log_native(registry);
        self.weierstrass.extend_rvr_log_native(registry);
    }
}

fn weierstrass_opcode(offset: usize, opcode: Rv64WeierstrassOpcode) -> VmOpcode {
    VmOpcode::from_usize(offset + opcode as usize)
}

fn register_curve<F: PrimeField32, RA, const BLOCKS: usize>(
    registry: &mut LogNativeAssemblerRegistry<F, RA>,
    offset: usize,
) where
    RA: WeierstrassRecordArena<F>
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        > + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 1, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 1, BLOCKS, BLOCKS>,
        >,
{
    let addne_opcodes = [
        Rv64WeierstrassOpcode::EC_ADD_NE,
        Rv64WeierstrassOpcode::SETUP_EC_ADD_NE,
    ]
    .map(|opcode| weierstrass_opcode(offset, opcode));
    registry.register_if(
        addne_opcodes,
        is_weierstrass_instruction,
        assemble_ec_add_ne::<F, RA, BLOCKS>,
    );
    registry.register_inline_arena_native(
        addne_opcodes,
        VecHeapRecordDescriptor::new_with_reads(BLOCKS * MEMORY_BLOCK_BYTES, 2).record_size,
        assemble_vec_heap_inline::<F, RA, 2, BLOCKS>,
        vec_heap_geometry::<F, 2, BLOCKS>(),
    );
    let double_opcodes = [
        Rv64WeierstrassOpcode::EC_DOUBLE,
        Rv64WeierstrassOpcode::SETUP_EC_DOUBLE,
    ]
    .map(|opcode| weierstrass_opcode(offset, opcode));
    registry.register_if(
        double_opcodes,
        is_weierstrass_instruction,
        assemble_ec_double::<F, RA, BLOCKS>,
    );
    registry.register_inline_arena_native(
        double_opcodes,
        VecHeapRecordDescriptor::new_with_reads(BLOCKS * MEMORY_BLOCK_BYTES, 1).record_size,
        assemble_vec_heap_inline::<F, RA, 1, BLOCKS>,
        vec_heap_geometry::<F, 1, BLOCKS>(),
    );
}

fn is_weierstrass_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.d.as_canonical_u32() == RV64_REGISTER_AS
        && instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

fn assemble_ec_add_ne<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: WeierstrassRecordArena<F>
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 2, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 2, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        weierstrass_local_opcode(instruction),
        [PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_READ],
    )
}

fn assemble_ec_double<F: PrimeField32, RA, const BLOCKS: usize>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: WeierstrassRecordArena<F>
        + for<'a> RecordArena<
            'a,
            VecHeapLayout<F, 1, BLOCKS, BLOCKS>,
            VecHeapRecordMut<'a, 1, BLOCKS, BLOCKS>,
        >,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 1, BLOCKS, BLOCKS>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        weierstrass_local_opcode(instruction),
        [PREFLIGHT_MEMORY_KIND_READ],
    )
}

fn weierstrass_local_opcode<F: PrimeField32>(instruction: &Instruction<F>) -> u8 {
    (instruction
        .opcode
        .as_usize()
        .wrapping_sub(Rv64WeierstrassOpcode::CLASS_OFFSET)
        % Rv64WeierstrassOpcode::COUNT) as u8
}
