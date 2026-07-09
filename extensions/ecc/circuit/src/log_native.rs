//! Log-native record assembly for short-Weierstrass operations.

use openvm_algebra_circuit::log_native::{
    assemble_rv64_vec_heap_field_expression, ModularRecordArena, VecHeapLayout, VecHeapRecordMut,
};
use openvm_circuit::arch::{
    rvr::{
        LogNativeAccessView, LogNativeAssemblerRegistry, VmRvrLogNativeExtension,
        PREFLIGHT_MEMORY_KIND_READ,
    },
    Arena, ExecutionError, RecordArena,
};
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode, VmOpcode,
};
use openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena};
use openvm_stark_backend::p3_field::PrimeField32;
use strum::EnumCount;

use crate::{Rv64WeierstrassConfig, WeierstrassExtension, ECC_BLOCKS_32, NUM_LIMBS_32};

/// Record-arena requirements contributed by 32-byte Weierstrass curves.
///
/// One point occupies eight memory blocks: four blocks for each 32-byte
/// coordinate. The bounds remain arena-agnostic for CPU and GPU reuse.
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
            assert!(
                bytes <= NUM_LIMBS_32,
                "rvr log-native Weierstrass preflight currently supports only 32-byte curves"
            );
            let offset =
                Rv64WeierstrassOpcode::CLASS_OFFSET + curve_idx * Rv64WeierstrassOpcode::COUNT;
            registry.register_if(
                [
                    Rv64WeierstrassOpcode::EC_ADD_NE,
                    Rv64WeierstrassOpcode::SETUP_EC_ADD_NE,
                ]
                .map(|opcode| weierstrass_opcode(offset, opcode)),
                is_weierstrass_instruction,
                assemble_ec_add_ne::<F, RA>,
            );
            registry.register_if(
                [
                    Rv64WeierstrassOpcode::EC_DOUBLE,
                    Rv64WeierstrassOpcode::SETUP_EC_DOUBLE,
                ]
                .map(|opcode| weierstrass_opcode(offset, opcode)),
                is_weierstrass_instruction,
                assemble_ec_double::<F, RA>,
            );
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

fn is_weierstrass_instruction<F: PrimeField32>(instruction: &Instruction<F>) -> bool {
    instruction.d.as_canonical_u32() == RV64_REGISTER_AS
        && instruction.e.as_canonical_u32() == RV64_MEMORY_AS
}

fn assemble_ec_add_ne<F: PrimeField32, RA>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: WeierstrassRecordArena<F>,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 2, ECC_BLOCKS_32, ECC_BLOCKS_32>(
        arena,
        access,
        instruction,
        pc,
        timestamp,
        weierstrass_local_opcode(instruction),
        [PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_READ],
    )
}

fn assemble_ec_double<F: PrimeField32, RA>(
    arena: &mut RA,
    access: &LogNativeAccessView<'_, F>,
    instruction: &Instruction<F>,
    pc: u32,
    timestamp: u32,
) -> Result<(), ExecutionError>
where
    RA: WeierstrassRecordArena<F>,
{
    assemble_rv64_vec_heap_field_expression::<F, RA, 1, ECC_BLOCKS_32, ECC_BLOCKS_32>(
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
