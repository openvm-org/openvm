use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use num_bigint::BigUint;
use openvm_algebra_circuit::fields::{get_field_type, FieldType};
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_mod_circuit_builder::{run_field_expression_precomputed, FieldExpressionProgram};
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_circuit::adapters::rv64_bytes_to_u32;
use openvm_stark_backend::p3_field::PrimeField32;

use super::EcAddExecutor;
use crate::weierstrass_chip::curves::ec_add_proj;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcAddPreCompute<'a> {
    program: &'a FieldExpressionProgram,
    rs_addrs: [u8; 2],
    a: u8,
    flag_idx: u8,
}

impl<'a, const BLOCKS: usize> EcAddExecutor<BLOCKS> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcAddPreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        // Validate instruction format
        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        if d != RV64_REGISTER_AS || e != RV64_MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.offset);

        // Pre-compute flag_idx
        let needs_setup = self.program().needs_setup();
        let mut flag_idx = self.program().num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.opcode_flag_idx.len() {
                    flag_idx = self.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = EcAddPreCompute {
            program: self.program(),
            rs_addrs,
            a: a as u8,
            flag_idx,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        let is_setup = local_opcode == Rv64WeierstrassOpcode::SETUP_SW_EC_ADD_PROJ as usize;

        Ok(is_setup)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $pre_compute:ident, $is_setup:ident) => {
        if let Some(field_type) = {
            let modulus = $pre_compute.program.prime();
            get_field_type(modulus)
        } {
            match ($is_setup, field_type) {
                (true, FieldType::K256Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::K256Coordinate as u8 }, true>)
                }
                (true, FieldType::P256Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::P256Coordinate as u8 }, true>)
                }
                (true, FieldType::BN254Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::BN254Coordinate as u8 }, true>)
                }
                (true, FieldType::BLS12_381Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::BLS12_381Coordinate as u8 }, true>)
                }
                (false, FieldType::K256Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::K256Coordinate as u8 }, false>)
                }
                (false, FieldType::P256Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::P256Coordinate as u8 }, false>)
                }
                (false, FieldType::BN254Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::BN254Coordinate as u8 }, false>)
                }
                (false, FieldType::BLS12_381Coordinate) => {
                    Ok($execute_impl::<_, BLOCKS, { FieldType::BLS12_381Coordinate as u8 }, false>)
                }
                _ => panic!("Unsupported field type"),
            }
        } else if $is_setup {
            Ok($execute_impl::<_, BLOCKS, { u8::MAX }, true>)
        } else {
            Ok($execute_impl::<_, BLOCKS, { u8::MAX }, false>)
        }
    };
}
impl<F: PrimeField32, const BLOCKS: usize> InterpreterExecutor<F> for EcAddExecutor<BLOCKS> {
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcAddPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EcAddPreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut EcAddPreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;

        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize> InterpreterMeteredExecutor<F> for EcAddExecutor<BLOCKS> {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcAddPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EcAddPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<EcAddPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let pre_compute_pure = &mut pre_compute.data;
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute_pure)?;
        dispatch!(execute_e2_handler, pre_compute_pure, is_setup)
    }
}
#[inline(always)]
unsafe fn execute_e12_impl<
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const FIELD_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &EcAddPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| rv64_bytes_to_u32(exec_state.vm_read_bytes(RV64_REGISTER_AS, addr as u32)));

    // Read memory values for both points
    let read_data: [[[u8; MEMORY_BLOCK_BYTES]; BLOCKS]; 2] = rs_vals.map(|address| {
        debug_assert!(address as usize + MEMORY_BLOCK_BYTES * BLOCKS - 1 < MEM_SIZE);
        from_fn(|i| {
            exec_state.vm_read_bytes(RV64_MEMORY_AS, address + (i * MEMORY_BLOCK_BYTES) as u32)
        })
    });

    if IS_SETUP {
        // For projective coordinates, BLOCKS = 3 * blocks_per_coord.
        // Setup input (first point) contains: X=modulus, Y=a, Z=b.
        let blocks_per_coord = BLOCKS / 3;

        // Validate X coordinate = modulus
        let input_prime = BigUint::from_bytes_le(read_data[0][..blocks_per_coord].as_flattened());
        if &input_prime != pre_compute.program.prime() {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcAdd: mismatched prime",
            };
            return Err(err);
        }

        // Validate Y coordinate = a coefficient
        let input_a = BigUint::from_bytes_le(
            read_data[0][blocks_per_coord..2 * blocks_per_coord].as_flattened(),
        );
        let coeff_a = &pre_compute.program.setup_values()[0];
        if input_a != *coeff_a {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcAdd: mismatched coeff_a",
            };
            return Err(err);
        }

        // Validate Z coordinate = b coefficient
        let input_b = BigUint::from_bytes_le(read_data[0][2 * blocks_per_coord..].as_flattened());
        let coeff_b = &pre_compute.program.setup_values()[1];
        if input_b != *coeff_b {
            let err = ExecutionError::Fail {
                pc,
                msg: "EcAdd: mismatched coeff_b",
            };
            return Err(err);
        }
    }

    let output_data = if FIELD_TYPE == u8::MAX || IS_SETUP {
        let read_data: DynArray<u8> = read_data.into();
        run_field_expression_precomputed::<true>(
            pre_compute.program,
            pre_compute.flag_idx as usize,
            &read_data.0,
        )
        .into()
    } else {
        ec_add_proj::<FIELD_TYPE, BLOCKS>(read_data)
    };

    let rd_val =
        rv64_bytes_to_u32(exec_state.vm_read_bytes(RV64_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + MEMORY_BLOCK_BYTES * BLOCKS - 1 < MEM_SIZE);

    // Write output data to memory
    for (i, block) in output_data.into_iter().enumerate() {
        exec_state.vm_write_bytes(
            RV64_MEMORY_AS,
            rd_val + (i * MEMORY_BLOCK_BYTES) as u32,
            &block,
        );
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    Ok(())
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const FIELD_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &EcAddPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<EcAddPreCompute>()).borrow();
    execute_e12_impl::<_, BLOCKS, FIELD_TYPE, IS_SETUP>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const FIELD_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let e2_pre_compute: &E2PreCompute<EcAddPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EcAddPreCompute>>())
            .borrow();
    exec_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, BLOCKS, FIELD_TYPE, IS_SETUP>(&e2_pre_compute.data, exec_state)
}
