//! Execution for `EC_MUL` and `SETUP_EC_MUL`.
//!
//! Instruction fields are parsed once at program load, after which a
//! handler monomorphised on the curve is selected, so the hot path branches on nothing.
//!
//! For a recognised curve the result comes from native field arithmetic; otherwise the ladder is
//! interpreted through the field expression program. Either way the bytes written here must match
//! what trace generation records, since the memory argument compares them.
//!
//! Metered execution reports the full row count rather than one row per instruction. An
//! incorrect height here does not fail any unit test; it mis-sizes segments.

use std::borrow::{Borrow, BorrowMut};

use num_bigint::BigUint;
use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_mod_circuit_builder::FieldExpressionProgram;
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_circuit::adapters::{try_bytes_to_u32, validate_memory_block_byte_ptr};
use openvm_stark_backend::p3_field::PrimeField32;
use strum::EnumCount;

use super::{
    setup_row_inputs, EcMulExecutor, EC_MUL_COMPUTE_ROWS, EC_MUL_SCALAR_BITS, EC_MUL_SIGN_PATTERNS,
    EC_MUL_STEPS_PER_ROW, SCALAR_BLOCKS, SCALAR_LIMBS,
};
use crate::weierstrass_chip::curves::{ec_mul, get_curve_type, CurveType};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct EcMulPreCompute<'a> {
    program: &'a FieldExpressionProgram,
    /// `[rs1, rs2]`: base point pointer and scalar pointer.
    rs_addrs: [u8; 2],
    /// `rd`: destination pointer register.
    a: u8,
}

fn checked_register_pointer(pc: u32, pointer: u32) -> Result<u8, StaticProgramError> {
    let pointer = u8::try_from(pointer).map_err(|_| StaticProgramError::InvalidInstruction(pc))?;
    if !pointer.is_multiple_of(REGISTER_NUM_LIMBS as u8) {
        return Err(StaticProgramError::InvalidInstruction(pc));
    }
    Ok(pointer)
}

fn pointer_from_register(pc: u32, value: [u8; REGISTER_NUM_LIMBS]) -> Result<u32, ExecutionError> {
    try_bytes_to_u32(value).ok_or(ExecutionError::Fail {
        pc,
        msg: "EC_MUL pointer register has nonzero upper 32 bits",
    })
}

fn validate_memory_span(pc: u32, address: u32, bytes: usize) -> Result<(), ExecutionError> {
    let in_bounds = (address as usize)
        .checked_add(bytes)
        .is_some_and(|end| end <= MEM_SIZE);
    if !in_bounds {
        return Err(ExecutionError::Fail {
            pc,
            msg: "EC_MUL memory access is out of bounds",
        });
    }
    Ok(())
}

impl<'a, const BLOCKS: usize> EcMulExecutor<BLOCKS> {
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut EcMulPreCompute<'a>,
    ) -> Result<bool, StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        if d.as_canonical_u32() != REGISTER_AS || e.as_canonical_u32() != MEMORY_AS {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *data = EcMulPreCompute {
            program: &self.program,
            rs_addrs: [
                checked_register_pointer(pc, b)?,
                checked_register_pointer(pc, c)?,
            ],
            a: checked_register_pointer(pc, a)?,
        };

        let local_opcode = opcode.local_opcode_idx(self.offset);
        Ok(local_opcode == WeierstrassOpcode::SETUP_EC_MUL as usize)
    }
}

macro_rules! dispatch {
    ($execute_impl:ident, $pre_compute:ident, $is_setup:ident) => {
        // Identified by the (modulus, a) pair the chip was built with. Modulus alone would
        // misclassify a curve sharing a modulus but differing in `a`.
        match (
            $is_setup,
            get_curve_type(
                $pre_compute.program.prime(),
                &$pre_compute.program.setup_values()[0],
            ),
        ) {
            (true, _) => Ok($execute_impl::<_, BLOCKS, { u8::MAX }, true>),
            (false, Some(CurveType::K256)) => {
                Ok($execute_impl::<_, BLOCKS, { CurveType::K256 as u8 }, false>)
            }
            (false, Some(CurveType::P256)) => {
                Ok($execute_impl::<_, BLOCKS, { CurveType::P256 as u8 }, false>)
            }
            (false, Some(CurveType::BN254)) => {
                Ok($execute_impl::<_, BLOCKS, { CurveType::BN254 as u8 }, false>)
            }
            (false, Some(CurveType::BLS12_381)) => {
                Ok($execute_impl::<_, BLOCKS, { CurveType::BLS12_381 as u8 }, false>)
            }
            // Unrecognised curve: interpret the field expression instead.
            (false, None) => Ok($execute_impl::<_, BLOCKS, { u8::MAX }, false>),
        }
    };
}

impl<F: PrimeField32, const BLOCKS: usize> InterpreterExecutor<F> for EcMulExecutor<BLOCKS> {
    /// Distinguishes the setup row from a real multiplication, keeping execution histograms
    /// readable.
    fn get_opcode_name(&self, opcode: usize) -> String {
        let local = opcode.wrapping_sub(WeierstrassOpcode::CLASS_OFFSET) % WeierstrassOpcode::COUNT;
        if local == WeierstrassOpcode::SETUP_EC_MUL as usize {
            "SetupEcMul".to_string()
        } else {
            "EcMul".to_string()
        }
    }

    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<EcMulPreCompute>()
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
        let pre_compute: &mut EcMulPreCompute = data.borrow_mut();
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
        let pre_compute: &mut EcMulPreCompute = data.borrow_mut();
        let is_setup = self.pre_compute_impl(pc, inst, pre_compute)?;
        dispatch!(execute_e1_handler, pre_compute, is_setup)
    }
}

impl<F: PrimeField32, const BLOCKS: usize> InterpreterMeteredExecutor<F> for EcMulExecutor<BLOCKS> {
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<EcMulPreCompute>>()
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
        let pre_compute: &mut E2PreCompute<EcMulPreCompute> = data.borrow_mut();
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
        let pre_compute: &mut E2PreCompute<EcMulPreCompute> = data.borrow_mut();
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
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: &EcMulPreCompute,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pc = exec_state.pc();

    // rs1 holds the base point pointer, rs2 the scalar pointer, rd the destination.
    let rs_vals = [
        pointer_from_register(
            pc,
            exec_state.vm_read_bytes(REGISTER_AS, pre_compute.rs_addrs[0] as u32),
        )?,
        pointer_from_register(
            pc,
            exec_state.vm_read_bytes(REGISTER_AS, pre_compute.rs_addrs[1] as u32),
        )?,
    ];
    for &address in &rs_vals {
        validate_memory_block_byte_ptr(pc, address)?;
    }
    let rd_val = validate_memory_block_byte_ptr(
        pc,
        pointer_from_register(
            pc,
            exec_state.vm_read_bytes(REGISTER_AS, pre_compute.a as u32),
        )?,
    )?;

    validate_memory_span(pc, rs_vals[0], MEMORY_BLOCK_BYTES * BLOCKS)?;
    validate_memory_span(pc, rs_vals[1], MEMORY_BLOCK_BYTES * SCALAR_BLOCKS)?;
    validate_memory_span(pc, rd_val, MEMORY_BLOCK_BYTES * BLOCKS)?;

    let point_data: [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] = std::array::from_fn(|i| {
        exec_state.vm_read_bytes(MEMORY_AS, rs_vals[0] + (i * MEMORY_BLOCK_BYTES) as u32)
    });

    let scalar_blocks: [[u8; MEMORY_BLOCK_BYTES]; SCALAR_BLOCKS] = std::array::from_fn(|i| {
        exec_state.vm_read_bytes(MEMORY_AS, rs_vals[1] + (i * MEMORY_BLOCK_BYTES) as u32)
    });
    let scalar: &[u8; SCALAR_LIMBS] = scalar_blocks.as_flattened().try_into().unwrap();
    if !IS_SETUP {
        debug_assert_eq!(scalar[0] & 1, 1, "EC_MUL scalar must be odd");
    }

    if IS_SETUP {
        // The point operand carries (modulus, a), as for the other setup opcodes; a mismatch
        // reports a clear error rather than an unsatisfiable trace. The scalar operand is unused.
        let coord_blocks = BLOCKS / 2;
        let input_prime = BigUint::from_bytes_le(point_data[..coord_blocks].as_flattened());
        if &input_prime != pre_compute.program.prime() {
            return Err(ExecutionError::Fail {
                pc,
                msg: "EcMul: mismatched prime",
            });
        }
        let input_a = BigUint::from_bytes_le(point_data[coord_blocks..].as_flattened());
        if input_a != pre_compute.program.setup_values()[0] {
            return Err(ExecutionError::Fail {
                pc,
                msg: "EcMul: mismatched curve coefficient a",
            });
        }
    }

    let output_data: [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] = if CURVE_TYPE == u8::MAX {
        // Run the ladder through the field expression so the bytes match trace generation.
        run_ladder_via_expr::<BLOCKS>(pre_compute.program, &point_data, scalar, IS_SETUP)
    } else {
        ec_mul::<CURVE_TYPE, BLOCKS>(point_data, scalar)
    };

    for (i, block) in output_data.into_iter().enumerate() {
        exec_state.vm_write_bytes(MEMORY_AS, rd_val + (i * MEMORY_BLOCK_BYTES) as u32, &block);
    }

    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    Ok(())
}

/// Interprets the ladder through the field expression, selecting the one-hot sign flag as the AIR
/// and trace filler do. Used for unrecognised curves and for setup.
fn run_ladder_via_expr<const BLOCKS: usize>(
    program: &FieldExpressionProgram,
    point_data: &[[u8; MEMORY_BLOCK_BYTES]; BLOCKS],
    scalar: &[u8; SCALAR_LIMBS],
    is_setup: bool,
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    let coord_bytes = BLOCKS / 2 * MEMORY_BLOCK_BYTES;
    let flat = point_data.as_flattened();
    let px = BigUint::from_bytes_le(&flat[..coord_bytes]);
    let py = BigUint::from_bytes_le(&flat[coord_bytes..]);
    let outs = program.output_indices();

    let (rx, ry) = if is_setup {
        // All setup rows are equal. Evaluate the row once.
        let vars = program.execute(&setup_row_inputs(program), &[false; EC_MUL_SIGN_PATTERNS]);
        (vars[outs[0]].clone(), vars[outs[1]].clone())
    } else {
        // The most significant digit is `+1`, so the accumulator seeds itself from `P`.
        let mut rx = px.clone();
        let mut ry = py.clone();
        for row in 0..EC_MUL_COMPUTE_ROWS {
            let mut flags = vec![false; EC_MUL_SIGN_PATTERNS];
            flags[sign_pattern_for_row(scalar, row)] = true;
            let inputs = vec![px.clone(), py.clone(), rx, ry];
            let vars = program.execute(&inputs, &flags);
            rx = vars[outs[0]].clone();
            ry = vars[outs[1]].clone();
        }
        (rx, ry)
    };

    let mut output = [[0u8; MEMORY_BLOCK_BYTES]; BLOCKS];
    let flat = output.as_flattened_mut();
    for (dst, byte) in flat[..coord_bytes].iter_mut().zip(rx.to_bytes_le()) {
        *dst = byte;
    }
    for (dst, byte) in flat[coord_bytes..].iter_mut().zip(ry.to_bytes_le()) {
        *dst = byte;
    }
    output
}

/// The one-hot flag index for compute row `row`, packing its digits most significant first.
/// Digit `i` is bit `i + 1` of the scalar, since the ladder's value is `2B + 1`.
pub(super) fn sign_pattern_for_row(scalar: &[u8], row: usize) -> usize {
    let mut pattern = 0usize;
    for step in 0..EC_MUL_STEPS_PER_ROW {
        let i = EC_MUL_SCALAR_BITS - 1 - (row * EC_MUL_STEPS_PER_ROW + step);
        let j = i + 1;
        let bit = j < EC_MUL_SCALAR_BITS && (scalar[j / 8] >> (j % 8)) & 1 == 1;
        pattern |= (bit as usize) << (EC_MUL_STEPS_PER_ROW - 1 - step);
    }
    pattern
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn validates_register_pointers() {
        assert!(checked_register_pointer(0, 256).is_err());
        assert!(checked_register_pointer(0, 1).is_err());
        assert_eq!(checked_register_pointer(0, 248).unwrap(), 248);
    }

    #[test]
    fn rejects_wide_pointer_registers() {
        let mut bytes = [0u8; REGISTER_NUM_LIMBS];
        bytes[4] = 1;
        assert!(pointer_from_register(0, bytes).is_err());
    }

    #[test]
    fn validates_the_complete_memory_span() {
        let last_block = (MEM_SIZE - MEMORY_BLOCK_BYTES) as u32;
        assert!(validate_memory_span(0, last_block, MEMORY_BLOCK_BYTES).is_ok());
        assert!(validate_memory_span(0, last_block, 2 * MEMORY_BLOCK_BYTES).is_err());
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<
    CTX: ExecutionCtxTrait,
    const BLOCKS: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let pre_compute: &EcMulPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<EcMulPreCompute>()).borrow();
    execute_e12_impl::<_, BLOCKS, CURVE_TYPE, IS_SETUP>(pre_compute, exec_state)
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<
    CTX: MeteredExecutionCtxTrait,
    const BLOCKS: usize,
    const CURVE_TYPE: u8,
    const IS_SETUP: bool,
>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<GuestMemory, CTX>,
) -> Result<(), ExecutionError> {
    let e2_pre_compute: &E2PreCompute<EcMulPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<EcMulPreCompute>>())
            .borrow();
    // One instruction contributes EC_MUL_COMPUTE_ROWS rows, not one.
    exec_state
        .ctx
        .on_height_change(e2_pre_compute.chip_idx as usize, EC_MUL_COMPUTE_ROWS as u32);
    execute_e12_impl::<_, BLOCKS, CURVE_TYPE, IS_SETUP>(&e2_pre_compute.data, exec_state)
}
