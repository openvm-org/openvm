use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
};

use openvm_circuit::{
    arch::{
        execution::ExecuteFunc,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
        DynArray, E2PreCompute, ExecutionError, InsExecutorE1, InsExecutorE2, Result,
        VmSegmentState,
    },
    system::memory::{online::GuestMemory, POINTER_MAX_BITS},
};
use openvm_circuit_derive::InstructionExecutor;
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpr, FieldExpressionStep,
};
use openvm_rv32_adapters::Rv32VecHeapAdapterStep;
use openvm_stark_backend::p3_field::PrimeField32;

pub mod fp2_chip;
pub mod modular_chip;

mod fp2;
pub use fp2::*;
mod modular_extension;
pub use modular_extension::*;
mod fp2_extension;
pub use fp2_extension::*;
mod config;
pub use config::*;

pub struct AlgebraCpuProverExt;

#[derive(Clone, InstructionExecutor)]
pub struct FieldExprVecHeapStep<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(FieldExpressionStep<Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>);

impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        adapter: Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        name: &str,
    ) -> Self {
        Self(FieldExpressionStep::new(
            adapter,
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            name,
        ))
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FieldExpressionPreCompute<'a, const NUM_READS: usize> {
    expr: &'a FieldExpr,
    // NUM_READS <= 2 as in Rv32VecHeapAdapter
    rs_addrs: [u8; NUM_READS],
    a: u8,
    flag_idx: u8,
}

impl<'a, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    fn pre_compute_impl<F: PrimeField32>(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut FieldExpressionPreCompute<'a, NUM_READS>,
    ) -> Result<bool> {
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
        if d != RV32_REGISTER_AS || e != RV32_MEMORY_AS {
            return Err(ExecutionError::InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.0.offset);

        // Pre-compute flag_idx
        let needs_setup = self.0.expr.needs_setup();
        let mut flag_idx = self.0.expr.num_flags() as u8;
        if needs_setup {
            // Find which opcode this is in our local_opcode_idx list
            if let Some(opcode_position) = self
                .0
                .local_opcode_idx
                .iter()
                .position(|&idx| idx == local_opcode)
            {
                // If this is NOT the last opcode (setup), get the corresponding flag_idx
                if opcode_position < self.0.opcode_flag_idx.len() {
                    flag_idx = self.0.opcode_flag_idx[opcode_position] as u8;
                }
            }
        }

        let rs_addrs = from_fn(|i| if i == 0 { b } else { c } as u8);
        *data = FieldExpressionPreCompute {
            a: a as u8,
            rs_addrs,
            expr: &self.0.expr,
            flag_idx,
        };

        Ok(needs_setup)
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    InsExecutorE1<F> for FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<FieldExpressionPreCompute<NUM_READS>>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1ExecutionCtx,
    {
        let pre_compute: &mut FieldExpressionPreCompute<NUM_READS> = data.borrow_mut();

        let needs_setup = self.pre_compute_impl(pc, inst, pre_compute)?;
        let fn_ptr = if needs_setup {
            execute_e1_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, true>
        } else {
            execute_e1_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, false>
        };

        Ok(fn_ptr)
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    InsExecutorE2<F> for FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    #[inline(always)]
    fn e2_pre_compute_size(&self) -> usize {
        std::mem::size_of::<E2PreCompute<FieldExpressionPreCompute<NUM_READS>>>()
    }

    fn pre_compute_e2<Ctx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E2ExecutionCtx,
    {
        let pre_compute: &mut E2PreCompute<FieldExpressionPreCompute<NUM_READS>> =
            data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        let needs_setup = self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;
        let fn_ptr = if needs_setup {
            execute_e2_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, true>
        } else {
            execute_e2_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, false>
        };

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const NEEDS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &FieldExpressionPreCompute<NUM_READS> = pre_compute.borrow();

    execute_e12_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, NEEDS_SETUP>(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<
    F: PrimeField32,
    CTX: E2ExecutionCtx,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const NEEDS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<FieldExpressionPreCompute<NUM_READS>> = pre_compute.borrow();
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, 1);
    execute_e12_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, NEEDS_SETUP>(
        &pre_compute.data,
        vm_state,
    );
}

unsafe fn execute_e12_impl<
    F: PrimeField32,
    CTX: E1ExecutionCtx,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const NEEDS_SETUP: bool,
>(
    pre_compute: &FieldExpressionPreCompute<NUM_READS>,
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    // Read register values
    let rs_vals = pre_compute
        .rs_addrs
        .map(|addr| u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, addr as u32)));

    // Read memory values
    let read_data: [[[u8; BLOCK_SIZE]; BLOCKS]; NUM_READS] = rs_vals.map(|address| {
        debug_assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));
        from_fn(|i| vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32))
    });
    let read_data: DynArray<u8> = read_data.into();

    let writes = run_field_expression_precomputed::<NEEDS_SETUP>(
        pre_compute.expr,
        pre_compute.flag_idx as usize,
        &read_data.0,
    );

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32));
    debug_assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << POINTER_MAX_BITS));

    // Write output data to memory
    let data: [[u8; BLOCK_SIZE]; BLOCKS] = writes.into();
    for (i, block) in data.into_iter().enumerate() {
        vm_state.vm_write(RV32_MEMORY_AS, rd_val + (i * BLOCK_SIZE) as u32, &block);
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> Deref
    for FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    type Target = FieldExpressionStep<
        Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    >;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> DerefMut
    for FieldExprVecHeapStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
