mod add_ne;
mod double;

pub use add_ne::*;
pub use double::*;

#[cfg(test)]
mod tests;

use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::{
    execution::ExecuteFunc,
    execution_mode::E1E2ExecutionCtx,
    instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    ExecutionError::InvalidInstruction,
    MatrixRecordArena, NewVmChipWrapper, Result, StepExecutorE1, VmAirWrapper, VmSegmentState,
};
use openvm_circuit_derive::{TraceFiller, TraceStep};
use openvm_circuit_primitives::{var_range::SharedVariableRangeCheckerChip, AlignedBytesBorrow};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_mod_circuit_builder::{
    run_field_expression_precomputed, FieldExpr, FieldExpressionCoreAir, FieldExpressionStep,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterStep};
use openvm_stark_backend::p3_field::PrimeField32;

pub(crate) type WeierstrassAir<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = VmAirWrapper<
    Rv32VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
    FieldExpressionCoreAir,
>;

#[derive(TraceStep, TraceFiller)]
pub(crate) struct WeierstrassStep<
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(FieldExpressionStep<Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>>);

impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    WeierstrassStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter: Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        range_checker: SharedVariableRangeCheckerChip,
        name: &str,
        should_finalize: bool,
    ) -> Self {
        Self(openvm_mod_circuit_builder::FieldExpressionStep::new(
            adapter,
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            name,
            should_finalize,
        ))
    }
}

#[derive(AlignedBytesBorrow)]
struct WeierstrassPreCompute {
    a: u32,
    b: u32,
    c: u32,
    expr: FieldExpr,
    flag_idx: usize,
}

impl<F: PrimeField32, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    StepExecutorE1<F> for WeierstrassStep<NUM_READS, BLOCKS, BLOCK_SIZE>
{
    fn pre_compute_size(&self) -> usize {
        std::mem::size_of::<WeierstrassPreCompute>()
    }

    fn pre_compute_e1<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let data: &mut WeierstrassPreCompute = data.borrow_mut();
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
            return Err(InvalidInstruction(pc));
        }

        let local_opcode = opcode.local_opcode_idx(self.0.offset);

        // Pre-compute flag_idx
        let needs_setup = self.0.expr.needs_setup();
        let mut flag_idx = self.0.expr.num_flags();
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
                    flag_idx = self.0.opcode_flag_idx[opcode_position];
                }
            }
        }

        *data = WeierstrassPreCompute {
            a,
            b,
            c,
            expr: self.0.expr.clone(),
            flag_idx,
        };

        let fn_ptr = if needs_setup {
            execute_e1_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, true>
        } else {
            execute_e1_impl::<_, _, NUM_READS, BLOCKS, BLOCK_SIZE, false>
        };

        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<
    F: PrimeField32,
    CTX: E1E2ExecutionCtx,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const NEEDS_SETUP: bool,
>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &WeierstrassPreCompute = pre_compute.borrow();

    // Read register values
    let rs_vals = [
        u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.b)),
        u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.c)),
    ];

    // Read memory values
    let mut read_data = Vec::new();
    for address in rs_vals {
        // TODO(ayush): add this back
        // assert!(address as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << self.0.pointer_max_bits));
        for i in 0..BLOCKS {
            let data: [u8; BLOCK_SIZE] =
                vm_state.vm_read(RV32_MEMORY_AS, address + (i * BLOCK_SIZE) as u32);
            read_data.extend_from_slice(&data);
        }
    }

    let writes = run_field_expression_precomputed::<NEEDS_SETUP>(
        &pre_compute.expr,
        pre_compute.flag_idx,
        &read_data,
    );

    let rd_val = u32::from_le_bytes(vm_state.vm_read(RV32_REGISTER_AS, pre_compute.a));
    // assert!(rd_val as usize + BLOCK_SIZE * BLOCKS - 1 < (1 << self.0.pointer_max_bits));

    // Write output data to memory
    for (i, chunk) in writes.0.chunks_exact(BLOCK_SIZE).enumerate() {
        let chunk_array: &[u8; BLOCK_SIZE] = chunk.try_into().unwrap();
        vm_state.vm_write(
            RV32_MEMORY_AS,
            rd_val + (i * BLOCK_SIZE) as u32,
            chunk_array,
        );
    }

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;
}

pub(crate) type WeierstrassChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> = NewVmChipWrapper<
    F,
    WeierstrassAir<NUM_READS, BLOCKS, BLOCK_SIZE>,
    WeierstrassStep<NUM_READS, BLOCKS, BLOCK_SIZE>,
    MatrixRecordArena<F>,
>;
