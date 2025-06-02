use std::{
    borrow::{Borrow, BorrowMut},
    ops::Deref,
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        ExecutionBridge, ExecutionError, ExecutionState, NewVmChipWrapper, PcIncOrSet, Result,
        StepExecutorE1, TraceStep, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerBus,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, NativeJalOpcode, NativeRangeCheckOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use static_assertions::const_assert_eq;
use AS::Native;

use crate::adapters::{memory_read_native, memory_write_native, tracing_write_native};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(AlignedBorrow)]
struct JalRangeCheckCols<T> {
    is_jal: T,
    is_range_check: T,
    a_pointer: T,
    state: ExecutionState<T>,
    // Write when is_jal, read when is_range_check.
    writes_aux: MemoryWriteAuxCols<T, 1>,
    b: T,
    // Only used by range check.
    c: T,
    // Only used by range check.
    y: T,
}

const OVERALL_WIDTH: usize = JalRangeCheckCols::<u8>::width();
const_assert_eq!(OVERALL_WIDTH, 12);

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct JalRangeCheckAir {
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for JalRangeCheckAir {
    fn width(&self) -> usize {
        OVERALL_WIDTH
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JalRangeCheckAir {}
impl<F: Field> PartitionedBaseAir<F> for JalRangeCheckAir {}
impl<AB: InteractionBuilder> Air<AB> for JalRangeCheckAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_slice = local.deref();
        let local: &JalRangeCheckCols<AB::Var> = local_slice.borrow();
        builder.assert_bool(local.is_jal);
        builder.assert_bool(local.is_range_check);
        let is_valid = local.is_jal + local.is_range_check;
        builder.assert_bool(is_valid.clone());

        let d = AB::Expr::from_canonical_u32(Native as u32);
        let a_val = local.writes_aux.prev_data()[0];
        // if is_jal, write pc + DEFAULT_PC_STEP, else if is_range_check, read a_val.
        let write_val = local.is_jal
            * (local.state.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP))
            + local.is_range_check * a_val;
        self.memory_bridge
            .write(
                MemoryAddress::new(d.clone(), local.a_pointer),
                [write_val],
                local.state.timestamp,
                &local.writes_aux,
            )
            .eval(builder, is_valid.clone());

        let opcode = local.is_jal
            * AB::F::from_canonical_usize(NativeJalOpcode::JAL.global_opcode().as_usize())
            + local.is_range_check
                * AB::F::from_canonical_usize(
                    NativeRangeCheckOpcode::RANGE_CHECK
                        .global_opcode()
                        .as_usize(),
                );
        // Increment pc by b if is_jal, else by DEFAULT_PC_STEP if is_range_check.
        let pc_inc = local.is_jal * local.b
            + local.is_range_check * AB::F::from_canonical_u32(DEFAULT_PC_STEP);
        builder.when(local.is_jal).assert_zero(local.c);
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                opcode,
                [local.a_pointer.into(), local.b.into(), local.c.into(), d],
                local.state,
                AB::F::ONE,
                PcIncOrSet::Inc(pc_inc),
            )
            .eval(builder, is_valid);

        // Range check specific:
        // a_val = x + y * (1 << 16)
        let x = a_val - local.y * AB::Expr::from_canonical_u32(1 << 16);
        self.range_bus
            .send(x.clone(), local.b)
            .eval(builder, local.is_range_check);
        // Assert y < (1 << c), where c <= 14.
        self.range_bus
            .send(local.y, local.c)
            .eval(builder, local.is_range_check);
    }
}

/// Chip for JAL and RANGE_CHECK. These opcodes are logically irrelevant. Putting these opcodes into
/// the same chip is just to save columns.
pub struct JalRangeCheckStep {
    range_checker_chip: SharedVariableRangeCheckerChip,
    /// If true, ignore execution errors.
    debug: bool,
}

impl JalRangeCheckStep {
    pub fn new(range_checker_chip: SharedVariableRangeCheckerChip) -> Self {
        Self {
            range_checker_chip,
            debug: false,
        }
    }
    pub fn set_debug(&mut self) {
        self.debug = true;
    }
}

impl<F, CTX> TraceStep<F, CTX> for JalRangeCheckStep
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        let jal_opcode = NativeJalOpcode::JAL.global_opcode().as_usize();
        let range_check_opcode = NativeRangeCheckOpcode::RANGE_CHECK
            .global_opcode()
            .as_usize();
        if opcode == jal_opcode {
            return String::from("JAL");
        }
        if opcode == range_check_opcode {
            return String::from("RANGE_CHECK");
        }
        panic!("Unknown opcode {}", opcode);
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction {
            opcode, a, b, c, ..
        } = instruction;

        debug_assert!(
            opcode == NativeJalOpcode::JAL.global_opcode()
                || opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode()
        );

        let row: &mut JalRangeCheckCols<F> =
            trace[*trace_offset..*trace_offset + width].borrow_mut();

        row.state.pc = F::from_canonical_u32(*state.pc);
        row.state.timestamp = F::from_canonical_u32(state.memory.timestamp);

        row.a_pointer = a;
        row.b = b;

        if opcode == NativeJalOpcode::JAL.global_opcode() {
            row.is_jal = F::ONE;
            row.c = F::ZERO;

            tracing_write_native(
                state.memory,
                a.as_canonical_u32(),
                &[F::from_canonical_u32(
                    state.pc.wrapping_add(DEFAULT_PC_STEP),
                )],
                &mut row.writes_aux,
            );
            // TODO(ayush): can this addition be done in u32 instead of F
            *state.pc = (F::from_canonical_u32(*state.pc) + b).as_canonical_u32();
        } else if opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            row.is_jal = F::ZERO;
            row.c = c;

            let [a_val]: [F; 1] = memory_read_native(state.memory.data(), a.as_canonical_u32());
            tracing_write_native(
                state.memory,
                a.as_canonical_u32(),
                &[a_val],
                &mut row.writes_aux,
            );

            // TODO(ayush): should this debug stuff be removed?
            let a_val = a_val.as_canonical_u32();
            let b = b.as_canonical_u32();
            let c = c.as_canonical_u32();

            debug_assert!(!self.debug || b <= 16);
            debug_assert!(!self.debug || c <= 14);

            let x = a_val & ((1 << 16) - 1);
            if !self.debug && x >= 1 << b {
                return Err(ExecutionError::Fail { pc: *state.pc });
            }
            let y = a_val >> 16;
            if !self.debug && y >= 1 << c {
                return Err(ExecutionError::Fail { pc: *state.pc });
            }

            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        *trace_offset += width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let row: &mut JalRangeCheckCols<_> = row_slice.borrow_mut();

        let timestamp = row.state.timestamp.as_canonical_u32();
        mem_helper.fill_from_prev(timestamp, row.writes_aux.as_mut());

        row.is_range_check = F::ONE - row.is_jal;

        if row.is_range_check.is_one() {
            let a_val = row.writes_aux.prev_data()[0];
            let a_val_u32 = a_val.as_canonical_u32();
            let y = a_val_u32 >> 16;
            let x = a_val_u32 & ((1 << 16) - 1);
            self.range_checker_chip
                .add_count(x, row.b.as_canonical_u32() as usize);
            self.range_checker_chip
                .add_count(y, row.c.as_canonical_u32() as usize);
            row.y = F::from_canonical_u32(y);
        }
    }
}

impl<F> StepExecutorE1<F> for JalRangeCheckStep
where
    F: PrimeField32,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction { opcode, a, b, .. } = instruction;

        debug_assert!(
            opcode == NativeJalOpcode::JAL.global_opcode()
                || opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode()
        );

        if opcode == NativeJalOpcode::JAL.global_opcode() {
            memory_write_native(
                state.memory,
                a.as_canonical_u32(),
                &[F::from_canonical_u32(
                    state.pc.wrapping_add(DEFAULT_PC_STEP),
                )],
            );
            // TODO(ayush): can this addition be done in u32 instead of F
            *state.pc = (F::from_canonical_u32(*state.pc) + b).as_canonical_u32();
        } else if opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            // TODO(ayush): should this not call memory callback?
            let [a_val]: [F; 1] = memory_read_native(state.memory, a.as_canonical_u32());

            memory_write_native(state.memory, a.as_canonical_u32(), &[a_val]);

            let a_val = a_val.as_canonical_u32();
            let b = instruction.b.as_canonical_u32();
            let c = instruction.c.as_canonical_u32();

            debug_assert!(!self.debug || b <= 16);
            debug_assert!(!self.debug || c <= 14);

            let x = a_val & ((1 << 16) - 1);
            if !self.debug && x >= 1 << b {
                return Err(ExecutionError::Fail { pc: *state.pc });
            }
            let y = a_val >> 16;
            if !self.debug && y >= 1 << c {
                return Err(ExecutionError::Fail { pc: *state.pc });
            }

            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

pub type JalRangeCheckChip<F> = NewVmChipWrapper<F, JalRangeCheckAir, JalRangeCheckStep>;
