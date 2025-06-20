use std::{
    borrow::{Borrow, BorrowMut},
    ops::Deref,
};

use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        get_record_from_slice, ExecutionBridge, ExecutionState, MatrixRecordArena, MultiRowLayout,
        NewVmChipWrapper, PcIncOrSet, RecordArena, Result, StepExecutorE1, TraceFiller, TraceStep,
        VmStateMut,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols, MemoryWriteAuxRecord},
            online::{GuestMemory, TracingMemory},
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{
            memory_read_native, memory_write_native_from_state, tracing_write_native,
        },
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow,
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

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct JalRangeCheckRecord<F> {
    pub is_jal: bool,
    pub a: F,
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub write: MemoryWriteAuxRecord<F, 1>,
    pub b: F,
    pub c: F,
}

/// Chip for JAL and RANGE_CHECK. These opcodes are logically irrelevant. Putting these opcodes into
/// the same chip is just to save columns.
#[derive(derive_new::new)]
pub struct JalRangeCheckStep {
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, CTX> TraceStep<F, CTX> for JalRangeCheckStep
where
    F: PrimeField32,
{
    // Will use MultiRowLayout with `num_rows = 1`
    type RecordLayout = MultiRowLayout<()>;
    type RecordMut<'a> = &'a mut JalRangeCheckRecord<F>;

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

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<F, TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let &Instruction {
            opcode, a, b, c, ..
        } = instruction;

        debug_assert!(
            opcode == NativeJalOpcode::JAL.global_opcode()
                || opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode()
        );

        let record = arena.alloc(MultiRowLayout::default());

        record.from_pc = *state.pc;
        record.from_timestamp = state.memory.timestamp;

        record.a = a;
        record.b = b;

        if opcode == NativeJalOpcode::JAL.global_opcode() {
            record.is_jal = true;
            record.c = F::ZERO;

            tracing_write_native(
                state.memory,
                a.as_canonical_u32(),
                &[F::from_canonical_u32(
                    state.pc.wrapping_add(DEFAULT_PC_STEP),
                )],
                &mut record.write.prev_timestamp,
                &mut record.write.prev_data,
            );
            *state.pc = (F::from_canonical_u32(*state.pc) + b).as_canonical_u32();
        } else if opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            record.is_jal = false;
            record.c = c;

            let a_ptr = a.as_canonical_u32();
            let [a_val]: [F; 1] = memory_read_native(state.memory.data(), a_ptr);
            tracing_write_native(
                state.memory,
                a_ptr,
                &[a_val],
                &mut record.write.prev_timestamp,
                &mut record.write.prev_data,
            );
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }
}

impl<F: PrimeField32, CTX> TraceFiller<F, CTX> for JalRangeCheckStep {
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        let record: &mut JalRangeCheckRecord<F> =
            unsafe { get_record_from_slice(&mut row_slice, ()) };
        let cols: &mut JalRangeCheckCols<F> = row_slice.borrow_mut();

        // Writing in reverse order to avoid overwriting the `record`
        if record.is_jal {
            cols.y = F::ZERO;
            cols.c = F::ZERO;
            cols.b = record.b;
            cols.writes_aux.set_prev_data(record.write.prev_data);
            mem_helper.fill(
                record.write.prev_timestamp,
                record.from_timestamp,
                cols.writes_aux.as_mut(),
            );
            cols.state.timestamp = F::from_canonical_u32(record.from_timestamp);
            cols.state.pc = F::from_canonical_u32(record.from_pc);
            cols.a_pointer = record.a;
            cols.is_range_check = F::ZERO;
            cols.is_jal = F::ONE;
        } else {
            let a_val = record.write.prev_data[0].as_canonical_u32();
            let b = record.b.as_canonical_u32();
            let c = record.c.as_canonical_u32();
            let x = a_val & 0xffff;
            let y = a_val >> 16;
            #[cfg(debug_assertions)]
            {
                assert!(b <= 16);
                assert!(c <= 14);
                assert!(x < (1 << b));
                assert!(y < (1 << c));
            }

            self.range_checker_chip.add_count(x, b as usize);
            self.range_checker_chip.add_count(y, c as usize);

            cols.y = F::from_canonical_u32(y);
            cols.c = record.c;
            cols.b = record.b;
            cols.writes_aux.set_prev_data(record.write.prev_data);
            mem_helper.fill(
                record.write.prev_timestamp,
                record.from_timestamp,
                cols.writes_aux.as_mut(),
            );
            cols.state.timestamp = F::from_canonical_u32(record.from_timestamp);
            cols.state.pc = F::from_canonical_u32(record.from_pc);
            cols.a_pointer = record.a;
            cols.is_range_check = F::ONE;
            cols.is_jal = F::ZERO;
        }
    }
}

impl<F> StepExecutorE1<F> for JalRangeCheckStep
where
    F: PrimeField32,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
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
            memory_write_native_from_state(
                state,
                a.as_canonical_u32(),
                &[F::from_canonical_u32(
                    state.pc.wrapping_add(DEFAULT_PC_STEP),
                )],
            );
            *state.pc = (F::from_canonical_u32(*state.pc) + b).as_canonical_u32();
        } else if opcode == NativeRangeCheckOpcode::RANGE_CHECK.global_opcode() {
            let [a_val]: [F; 1] = memory_read_native(state.memory, a.as_canonical_u32());

            memory_write_native_from_state(state, a.as_canonical_u32(), &[a_val]);

            #[cfg(debug_assertions)]
            {
                let a_val = a_val.as_canonical_u32();
                let b = instruction.b.as_canonical_u32();
                let c = instruction.c.as_canonical_u32();
                let x = a_val & 0xffff;
                let y = a_val >> 16;

                assert!(b <= 16);
                assert!(c <= 14);
                assert!(x < (1 << b));
                assert!(y < (1 << c));
            }
            *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        }

        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<F, GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

pub type JalRangeCheckChip<F> =
    NewVmChipWrapper<F, JalRangeCheckAir, JalRangeCheckStep, MatrixRecordArena<F>>;
