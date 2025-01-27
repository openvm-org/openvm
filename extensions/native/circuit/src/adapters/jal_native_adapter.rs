use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, ImmInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols},
            MemoryAddress, MemoryController, OfflineMemory,
        },
        native_adapter::NativeWriteRecord,
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

#[derive(Debug)]
pub struct JalNativeAdapterChip<F: Field> {
    pub air: JalNativeAdapterAir,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> JalNativeAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: JalNativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct JalNativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub writes_aux: MemoryWriteAuxCols<T, 1>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct JalNativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for JalNativeAdapterAir {
    fn width(&self) -> usize {
        JalNativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for JalNativeAdapterAir {
    type Interface = BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &JalNativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let d = AB::Expr::from_canonical_u32(AS::Native as u32);

        self.memory_bridge
            .write(
                MemoryAddress::new(d.clone(), cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a_pointer.into(),
                    ctx.instruction.immediate,
                    AB::Expr::ZERO,
                    d,
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &JalNativeAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for JalNativeAdapterChip<F> {
    type ReadRecord = ();
    type WriteRecord = NativeWriteRecord<F, 1>;
    type Air = JalNativeAdapterAir;
    type Interface = BasicAdapterInterface<F, ImmInstruction<F>, 0, 1, 1, 1>;

    fn preprocess(
        &mut self,
        _memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        Ok(([], ()))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { a, d, .. } = *_instruction;
        let writes = vec![memory.write(d, a, output.writes[0])];

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord {
                from_state,
                writes: writes.try_into().unwrap(),
            },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let aux_cols_factory = memory.aux_cols_factory();
        let row_slice: &mut JalNativeAdapterCols<_> = row_slice.borrow_mut();

        let write = memory.record_by_id(write_record.writes[0].0);
        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
        row_slice.a_pointer = write.pointer;
        aux_cols_factory.generate_write_aux(write, &mut row_slice.writes_aux);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
