use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryController, OfflineMemory,
        },
        native_adapter::{NativeReadRecord, NativeWriteRecord},
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
pub struct AluNativeAdapterChip<F: Field> {
    pub air: AluNativeAdapterAir,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> AluNativeAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
    ) -> Self {
        Self {
            air: AluNativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct AluNativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub b_pointer: T,
    pub c_pointer: T,
    pub e_as: T,
    pub f_as: T,
    pub reads_aux: [MemoryReadOrImmediateAuxCols<T>; 2],
    pub write_aux: MemoryWriteAuxCols<T, 1>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct AluNativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for AluNativeAdapterAir {
    fn width(&self) -> usize {
        AluNativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for AluNativeAdapterAir {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &AluNativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.e_as, cols.b_pointer),
                ctx.reads[0][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_or_immediate(
                MemoryAddress::new(cols.f_as, cols.c_pointer),
                ctx.reads[1][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.write_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a_pointer.into(),
                    cols.b_pointer.into(),
                    cols.c_pointer.into(),
                    native_as.clone(),
                    cols.e_as.into(),
                    cols.f_as.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &AluNativeAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for AluNativeAdapterChip<F> {
    type ReadRecord = NativeReadRecord<F, 2>;
    type WriteRecord = NativeWriteRecord<F, 1>;
    type Air = AluNativeAdapterAir;
    type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 1, 1>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { b, c, e, f, .. } = *instruction;

        let reads = vec![memory.read::<F, 1>(e, b), memory.read::<F, 1>(f, c)];
        let i_reads: [_; 2] = std::array::from_fn(|i| reads[i].1);

        Ok((
            i_reads,
            Self::ReadRecord {
                reads: reads.try_into().unwrap(),
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction { a, .. } = *_instruction;
        let writes = vec![memory.write(
            F::from_canonical_u32(AS::Native as u32),
            a,
            &output.writes[0],
        )];

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
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        let row_slice: &mut AluNativeAdapterCols<_> = row_slice.borrow_mut();
        let aux_cols_factory = memory.aux_cols_factory();

        row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

        row_slice.a_pointer = memory.record_by_id(write_record.writes[0].0).pointer;
        row_slice.b_pointer = memory.record_by_id(read_record.reads[0].0).pointer;
        row_slice.c_pointer = memory.record_by_id(read_record.reads[1].0).pointer;
        row_slice.e_as = memory.record_by_id(read_record.reads[0].0).address_space;
        row_slice.f_as = memory.record_by_id(read_record.reads[1].0).address_space;

        for (i, x) in read_record.reads.iter().enumerate() {
            let read = memory.record_by_id(x.0);
            aux_cols_factory.generate_read_or_immediate_aux(read, &mut row_slice.reads_aux[i]);
        }

        let write = memory.record_by_id(write_record.writes[0].0);
        aux_cols_factory.generate_write_aux(write, &mut row_slice.write_aux);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
