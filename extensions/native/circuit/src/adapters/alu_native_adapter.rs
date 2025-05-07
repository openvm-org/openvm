use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterRuntimeContext, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, ExecutionBus, ExecutionState, MinimalInstruction,
        Result, VmAdapterAir, VmAdapterChip, VmAdapterInterface,
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

use super::{tracing_read_or_imm, tracing_write, tracing_write_reg};

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

#[derive(derive_new::new)]
pub struct AluNativeAdapterStep;

impl<F, CTX> AdapterTraceStep<F, CTX> for AluNativeAdapterStep
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<AluNativeAdapterCols<u8>>();
    type ReadData = [F; 2];
    type WriteData = [F; 1];
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction { b, c, e, f, .. } = instruction;

        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();

        let read1 = tracing_read_or_imm(
            memory,
            e.as_canonical_u32(),
            b.as_canonical_u32(),
            &mut adapter_row.e_as,
            (&mut adapter_row.b_pointer, &mut adapter_row.reads_aux[0]),
        );
        let read2 = tracing_read_or_imm(
            memory,
            f.as_canonical_u32(),
            c.as_canonical_u32(),
            &mut adapter_row.f_as,
            (&mut adapter_row.c_pointer, &mut adapter_row.reads_aux[1]),
        );
        [read1, read2]
    }

    #[inline(always)]
    fn write(
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction { a, .. } = instruction;

        let adapter_row: &mut AluNativeAdapterCols<F> = adapter_row.borrow_mut();
        tracing_write(
            memory,
            a.as_canonical_u32(),
            data,
            (&mut adapter_row.a_pointer, &mut adapter_row.write_aux),
        );
    }

    #[inline(always)]
    fn fill_trace_row(
        mem_helper: &MemoryAuxColsFactory<F>,
        bitwise_lookup_chip: &BitwiseOperationLookupChip<LIMB_BITS>,
        adapter_row: &mut [F],
    ) {
        todo!("Implement fill_trace_row")
    }
}

impl<Mem, F> AdapterExecutorE1<Mem, F> for AluNativeAdapterStep
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    type ReadData = (F, F);
    type WriteData = F;

    fn read(memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { b, c, e, f, .. } = instruction;

        let [read1]: [F; 1] = unsafe { memory.read(e.as_canonical_u32(), b.as_canonical_u32()) };
        let [read2]: [F; 1] = unsafe { memory.read(f.as_canonical_u32(), c.as_canonical_u32()) };

        (read1, read2)
    }

    fn write(memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData) {
        let Instruction { a, .. } = instruction;

        unsafe { memory.write(AS::Native, a.as_canonical_u32(), &[data]) };
    }
}

// impl<F: PrimeField32> VmAdapterChip<F> for AluNativeAdapterChip<F> {
//     type ReadRecord = NativeReadRecord<F, 2>;
//     type WriteRecord = NativeWriteRecord<F, 1>;
//     type Air = AluNativeAdapterAir;
//     type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, 1, 1>;

//     fn preprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//     ) -> Result<(
//         <Self::Interface as VmAdapterInterface<F>>::Reads,
//         Self::ReadRecord,
//     )> {
//         let Instruction { b, c, e, f, .. } = *instruction;

//         let reads = vec![memory.read::<F, 1>(e, b), memory.read::<F, 1>(f, c)];
//         let i_reads: [_; 2] = std::array::from_fn(|i| reads[i].1);

//         Ok((
//             i_reads,
//             Self::ReadRecord {
//                 reads: reads.try_into().unwrap(),
//             },
//         ))
//     }

//     fn postprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         _instruction: &Instruction<F>,
//         from_state: ExecutionState<u32>,
//         output: AdapterRuntimeContext<F, Self::Interface>,
//         _read_record: &Self::ReadRecord,
//     ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
//         let Instruction { a, .. } = *_instruction;
//         let writes = vec![memory.write(
//             F::from_canonical_u32(AS::Native as u32),
//             a,
//             &output.writes[0],
//         )];

//         Ok((
//             ExecutionState {
//                 pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
//                 timestamp: memory.timestamp(),
//             },
//             Self::WriteRecord {
//                 from_state,
//                 writes: writes.try_into().unwrap(),
//             },
//         ))
//     }

//     fn generate_trace_row(
//         &self,
//         row_slice: &mut [F],
//         read_record: Self::ReadRecord,
//         write_record: Self::WriteRecord,
//         memory: &OfflineMemory<F>,
//     ) {
//         let row_slice: &mut AluNativeAdapterCols<_> = row_slice.borrow_mut();
//         let aux_cols_factory = memory.aux_cols_factory();

//         row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);

//         row_slice.a_pointer = memory.record_by_id(write_record.writes[0].0).pointer;
//         row_slice.b_pointer = memory.record_by_id(read_record.reads[0].0).pointer;
//         row_slice.c_pointer = memory.record_by_id(read_record.reads[1].0).pointer;
//         row_slice.e_as = memory.record_by_id(read_record.reads[0].0).address_space;
//         row_slice.f_as = memory.record_by_id(read_record.reads[1].0).address_space;

//         for (i, x) in read_record.reads.iter().enumerate() {
//             let read = memory.record_by_id(x.0);
//             aux_cols_factory.generate_read_or_immediate_aux(read, &mut row_slice.reads_aux[i]);
//         }

//         let write = memory.record_by_id(write_record.writes[0].0);
//         aux_cols_factory.generate_write_aux(write, &mut row_slice.write_aux);
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }
