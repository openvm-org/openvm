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
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryController, OfflineMemory, RecordId,
        },
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
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeVectorizedReadRecord<const N: usize> {
    pub b: RecordId,
    pub c: RecordId,
}

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeVectorizedWriteRecord<const N: usize> {
    pub from_state: ExecutionState<u32>,
    pub a: RecordId,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeVectorizedAdapterCols<T, const N: usize> {
    pub from_state: ExecutionState<T>,
    pub a_pointer: T,
    pub b_pointer: T,
    pub c_pointer: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: [MemoryWriteAuxCols<T, N>; 1],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeVectorizedAdapterAir<const N: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const N: usize> BaseAir<F> for NativeVectorizedAdapterAir<N> {
    fn width(&self) -> usize {
        NativeVectorizedAdapterCols::<F, N>::width()
    }
}

impl<AB: InteractionBuilder, const N: usize> VmAdapterAir<AB> for NativeVectorizedAdapterAir<N> {
    type Interface = BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 2, 1, N, N>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.b_pointer),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.c_pointer),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &cols.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.a_pointer),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux[0],
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
                    native_as.clone(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &NativeVectorizedAdapterCols<_, N> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct NativeVectorizedAdapterChip<const N: usize>;

impl<F, CTX, const N: usize> AdapterTraceStep<F, CTX> for NativeVectorizedAdapterChip<N>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<NativeVectorizedAdapterCols<u8, NUM_CELLS>>();
    type ReadData = [[F; N]; 2];
    type WriteData = [F; N];
    // TODO(ayush): what's this?
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_row: &mut NativeVectorizedAdapterCols<F> = adapter_row.borrow_mut();

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
        let adapter_row: &mut NativeVectorizedAdapterCols<F> = adapter_row.borrow_mut();

        // TODO(ayush): create similar `tracing_read_reg_or_imm` for F
        let y_val = tracing_read_reg_or_imm(
            memory,
            d.as_canonical_u32(),
            b.as_canonical_u32(),
            // TODO(ayush): why no address space pointer? Should this be hardcoded?
            &mut adapter_row.d_as,
            (&mut adapter_row.b_pointer, &mut adapter_row.reads_aux[0]),
        );
        let z_val = tracing_read_reg_or_imm(
            memory,
            e.as_canonical_u32(),
            c.as_canonical_u32(),
            // TODO(ayush): why no address space pointer? Should this be hardcoded?
            &mut adapter_row.e_as,
            (&mut adapter_row.c_pointer, &mut adapter_row.reads_aux[1]),
        );

        [y_val, z_val]
    }

    #[inline(always)]
    fn write(
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let Instruction { a, d, .. } = instruction;
        let adapter_row: &mut NativeVectorizedAdapterCols<F> = adapter_row.borrow_mut();

        // TODO(ayush): create similar `tracing_read_reg_or_imm` for F
        tracing_write_reg(
            memory,
            d.as_canonical_u32(),
            a.as_canonical_u32(),
            data,
            (&mut adapter_row.a_pointer, &mut adapter_row.writes_aux[0]),
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

impl<F, const N: usize> AdapterExecutorE1<F> for NativeVectorizedAdapterChip<N>
where
    F: PrimeField32,
{
    type ReadData = ([F; N], [F; N]);
    type WriteData = [F; N];

    fn read(memory: &mut GuestMemory, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction { b, c, d, e, .. } = instruction;

        let y_val: [F; N] = unsafe { memory.read(d.as_canonical_u32(), b.as_cas_canonical_u32()) };
        let z_val: [F; N] =
            unsafe { memory.read(e.as_cas_canonical_u32(), c.as_cas_canonical_u32()) };

        (y_val, z_val)
    }

    fn write(memory: &mut GuestMemory, instruction: &Instruction<F>, data: &Self::WriteData) {
        let Instruction { a, d, .. } = instruction;

        unsafe { memory.write(d.as_canonical_u32(), a.as_cas_canonical_u32(), &data) };
    }
}

// impl<F: PrimeField32, const N: usize> VmAdapterChip<F> for NativeVectorizedAdapterChip<F, N> {
//     type ReadRecord = NativeVectorizedReadRecord<N>;
//     type WriteRecord = NativeVectorizedWriteRecord<N>;
//     type Air = NativeVectorizedAdapterAir<N>;
//     type Interface = BasicAdapterInterface<F, MinimalInstruction<F>, 2, 1, N, N>;

//     fn preprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//     ) -> Result<(
//         <Self::Interface as VmAdapterInterface<F>>::Reads,
//         Self::ReadRecord,
//     )> {
//         let Instruction { b, c, d, e, .. } = *instruction;

//         let y_val = memory.read::<F, N>(d, b);
//         let z_val = memory.read::<F, N>(e, c);

//         Ok((
//             [y_val.1, z_val.1],
//             Self::ReadRecord {
//                 b: y_val.0,
//                 c: z_val.0,
//             },
//         ))
//     }

//     fn postprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//         from_state: ExecutionState<u32>,
//         output: AdapterRuntimeContext<F, Self::Interface>,
//         _read_record: &Self::ReadRecord,
//     ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
//         let Instruction { a, d, .. } = *instruction;
//         let (a_val, _) = memory.write(d, a, &output.writes[0]);

//         Ok((
//             ExecutionState {
//                 pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
//                 timestamp: memory.timestamp(),
//             },
//             Self::WriteRecord {
//                 from_state,
//                 a: a_val,
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
//         let aux_cols_factory = memory.aux_cols_factory();
//         let row_slice: &mut NativeVectorizedAdapterCols<_, N> = row_slice.borrow_mut();

//         let b_record = memory.record_by_id(read_record.b);
//         let c_record = memory.record_by_id(read_record.c);
//         let a_record = memory.record_by_id(write_record.a);
//         row_slice.from_state = write_record.from_state.map(F::from_canonical_u32);
//         row_slice.a_pointer = a_record.pointer;
//         row_slice.b_pointer = b_record.pointer;
//         row_slice.c_pointer = c_record.pointer;
//         aux_cols_factory.generate_read_aux(b_record, &mut row_slice.reads_aux[0]);
//         aux_cols_factory.generate_read_aux(c_record, &mut row_slice.reads_aux[1]);
//         aux_cols_factory.generate_write_aux(a_record, &mut row_slice.writes_aux[0]);
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }
