use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        instructions::LocalOpcode, AdapterAirContext, AdapterExecutorE1, AdapterRuntimeContext,
        ExecutionBridge, ExecutionBus, ExecutionState, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
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
use openvm_native_compiler::{
    conversion::AS,
    NativeLoadStoreOpcode::{self, *},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};
use serde::{Deserialize, Serialize};

pub struct NativeLoadStoreInstruction<T> {
    pub is_valid: T,
    // Absolute opcode number
    pub opcode: T,
    pub is_loadw: T,
    pub is_storew: T,
    pub is_hint_storew: T,
}

pub struct NativeLoadStoreAdapterInterface<T, const NUM_CELLS: usize>(PhantomData<T>);

impl<T, const NUM_CELLS: usize> VmAdapterInterface<T>
    for NativeLoadStoreAdapterInterface<T, NUM_CELLS>
{
    type Reads = (T, [T; NUM_CELLS]);
    type Writes = [T; NUM_CELLS];
    type ProcessedInstruction = NativeLoadStoreInstruction<T>;
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct NativeLoadStoreReadRecord<F: Field, const NUM_CELLS: usize> {
    pub pointer_read: RecordId,
    pub data_read: Option<RecordId>,
    pub write_as: F,
    pub write_ptr: F,

    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
}

#[repr(C)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct NativeLoadStoreWriteRecord<F: Field, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<F>,
    pub write_id: RecordId,
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct NativeLoadStoreAdapterCols<T, const NUM_CELLS: usize> {
    pub from_state: ExecutionState<T>,
    pub a: T,
    pub b: T,
    pub c: T,

    pub data_write_pointer: T,

    pub pointer_read_aux_cols: MemoryReadAuxCols<T>,
    pub data_read_aux_cols: MemoryReadAuxCols<T>,
    pub data_write_aux_cols: MemoryWriteAuxCols<T, NUM_CELLS>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct NativeLoadStoreAdapterAir<const NUM_CELLS: usize> {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for NativeLoadStoreAdapterAir<NUM_CELLS> {
    fn width(&self) -> usize {
        NativeLoadStoreAdapterCols::<F, NUM_CELLS>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_CELLS: usize> VmAdapterAir<AB>
    for NativeLoadStoreAdapterAir<NUM_CELLS>
{
    type Interface = NativeLoadStoreAdapterInterface<AB::Expr, NUM_CELLS>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = AB::Expr::from_canonical_usize(0);

        let is_valid = ctx.instruction.is_valid;
        let is_loadw = ctx.instruction.is_loadw;
        let is_storew = ctx.instruction.is_storew;
        let is_hint_storew = ctx.instruction.is_hint_storew;

        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);

        let ptr = ctx.reads.0;
        // Here we ignore ctx.reads.1 and we use `ctx.writes` as the data for both the write and the
        // second read (in the case of load/store when it exists).
        let data = ctx.writes;

        // first pointer read is always [c]_d
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), cols.c),
                [ptr.clone()],
                timestamp + timestamp_delta.clone(),
                &cols.pointer_read_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    native_as.clone(),
                    is_storew.clone() * cols.a + is_loadw.clone() * (ptr.clone() + cols.b),
                ),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_read_aux_cols,
            )
            .eval(builder, is_valid.clone() - is_hint_storew.clone());
        timestamp_delta += is_valid.clone() - is_hint_storew.clone();

        builder.assert_eq(
            is_valid.clone() * cols.data_write_pointer,
            is_loadw.clone() * cols.a
                + (is_storew.clone() + is_hint_storew.clone()) * (ptr.clone() + cols.b),
        );

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), cols.data_write_pointer),
                data.clone(),
                timestamp + timestamp_delta.clone(),
                &cols.data_write_aux_cols,
            )
            .eval(builder, is_valid.clone());
        timestamp_delta += is_valid.clone();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.a.into(),
                    cols.b.into(),
                    cols.c.into(),
                    native_as.clone(),
                    native_as.clone(),
                ],
                cols.from_state,
                timestamp_delta.clone(),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let local_cols: &NativeLoadStoreAdapterCols<_, NUM_CELLS> = local.borrow();
        local_cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct NativeLoadStoreAdapterStep<const NUM_CELLS: usize> {
    offset: usize,
}

impl<F, CTX, const NUM_CELLS: usize> AdapterTraceStep<F, CTX>
    for NativeLoadStoreAdapterStep<NUM_CELLS>
where
    F: PrimeField32,
{
    const WIDTH: usize = size_of::<NativeLoadStoreAdapterCols<u8, NUM_CELLS>>();
    type ReadData = (F, [F; NUM_CELLS]);
    type WriteData = [F; NUM_CELLS];
    // TODO(ayush): what's this?
    type TraceContext<'a> = &'a BitwiseOperationLookupChip<LIMB_BITS>;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, adapter_row: &mut [F]) {
        let adapter_row: &mut NativeLoadStoreAdapterCols<F> = adapter_row.borrow_mut();

        adapter_row.from_state.pc = F::from_canonical_u32(pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    #[inline(always)]
    fn read(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        todo!("Implement read method");
    }

    #[inline(always)]
    fn write(
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        todo!("Implement write method");
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

impl<Mem, F, const READ_SIZE: usize, const WRITE_SIZE: usize> AdapterExecutorE1<Mem, F>
    for NativeLoadStoreAdapterStep<READ_SIZE, WRITE_SIZE>
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    type ReadData = (F, [F; NUM_CELLS]);
    type WriteData = [F; NUM_CELLS];

    fn read(memory: &mut Mem, instruction: &Instruction<F>) -> Self::ReadData {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        // TODO(ayush): how to handle self.offset?
        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [read_cell]: [F; 1] =
            unsafe { memory.read(d.as_canonical_u32(), c.as_canonical_u32()) };

        let (data_read_as, data_write_as) = {
            match local_opcode {
                LOADW => (e, d),
                STOREW | HINT_STOREW => (d, e),
            }
        };
        let (data_read_ptr, data_write_ptr) = {
            match local_opcode {
                LOADW => (read_cell + b, a),
                STOREW | HINT_STOREW => (a, read_cell + b),
            }
        };

        let data_read: [F; NUM_CELLS] = match local_opcode {
            HINT_STOREW => [F::ZERO; NUM_CELLS],
            LOADW | STOREW => unsafe {
                memory.read(
                    data_read_as.as_canonical_u32(),
                    data_read_ptr.as_canonical_u32(),
                )
            },
        };

        (read_cell, data_read)
    }

    fn write(memory: &mut Mem, instruction: &Instruction<F>, data: &Self::WriteData) {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        // TODO(ayush): how to handle self.offset?
        let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let [read_cell]: [F; 1] =
            unsafe { memory.read(d.as_canonical_u32(), c.as_canonical_u32()) };

        let (data_read_as, data_write_as) = {
            match local_opcode {
                LOADW => (e, d),
                STOREW | HINT_STOREW => (d, e),
            }
        };
        let (data_read_ptr, data_write_ptr) = {
            match local_opcode {
                LOADW => (read_cell + b, a),
                STOREW | HINT_STOREW => (a, read_cell + b),
            }
        };

        unsafe {
            memory.write(
                data_write_as.as_canonical_u32(),
                data_write_ptr.as_cas_canonical_u32(),
                &data,
            )
        };
    }
}

// impl<F: PrimeField32, const NUM_CELLS: usize> VmAdapterChip<F>
//     for NativeLoadStoreAdapterChip<F, NUM_CELLS>
// {
//     type ReadRecord = NativeLoadStoreReadRecord<F, NUM_CELLS>;
//     type WriteRecord = NativeLoadStoreWriteRecord<F, NUM_CELLS>;
//     type Air = NativeLoadStoreAdapterAir<NUM_CELLS>;
//     type Interface = NativeLoadStoreAdapterInterface<F, NUM_CELLS>;

//     fn preprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         instruction: &Instruction<F>,
//     ) -> Result<(
//         <Self::Interface as VmAdapterInterface<F>>::Reads,
//         Self::ReadRecord,
//     )> {
//         let Instruction {
//             opcode,
//             a,
//             b,
//             c,
//             d,
//             e,
//             ..
//         } = *instruction;
//         let local_opcode = NativeLoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

//         let read_as = d;
//         let read_ptr = c;
//         let read_cell = memory.read_cell(read_as, read_ptr);

//         let (data_read_as, data_write_as) = {
//             match local_opcode {
//                 LOADW => (e, d),
//                 STOREW | HINT_STOREW => (d, e),
//             }
//         };
//         let (data_read_ptr, data_write_ptr) = {
//             match local_opcode {
//                 LOADW => (read_cell.1 + b, a),
//                 STOREW | HINT_STOREW => (a, read_cell.1 + b),
//             }
//         };

//         let data_read = match local_opcode {
//             HINT_STOREW => None,
//             LOADW | STOREW => Some(memory.read::<F, NUM_CELLS>(data_read_as, data_read_ptr)),
//         };
//         let record = NativeLoadStoreReadRecord {
//             pointer_read: read_cell.0,
//             data_read: data_read.map(|x| x.0),
//             write_as: data_write_as,
//             write_ptr: data_write_ptr,
//             a,
//             b,
//             c,
//             d,
//             e,
//         };

//         Ok((
//             (read_cell.1, data_read.map_or([F::ZERO; NUM_CELLS], |x| x.1)),
//             record,
//         ))
//     }

//     fn postprocess(
//         &mut self,
//         memory: &mut MemoryController<F>,
//         _instruction: &Instruction<F>,
//         from_state: ExecutionState<u32>,
//         output: AdapterRuntimeContext<F, Self::Interface>,
//         read_record: &Self::ReadRecord,
//     ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
//         let (write_id, _) = memory.write::<F, NUM_CELLS>(
//             read_record.write_as,
//             read_record.write_ptr,
//             &output.writes,
//         );
//         Ok((
//             ExecutionState {
//                 pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
//                 timestamp: memory.timestamp(),
//             },
//             Self::WriteRecord {
//                 from_state: from_state.map(F::from_canonical_u32),
//                 write_id,
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
//         let cols: &mut NativeLoadStoreAdapterCols<_, NUM_CELLS> = row_slice.borrow_mut();
//         cols.from_state = write_record.from_state;
//         cols.a = read_record.a;
//         cols.b = read_record.b;
//         cols.c = read_record.c;

//         let data_read = read_record.data_read.map(|read| memory.record_by_id(read));
//         if let Some(data_read) = data_read {
//             aux_cols_factory.generate_read_aux(data_read, &mut cols.data_read_aux_cols);
//         }

//         let write = memory.record_by_id(write_record.write_id);
//         cols.data_write_pointer = write.pointer;

//         aux_cols_factory.generate_read_aux(
//             memory.record_by_id(read_record.pointer_read),
//             &mut cols.pointer_read_aux_cols,
//         );
//         aux_cols_factory.generate_write_aux(write, &mut cols.data_write_aux_cols);
//     }

//     fn air(&self) -> &Self::Air {
//         &self.air
//     }
// }
