use std::{array::from_fn, marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{
    batch_read_rv32_registers, read_rv32_register, Rv32RegisterHeapReadAuxCols,
    Rv32RegisterHeapReadRecord, Rv32RegisterHeapWriteAuxCols, Rv32RegisterHeapWriteRecord,
    RV32_REGISTER_NUM_LANES,
};
use crate::{
    arch::{
        AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge, ExecutionState, Result,
        VmAdapterChip, VmAdapterInterface,
    },
    rv32im::adapters::{read_heap_from_rv32_register, write_heap_from_rv32_register},
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            HeapAddress, MemoryController, MemoryReadRecord, MemoryWriteRecord,
        },
        program::Instruction,
    },
};

#[derive(Clone)]
pub struct Rv32VecHeapAdapter<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CELLS: usize,
    const WRITE_CELLS: usize,
> {
    pub air: Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>,
    _marker: PhantomData<F>,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_CELLS: usize,
        const WRITE_CELLS: usize,
    > Rv32VecHeapAdapter<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE, READ_CELLS, WRITE_CELLS>
{
    /// ## Panics
    /// If `READ_CELLS != NUM_READS * READ_SIZE` or `WRITE_CELLS != NUM_WRITES * WRITE_SIZE`.
    /// This is a runtime assertion until Rust const generics expressions are stabilized.
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        assert_eq!(READ_CELLS, NUM_READS * READ_SIZE);
        assert_eq!(WRITE_CELLS, NUM_WRITES * WRITE_SIZE);
        Self {
            air: Rv32VecHeapAdapterAir::new(execution_bridge, memory_bridge),
            _marker: PhantomData,
        }
    }
}

/// Represents first reads a RV register, and then a batch read at the pointer.
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapBatchReadRecord<T, const READ_SIZE: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_read: MemoryReadRecord<T, READ_SIZE>,
}

/// Represents first reads a RV register, and then a batch write at the pointer.
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapBatchWriteRecord<T, const WRITE_SIZE: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_write: MemoryWriteRecord<T, WRITE_SIZE>,
}

#[derive(Clone)]
pub struct Rv32VecHeapAdapterAir<
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        Self {
            _execution_bridge: execution_bridge,
            _memory_bridge: memory_bridge,
        }
    }
}

impl<
        F,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F> for Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        size_of::<Rv32VecHeapAdapterCols<u8, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>>()
    }
}

// pub struct Rv32VecHeapAdapterInterface<
//     T,
//     const NUM_READS: usize,
//     const NUM_WRITES: usize,
//     const READ_SIZE: usize,
//     const WRITE_SIZE: usize,
// > {
//     _marker: PhantomData<T>,
// }

// impl<
//         T: AbstractField,
//         const NUM_READS: usize,
//         const NUM_WRITES: usize,
//         const READ_SIZE: usize,
//         const WRITE_SIZE: usize,
//     > VmAdapterInterface<T>
//     for BasicAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
// {
//     type Reads = [[T; READ_SIZE]; NUM_READS];
//     type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
//     type ProcessedInstruction = ();
// }

pub struct Rv32VecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub read_aux: [Rv32RegisterHeapBatchReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub write_aux: [Rv32RegisterHeapBatchWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
    pub read_addresses: [HeapAddress<T, T>; NUM_READS],
    pub write_addresses: [HeapAddress<T, T>; NUM_WRITES],
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct Rv32RegisterHeapBatchReadAuxCols<T, const READ_SIZE: usize> {
    pub address: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub data: MemoryReadAuxCols<T, READ_SIZE>,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapBatchWriteAuxCols<T, const WRITE_SIZE: usize> {
    pub address: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub data: MemoryWriteAuxCols<T, WRITE_SIZE>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_CELLS: usize,
        const WRITE_CELLS: usize,
    > VmAdapterChip<F>
    for Rv32VecHeapAdapter<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE, READ_CELLS, WRITE_CELLS>
{
    type ReadRecord = Rv32RegisterHeapReadRecord<F, READ_CELLS>;
    type WriteRecord = Rv32RegisterHeapWriteRecord<F, WRITE_CELLS>;
    type Interface = BasicAdapterInterface<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;
    type Air = Rv32VecHeapAdapterAir<NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_b: address_ptr,
            d,
            e,
            ..
        } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        let read_record =
            batch_read_heap_from_rv32_register::<F, READ_CELLS>(memory, d, e, address_ptr);
        // let reads = read_record
        //     .iter()
        //     .map(|x| x.data_read.data)
        //     .collect::<Vec<_>>();
        let mut read_record_it = read_record.data_read.data.into_iter();
        let reads: [[F; READ_SIZE]; NUM_READS] =
            from_fn(|_| from_fn(|_| read_record_it.next().unwrap()));
        // let read_record: [Rv32RegisterHeapReadRecord<F, READ_SIZE>; NUM_READS] =
        //     read_record.try_into().unwrap();

        Ok((reads, read_record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let Instruction {
            op_a: address_out,
            d,
            e,
            ..
        } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), 1);

        let write_record = batch_write_heap_from_rv32_register::<
            F,
            NUM_WRITES,
            WRITE_SIZE,
            WRITE_CELLS,
        >(memory, d, e, address_out, output.writes);
        let write_record: [Rv32RegisterHeapWriteRecord<F, WRITE_SIZE>; NUM_WRITES] =
            write_record.try_into().unwrap();

        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + 4),
                timestamp: memory.timestamp(),
            },
            write_record,
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut [F],
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

/// First lookup the heap pointer from register, and then read the data at the pointer.
pub fn batch_read_heap_from_rv32_register<
    F: PrimeField32,
    // const NUM_READS: usize,
    // const READ_SIZE: usize,
    const READ_CELLS: usize,
>(
    memory: &mut MemoryController<F>,
    ptr_address_space: F,
    data_address_space: F,
    ptr_pointer: F,
) -> Rv32RegisterHeapReadRecord<F, READ_CELLS> {
    let (address_read, data_address) = read_rv32_register(memory, ptr_address_space, ptr_pointer);
    let data_read =
        memory.read::<READ_CELLS>(data_address_space, F::from_canonical_u32(data_address));

    Rv32RegisterHeapReadRecord {
        address_read,
        data_read,
    }
}

/// First lookup the heap pointer from register, and then write the data at the pointer.
pub fn batch_write_heap_from_rv32_register<
    F: PrimeField32,
    const NUM_WRITES: usize,
    const WRITE_SIZE: usize,
    const WRITE_CELLS: usize,
>(
    memory: &mut MemoryController<F>,
    ptr_address_space: F,
    data_address_space: F,
    ptr_pointer: F,
    data: [[F; WRITE_SIZE]; NUM_WRITES],
) -> Rv32RegisterHeapWriteRecord<F, WRITE_CELLS> {
    let (address_read, val) = read_rv32_register(memory, ptr_address_space, ptr_pointer);
    let data_write =
        memory.write::<WRITE_CELLS>(data_address_space, F::from_canonical_u32(val), data);

    Rv32RegisterHeapWriteRecord {
        address_read,
        data_write,
    }
}
