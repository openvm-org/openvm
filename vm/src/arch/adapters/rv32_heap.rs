use std::{marker::PhantomData, mem::size_of};

use afs_derive::AlignedBorrow;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{read_rv32_register, RV32_REGISTER_NUM_LANES};
use crate::{
    arch::{
        AdapterRuntimeContext, ExecutionBridge, ExecutionState, Result, VmAdapterChip,
        VmAdapterInterface,
    },
    memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        HeapAddress, MemoryChip, MemoryReadRecord, MemoryWriteRecord,
    },
    program::Instruction,
};

// Assuming two reads 1 write.

/// Reads `NUM_READS` register values and uses each register value as a pointer to batch read `READ_SIZE` memory cells from
/// address starting at the pointer value.
/// Reads `NUM_WRITES` register values and uses each register value as a pointer to batch write `WRITE_SIZE` memory cells
/// with address starting at the pointer value.
#[derive(Clone)]
pub struct Rv32HeapAdapter<
    F: Field,
    // const NUM_READS: usize,
    // const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    air: Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE>,
    _marker: PhantomData<F>,
}

impl<F: Field, const READ_SIZE: usize, const WRITE_SIZE: usize>
    Rv32HeapAdapter<F, READ_SIZE, WRITE_SIZE>
{
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        let air = Rv32HeapAdapterAir::new(execution_bridge, memory_bridge);
        Self {
            air,
            _marker: PhantomData,
        }
    }
}

/// Represents first reads a RV register, and then a batch read at the pointer.
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapReadRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_read: MemoryReadRecord<T, N>,
}

/// Represents first reads a RV register, and then a batch write at the pointer.
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapWriteRecord<T, const N: usize> {
    pub address_read: MemoryReadRecord<T, RV32_REGISTER_NUM_LANES>,
    pub data_write: MemoryWriteRecord<T, N>,
}

#[derive(Clone, Copy)]
pub struct Rv32HeapAdapterAir<const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<const READ_SIZE: usize, const WRITE_SIZE: usize> Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE> {
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        Self {
            _execution_bridge: execution_bridge,
            _memory_bridge: memory_bridge,
        }
    }
}

impl<F, const READ_SIZE: usize, const WRITE_SIZE: usize> BaseAir<F>
    for Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        size_of::<Rv32HeapAdapterCols<u8, READ_SIZE, WRITE_SIZE>>()
    }
}

pub struct Rv32HeapAdapterInterface<
    T: AbstractField,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    _marker: PhantomData<T>,
}

impl<T: AbstractField, const READ_SIZE: usize, const WRITE_SIZE: usize> VmAdapterInterface<T>
    for Rv32HeapAdapterInterface<T, READ_SIZE, WRITE_SIZE>
{
    type Reads = ([T; READ_SIZE], [T; READ_SIZE]);
    type Writes = [T; WRITE_SIZE];
    type ProcessedInstruction = ();
}

pub struct Rv32HeapAdapterCols<T, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    // TODO: we should save on address space as register is 1, and the data is fixed for all (e = 2).
    pub read_aux: [Rv32RegisterHeapReadAuxCols<T, READ_SIZE>; 2],
    pub write_aux: Rv32RegisterHeapWriteAuxCols<T, WRITE_SIZE>,
    pub read_addresses: [HeapAddress<T, T>; 2],
    pub write_addresses: HeapAddress<T, T>,
}

#[repr(C)]
#[derive(Clone, Debug, AlignedBorrow)]
pub struct Rv32RegisterHeapReadAuxCols<T, const N: usize> {
    pub address: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub data: MemoryReadAuxCols<T, N>,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Rv32RegisterHeapWriteAuxCols<T, const N: usize> {
    pub address: MemoryReadAuxCols<T, RV32_REGISTER_NUM_LANES>,
    pub data: MemoryWriteAuxCols<T, N>,
}

impl<
        F: PrimeField32,
        // const NUM_READS: usize,
        // const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterChip<F> for Rv32HeapAdapter<F, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = [Rv32RegisterHeapReadRecord<F, READ_SIZE>; 2];
    type WriteRecord = [Rv32RegisterHeapWriteRecord<F, WRITE_SIZE>; 1];
    type Interface<T: AbstractField> = Rv32HeapAdapterInterface<T, READ_SIZE, WRITE_SIZE>;
    type Air = Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            op_a: _z_address_ptr,
            op_b: x_address_ptr,
            op_c: y_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        debug_assert_eq!(d.as_canonical_u32(), 1);
        let x_read = read_heap_from_rv32_register::<_, READ_SIZE>(memory, d, e, x_address_ptr);
        let y_read = read_heap_from_rv32_register::<_, READ_SIZE>(memory, d, e, y_address_ptr);

        Ok((
            (x_read.data_read.data, y_read.data_read.data),
            [x_read, y_read],
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: AdapterRuntimeContext<F, Self::Interface<F>>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        let Instruction {
            op_a: z_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        let z_write = write_heap_from_rv32_register::<_, WRITE_SIZE>(
            memory,
            d,
            e,
            z_address_ptr,
            output.writes,
        );
        Ok((
            ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            [z_write],
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
pub fn read_heap_from_rv32_register<F: PrimeField32, const N: usize>(
    memory: &mut MemoryChip<F>,
    ptr_address_space: F,
    data_address_space: F,
    ptr_pointer: F,
) -> Rv32RegisterHeapReadRecord<F, N> {
    let (address_read, val) = read_rv32_register(memory, ptr_address_space, ptr_pointer);
    let data_read = memory.read(data_address_space, F::from_canonical_u32(val));

    Rv32RegisterHeapReadRecord {
        address_read,
        data_read,
    }
}

/// First lookup the heap pointer from register, and then write the data at the pointer.
pub fn write_heap_from_rv32_register<F: PrimeField32, const N: usize>(
    memory: &mut MemoryChip<F>,
    ptr_address_space: F,
    data_address_space: F,
    ptr_pointer: F,
    data: [F; N],
) -> Rv32RegisterHeapWriteRecord<F, N> {
    let (address_read, val) = read_rv32_register(memory, ptr_address_space, ptr_pointer);
    let data_write = memory.write(data_address_space, F::from_canonical_u32(val), data);

    Rv32RegisterHeapWriteRecord {
        address_read,
        data_write,
    }
}
