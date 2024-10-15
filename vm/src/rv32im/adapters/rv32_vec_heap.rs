use std::{marker::PhantomData, mem::size_of};

use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use super::{
    Rv32RegisterHeapReadAuxCols, Rv32RegisterHeapReadRecord, Rv32RegisterHeapWriteAuxCols,
    Rv32RegisterHeapWriteRecord,
};
use crate::{
    arch::{
        AdapterRuntimeContext, ExecutionBridge, ExecutionState, Result, VmAdapterChip,
        VmAdapterInterface,
    },
    rv32im::adapters::{read_heap_from_rv32_register, write_heap_from_rv32_register},
    system::{
        memory::{offline_checker::MemoryBridge, HeapAddress, MemoryController},
        program::Instruction,
    },
};

pub struct Rv32VecHeapAdapter<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
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
    > Rv32VecHeapAdapter<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        Self {
            air: Rv32VecHeapAdapterAir::new(execution_bridge, memory_bridge),
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Rv32VecHeapProcessedInstruction<T> {
    pub _marker: PhantomData<T>,
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

pub struct Rv32VecHeapAdapterInterface<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    _marker: PhantomData<T>,
}

impl<
        T: AbstractField,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterInterface<T>
    for Rv32VecHeapAdapterInterface<T, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type Reads = [[T; READ_SIZE]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = ();
}

pub struct Rv32VecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub read_aux: [Rv32RegisterHeapReadAuxCols<T, READ_SIZE>; NUM_READS],
    pub write_aux: [Rv32RegisterHeapWriteAuxCols<T, WRITE_SIZE>; NUM_WRITES],
    pub read_addresses: [HeapAddress<T, T>; NUM_READS],
    pub write_addresses: [HeapAddress<T, T>; NUM_WRITES],
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterChip<F> for Rv32VecHeapAdapter<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = [Rv32RegisterHeapReadRecord<F, READ_SIZE>; NUM_READS];
    type WriteRecord = [Rv32RegisterHeapWriteRecord<F, WRITE_SIZE>; NUM_WRITES];
    type Interface = Rv32VecHeapAdapterInterface<F, NUM_READS, NUM_WRITES, READ_SIZE, WRITE_SIZE>;
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

        let mut read_record = vec![];
        for _ in 0..NUM_READS {
            let x_read = read_heap_from_rv32_register::<F, READ_SIZE>(memory, d, e, address_ptr);
            read_record.push(x_read);
        }
        let reads = read_record
            .iter()
            .map(|x| x.data_read.data)
            .collect::<Vec<_>>();
        let reads: [[F; READ_SIZE]; NUM_READS] = reads.try_into().unwrap();
        let read_record: [Rv32RegisterHeapReadRecord<F, READ_SIZE>; NUM_READS] =
            read_record.try_into().unwrap();

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

        let mut write_record = vec![];
        for i in 0..NUM_WRITES {
            let x_write = write_heap_from_rv32_register::<F, WRITE_SIZE>(
                memory,
                d,
                e,
                address_out,
                output.writes[i],
            );
            write_record.push(x_write);
        }
        let write_record: [Rv32RegisterHeapWriteRecord<F, WRITE_SIZE>; NUM_WRITES] =
            write_record.try_into().unwrap();

        Ok((
            ExecutionState {
                pc: from_state.pc + 4 * NUM_WRITES as u32,
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
