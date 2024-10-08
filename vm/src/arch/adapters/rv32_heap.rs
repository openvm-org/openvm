use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{ExecutionBridge, MachineAdapter, MachineAdapterInterface},
    memory::{
        offline_checker::{MemoryBridge, MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapReadRecord, MemoryHeapWriteRecord,
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
    _marker: std::marker::PhantomData<F>,
    air: Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE>,
}

impl<F: Field, const READ_SIZE: usize, const WRITE_SIZE: usize>
    Rv32HeapAdapter<F, READ_SIZE, WRITE_SIZE>
{
    pub fn new(execution_bridge: ExecutionBridge, memory_bridge: MemoryBridge) -> Self {
        let air = Rv32HeapAdapterAir::new(execution_bridge, memory_bridge);
        Self {
            air,
            _marker: std::marker::PhantomData,
        }
    }
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
        std::mem::size_of::<Rv32HeapAdapterCols<u8, READ_SIZE, WRITE_SIZE>>()
    }
}

pub struct Rv32HeapAdapterInterface<
    T: AbstractField,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: AbstractField, const READ_SIZE: usize, const WRITE_SIZE: usize> MachineAdapterInterface<T>
    for Rv32HeapAdapterInterface<T, READ_SIZE, WRITE_SIZE>
{
    type Reads = ([T; READ_SIZE], [T; READ_SIZE]);
    type Writes = [T; WRITE_SIZE];
    type ProcessedInstruction = ();
}

pub struct AddressAndSpace<T> {
    pub address: T,
    pub address_space: T,
}

pub struct HeapAddresses<T> {
    pub address_address: AddressAndSpace<T>,
    pub data_address: AddressAndSpace<T>,
}

pub struct Rv32HeapAdapterCols<T, const READ_SIZE: usize, const WRITE_SIZE: usize> {
    pub x_read_aux: MemoryHeapReadAuxCols<T, READ_SIZE>,
    pub y_read_aux: MemoryHeapReadAuxCols<T, READ_SIZE>,
    pub z_write_aux: MemoryHeapWriteAuxCols<T, WRITE_SIZE>,
    pub x_addresses: HeapAddresses<T>,
    pub y_addresses: HeapAddresses<T>,
    pub z_addresses: HeapAddresses<T>,
}

impl<
        F: PrimeField32,
        // const NUM_READS: usize,
        // const NUM_WRITES: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > MachineAdapter<F> for Rv32HeapAdapter<F, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = [MemoryHeapReadRecord<F, READ_SIZE>; 2];
    type WriteRecord = [MemoryHeapWriteRecord<F, WRITE_SIZE>; 1];
    type Cols<T> = Rv32HeapAdapterCols<T, READ_SIZE, WRITE_SIZE>;
    type Interface<T: AbstractField> = Rv32HeapAdapterInterface<T, READ_SIZE, WRITE_SIZE>;
    type Air = Rv32HeapAdapterAir<READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut crate::memory::MemoryChip<F>,
        instruction: &crate::program::Instruction<F>,
    ) -> crate::arch::Result<(
        <Self::Interface<F> as MachineAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            opcode: _,
            op_a: _z_address_ptr,
            op_b: x_address_ptr,
            op_c: y_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        let x_read = memory.read_heap::<READ_SIZE>(d, e, x_address_ptr);
        let y_read = memory.read_heap::<READ_SIZE>(d, e, y_address_ptr);

        Ok((
            (x_read.data_read.data, y_read.data_read.data),
            [x_read, y_read],
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut crate::memory::MemoryChip<F>,
        instruction: &crate::program::Instruction<F>,
        from_state: crate::arch::ExecutionState<usize>,
        // we aren't really using output.to_pc?
        output: crate::arch::InstructionOutput<F, Self::Interface<F>>,
    ) -> crate::arch::Result<(crate::arch::ExecutionState<usize>, Self::WriteRecord)> {
        let Instruction {
            opcode: _,
            op_a: z_address_ptr,
            op_b: _x_address_ptr,
            op_c: _y_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        let z_write = memory.write_heap::<WRITE_SIZE>(d, e, z_address_ptr, output.writes);
        Ok((
            crate::arch::ExecutionState {
                pc: from_state.pc + 4,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            [z_write],
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut Self::Cols<F>,
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!()
    }

    fn eval_adapter_constraints<
        AB: afs_stark_backend::interaction::InteractionBuilder<F = F>
            + p3_air::PairBuilder
            + p3_air::AirBuilderWithPublicValues,
    >(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _interface: crate::arch::IntegrationInterface<AB::Expr, Self::Interface<AB::Expr>>,
    ) -> AB::Expr {
        todo!()
    }

    fn air(&self) -> Self::Air {
        self.air
    }
}
