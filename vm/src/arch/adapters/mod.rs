//! Under construction
use std::mem::size_of;

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};

use crate::{
    arch::{ExecutionBridge, ExecutionState, MachineAdapterInterface},
    memory::offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
};

pub mod risc_alu;

/// RISC-V Register Adapter
///
/// Any RISC-V adapter that only reads and writes to registers can use the following structs as
/// their Interface, Cols, and Air types respectively. Reads NUM_READS 4-byte registers, and
/// writes to NUM_WRITES 4-byte registers.
///
pub struct Rv32RegisterAdapterInterface<T, const NUM_READS: usize, const NUM_WRITES: usize> {
    _marker: std::marker::PhantomData<T>,
}

impl<F: AbstractField, const NUM_READS: usize, const NUM_WRITES: usize> MachineAdapterInterface<F>
    for Rv32RegisterAdapterInterface<F, NUM_READS, NUM_WRITES>
{
    type Reads = [[F; 4]; NUM_READS];
    type Writes = [[F; 4]; NUM_WRITES];
    type ProcessedInstruction = u32;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32RegisterAdapterCols<T, const NUM_READS: usize, const NUM_WRITES: usize> {
    pub from_state: ExecutionState<T>,
    pub address_space: T,
    pub reads: [Rv32RegisterData<T>; NUM_READS],
    pub writes: [Rv32RegisterData<T>; NUM_WRITES],
    pub reads_aux: [MemoryReadAuxCols<T, 4>; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, 4>; NUM_WRITES],
}

impl<T, const NUM_READS: usize, const NUM_WRITES: usize>
    Rv32RegisterAdapterCols<T, NUM_READS, NUM_WRITES>
{
    pub fn width() -> usize {
        size_of::<Rv32RegisterAdapterCols<u8, NUM_READS, NUM_WRITES>>()
    }
}

#[derive(Clone, Copy)]
pub struct Rv32RegisterAdapterAir<const NUM_READS: usize, const NUM_WRITES: usize> {
    pub(super) _execution_bridge: ExecutionBridge,
    pub(super) _memory_bridge: MemoryBridge,
}

impl<F: Field, const NUM_READS: usize, const NUM_WRITES: usize> BaseAir<F>
    for Rv32RegisterAdapterAir<NUM_READS, NUM_WRITES>
{
    fn width(&self) -> usize {
        Rv32RegisterAdapterCols::<F, NUM_READS, NUM_WRITES>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_READS: usize, const NUM_WRITES: usize> Air<AB>
    for Rv32RegisterAdapterAir<NUM_READS, NUM_WRITES>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32RegisterData<T> {
    pub data: [T; 4],
    pub reg_idx: T,
}

/// Reads `NUM_READS` register values and uses each register value as a pointer to batch read `READ_SIZE` memory cells from
/// address starting at the pointer value.
/// Reads `NUM_WRITES` register values and uses each register value as a pointer to batch write `WRITE_SIZE` memory cells
/// with address starting at the pointer value.
pub struct Rv32HeapAdapter<
    F: Field,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    _marker: std::marker::PhantomData<F>,
}
