use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller, BasicAdapterInterface,
        ExecutionBridge, ImmInstruction, VmAdapterAir,
    },
    system::memory::{offline_checker::MemoryBridge, online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
};

use crate::{
    Rv32VecHeapBranchAdapterAir, Rv32VecHeapBranchAdapterCols, Rv32VecHeapBranchAdapterExecutor,
    Rv32VecHeapBranchAdapterFiller, Rv32VecHeapBranchAdapterRecord,
};

/// This adapter reads from NUM_READS <= 2 pointers (for branch operations).
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads are from the addresses in `rs[0]` (and `rs[1]` if `NUM_READS = 2`).
/// * No writes are performed (branch operations only compare values).
///
/// This is a wrapper over `Rv32VecHeapBranchAdapterAir` with `BLOCKS_PER_READ = 1`.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HeapBranchAdapterAir<const NUM_READS: usize, const READ_SIZE: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    address_bits: usize,
}

impl<F: Field, const NUM_READS: usize, const READ_SIZE: usize> BaseAir<F>
    for Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>
{
    fn width(&self) -> usize {
        Rv32VecHeapBranchAdapterCols::<F, NUM_READS, 1, READ_SIZE>::width()
    }
}

impl<AB: InteractionBuilder, const NUM_READS: usize, const READ_SIZE: usize> VmAdapterAir<AB>
    for Rv32HeapBranchAdapterAir<NUM_READS, READ_SIZE>
{
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, NUM_READS, 0, READ_SIZE, 0>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let vec_heap_air: Rv32VecHeapBranchAdapterAir<NUM_READS, 1, READ_SIZE> =
            Rv32VecHeapBranchAdapterAir::new(
                self.execution_bridge,
                self.memory_bridge,
                self.bus,
                self.address_bits,
            );
        vec_heap_air.eval(builder, local, ctx.into());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32VecHeapBranchAdapterCols<_, NUM_READS, 1, READ_SIZE> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, Copy)]
pub struct Rv32HeapBranchAdapterExecutor<const NUM_READS: usize, const READ_SIZE: usize>(
    Rv32VecHeapBranchAdapterExecutor<NUM_READS, 1, READ_SIZE>,
);

impl<const NUM_READS: usize, const READ_SIZE: usize>
    Rv32HeapBranchAdapterExecutor<NUM_READS, READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        assert!(NUM_READS <= 2);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
        );
        Rv32HeapBranchAdapterExecutor(Rv32VecHeapBranchAdapterExecutor::new(pointer_max_bits))
    }
}

pub struct Rv32HeapBranchAdapterFiller<const NUM_READS: usize, const READ_SIZE: usize>(
    Rv32VecHeapBranchAdapterFiller<NUM_READS, 1, READ_SIZE>,
);

impl<const NUM_READS: usize, const READ_SIZE: usize>
    Rv32HeapBranchAdapterFiller<NUM_READS, READ_SIZE>
{
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
        );
        Rv32HeapBranchAdapterFiller(Rv32VecHeapBranchAdapterFiller::new(
            pointer_max_bits,
            bitwise_lookup_chip,
        ))
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize> AdapterTraceExecutor<F>
    for Rv32HeapBranchAdapterExecutor<NUM_READS, READ_SIZE>
{
    const WIDTH: usize = Rv32VecHeapBranchAdapterCols::<F, NUM_READS, 1, READ_SIZE>::width();
    type ReadData = [[u8; READ_SIZE]; NUM_READS];
    type WriteData = ();
    type RecordMut<'a> = &'a mut Rv32VecHeapBranchAdapterRecord<NUM_READS, 1, READ_SIZE>;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let read_data = AdapterTraceExecutor::<F>::read(&self.0, memory, instruction, record);
        // Flatten from [[[u8; READ_SIZE]; 1]; NUM_READS] to [[u8; READ_SIZE]; NUM_READS]
        read_data.map(|r| r[0])
    }

    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // Branch adapters don't write anything
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize> AdapterTraceFiller<F>
    for Rv32HeapBranchAdapterFiller<NUM_READS, READ_SIZE>
{
    const WIDTH: usize = Rv32VecHeapBranchAdapterCols::<F, NUM_READS, 1, READ_SIZE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]) {
        AdapterTraceFiller::<F>::fill_trace_row(&self.0, mem_helper, adapter_row);
    }
}
