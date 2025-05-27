use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterExecutorE1, AdapterTraceFiller, AdapterTraceStep,
        BasicAdapterInterface, ExecutionBridge, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::MemoryBridge,
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
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
    Rv32VecHeapAdapterAir, Rv32VecHeapAdapterCols, Rv32VecHeapAdapterRecord, Rv32VecHeapAdapterStep,
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads are from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes are to the address in `rd`.

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HeapAdapterAir<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<F: Field, const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize> BaseAir<F>
    for Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        Rv32VecHeapAdapterCols::<F, NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB> for Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        READ_SIZE,
        WRITE_SIZE,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let vec_heap_air: Rv32VecHeapAdapterAir<NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE> =
            Rv32VecHeapAdapterAir::new(
                self.execution_bridge,
                self.memory_bridge,
                self.bus,
                self.address_bits,
            );
        vec_heap_air.eval(builder, local, ctx.into());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32VecHeapAdapterCols<_, NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

pub struct Rv32HeapAdapterStep<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(Rv32VecHeapAdapterStep<NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>);

impl<const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize>
    Rv32HeapAdapterStep<NUM_READS, READ_SIZE, WRITE_SIZE>
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
        Rv32HeapAdapterStep(Rv32VecHeapAdapterStep::new(
            pointer_max_bits,
            bitwise_lookup_chip,
        ))
    }
}

impl<
        F: PrimeField32,
        CTX,
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceStep<F, CTX> for Rv32HeapAdapterStep<NUM_READS, READ_SIZE, WRITE_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize =
        Rv32VecHeapAdapterCols::<F, NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>::width();
    type ReadData = [[u8; READ_SIZE]; NUM_READS];
    type WriteData = [[u8; WRITE_SIZE]; 1];
    type RecordMut<'a> = &'a mut Rv32VecHeapAdapterRecord<NUM_READS, 1, 1, READ_SIZE, WRITE_SIZE>;

    fn start(pc: u32, memory: &TracingMemory<F>, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc.into();
        record.from_timestamp = memory.timestamp.into();
    }

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let read_data = AdapterTraceStep::<F, CTX>::read(&self.0, memory, instruction, record);
        read_data.map(|r| r[0])
    }

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        AdapterTraceStep::<F, CTX>::write(&self.0, memory, instruction, data, record);
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize>
    AdapterTraceFiller<F> for Rv32HeapAdapterStep<NUM_READS, READ_SIZE, WRITE_SIZE>
{
    const WIDTH: usize = <Self as AdapterTraceStep<F, ()>>::WIDTH;

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]) {
        AdapterTraceFiller::<F>::fill_trace_row(&self.0, mem_helper, adapter_row);
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize>
    AdapterExecutorE1<F> for Rv32HeapAdapterStep<NUM_READS, READ_SIZE, WRITE_SIZE>
{
    type ReadData = [[u8; READ_SIZE]; NUM_READS];
    type WriteData = [[u8; WRITE_SIZE]; 1];

    #[inline(always)]
    fn read(&self, memory: &mut GuestMemory, instruction: &Instruction<F>) -> Self::ReadData {
        let read_data = AdapterExecutorE1::<F>::read(&self.0, memory, instruction);
        read_data.map(|r| r[0])
    }

    #[inline(always)]
    fn write(
        &self,
        memory: &mut GuestMemory,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) {
        AdapterExecutorE1::<F>::write(&self.0, memory, instruction, data);
    }
}
