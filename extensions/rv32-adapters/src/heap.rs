use std::{
    array::{self, from_fn},
    borrow::Borrow,
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, MinimalInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{offline_checker::MemoryBridge, MemoryController, OfflineMemory},
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::adapters::read_rv32_register;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
};

use super::{
    vec_heap_generate_trace_row_impl, Rv32VecHeapAdapterAir, Rv32VecHeapAdapterCols,
    Rv32VecHeapReadRecord, Rv32VecHeapWriteRecord,
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

pub struct Rv32HeapAdapterChip<
    F: Field,
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub air: Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize>
    Rv32HeapAdapterChip<F, NUM_READS, READ_SIZE, WRITE_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        address_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - address_bits < RV32_CELL_BITS,
            "address_bits={address_bits} needs to be large enough for high limb range check"
        );
        Self {
            air: Rv32HeapAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                bus: bitwise_lookup_chip.bus(),
                address_bits,
            },
            bitwise_lookup_chip,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const READ_SIZE: usize, const WRITE_SIZE: usize>
    VmAdapterChip<F> for Rv32HeapAdapterChip<F, NUM_READS, READ_SIZE, WRITE_SIZE>
{
    type ReadRecord = Rv32VecHeapReadRecord<F, NUM_READS, 1, READ_SIZE>;
    type WriteRecord = Rv32VecHeapWriteRecord<1, WRITE_SIZE>;
    type Air = Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE>;
    type Interface =
        BasicAdapterInterface<F, MinimalInstruction<F>, NUM_READS, 1, READ_SIZE, WRITE_SIZE>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let mut rs_vals = [0; NUM_READS];
        let rs_records: [_; NUM_READS] = from_fn(|i| {
            let addr = if i == 0 { b } else { c };
            let (record, val) = read_rv32_register(memory, d, addr);
            rs_vals[i] = val;
            record
        });
        let (rd_record, rd_val) = read_rv32_register(memory, d, a);

        let read_records = rs_vals.map(|address| {
            debug_assert!(address as usize + READ_SIZE - 1 < (1 << self.air.address_bits));
            [memory.read::<READ_SIZE>(e, F::from_canonical_u32(address))]
        });
        let read_data = read_records.map(|r| r[0].1);

        let record = Rv32VecHeapReadRecord {
            rs: rs_records,
            rd: rd_record,
            rd_val: F::from_canonical_u32(rd_val),
            reads: read_records.map(|r| array::from_fn(|i| r[i].0)),
        };

        Ok((read_data, record))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        let e = instruction.e;
        let writes = [memory.write(e, read_record.rd_val, output.writes[0]).0];

        let timestamp_delta = memory.timestamp() - from_state.timestamp;
        debug_assert!(
            timestamp_delta == 6,
            "timestamp delta is {}, expected 6",
            timestamp_delta
        );

        Ok((
            ExecutionState {
                pc: from_state.pc + DEFAULT_PC_STEP,
                timestamp: memory.timestamp(),
            },
            Self::WriteRecord { from_state, writes },
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    ) {
        vec_heap_generate_trace_row_impl(
            row_slice,
            &read_record,
            &write_record,
            self.bitwise_lookup_chip.clone(),
            self.air.address_bits,
            memory,
        );
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
