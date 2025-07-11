use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::zip,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, get_record_from_slice, AdapterAirContext,
        AdapterExecutorE1, AdapterTraceFiller, AdapterTraceStep, ExecutionBridge, ExecutionState,
        VecHeapTwoReadsAdapterInterface, VmAdapterAir, VmStateMut,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::adapters::{
    abstract_compose, memory_read_from_state, memory_write_from_state,
    read_rv32_register_from_state, tracing_read, tracing_write, RV32_CELL_BITS,
    RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// This adapter reads from 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READX` consecutive reads of size `READ_SIZE` from the heap,
///   starting from the addresses in `rs[X]`
/// * NOTE that the two reads can read different numbers of blocks.
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive writes of size `WRITE_SIZE` to the
///   heap, starting from the address in `rd`.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32VecHeapTwoReadsAdapterCols<
    T,
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub rd_ptr: T,

    pub rs1_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs2_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs1_read_aux: MemoryReadAuxCols<T>,
    pub rs2_read_aux: MemoryReadAuxCols<T>,
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads1_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_READ1],
    pub reads2_aux: [MemoryReadAuxCols<T>; BLOCKS_PER_READ2],
    pub writes_aux: [MemoryWriteAuxCols<T, WRITE_SIZE>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32VecHeapTwoReadsAdapterAir<
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F>
    for Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    fn width(&self) -> usize {
        Rv32VecHeapTwoReadsAdapterCols::<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >::width()
    }
}

impl<
        AB: InteractionBuilder,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv32VecHeapTwoReadsAdapterAir<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Interface = VecHeapTwoReadsAdapterInterface<
        AB::Expr,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32VecHeapTwoReadsAdapterCols<
            _,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let ptrs = [cols.rs1_ptr, cols.rs2_ptr, cols.rd_ptr];
        let vals = [cols.rs1_val, cols.rs2_val, cols.rd_val];
        let auxs = [&cols.rs1_read_aux, &cols.rs2_read_aux, &cols.rd_read_aux];

        // Read register values for rs1, rs2, rd
        for (ptr, val, aux) in izip!(ptrs, vals, auxs) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), ptr),
                    val,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Range checks: see Rv32VecHeaperAdapterAir
        let need_range_check = [&cols.rs1_val, &cols.rs2_val, &cols.rd_val, &cols.rd_val]
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1]);

        // range checks constrain to RV32_CELL_BITS bits, so we need to shift the limbs to constrain
        // the correct amount of bits
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        // Note: since limbs are read from memory we already know that limb[i] < 2^RV32_CELL_BITS
        //       thus range checking limb[i] * shift < 2^RV32_CELL_BITS, gives us that
        //       limb[i] < 2^(addr_bits - (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs1_val_f: AB::Expr = abstract_compose(cols.rs1_val);
        let rs2_val_f: AB::Expr = abstract_compose(cols.rs2_val);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);
        // Reads from heap
        for (i, (read, aux)) in zip(ctx.reads.0, &cols.reads1_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        rs1_val_f.clone() + AB::Expr::from_canonical_usize(i * READ_SIZE),
                    ),
                    read,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }
        for (i, (read, aux)) in zip(ctx.reads.1, &cols.reads2_aux).enumerate() {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e,
                        rs2_val_f.clone() + AB::Expr::from_canonical_usize(i * READ_SIZE),
                    ),
                    read,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        rd_val_f.clone() + AB::Expr::from_canonical_usize(i * WRITE_SIZE),
                    ),
                    write,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rd_ptr.into(),
                    cols.rs1_ptr.into(),
                    cols.rs2_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32VecHeapTwoReadsAdapterCols<
            _,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32VecHeapTwoReadsAdapterRecord<
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub rd_ptr: u32,

    pub rs1_val: u32,
    pub rs2_val: u32,
    pub rd_val: u32,

    pub rs1_read_aux: MemoryReadAuxRecord,
    pub rs2_read_aux: MemoryReadAuxRecord,
    pub rd_read_aux: MemoryReadAuxRecord,

    pub reads1_aux: [MemoryReadAuxRecord; BLOCKS_PER_READ1],
    pub reads2_aux: [MemoryReadAuxRecord; BLOCKS_PER_READ2],
    pub writes_aux: [MemoryWriteBytesAuxRecord<WRITE_SIZE>; BLOCKS_PER_WRITE],
}

pub struct Rv32VecHeapTwoReadsAdapterStep<
    const BLOCKS_PER_READ1: usize,
    const BLOCKS_PER_READ2: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    >
    Rv32VecHeapTwoReadsAdapterStep<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
        );
        Self {
            pointer_max_bits,
            bitwise_lookup_chip,
        }
    }
}

impl<
        F: PrimeField32,
        CTX,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceStep<F, CTX>
    for Rv32VecHeapTwoReadsAdapterStep<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    const WIDTH: usize = Rv32VecHeapTwoReadsAdapterCols::<
        F,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >::width();

    type ReadData = (
        [[u8; READ_SIZE]; BLOCKS_PER_READ1],
        [[u8; READ_SIZE]; BLOCKS_PER_READ2],
    );
    type WriteData = [[u8; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type RecordMut<'a> = &'a mut Rv32VecHeapTwoReadsAdapterRecord<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        WRITE_SIZE,
    >;

    fn start(pc: u32, memory: &TracingMemory<F>, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let Instruction { a, b, c, d, e, .. } = *instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        record.rs1_ptr = b.as_canonical_u32();
        record.rs1_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.rs1_ptr,
            &mut record.rs1_read_aux.prev_timestamp,
        ));
        record.rs2_ptr = c.as_canonical_u32();
        record.rs2_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.rs2_ptr,
            &mut record.rs2_read_aux.prev_timestamp,
        ));

        record.rd_ptr = a.as_canonical_u32();
        record.rd_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            record.rd_ptr,
            &mut record.rd_read_aux.prev_timestamp,
        ));
        assert!(
            record.rs1_val as usize + READ_SIZE * BLOCKS_PER_READ1 - 1
                < (1 << self.pointer_max_bits)
        );
        assert!(
            record.rs2_val as usize + READ_SIZE * BLOCKS_PER_READ2 - 1
                < (1 << self.pointer_max_bits)
        );

        (
            from_fn(|i| {
                tracing_read(
                    memory,
                    RV32_MEMORY_AS,
                    record.rs1_val + (i * READ_SIZE) as u32,
                    &mut record.reads1_aux[i].prev_timestamp,
                )
            }),
            from_fn(|i| {
                tracing_read(
                    memory,
                    RV32_MEMORY_AS,
                    record.rs2_val + (i * READ_SIZE) as u32,
                    &mut record.reads2_aux[i].prev_timestamp,
                )
            }),
        )
    }

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        _instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        assert!(
            record.rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1
                < (1 << self.pointer_max_bits)
        );

        for (i, block) in data.into_iter().enumerate() {
            tracing_write(
                memory,
                RV32_MEMORY_AS,
                record.rd_val + (i * WRITE_SIZE) as u32,
                block,
                &mut record.writes_aux[i].prev_timestamp,
                &mut record.writes_aux[i].prev_data,
            );
        }
    }
}

impl<
        F: PrimeField32,
        CTX,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceFiller<F, CTX>
    for Rv32VecHeapTwoReadsAdapterStep<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32VecHeapTwoReadsAdapterRecord<
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            WRITE_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv32VecHeapTwoReadsAdapterCols<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);

        const MSL_SHIFT: usize = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        self.bitwise_lookup_chip.request_range(
            (record.rs1_val >> MSL_SHIFT) << limb_shift_bits,
            (record.rs2_val >> MSL_SHIFT) << limb_shift_bits,
        );
        self.bitwise_lookup_chip.request_range(
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
        );

        let mut timestamp = record.from_timestamp
            + 2
            + (BLOCKS_PER_READ1 + BLOCKS_PER_READ2 + BLOCKS_PER_WRITE) as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // Writing everything in reverse order
        cols.writes_aux
            .iter_mut()
            .rev()
            .zip(record.writes_aux.iter().rev())
            .for_each(|(col, record)| {
                col.set_prev_data(record.prev_data.map(F::from_canonical_u8));
                mem_helper.fill(record.prev_timestamp, timestamp_mm(), col.as_mut());
            });

        cols.reads2_aux
            .iter_mut()
            .rev()
            .zip(record.reads2_aux.iter().rev())
            .for_each(|(col, record)| {
                mem_helper.fill(record.prev_timestamp, timestamp_mm(), col.as_mut());
            });

        cols.reads1_aux
            .iter_mut()
            .rev()
            .zip(record.reads1_aux.iter().rev())
            .for_each(|(col, record)| {
                mem_helper.fill(record.prev_timestamp, timestamp_mm(), col.as_mut());
            });

        mem_helper.fill(
            record.rd_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rd_read_aux.as_mut(),
        );
        mem_helper.fill(
            record.rs2_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rs2_read_aux.as_mut(),
        );
        mem_helper.fill(
            record.rs1_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rs1_read_aux.as_mut(),
        );

        cols.rd_val = record.rd_val.to_le_bytes().map(F::from_canonical_u8);
        cols.rs2_val = record.rs2_val.to_le_bytes().map(F::from_canonical_u8);
        cols.rs1_val = record.rs1_val.to_le_bytes().map(F::from_canonical_u8);
        cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        cols.rs2_ptr = F::from_canonical_u32(record.rs2_ptr);
        cols.rs1_ptr = F::from_canonical_u32(record.rs1_ptr);

        cols.from_state.timestamp = F::from_canonical_u32(timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}

impl<
        F: PrimeField32,
        const BLOCKS_PER_READ1: usize,
        const BLOCKS_PER_READ2: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterExecutorE1<F>
    for Rv32VecHeapTwoReadsAdapterStep<
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type ReadData = (
        [[u8; READ_SIZE]; BLOCKS_PER_READ1],
        [[u8; READ_SIZE]; BLOCKS_PER_READ2],
    );
    type WriteData = [[u8; WRITE_SIZE]; BLOCKS_PER_WRITE];

    fn read<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        let rs1_val = read_rv32_register_from_state(state, b.as_canonical_u32());
        let rs2_val = read_rv32_register_from_state(state, c.as_canonical_u32());

        assert!(rs1_val as usize + READ_SIZE * BLOCKS_PER_READ1 - 1 < (1 << self.pointer_max_bits));
        assert!(rs2_val as usize + READ_SIZE * BLOCKS_PER_READ2 - 1 < (1 << self.pointer_max_bits));
        // Read memory values
        let read_data1 = from_fn(|i| {
            memory_read_from_state(
                state,
                e.as_canonical_u32(),
                rs1_val + (i * READ_SIZE) as u32,
            )
        });
        let read_data2 = from_fn(|i| {
            memory_read_from_state(
                state,
                e.as_canonical_u32(),
                rs2_val + (i * READ_SIZE) as u32,
            )
        });

        (read_data1, read_data2)
    }

    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<F, GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { a, .. } = *instruction;

        let rd_val = read_rv32_register_from_state(state, a.as_canonical_u32());
        assert!(rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1 < (1 << self.pointer_max_bits));

        for (i, block) in data.into_iter().enumerate() {
            memory_write_from_state(
                state,
                RV32_MEMORY_AS,
                rd_val + (i * WRITE_SIZE) as u32,
                block,
            );
        }
    }
}
