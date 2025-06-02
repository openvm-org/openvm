use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::zip,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        execution_mode::E1E2ExecutionCtx, AdapterAirContext, AdapterExecutorE1, AdapterTraceStep,
        ExecutionBridge, ExecutionState, VecHeapTwoReadsAdapterInterface, VmAdapterAir, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        online::{GuestMemory, TracingMemory},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::adapters::{
    abstract_compose, memory_read_from_state, memory_write_from_state,
    new_read_rv32_register_from_state, tracing_read, tracing_write, RV32_CELL_BITS,
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
    type TraceContext<'a> = ();

    fn start(pc: u32, memory: &TracingMemory<F>, adapter_row: &mut [F]) {
        let adapter_cols: &mut Rv32VecHeapTwoReadsAdapterCols<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();
        adapter_cols.from_state.pc = F::from_canonical_u32(pc);
        adapter_cols.from_state.timestamp = F::from_canonical_u32(memory.timestamp);
    }

    fn read(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
    ) -> Self::ReadData {
        let Instruction { a, b, c, d, e, .. } = *instruction;

        let e = e.as_canonical_u32();
        let d = d.as_canonical_u32();
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        let cols: &mut Rv32VecHeapTwoReadsAdapterCols<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        // Read register values
        cols.rs1_ptr = b;
        let rs1_val = tracing_read(memory, d, b.as_canonical_u32(), &mut cols.rs1_read_aux);
        cols.rs1_val = rs1_val.map(F::from_canonical_u8);
        let rs1_val = u32::from_le_bytes(rs1_val);
        cols.rs2_ptr = c;
        let rs2_val = tracing_read(memory, d, c.as_canonical_u32(), &mut cols.rs2_read_aux);
        cols.rs2_val = rs2_val.map(F::from_canonical_u8);
        let rs2_val = u32::from_le_bytes(rs2_val);

        cols.rd_ptr = a;
        let rd_val = tracing_read(memory, d, a.as_canonical_u32(), &mut cols.rd_read_aux);
        cols.rd_val = rd_val.map(F::from_canonical_u8);
        assert!(rs1_val as usize + READ_SIZE * BLOCKS_PER_READ1 - 1 < (1 << self.pointer_max_bits));
        assert!(rs2_val as usize + READ_SIZE * BLOCKS_PER_READ2 - 1 < (1 << self.pointer_max_bits));

        (
            from_fn(|i| {
                tracing_read(
                    memory,
                    e,
                    rs1_val + (i * READ_SIZE) as u32,
                    &mut cols.reads1_aux[i],
                )
            }),
            from_fn(|i| {
                tracing_read(
                    memory,
                    e,
                    rs2_val + (i * READ_SIZE) as u32,
                    &mut cols.reads2_aux[i],
                )
            }),
        )
    }

    fn write(
        &self,
        memory: &mut TracingMemory<F>,
        instruction: &Instruction<F>,
        adapter_row: &mut [F],
        data: &Self::WriteData,
    ) {
        let e = instruction.e.as_canonical_u32();
        let cols: &mut Rv32VecHeapTwoReadsAdapterCols<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        let rd_val = u32::from_le_bytes(cols.rd_val.map(|x| x.as_canonical_u32() as u8));
        assert!(rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1 < (1 << self.pointer_max_bits));

        for (i, block) in data.iter().enumerate() {
            tracing_write(
                memory,
                e,
                rd_val + (i * WRITE_SIZE) as u32,
                block,
                &mut cols.writes_aux[i],
            );
        }
    }

    fn fill_trace_row(
        &self,
        mem_helper: &openvm_circuit::system::memory::MemoryAuxColsFactory<F>,
        _ctx: (),
        adapter_row: &mut [F],
    ) {
        let cols: &mut Rv32VecHeapTwoReadsAdapterCols<
            F,
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        let mut timestamp = cols.from_state.timestamp.as_canonical_u32();
        let mut timestamp_pp = || {
            timestamp += 1;
            timestamp - 1
        };

        mem_helper.fill_from_prev(timestamp_pp(), cols.rs1_read_aux.as_mut());
        mem_helper.fill_from_prev(timestamp_pp(), cols.rs2_read_aux.as_mut());
        mem_helper.fill_from_prev(timestamp_pp(), cols.rd_read_aux.as_mut());
        cols.reads1_aux.iter_mut().for_each(|aux| {
            mem_helper.fill_from_prev(timestamp_pp(), aux.as_mut());
        });
        cols.reads2_aux.iter_mut().for_each(|aux| {
            mem_helper.fill_from_prev(timestamp_pp(), aux.as_mut());
        });
        cols.writes_aux.iter_mut().for_each(|aux| {
            mem_helper.fill_from_prev(timestamp_pp(), aux.as_mut());
        });

        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);

        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        self.bitwise_lookup_chip.request_range(
            cols.rs1_val[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits,
            cols.rs2_val[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits,
        );
        self.bitwise_lookup_chip.request_range(
            cols.rd_val[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits,
            cols.rd_val[RV32_REGISTER_NUM_LIMBS - 1].as_canonical_u32() << limb_shift_bits,
        );
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
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Self::ReadData
    where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { b, c, d, e, .. } = *instruction;

        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        debug_assert_eq!(d, RV32_REGISTER_AS);
        debug_assert_eq!(e, RV32_MEMORY_AS);

        // Read register values
        let rs1_val = new_read_rv32_register_from_state(state, d, b.as_canonical_u32());
        let rs2_val = new_read_rv32_register_from_state(state, d, c.as_canonical_u32());

        assert!(rs1_val as usize + READ_SIZE * BLOCKS_PER_READ1 - 1 < (1 << self.pointer_max_bits));
        assert!(rs2_val as usize + READ_SIZE * BLOCKS_PER_READ2 - 1 < (1 << self.pointer_max_bits));
        // Read memory values
        let read_data1 =
            from_fn(|i| memory_read_from_state(state, e, rs1_val + (i * READ_SIZE) as u32));
        let read_data2 =
            from_fn(|i| memory_read_from_state(state, e, rs2_val + (i * READ_SIZE) as u32));

        (read_data1, read_data2)
    }

    fn write<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
        data: &Self::WriteData,
    ) where
        Ctx: E1E2ExecutionCtx,
    {
        let Instruction { a, d, e, .. } = *instruction;

        let rd_val =
            new_read_rv32_register_from_state(state, d.as_canonical_u32(), a.as_canonical_u32());
        assert!(rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1 < (1 << self.pointer_max_bits));

        for (i, block) in data.iter().enumerate() {
            memory_write_from_state(
                state,
                e.as_canonical_u32(),
                rd_val + (i * WRITE_SIZE) as u32,
                block,
            );
        }
    }
}
