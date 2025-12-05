use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::once,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
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
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::adapters::{abstract_compose, tracing_read, tracing_write};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// Fixed memory block size for all heap adapter memory bus interactions.
/// All reads and writes are sent in 4-byte chunks to avoid access adapters.
pub const HEAP_ADAPTER_BLOCK_SIZE: usize = 4;

/// This adapter reads from NUM_READS <= 2 pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads are from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes are to the address in `rd`.
///
/// Memory bus interactions are sent in 4-byte blocks to avoid needing access adapters.
/// READ_SIZE and WRITE_SIZE must be multiples of 4.
/// READ_BLOCKS must equal READ_SIZE / 4, WRITE_BLOCKS must equal WRITE_SIZE / 4.
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32HeapAdapterCols<
    T,
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    /// Aux columns for heap reads: READ_BLOCKS aux cols per read
    pub reads_aux: [[MemoryReadAuxCols<T>; READ_BLOCKS]; NUM_READS],
    /// Aux columns for heap writes: WRITE_BLOCKS aux cols
    pub writes_aux: [MemoryWriteAuxCols<T, HEAP_ADAPTER_BLOCK_SIZE>; WRITE_BLOCKS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HeapAdapterAir<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_BLOCKS: usize = { 8 },   // Default for 32-byte reads
    const WRITE_BLOCKS: usize = { 8 },  // Default for 32-byte writes
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    /// The max number of bits for an address in memory
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > BaseAir<F>
    for Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
{
    fn width(&self) -> usize {
        Rv32HeapAdapterCols::<F, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > VmAdapterAir<AB>
    for Rv32HeapAdapterAir<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
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
        debug_assert_eq!(READ_BLOCKS, READ_SIZE / HEAP_ADAPTER_BLOCK_SIZE);
        debug_assert_eq!(WRITE_BLOCKS, WRITE_SIZE / HEAP_ADAPTER_BLOCK_SIZE);

        let cols: &Rv32HeapAdapterCols<_, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), ptr),
                    val,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Range check the highest limbs of heap pointers
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .chain(std::iter::repeat_n(&cols.rd_val, 2))
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1])
            .collect();

        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register values into field elements
        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(abstract_compose);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        // Reads from heap - send READ_BLOCKS reads of 4 bytes each
        for (read_idx, (address, reads_aux)) in izip!(rs_val_f.clone(), &cols.reads_aux).enumerate()
        {
            let read_data = &ctx.reads[read_idx];
            for (block_idx, aux) in reads_aux.iter().enumerate() {
                let block_start = block_idx * HEAP_ADAPTER_BLOCK_SIZE;
                let block_data: [AB::Expr; HEAP_ADAPTER_BLOCK_SIZE] =
                    from_fn(|i| read_data[block_start + i].clone());
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
                            address.clone() + AB::Expr::from_canonical_usize(block_start),
                        ),
                        block_data,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap - send WRITE_BLOCKS writes of 4 bytes each
        let write_data = &ctx.writes[0];
        for (block_idx, aux) in cols.writes_aux.iter().enumerate() {
            let block_start = block_idx * HEAP_ADAPTER_BLOCK_SIZE;
            let block_data: [AB::Expr; HEAP_ADAPTER_BLOCK_SIZE] =
                from_fn(|i| write_data[block_start + i].clone());
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        rd_val_f.clone() + AB::Expr::from_canonical_usize(block_start),
                    ),
                    block_data,
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
                    cols.rs_ptr
                        .first()
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
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
        let cols: &Rv32HeapAdapterCols<_, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS> =
            local.borrow();
        cols.from_state.pc
    }
}

/// Record for heap adapter trace generation.
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32HeapAdapterRecord<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_BLOCKS: usize,
    const WRITE_BLOCKS: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptrs: [u32; NUM_READS],
    pub rd_ptr: u32,

    pub rs_vals: [u32; NUM_READS],
    pub rd_val: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub rd_read_aux: MemoryReadAuxRecord,

    pub reads_aux: [[MemoryReadAuxRecord; READ_BLOCKS]; NUM_READS],
    pub writes_aux: [MemoryWriteBytesAuxRecord<HEAP_ADAPTER_BLOCK_SIZE>; WRITE_BLOCKS],
}

#[derive(Clone, Copy)]
pub struct Rv32HeapAdapterExecutor<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_BLOCKS: usize = { 8 },
    const WRITE_BLOCKS: usize = { 8 },
> {
    pointer_max_bits: usize,
}

impl<
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > Rv32HeapAdapterExecutor<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(READ_BLOCKS, READ_SIZE / HEAP_ADAPTER_BLOCK_SIZE);
        assert_eq!(WRITE_BLOCKS, WRITE_SIZE / HEAP_ADAPTER_BLOCK_SIZE);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
        );
        Self { pointer_max_bits }
    }
}

pub struct Rv32HeapAdapterFiller<
    const NUM_READS: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_BLOCKS: usize = { 8 },
    const WRITE_BLOCKS: usize = { 8 },
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > Rv32HeapAdapterFiller<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
{
    pub fn new(
        pointer_max_bits: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(READ_BLOCKS, READ_SIZE / HEAP_ADAPTER_BLOCK_SIZE);
        assert_eq!(WRITE_BLOCKS, WRITE_SIZE / HEAP_ADAPTER_BLOCK_SIZE);
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
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > AdapterTraceExecutor<F>
    for Rv32HeapAdapterExecutor<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
{
    const WIDTH: usize =
        Rv32HeapAdapterCols::<F, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>::width();
    type ReadData = [[u8; READ_SIZE]; NUM_READS];
    type WriteData = [[u8; WRITE_SIZE]; 1];
    type RecordMut<'a> =
        &'a mut Rv32HeapAdapterRecord<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>;

    #[inline(always)]
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
        let &Instruction { a, b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        record.rs_vals = from_fn(|i| {
            record.rs_ptrs[i] = if i == 0 { b } else { c }.as_canonical_u32();
            u32::from_le_bytes(tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs_ptrs[i],
                &mut record.rs_read_aux[i].prev_timestamp,
            ))
        });

        record.rd_ptr = a.as_canonical_u32();
        record.rd_val = u32::from_le_bytes(tracing_read(
            memory,
            RV32_REGISTER_AS,
            a.as_canonical_u32(),
            &mut record.rd_read_aux.prev_timestamp,
        ));

        // Read memory values in 4-byte blocks
        from_fn(|read_idx| {
            debug_assert!(
                (record.rs_vals[read_idx] as usize + READ_SIZE - 1) < (1 << self.pointer_max_bits)
            );
            let mut result = [0u8; READ_SIZE];
            for block_idx in 0..READ_BLOCKS {
                let block_start = block_idx * HEAP_ADAPTER_BLOCK_SIZE;
                let block: [u8; HEAP_ADAPTER_BLOCK_SIZE] = tracing_read(
                    memory,
                    RV32_MEMORY_AS,
                    record.rs_vals[read_idx] + block_start as u32,
                    &mut record.reads_aux[read_idx][block_idx].prev_timestamp,
                );
                result[block_start..block_start + HEAP_ADAPTER_BLOCK_SIZE].copy_from_slice(&block);
            }
            result
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV32_MEMORY_AS);
        debug_assert!(record.rd_val as usize + WRITE_SIZE - 1 < (1 << self.pointer_max_bits));

        // Write in 4-byte blocks
        for block_idx in 0..WRITE_BLOCKS {
            let block_start = block_idx * HEAP_ADAPTER_BLOCK_SIZE;
            let block: [u8; HEAP_ADAPTER_BLOCK_SIZE] =
                data[0][block_start..block_start + HEAP_ADAPTER_BLOCK_SIZE]
                    .try_into()
                    .unwrap();
            tracing_write(
                memory,
                RV32_MEMORY_AS,
                record.rd_val + block_start as u32,
                block,
                &mut record.writes_aux[block_idx].prev_timestamp,
                &mut record.writes_aux[block_idx].prev_data,
            );
        }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_BLOCKS: usize,
        const WRITE_BLOCKS: usize,
    > AdapterTraceFiller<F>
    for Rv32HeapAdapterFiller<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>
{
    const WIDTH: usize =
        Rv32HeapAdapterCols::<F, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv32HeapAdapterRecord<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv32HeapAdapterCols<F, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS> =
            adapter_row.borrow_mut();

        // Range checks
        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: usize = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

        if NUM_READS > 1 {
            self.bitwise_lookup_chip.request_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits,
            );
            self.bitwise_lookup_chip.request_range(
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
            );
        } else {
            self.bitwise_lookup_chip.request_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
            );
        }

        let timestamp_delta = NUM_READS + 1 + NUM_READS * READ_BLOCKS + WRITE_BLOCKS;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // Fill in reverse order to avoid overwriting records
        for (write_aux, cols_write) in record
            .writes_aux
            .iter()
            .rev()
            .zip(cols.writes_aux.iter_mut().rev())
        {
            cols_write.set_prev_data(write_aux.prev_data.map(F::from_canonical_u8));
            mem_helper.fill(
                write_aux.prev_timestamp,
                timestamp_mm(),
                cols_write.as_mut(),
            );
        }

        for (reads, cols_reads) in record.reads_aux.iter().zip(cols.reads_aux.iter_mut()).rev() {
            for (read, cols_read) in reads.iter().zip(cols_reads.iter_mut()).rev() {
                mem_helper.fill(read.prev_timestamp, timestamp_mm(), cols_read.as_mut());
            }
        }

        mem_helper.fill(
            record.rd_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rd_read_aux.as_mut(),
        );

        for (aux, cols_aux) in record
            .rs_read_aux
            .iter()
            .zip(cols.rs_read_aux.iter_mut())
            .rev()
        {
            mem_helper.fill(aux.prev_timestamp, timestamp_mm(), cols_aux.as_mut());
        }

        cols.rd_val = record.rd_val.to_le_bytes().map(F::from_canonical_u8);
        for (cols_val, val) in cols
            .rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
        {
            *cols_val = val.to_le_bytes().map(F::from_canonical_u8);
        }

        cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        for (cols_ptr, ptr) in cols
            .rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptrs.iter().rev())
        {
            *cols_ptr = F::from_canonical_u32(*ptr);
        }

        cols.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
