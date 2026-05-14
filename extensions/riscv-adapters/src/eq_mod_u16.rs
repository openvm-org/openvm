//! Pattern B u16-shaped variant of [`Rv64IsEqualModAdapter`].
//!
//! Differences vs the u8 variant:
//! - Register-side `rs_val` columns are `[T; BLOCK_FE_WIDTH]` (4 u16 cells) instead of `[T;
//!   RV64_WORD_NUM_LIMBS]` (8 bytes). The register read passes the u16 cells directly to the memory
//!   bus via [`expand_to_rv64_block`].
//! - Heap reads are u16-cell-shaped: `BLOCK_SIZE` counts u16 cells per block (= `BLOCK_FE_WIDTH`),
//!   and the per-block byte stride is `j * BLOCK_SIZE * BUS_PTR_SCALE`. The 4 u16 cells of each
//!   read block are passed through to `read_4` without `pack_u8_for_bus`.
//! - The write to `rd` is `BLOCK_FE_WIDTH` u16 cells wide (passed through directly).
//! - The base-address composition uses base 2^16 per cell rather than 2^8 per byte.
//! - The pointer high-cell range check uses the same byte-pair bitwise lookup with the shift `1 <<
//!   (16 * BLOCK_FE_WIDTH - address_bits)`; for typical `address_bits < 56` this constrains the top
//!   u16 cell to 0.

use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH, BUS_PTR_SCALE,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    expand_to_rv64_block, tracing_read_reg_ptr, tracing_read_u16, tracing_write_u16, RV64_CELL_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// Number of bytes per u16 memory cell on the bus.
const BUS_BYTES_PER_CELL: usize = BUS_PTR_SCALE as usize;

/// U16-shaped variant of [`Rv64IsEqualModAdapterCols`].
///
/// * `BLOCK_SIZE` counts u16 cells per heap-read block and must equal [`BLOCK_FE_WIDTH`].
/// * `rs_val` holds `BLOCK_FE_WIDTH` u16 cells per register pointer.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64IsEqualModAdapterU16Cols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; BLOCK_FE_WIDTH]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64IsEqualModAdapterU16Cols<u8, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>)]
pub struct Rv64IsEqualModAdapterU16Air<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F>
    for Rv64IsEqualModAdapterU16Air<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv64IsEqualModAdapterU16Cols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv64IsEqualModAdapterU16Air<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        BLOCK_FE_WIDTH,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv64IsEqualModAdapterU16Cols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_u32(RV64_REGISTER_AS);
        let e = AB::F::from_u32(RV64_MEMORY_AS);

        // Read register values for rs. The cell width is BLOCK_FE_WIDTH (4 u16 cells), passed
        // through to the bus directly via `expand_to_rv64_block` (zero-pads when N < target).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(d, ptr),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the BLOCK_FE_WIDTH u16 cells of each register value into a single field element
        // used as the heap base byte-address: cell j contributes `cells[j] * 2^(16*j)`.
        let rs_val_f: [AB::Expr; NUM_READS] = from_fn(|i| {
            let mut acc = AB::Expr::ZERO;
            for j in 0..BLOCK_FE_WIDTH {
                acc = acc + cols.rs_val[i][j] * AB::F::from_u64(1u64 << (16 * j));
            }
            acc
        });

        let need_range_check: [_; 2] = from_fn(|i| {
            if i < NUM_READS {
                cols.rs_val[i][BLOCK_FE_WIDTH - 1].into()
            } else {
                AB::Expr::ZERO
            }
        });

        // Top u16 cell occupies bits 48..64; the bus check enforces
        //   top_cell << (64 - address_bits) < 2^8.
        // For typical `address_bits < 56` this constrains the top u16 cell to 0; for larger
        // address_bits it allows `(address_bits - 56)` non-zero bits in the top cell.
        let limb_shift = AB::F::from_u64(1u64 << (16 * BLOCK_FE_WIDTH - self.address_bits));

        self.bus
            .send_range(
                need_range_check[0].clone() * limb_shift,
                need_range_check[1].clone() * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Reads from heap. `BLOCK_SIZE` counts u16 cells per block and must equal BLOCK_FE_WIDTH;
        // the per-block byte stride is `BLOCK_SIZE * BUS_PTR_SCALE`.
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        assert_eq!(
            BLOCK_SIZE, BLOCK_FE_WIDTH,
            "Rv64IsEqualModU16 adapter only supports BLOCK_SIZE = BLOCK_FE_WIDTH"
        );
        let read_block_data: [[[_; BLOCK_SIZE]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });
        let block_ptr_offset: [_; BLOCKS_PER_READ] =
            from_fn(|i| AB::F::from_usize(i * BLOCK_SIZE * BUS_BYTES_PER_CELL));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                let data_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| data[j].clone());
                self.memory_bridge
                    .read_4(
                        MemoryAddress::new(e, ptr.clone() + offset),
                        data_array,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register. `ctx.writes[0]` is already BLOCK_FE_WIDTH u16 cells; pass through.
        self.memory_bridge
            .write_4(
                MemoryAddress::new(d, cols.rd_ptr),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &cols.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

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
                    d.into(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64IsEqualModAdapterU16Cols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64IsEqualModAdapterU16Record<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub from_pc: u32,
    pub timestamp: u32,

    pub rs_ptr: [u32; NUM_READS],
    pub rs_val: [u32; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: u32,
    pub writes_aux: MemoryWriteAuxRecord<u16, BLOCK_FE_WIDTH>,
}

#[derive(Clone, Copy)]
pub struct Rv64IsEqualModAdapterU16Executor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64IsEqualModAdapterU16Filler<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
}

impl<
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > Rv64IsEqualModAdapterU16Executor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        assert_eq!(
            BLOCK_SIZE, BLOCK_FE_WIDTH,
            "Rv64IsEqualModU16 adapter only supports BLOCK_SIZE = BLOCK_FE_WIDTH"
        );
        assert!(
            16 * BLOCK_FE_WIDTH >= pointer_max_bits,
            "pointer_max_bits={pointer_max_bits} cannot exceed the register width (64 bits)"
        );
        Self { pointer_max_bits }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv64IsEqualModAdapterU16Executor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize =
        Rv64IsEqualModAdapterU16Cols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();
    type ReadData = [[u16; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u16; BLOCK_FE_WIDTH];
    type RecordMut<'a> = &'a mut Rv64IsEqualModAdapterU16Record<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCK_SIZE,
        TOTAL_READ_SIZE,
    >;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let Instruction { b, c, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        // Read register values (still byte-addressed as u32 pointers in the record).
        record.rs_val = from_fn(|i| {
            record.rs_ptr[i] = if i == 0 { b } else { c }.as_canonical_u32();
            tracing_read_reg_ptr(
                memory,
                record.rs_ptr[i],
                &mut record.rs_read_aux[i].prev_timestamp,
                self.pointer_max_bits,
            )
        });

        // Read memory values as u16 cells; the per-block byte stride is `BLOCK_SIZE * 2`.
        let bytes_per_block = BLOCK_SIZE * BUS_BYTES_PER_CELL;
        from_fn(|i| {
            debug_assert!(
                (record.rs_val[i] as u64) + ((bytes_per_block * BLOCKS_PER_READ - 1) as u64)
                    < (1u64 << self.pointer_max_bits)
            );
            from_fn::<_, BLOCKS_PER_READ, _>(|j| {
                tracing_read_u16::<BLOCK_SIZE>(
                    memory,
                    RV64_MEMORY_AS,
                    record.rs_val[i] + (j * bytes_per_block) as u32,
                    &mut record.heap_read_aux[i][j].prev_timestamp,
                )
            })
            .concat()
            .try_into()
            .unwrap()
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let Instruction { a, .. } = *instruction;
        record.rd_ptr = a.as_canonical_u32();
        tracing_write_u16(
            memory,
            RV64_REGISTER_AS,
            record.rd_ptr,
            data,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceFiller<F>
    for Rv64IsEqualModAdapterU16Filler<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    const WIDTH: usize =
        Rv64IsEqualModAdapterU16Cols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv64IsEqualModAdapterU16Record<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCK_SIZE,
            TOTAL_READ_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64IsEqualModAdapterU16Cols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();

        let mut timestamp = record.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) as u32 + 1;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };
        // Do range checks before writing anything: mirror the AIR side, which checks
        //   top_u16_cell << (16 * BLOCK_FE_WIDTH - address_bits) < 2^8
        // via the 8-bit bitwise lookup chip.
        debug_assert!(self.pointer_max_bits <= 16 * BLOCK_FE_WIDTH);
        let limb_shift_bits = 16 * BLOCK_FE_WIDTH - self.pointer_max_bits;
        const MSL_SHIFT: usize = 16 * (BLOCK_FE_WIDTH - 1);
        // Cast to `u64` end-to-end so that both `MSL_SHIFT >= 32` (right shift) and
        // `limb_shift_bits >= 32` (left shift for small `pointer_max_bits`) are well-defined.
        // The bitwise lookup chip only consumes the low byte; for valid pointers
        // (< 2^pointer_max_bits with pointer_max_bits <= 48) the high u16 cell is 0 and the
        // shift result is 0, but we still want this code to be well-defined for any
        // `pointer_max_bits` configuration the AIR allows.
        let rs_val_hi: [u32; 2] = from_fn(|i| {
            if i < NUM_READS {
                (((record.rs_val[i] as u64) >> MSL_SHIFT) << limb_shift_bits) as u32
            } else {
                0
            }
        });
        self.bitwise_lookup_chip
            .request_range(rs_val_hi[0], rs_val_hi[1]);
        // Writing in reverse order. `writes_aux.prev_data` is already u16-cell-shaped, so no
        // byte→u16 packing is needed (unlike the u8 adapter, which packs pairs of bytes).
        cols.writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(|v| F::from_u32(v as u32)));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp_mm(),
            cols.writes_aux.as_mut(),
        );
        cols.rd_ptr = F::from_u32(record.rd_ptr);

        // **NOTE**: Must iterate everything in reverse order to avoid overwriting the records
        cols.heap_read_aux
            .iter_mut()
            .rev()
            .zip(record.heap_read_aux.iter().rev())
            .for_each(|(col_reads, record_reads)| {
                col_reads
                    .iter_mut()
                    .rev()
                    .zip(record_reads.iter().rev())
                    .for_each(|(col, record)| {
                        mem_helper.fill(record.prev_timestamp, timestamp_mm(), col.as_mut());
                    });
            });

        cols.rs_read_aux
            .iter_mut()
            .rev()
            .zip(record.rs_read_aux.iter().rev())
            .for_each(|(col, record)| {
                mem_helper.fill(record.prev_timestamp, timestamp_mm(), col.as_mut());
            });

        // Decompose the u32 pointer into BLOCK_FE_WIDTH u16 cells. Bytes 4..8 are zero by the
        // `pointer_max_bits` constraint (always <= 32 in practice).
        cols.rs_val = record.rs_val.map(|val| {
            let val_u64 = val as u64;
            from_fn(|j| F::from_u32(((val_u64 >> (16 * j)) & 0xffff) as u32))
        });
        cols.rs_ptr = record.rs_ptr.map(F::from_u32);

        cols.from_state.timestamp = F::from_u32(record.timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
