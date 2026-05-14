//! Pattern B u16-shaped variant of [`Rv64VecHeapAdapter`].
//!
//! Differences vs the u8 variant:
//! - `READ_SIZE` and `WRITE_SIZE` are counted in **u16 cells**, not bytes; the per-block byte
//!   stride is `BUS_PTR_SCALE * READ_SIZE = 2 * READ_SIZE`.
//! - `ReadData` / `WriteData` are `[[[u16; READ_SIZE]; ...]; ...]` and the bus call passes the u16
//!   cells through directly without `pack_u8_for_bus`.
//! - All other shape constants (rs_ptr / rs_val / rd_ptr / rd_val) are unchanged because the
//!   register-side reads still go through the byte-shaped register cells that the existing
//!   `Rv64VecHeapAdapter` already handles via `expand_to_rv64_register`.

use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::{once, zip},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir, BLOCK_FE_WIDTH,
        BUS_PTR_SCALE,
    },
    system::memory::{
        offline_checker::{
            pack_u8_for_bus, MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord,
            MemoryWriteAuxCols,
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
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
};
use openvm_riscv_circuit::adapters::{
    abstract_compose, expand_to_rv64_register, tracing_read_reg_ptr, tracing_read_u16,
    tracing_write_u16, RV64_CELL_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// See module-level docs. `READ_SIZE`/`WRITE_SIZE` are in **u16 cells**.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64VecHeapU16AdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    pub rs_val: [[T; RV64_WORD_NUM_LIMBS]; NUM_READS],
    pub rd_val: [T; RV64_WORD_NUM_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64VecHeapU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>)]
pub struct Rv64VecHeapU16AdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
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
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > BaseAir<F>
    for Rv64VecHeapU16AdapterAir<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    fn width(&self) -> usize {
        Rv64VecHeapU16AdapterCols::<
            F,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv64VecHeapU16AdapterAir<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    type Interface = VecHeapAdapterInterface<
        AB::Expr,
        NUM_READS,
        BLOCKS_PER_READ,
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
        let cols: &Rv64VecHeapU16AdapterCols<
            _,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Register reads (still u8-byte-shaped on the register side; packed for the bus).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(AB::F::from_u32(RV64_REGISTER_AS), ptr),
                    pack_u8_for_bus::<AB>(&expand_to_rv64_register(&val)),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Pointer high-byte range checks (unchanged).
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .chain(std::iter::repeat_n(&cols.rd_val, 2))
            .map(|val| val[RV64_WORD_NUM_LIMBS - 1])
            .collect();

        let limb_shift =
            AB::F::from_usize(1 << (RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.address_bits));

        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the 4 materialized bytes of each register value into a single field element
        // used as a memory byte address.
        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(abstract_compose);

        let e = AB::F::from_u32(RV64_MEMORY_AS);
        // Heap reads. `READ_SIZE` is u16 cells; the per-block byte stride is
        // `BUS_PTR_SCALE * READ_SIZE` (= 2 * READ_SIZE for u16 cells).
        const BUS_BYTES_PER_CELL: usize = BUS_PTR_SCALE as usize;
        let read_bytes_per_block = READ_SIZE * BUS_BYTES_PER_CELL;
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux,) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                debug_assert_eq!(
                    READ_SIZE, BLOCK_FE_WIDTH,
                    "VecHeapU16 adapter only supports READ_SIZE = BLOCK_FE_WIDTH"
                );
                let read_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| read[j].clone());
                self.memory_bridge
                    .read_4(
                        MemoryAddress::new(
                            e,
                            address.clone() + AB::Expr::from_usize(i * read_bytes_per_block),
                        ),
                        read_array,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        let write_bytes_per_block = WRITE_SIZE * BUS_BYTES_PER_CELL;
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            debug_assert_eq!(
                WRITE_SIZE, BLOCK_FE_WIDTH,
                "VecHeapU16 adapter only supports WRITE_SIZE = BLOCK_FE_WIDTH"
            );
            let write_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| write[j].clone());
            self.memory_bridge
                .write_4(
                    MemoryAddress::new(
                        e,
                        rd_val_f.clone() + AB::Expr::from_usize(i * write_bytes_per_block),
                    ),
                    write_array,
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
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64VecHeapU16AdapterCols<
            _,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64VecHeapU16AdapterRecord<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptrs: [u32; NUM_READS],
    pub rd_ptr: u32,

    pub rs_vals: [u32; NUM_READS],
    pub rd_val: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub rd_read_aux: MemoryReadAuxRecord,

    pub reads_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [Rv64VecHeapU16WriteAuxRecord<WRITE_SIZE>; BLOCKS_PER_WRITE],
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone, Copy)]
pub struct Rv64VecHeapU16WriteAuxRecord<const WRITE_SIZE: usize> {
    pub prev_timestamp: u32,
    pub prev_data: [u16; WRITE_SIZE],
}

impl<const WRITE_SIZE: usize> Default for Rv64VecHeapU16WriteAuxRecord<WRITE_SIZE> {
    fn default() -> Self {
        Self {
            prev_timestamp: 0,
            prev_data: [0u16; WRITE_SIZE],
        }
    }
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv64VecHeapU16AdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64VecHeapU16AdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv64VecHeapU16AdapterExecutor<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    const WIDTH: usize = Rv64VecHeapU16AdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >::width();
    type ReadData = [[[u16; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type WriteData = [[u16; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type RecordMut<'a> = &'a mut Rv64VecHeapU16AdapterRecord<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >;

    #[inline(always)]
    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut &mut Rv64VecHeapU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >,
    ) -> Self::ReadData {
        let &Instruction { a, b, c, d, e, .. } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV64_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV64_MEMORY_AS);

        // Read register values
        record.rs_vals = from_fn(|i| {
            record.rs_ptrs[i] = if i == 0 { b } else { c }.as_canonical_u32();
            tracing_read_reg_ptr(
                memory,
                record.rs_ptrs[i],
                &mut record.rs_read_aux[i].prev_timestamp,
                self.pointer_max_bits,
            )
        });

        record.rd_ptr = a.as_canonical_u32();
        record.rd_val = tracing_read_reg_ptr(
            memory,
            record.rd_ptr,
            &mut record.rd_read_aux.prev_timestamp,
            self.pointer_max_bits,
        );

        let bytes_per_block = READ_SIZE * 2;
        from_fn(|i| {
            debug_assert!(
                (record.rs_vals[i] as u64) + ((bytes_per_block * BLOCKS_PER_READ - 1) as u64)
                    < (1u64 << self.pointer_max_bits)
            );
            from_fn(|j| {
                tracing_read_u16(
                    memory,
                    RV64_MEMORY_AS,
                    record.rs_vals[i] + (j * bytes_per_block) as u32,
                    &mut record.reads_aux[i][j].prev_timestamp,
                )
            })
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv64VecHeapU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >,
    ) {
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV64_MEMORY_AS);

        let bytes_per_block = WRITE_SIZE * 2;
        debug_assert!(
            (record.rd_val as u64) + ((bytes_per_block * BLOCKS_PER_WRITE - 1) as u64)
                < (1u64 << self.pointer_max_bits)
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..BLOCKS_PER_WRITE {
            tracing_write_u16(
                memory,
                RV64_MEMORY_AS,
                record.rd_val + (i * bytes_per_block) as u32,
                data[i],
                &mut record.writes_aux[i].prev_timestamp,
                &mut record.writes_aux[i].prev_data,
            );
        }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceFiller<F>
    for Rv64VecHeapU16AdapterFiller<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    const WIDTH: usize = Rv64VecHeapU16AdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        let record: &Rv64VecHeapU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64VecHeapU16AdapterCols<
            F,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        // Pointer high-byte range checks (unchanged from u8 adapter).
        debug_assert!(self.pointer_max_bits <= RV64_CELL_BITS * RV64_WORD_NUM_LIMBS);
        let limb_shift_bits = RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: usize = RV64_CELL_BITS * (RV64_WORD_NUM_LIMBS - 1);
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

        let timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        record
            .writes_aux
            .iter()
            .rev()
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|(write, cols_write)| {
                // `prev_data` is already u16-cell-shaped; no byte→u16 packing needed.
                // WRITE_SIZE == BLOCK_FE_WIDTH is enforced by the AIR-side debug_assert above;
                // build the BLOCK_FE_WIDTH-shaped array explicitly so the type-checker is happy.
                debug_assert_eq!(WRITE_SIZE, BLOCK_FE_WIDTH);
                let prev: [F; BLOCK_FE_WIDTH] = from_fn(|i| F::from_u32(write.prev_data[i] as u32));
                cols_write.set_prev_data(prev);
                mem_helper.fill(write.prev_timestamp, timestamp_mm(), cols_write.as_mut());
            });

        record
            .reads_aux
            .iter()
            .zip(cols.reads_aux.iter_mut())
            .rev()
            .for_each(|(reads, cols_reads)| {
                reads
                    .iter()
                    .zip(cols_reads.iter_mut())
                    .rev()
                    .for_each(|(read, cols_read)| {
                        mem_helper.fill(read.prev_timestamp, timestamp_mm(), cols_read.as_mut());
                    });
            });

        mem_helper.fill(
            record.rd_read_aux.prev_timestamp,
            timestamp_mm(),
            cols.rd_read_aux.as_mut(),
        );

        record
            .rs_read_aux
            .iter()
            .zip(cols.rs_read_aux.iter_mut())
            .rev()
            .for_each(|(aux, cols_aux)| {
                mem_helper.fill(aux.prev_timestamp, timestamp_mm(), cols_aux.as_mut());
            });

        cols.rd_val = record.rd_val.to_le_bytes().map(F::from_u8);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = val.to_le_bytes().map(F::from_u8);
            });
        cols.rd_ptr = F::from_u32(record.rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_u32(record.from_timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
