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
        MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            pack_u8_for_bus, MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord,
            MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
};
use openvm_riscv_circuit::adapters::{
    tracing_read, tracing_read_reg_ptr, tracing_write, RV64_CELL_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// Number of u16 cells holding the low 32 bits of a register pointer.
pub const REG_PTR_U16S: usize = RV64_WORD_NUM_LIMBS / 2;

/// This adapter reads from R (R <= 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size `READ_SIZE` from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive writes of size `WRITE_SIZE` to the
///   heap, starting from the address in `rd`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64VecHeapAdapterCols<
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

    /// Low 32 bits of rs registers, packed as 2 u16 cells (matches the memory bus payload).
    pub rs_val: [[T; REG_PTR_U16S]; NUM_READS],
    /// Low 32 bits of rd register, packed as 2 u16 cells (matches the memory bus payload).
    pub rd_val: [T; REG_PTR_U16S],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64VecHeapAdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>)]
pub struct Rv64VecHeapAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
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
    for Rv64VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>
{
    fn width(&self) -> usize {
        Rv64VecHeapAdapterCols::<
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
    for Rv64VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE, READ_SIZE, WRITE_SIZE>
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
        let cols: &Rv64VecHeapAdapterCols<
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

        // Read register values for rs, rd: low 32 bits as 2 u16 cells, zero-extended to
        // BLOCK_FE_WIDTH=4 cells on the memory bus.
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            let bus_payload: [AB::Expr; BLOCK_FE_WIDTH] =
                [val[0].into(), val[1].into(), AB::Expr::ZERO, AB::Expr::ZERO];
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::F::from_u32(RV64_REGISTER_AS), ptr),
                    bus_payload,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Constrain the high u16 of each materialized pointer (`val[1]`) to be less than
        // `2^(addr_bits - 16)`. Combined with the zero padding on the upper 4 bytes in the
        // memory bus interaction above, this ensures every heap address accessed below is
        // less than `2^addr_bits` without any extra explicit assertion.
        let u16_bits = RV64_CELL_BITS * 2;
        // Memory-read u16 cells are already < 2^16, so range-checking `val[1] * shift < 2^16`
        // gives `val[1] < 2^(addr_bits - 16)`.
        let limb_shift = AB::F::from_u32(1 << (u16_bits * 2 - self.address_bits));
        for val in cols.rs_val.iter().chain(once(&cols.rd_val)) {
            self.range_bus
                .range_check(val[1] * limb_shift, u16_bits)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the 2 u16 cells of each register value into a single field element used as a
        // memory address.
        let compose = |val: [AB::Var; REG_PTR_U16S]| -> AB::Expr {
            val[0] + val[1] * AB::F::from_u32(1 << u16_bits)
        };
        let rd_val_f: AB::Expr = compose(cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(compose);

        let e = AB::F::from_u32(RV64_MEMORY_AS);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux,) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                debug_assert_eq!(
                    READ_SIZE, MEMORY_BLOCK_BYTES,
                    "VecHeap adapter only supports READ_SIZE = MEMORY_BLOCK_BYTES"
                );
                let read_array: [AB::Expr; MEMORY_BLOCK_BYTES] =
                    std::array::from_fn(|j| read[j].clone());
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
                            address.clone() + AB::Expr::from_usize(i * READ_SIZE),
                        ),
                        pack_u8_for_bus::<AB>(&read_array),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            debug_assert_eq!(
                WRITE_SIZE, MEMORY_BLOCK_BYTES,
                "VecHeap adapter only supports WRITE_SIZE = MEMORY_BLOCK_BYTES"
            );
            let write_array: [AB::Expr; MEMORY_BLOCK_BYTES] =
                std::array::from_fn(|j| write[j].clone());
            self.memory_bridge
                .write(
                    MemoryAddress::new(e, rd_val_f.clone() + AB::Expr::from_usize(i * WRITE_SIZE)),
                    pack_u8_for_bus::<AB>(&write_array),
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
        let cols: &Rv64VecHeapAdapterCols<
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

// Intermediate type that should not be copied or cloned and should be directly written to
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64VecHeapAdapterRecord<
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
    pub writes_aux: [MemoryWriteBytesAuxRecord<WRITE_SIZE>; BLOCKS_PER_WRITE],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv64VecHeapAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64VecHeapAdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv64VecHeapAdapterExecutor<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    const WIDTH: usize = Rv64VecHeapAdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >::width();
    type ReadData = [[[u8; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type WriteData = [[u8; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type RecordMut<'a> = &'a mut Rv64VecHeapAdapterRecord<
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
        record: &mut &mut Rv64VecHeapAdapterRecord<
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

        // Read memory values
        from_fn(|i| {
            debug_assert!(
                (record.rs_vals[i] as u64) + ((READ_SIZE * BLOCKS_PER_READ - 1) as u64)
                    < (1u64 << self.pointer_max_bits)
            );
            from_fn(|j| {
                tracing_read(
                    memory,
                    RV64_MEMORY_AS,
                    record.rs_vals[i] + (j * READ_SIZE) as u32,
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
        record: &mut &mut Rv64VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        >,
    ) {
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV64_MEMORY_AS);

        debug_assert!(
            (record.rd_val as u64) + ((WRITE_SIZE * BLOCKS_PER_WRITE - 1) as u64)
                < (1u64 << self.pointer_max_bits)
        );

        #[allow(clippy::needless_range_loop)]
        for i in 0..BLOCKS_PER_WRITE {
            tracing_write(
                memory,
                RV64_MEMORY_AS,
                record.rd_val + (i * WRITE_SIZE) as u32,
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
    for Rv64VecHeapAdapterFiller<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >
{
    const WIDTH: usize = Rv64VecHeapAdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
    >::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv64VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64VecHeapAdapterCols<
            F,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
        > = adapter_row.borrow_mut();

        // Range checks: `(reg >> 16) << shift_bits < 2^16` for each register pointer.
        // **NOTE**: Must do the range checks before overwriting the records.
        let u16_bits = RV64_CELL_BITS * 2;
        debug_assert!(self.pointer_max_bits <= u16_bits * 2);
        let limb_shift_bits = u16_bits * 2 - self.pointer_max_bits;
        for &v in record.rs_vals.iter().chain(once(&record.rd_val)) {
            self.range_checker_chip
                .add_count((v >> u16_bits) << limb_shift_bits, u16_bits);
        }

        let timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // **NOTE**: Must iterate everything in reverse order to avoid overwriting the records
        record
            .writes_aux
            .iter()
            .rev()
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|(write, cols_write)| {
                cols_write.set_prev_data(from_fn(|i| {
                    F::from_u32(
                        write.prev_data[2 * i] as u32 + 256 * write.prev_data[2 * i + 1] as u32,
                    )
                }));
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

        // Pack the low 32 bits of each register into 2 u16 cells.
        let pack_u16s = |val: u32| -> [F; REG_PTR_U16S] {
            let bytes = val.to_le_bytes();
            from_fn(|i| F::from_u16(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
        };
        cols.rd_val = pack_u16s(record.rd_val);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = pack_u16s(*val);
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
