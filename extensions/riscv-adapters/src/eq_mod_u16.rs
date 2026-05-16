//! U16-shaped equality-mod adapter. Register and heap payloads are `BLOCK_FE_WIDTH` u16 cells and
//! are sent to the memory bus directly.

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
    tracing_read_reg_ptr, tracing_read_u16, tracing_write_u16, RV64_CELL_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// Number of u16 cells holding the low 32 bits of a register pointer.
///
/// Mirrors `vec_heap::REG_PTR_U16S` but kept private to avoid a duplicate `pub use` from the crate
/// root.
const REG_PTR_U16S: usize = RV64_WORD_NUM_LIMBS / 2;

/// U16-shaped variant of [`Rv64IsEqualModAdapterCols`].
///
/// * `BLOCK_SIZE` counts u16 cells per heap-read block and must equal [`BLOCK_FE_WIDTH`].
/// * `rs_val` holds the low 32 bits of the register pointer as 2 u16 cells; the bus payload
///   zero-extends to `BLOCK_FE_WIDTH` u16 cells.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64IsEqualModU16AdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    /// Low 32 bits of rs registers, packed as 2 u16 cells (matches the memory bus payload).
    pub rs_val: [[T; REG_PTR_U16S]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64IsEqualModU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>)]
pub struct Rv64IsEqualModU16AdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    address_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F>
    for Rv64IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv64IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
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
        let cols: &Rv64IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
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

        // Read register values for rs: low 32 bits as 2 u16 cells, zero-extended to 4 cells.
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            let bus_payload: [AB::Expr; BLOCK_FE_WIDTH] =
                [val[0].into(), val[1].into(), AB::Expr::ZERO, AB::Expr::ZERO];
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), bus_payload, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the 2 u16 cells of each register value into a single field element used as the
        // heap base byte address.
        let u16_bits = RV64_CELL_BITS * 2;
        let compose = |val: [AB::Var; REG_PTR_U16S]| -> AB::Expr {
            val[0] + val[1] * AB::F::from_u32(1 << u16_bits)
        };
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(compose);

        // Range-check the high u16 of each pointer.
        let limb_shift = AB::F::from_u32(1 << (u16_bits * 2 - self.address_bits));
        for val in cols.rs_val.iter() {
            self.range_bus
                .range_check(val[1] * limb_shift, u16_bits)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Reads from heap. `BLOCK_SIZE` counts u16 cells per block.
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
            from_fn(|i| AB::F::from_usize(i * BLOCK_SIZE * BUS_PTR_SCALE));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                let data_array: [AB::Expr; BLOCK_FE_WIDTH] = from_fn(|j| data[j].clone());
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, ptr.clone() + offset),
                        data_array,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register.
        self.memory_bridge
            .write(
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
        let cols: &Rv64IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64IsEqualModU16AdapterRecord<
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
pub struct Rv64IsEqualModU16AdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64IsEqualModU16AdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > Rv64IsEqualModU16AdapterExecutor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
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
    for Rv64IsEqualModU16AdapterExecutor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize =
        Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();
    type ReadData = [[u16; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u16; BLOCK_FE_WIDTH];
    type RecordMut<'a> = &'a mut Rv64IsEqualModU16AdapterRecord<
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
        let bytes_per_block = BLOCK_SIZE * BUS_PTR_SCALE;
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
    for Rv64IsEqualModU16AdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    const WIDTH: usize =
        Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv64IsEqualModU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCK_SIZE,
            TOTAL_READ_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64IsEqualModU16AdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();

        let mut timestamp = record.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) as u32 + 1;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };
        // Do range checks before writing anything: range-check the high u16 of each pointer
        // shifted to fit in 16 bits.
        let u16_bits = RV64_CELL_BITS * 2;
        debug_assert!(self.pointer_max_bits <= u16_bits * 2);
        let limb_shift_bits = u16_bits * 2 - self.pointer_max_bits;
        for &v in record.rs_val.iter() {
            self.range_checker_chip
                .add_count((v >> u16_bits) << limb_shift_bits, u16_bits);
        }
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

        // Pack the low 32 bits of each register pointer into 2 u16 cells.
        cols.rs_val = record.rs_val.map(|val| {
            let bytes = val.to_le_bytes();
            from_fn(|i| F::from_u16(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])))
        });
        cols.rs_ptr = record.rs_ptr.map(F::from_u32);

        cols.from_state.timestamp = F::from_u32(record.timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
