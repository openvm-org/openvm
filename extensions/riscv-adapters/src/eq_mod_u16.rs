use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE,
    },
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteU16AuxRecord,
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
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    add_const_u16_limbs_value, byte_ptr_limbs_to_cell_ptr_limbs_value, byte_ptr_to_u16_ptr_value,
    cell_ptr_hi_bits, eval_add_const_u16_limbs, eval_byte_ptr_limbs_to_u16_cell_ptr_limbs,
    expand_to_rv64_block, ptr_to_field_u16_limbs, ptr_to_u16_limbs, reg_byte_ptr_to_cell_ptr_limbs,
    tracing_read_reg_ptr, tracing_read_u16, tracing_write_u16, RV64_PTR_BITS, RV64_PTR_U16_LIMBS,
    U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// U16-shaped equality-mod adapter. Heap reads and register writes are
/// `BLOCK_FE_WIDTH` u16 cells per memory-bus message.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64IsEqualModU16AdapterCols<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    /// Carry for converting each base byte pointer to AS-native u16 *cell* pointer limbs.
    pub rs_cell_carry: [T; NUM_READS],
    /// Per-block carry for adding the cell offset `j * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to
    /// each base cell pointer (block `j`'s carry into the high cell limb).
    pub reads_add_carry: [[T; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64IsEqualModU16AdapterCols<u8, NUM_READS, BLOCKS_PER_READ>)]
pub struct Rv64IsEqualModU16AdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pointer_max_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F> for Rv64IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv64IsEqualModU16AdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
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
        const {
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * BLOCK_FE_WIDTH,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * BLOCK_FE_WIDTH"
            );
        }
        let cols: &Rv64IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_u32(RV64_REGISTER_AS);
        let e = AB::F::from_u32(RV64_MEMORY_AS);

        // Read register values for rs (register pointers are small).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(d, reg_byte_ptr_to_cell_ptr_limbs::<AB>(ptr)),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let byte_ptr_max_bits = self.pointer_max_bits;
        // Cell offset (in u16 cells) between consecutive heap blocks.
        let cell_ptr_block_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;

        // Convert each base *byte* pointer to base AS-native u16 *cell* pointer limbs.
        let rs_base_cell: [[AB::Expr; 2]; NUM_READS] = from_fn(|i| {
            eval_byte_ptr_limbs_to_u16_cell_ptr_limbs::<AB>(
                builder,
                self.range_bus,
                cols.rs_val[i].map(Into::into),
                cols.rs_cell_carry[i],
                byte_ptr_max_bits,
                ctx.instruction.is_valid.clone(),
            )
        });

        // Reads from heap: block `j` is at base cell pointer + `j * cell_ptr_block_stride`.
        let read_block_data: [[[_; BLOCK_FE_WIDTH]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });

        for (base_cell, block_data, block_aux, add_carry) in izip!(
            rs_base_cell,
            read_block_data,
            &cols.heap_read_aux,
            &cols.reads_add_carry
        ) {
            for (j, (data, aux, carry)) in izip!(block_data, block_aux, add_carry).enumerate() {
                let block_cell_ptr = eval_add_const_u16_limbs::<AB>(
                    builder,
                    self.range_bus,
                    base_cell.clone(),
                    j as u32 * cell_ptr_block_stride,
                    *carry,
                    ctx.instruction.is_valid.clone(),
                );
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, block_cell_ptr),
                        data,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register (register pointer is small).
        self.memory_bridge
            .write(
                MemoryAddress::new(d, reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.rd_ptr)),
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
        let cols: &Rv64IsEqualModU16AdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64IsEqualModU16AdapterRecord<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_pc: u32,
    pub timestamp: u32,

    pub rs_ptr: [u32; NUM_READS],
    pub rs_val: [u32; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: u32,
    pub writes_aux: MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH>,
}

#[derive(Clone, Copy)]
pub struct Rv64IsEqualModU16AdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64IsEqualModU16AdapterFiller<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const TOTAL_READ_SIZE: usize>
    Rv64IsEqualModU16AdapterExecutor<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        const {
            assert!(NUM_READS <= 2);
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * BLOCK_FE_WIDTH,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * BLOCK_FE_WIDTH"
            );
        }
        assert!(
            (U16_BITS..=RV64_PTR_BITS).contains(&pointer_max_bits),
            "pointer_max_bits must be in [16, 32]"
        );
        Self { pointer_max_bits }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv64IsEqualModU16AdapterExecutor<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    const WIDTH: usize = Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width();
    type ReadData = [[u16; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u16; BLOCK_FE_WIDTH];
    type RecordMut<'a> = &'a mut Rv64IsEqualModU16AdapterRecord<NUM_READS, BLOCKS_PER_READ>;

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

        // Read register values.
        record.rs_val = from_fn(|i| {
            record.rs_ptr[i] = if i == 0 { b } else { c }.as_canonical_u32();
            tracing_read_reg_ptr(
                memory,
                record.rs_ptr[i],
                &mut record.rs_read_aux[i].prev_timestamp,
                self.pointer_max_bits,
            )
        });

        // Reads from heap
        from_fn(|i| {
            debug_assert!(
                (record.rs_val[i] as u64) + ((MEMORY_BLOCK_BYTES * BLOCKS_PER_READ - 1) as u64)
                    < (1u64 << self.pointer_max_bits)
            );
            from_fn::<_, BLOCKS_PER_READ, _>(|j| {
                tracing_read_u16::<BLOCK_FE_WIDTH>(
                    memory,
                    RV64_MEMORY_AS,
                    byte_ptr_to_u16_ptr_value(record.rs_val[i] + (j * MEMORY_BLOCK_BYTES) as u32),
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
            byte_ptr_to_u16_ptr_value(record.rd_ptr),
            data,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const BLOCKS_PER_READ: usize> AdapterTraceFiller<F>
    for Rv64IsEqualModU16AdapterFiller<NUM_READS, BLOCKS_PER_READ>
{
    const WIDTH: usize = Rv64IsEqualModU16AdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv64IsEqualModU16AdapterRecord<NUM_READS, BLOCKS_PER_READ> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64IsEqualModU16AdapterCols<F, NUM_READS, BLOCKS_PER_READ> =
            adapter_row.borrow_mut();

        // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
        // range-check counts. **NOTE**: Must read the record values before overwriting them below.
        // `carry` columns are written near the end (after all record reads), so store them here.
        let hi_bits = cell_ptr_hi_bits(self.pointer_max_bits);
        let cell_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;
        let rs_carries: [(u32, Vec<u32>); NUM_READS] = from_fn(|i| {
            let byte_limbs = ptr_to_u16_limbs(record.rs_val[i]).map(u32::from);
            let (conv_carry, base_cell) = byte_ptr_limbs_to_cell_ptr_limbs_value(byte_limbs);
            self.range_checker_chip.add_count(base_cell[1], hi_bits);
            let add_carries = (0..BLOCKS_PER_READ)
                .map(|j| {
                    let (add_carry, block_cell_ptr) =
                        add_const_u16_limbs_value(base_cell, j as u32 * cell_stride);
                    self.range_checker_chip
                        .add_count(block_cell_ptr[0], U16_BITS);
                    add_carry
                })
                .collect();
            (conv_carry, add_carries)
        });

        let mut timestamp = record.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) as u32 + 1;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // Write auxiliary columns are filled in reverse timestamp order.
        cols.writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_u16));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp_mm(),
            cols.writes_aux.as_mut(),
        );
        cols.rd_ptr = F::from_u32(record.rd_ptr);

        // Remaining auxiliary columns are filled in reverse timestamp order.
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

        cols.rs_val = record.rs_val.map(ptr_to_field_u16_limbs);
        cols.rs_ptr = record.rs_ptr.map(F::from_u32);

        // Pointer-conversion / block-offset carry columns (computed above).
        for (i, (conv, add)) in rs_carries.iter().enumerate() {
            cols.rs_cell_carry[i] = F::from_u32(*conv);
            for (col, &c) in cols.reads_add_carry[i].iter_mut().zip(add.iter()) {
                *col = F::from_u32(c);
            }
        }

        cols.from_state.timestamp = F::from_u32(record.timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
