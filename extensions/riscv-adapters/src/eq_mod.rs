use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
        BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            pack_u8_block, pack_u8_block_bytes, MemoryBridge, MemoryReadAuxCols,
            MemoryReadAuxRecord, MemoryWriteAuxCols, MemoryWriteBytesAuxRecord,
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
    byte_ptr_to_u16_ptr, expand_to_rv64_block, ptr_bound_from_high_u16_expr, ptr_bound_from_ptr,
    ptr_to_field_u16_limbs, tracing_read, tracing_read_reg_ptr, tracing_write, u16_limbs_to_ptr,
    RV64_PTR_BITS, RV64_PTR_U16_LIMBS, RV64_REGISTER_NUM_LIMBS, U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive heap reads.
/// * Writes are to 64-bit register rd (8 bytes).
///
/// The materialized pointer values are stored as two u16 cells.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64IsEqualModAdapterCols<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64IsEqualModAdapterCols<u8, NUM_READS, BLOCKS_PER_READ>)]
pub struct Rv64IsEqualModAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    byte_ptr_max_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > BaseAir<F> for Rv64IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv64IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB> for Rv64IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        RV64_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        const {
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * MEMORY_BLOCK_BYTES,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * MEMORY_BLOCK_BYTES"
            );
        }
        let cols: &Rv64IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_u32(RV64_REGISTER_AS);
        let e = AB::F::from_u32(RV64_MEMORY_AS);

        // Read register values for rs.
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(d, byte_ptr_to_u16_ptr::<AB>(ptr)),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the two u16 cells into low 32-bit heap pointers.
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(|limbs| u16_limbs_to_ptr(&limbs));

        // Each materialized pointer is stored as two u16 cells. Bound the high
        // cell against the guest byte-pointer limit.
        for val in cols.rs_val.iter() {
            self.range_bus
                .range_check(
                    ptr_bound_from_high_u16_expr(val[1], self.byte_ptr_max_bits),
                    U16_BITS,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Reads from heap
        let read_block_data: [[[_; MEMORY_BLOCK_BYTES]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });
        let block_ptr_offset: [_; BLOCKS_PER_READ] =
            from_fn(|i| AB::F::from_usize(i * MEMORY_BLOCK_BYTES));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, byte_ptr_to_u16_ptr::<AB>(ptr.clone() + offset)),
                        pack_u8_block::<AB>(&data),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register
        self.memory_bridge
            .write(
                MemoryAddress::new(d, byte_ptr_to_u16_ptr::<AB>(cols.rd_ptr)),
                pack_u8_block::<AB>(&ctx.writes[0].clone()),
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
        let cols: &Rv64IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ> = local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv64IsEqualModAdapterRecord<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_pc: u32,
    pub timestamp: u32,

    pub rs_ptr: [u32; NUM_READS],
    pub rs_val: [u32; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: u32,
    pub writes_aux: MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy)]
pub struct Rv64IsEqualModAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const TOTAL_READ_SIZE: usize,
> {
    byte_ptr_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64IsEqualModAdapterFiller<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    byte_ptr_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const TOTAL_READ_SIZE: usize>
    Rv64IsEqualModAdapterExecutor<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    pub fn new(byte_ptr_max_bits: usize) -> Self {
        const {
            assert!(NUM_READS <= 2);
            assert!(
                TOTAL_READ_SIZE == BLOCKS_PER_READ * MEMORY_BLOCK_BYTES,
                "TOTAL_READ_SIZE must equal BLOCKS_PER_READ * MEMORY_BLOCK_BYTES"
            );
        }
        assert!(
            (U16_BITS..=RV64_PTR_BITS).contains(&byte_ptr_max_bits),
            "byte_ptr_max_bits must be in [16, 32]"
        );
        Self { byte_ptr_max_bits }
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const TOTAL_READ_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv64IsEqualModAdapterExecutor<NUM_READS, BLOCKS_PER_READ, TOTAL_READ_SIZE>
{
    const WIDTH: usize = Rv64IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width();
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u8; RV64_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv64IsEqualModAdapterRecord<NUM_READS, BLOCKS_PER_READ>;

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
                self.byte_ptr_max_bits,
            )
        });

        // Read memory values
        from_fn(|i| {
            debug_assert!(
                (record.rs_val[i] as u64) + ((TOTAL_READ_SIZE - 1) as u64)
                    < (1u64 << self.byte_ptr_max_bits)
            );
            from_fn::<_, BLOCKS_PER_READ, _>(|j| {
                tracing_read::<MEMORY_BLOCK_BYTES>(
                    memory,
                    RV64_MEMORY_AS,
                    record.rs_val[i] + (j * MEMORY_BLOCK_BYTES) as u32,
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
        tracing_write(
            memory,
            RV64_REGISTER_AS,
            record.rd_ptr,
            data,
            &mut record.writes_aux.prev_timestamp,
            &mut record.writes_aux.prev_data,
        );
    }
}

impl<F: PrimeField32, const NUM_READS: usize, const BLOCKS_PER_READ: usize> AdapterTraceFiller<F>
    for Rv64IsEqualModAdapterFiller<NUM_READS, BLOCKS_PER_READ>
{
    const WIDTH: usize = Rv64IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv64IsEqualModAdapterRecord<NUM_READS, BLOCKS_PER_READ> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv64IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ> =
            adapter_row.borrow_mut();

        for &ptr in record.rs_val.iter() {
            self.range_checker_chip
                .add_count(ptr_bound_from_ptr(ptr, self.byte_ptr_max_bits), U16_BITS);
        }

        let mut timestamp = record.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) as u32 + 1;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // Write auxiliary columns are filled in reverse timestamp order.
        cols.writes_aux
            .set_prev_data(pack_u8_block_bytes(&record.writes_aux.prev_data));
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
        cols.rs_ptr = record.rs_ptr.map(|ptr| F::from_u32(ptr));

        cols.from_state.timestamp = F::from_u32(record.timestamp);
        cols.from_state.pc = F::from_u32(record.from_pc);
    }
}
