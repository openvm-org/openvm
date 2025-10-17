use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
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
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_rv32im_circuit::adapters::{
    tracing_read, tracing_write, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// This adapter reads from NUM_READS <= 2 pointers and writes to a register.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size `BLOCK_SIZE` from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes are to 32-bit register rd.
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32IsEqualModAdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],

    pub rd_ptr: T,
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32IsEqualModAdapterAir<
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
    for Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    fn width(&self) -> usize {
        Rv32IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width()
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > VmAdapterAir<AB>
    for Rv32IsEqualModAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        NUM_READS,
        1,
        TOTAL_READ_SIZE,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Address spaces
        let d = AB::F::from_canonical_u32(RV32_REGISTER_AS);
        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        // Read register values for rs
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), val, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with
        // a range check on the highest limb.
        let rs_val_f = cols.rs_val.map(|decomp| {
            decomp.iter().rev().fold(AB::Expr::ZERO, |acc, &limb| {
                acc * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS) + limb
            })
        });

        let need_range_check: [_; 2] = from_fn(|i| {
            if i < NUM_READS {
                cols.rs_val[i][RV32_REGISTER_NUM_LIMBS - 1].into()
            } else {
                AB::Expr::ZERO
            }
        });

        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        self.bus
            .send_range(
                need_range_check[0].clone() * limb_shift,
                need_range_check[1].clone() * limb_shift,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Reads from heap
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        let read_block_data: [[[_; BLOCK_SIZE]; BLOCKS_PER_READ]; NUM_READS] =
            ctx.reads.map(|r: [AB::Expr; TOTAL_READ_SIZE]| {
                let mut r_it = r.into_iter();
                from_fn(|_| from_fn(|_| r_it.next().unwrap()))
            });
        let block_ptr_offset: [_; BLOCKS_PER_READ] =
            from_fn(|i| AB::F::from_canonical_usize(i * BLOCK_SIZE));

        for (ptr, block_data, block_aux) in izip!(rs_val_f, read_block_data, &cols.heap_read_aux) {
            for (offset, data, aux) in izip!(block_ptr_offset, block_data, block_aux) {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, ptr.clone() + offset),
                        data,
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Write to rd register
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
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32IsEqualModAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32IsEqualModAdapterRecord<
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
    pub writes_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
}

#[derive(Clone, Copy)]
pub struct Rv32IsEqualModAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv32IsEqualModAdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCK_SIZE: usize,
    const TOTAL_READ_SIZE: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCK_SIZE: usize,
        const TOTAL_READ_SIZE: usize,
    > Rv32IsEqualModAdapterExecutor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        assert!(NUM_READS <= 2);
        assert_eq!(TOTAL_READ_SIZE, BLOCKS_PER_READ * BLOCK_SIZE);
        assert!(
            RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits < RV32_CELL_BITS,
            "pointer_max_bits={pointer_max_bits} needs to be large enough for high limb range check"
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
    for Rv32IsEqualModAdapterExecutor<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
where
    F: PrimeField32,
{
    const WIDTH: usize =
        Rv32IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();
    type ReadData = [[u8; TOTAL_READ_SIZE]; NUM_READS];
    type WriteData = [u8; RV32_REGISTER_NUM_LIMBS];
    type RecordMut<'a> = &'a mut Rv32IsEqualModAdapterRecord<
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

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        record.rs_val = from_fn(|i| {
            record.rs_ptr[i] = if i == 0 { b } else { c }.as_canonical_u32();

            u32::from_le_bytes(tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs_ptr[i],
                &mut record.rs_read_aux[i].prev_timestamp,
            ))
        });

        // Read memory values
        from_fn(|i| {
            debug_assert!(
                record.rs_val[i] as usize + TOTAL_READ_SIZE - 1 < (1 << self.pointer_max_bits)
            );
            from_fn::<_, BLOCKS_PER_READ, _>(|j| {
                tracing_read::<BLOCK_SIZE>(
                    memory,
                    RV32_MEMORY_AS,
                    record.rs_val[i] + (j * BLOCK_SIZE) as u32,
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
            RV32_REGISTER_AS,
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
    for Rv32IsEqualModAdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE, TOTAL_READ_SIZE>
{
    const WIDTH: usize =
        Rv32IsEqualModAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv32IsEqualModAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCK_SIZE,
            TOTAL_READ_SIZE,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv32IsEqualModAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCK_SIZE> =
            adapter_row.borrow_mut();

        let mut timestamp = record.timestamp + (NUM_READS + NUM_READS * BLOCKS_PER_READ) as u32 + 1;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };
        // Do range checks before writing anything:
        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: usize = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        self.bitwise_lookup_chip.request_range(
            (record.rs_val[0] >> MSL_SHIFT) << limb_shift_bits,
            if NUM_READS > 1 {
                (record.rs_val[1] >> MSL_SHIFT) << limb_shift_bits
            } else {
                0
            },
        );
        // Writing in reverse order
        cols.writes_aux
            .set_prev_data(record.writes_aux.prev_data.map(F::from_canonical_u8));
        mem_helper.fill(
            record.writes_aux.prev_timestamp,
            timestamp_mm(),
            cols.writes_aux.as_mut(),
        );
        cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);

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

        cols.rs_val = record
            .rs_val
            .map(|val| val.to_le_bytes().map(F::from_canonical_u8));
        cols.rs_ptr = record.rs_ptr.map(|ptr| F::from_canonical_u32(ptr));

        cols.from_state.timestamp = F::from_canonical_u32(record.timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
