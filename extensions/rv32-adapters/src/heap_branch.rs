use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir, VmAdapterInterface,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord},
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
use openvm_rv32im_circuit::adapters::{tracing_read, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

pub type Rv32HeapBranchAdapterCols<T, const NUM_READS: usize, const READ_SIZE: usize> =
    Rv32HeapBranchAdapterColsGeneric<T, NUM_READS, 1, READ_SIZE>;
pub type Rv32HeapBranchAdapterAir<const NUM_READS: usize, const READ_SIZE: usize> =
    Rv32HeapBranchAdapterAirGeneric<NUM_READS, 1, READ_SIZE>;
pub type Rv32HeapBranchAdapterRecord<const NUM_READS: usize> =
    Rv32HeapBranchAdapterRecordGeneric<NUM_READS, 1>;
pub type Rv32HeapBranchAdapterExecutor<const NUM_READS: usize, const READ_SIZE: usize> =
    Rv32HeapBranchAdapterExecutorGeneric<NUM_READS, 1, READ_SIZE>;
pub type Rv32HeapBranchAdapterFiller<const NUM_READS: usize, const READ_SIZE: usize> =
    Rv32HeapBranchAdapterFillerGeneric<NUM_READS, 1, READ_SIZE>;

pub struct VecHeapBranchAdapterInterface<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const READ_SIZE: usize,
>(PhantomData<T>);

impl<T, const NUM_READS: usize, const BLOCKS_PER_READ: usize, const READ_SIZE: usize>
    VmAdapterInterface<T> for VecHeapBranchAdapterInterface<T, NUM_READS, BLOCKS_PER_READ, READ_SIZE>
{
    type Reads = [[[T; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type Writes = [[T; 0]; 0];
    type ProcessedInstruction = ImmInstruction<T>;
}

/// Adapter cols for branching on heap values read in NUM_READS pointers.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32HeapBranchAdapterColsGeneric<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const READ_SIZE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],

    pub heap_read_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32HeapBranchAdapterAirGeneric<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const READ_SIZE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub bus: BitwiseOperationLookupBus,
    address_bits: usize,
}

impl<F: Field, const NUM_READS: usize, const BLOCKS_PER_READ: usize, const READ_SIZE: usize>
    BaseAir<F> for Rv32HeapBranchAdapterAirGeneric<NUM_READS, BLOCKS_PER_READ, READ_SIZE>
{
    fn width(&self) -> usize {
        Rv32HeapBranchAdapterColsGeneric::<F, NUM_READS, BLOCKS_PER_READ, READ_SIZE>::width()
    }
}

impl<AB, const NUM_READS: usize, const BLOCKS_PER_READ: usize, const READ_SIZE: usize> VmAdapterAir<AB>
    for Rv32HeapBranchAdapterAirGeneric<NUM_READS, BLOCKS_PER_READ, READ_SIZE>
where
    AB: InteractionBuilder,
{
    type Interface =
        VecHeapBranchAdapterInterface<AB::Expr, NUM_READS, BLOCKS_PER_READ, READ_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv32HeapBranchAdapterColsGeneric<_, NUM_READS, BLOCKS_PER_READ, READ_SIZE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        let d = AB::F::from_canonical_u32(RV32_REGISTER_AS);
        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        for (ptr, data, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux) {
            self.memory_bridge
                .read(MemoryAddress::new(d, ptr), data, timestamp_pp(), aux)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // We constrain the highest limbs of heap pointers to be less than 2^(addr_bits -
        // (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))). This ensures that no overflow
        // occurs when computing memory pointers. Since the number of cells accessed with each
        // address will be small enough, and combined with the memory argument, it ensures
        // that all the cells accessed in the memory are less than 2^addr_bits.
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .map(|val| val[RV32_REGISTER_NUM_LIMBS - 1])
            .collect();

        // range checks constrain to RV32_CELL_BITS bits, so we need to shift the limbs to constrain
        // the correct amount of bits
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits),
        );

        // Note: since limbs are read from memory we already know that limb[i] < 2^RV32_CELL_BITS
        //       thus range checking limb[i] * shift < 2^RV32_CELL_BITS, gives us that
        //       limb[i] < 2^(addr_bits - (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1)))
        for pair in need_range_check.chunks(2) {
            self.bus
                .send_range(
                    pair[0] * limb_shift,
                    pair.get(1).map(|x| (*x).into()).unwrap_or(AB::Expr::ZERO) * limb_shift, // in case NUM_READS is odd
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let heap_ptr = cols.rs_val.map(|r| {
            r.iter().rev().fold(AB::Expr::ZERO, |acc, limb| {
                acc * AB::F::from_canonical_u32(1 << RV32_CELL_BITS) + (*limb)
            })
        });
        for (ptr, data_blocks, aux_blocks) in izip!(heap_ptr.iter(), ctx.reads.iter(), cols.heap_read_aux.iter()) {
            for (block_idx, (data, aux)) in data_blocks.iter().zip(aux_blocks).enumerate() {
                let offset = AB::Expr::from_canonical_usize(block_idx * READ_SIZE);
                self.memory_bridge
                    .read(
                        MemoryAddress::new(e, ptr.clone() + offset),
                        data.clone(),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rs_ptr
                        .first()
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    cols.rs_ptr
                        .get(1)
                        .map(|&x| x.into())
                        .unwrap_or(AB::Expr::ZERO),
                    ctx.instruction.immediate,
                    d.into(),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32HeapBranchAdapterColsGeneric<_, NUM_READS, BLOCKS_PER_READ, READ_SIZE> =
            local.borrow();
        cols.from_state.pc
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32HeapBranchAdapterRecordGeneric<const NUM_READS: usize, const BLOCKS_PER_READ: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptr: [u32; NUM_READS],
    pub rs_vals: [u32; NUM_READS],

    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub heap_read_aux: [[MemoryReadAuxRecord; BLOCKS_PER_READ]; NUM_READS],
}

#[derive(Clone, Copy)]
pub struct Rv32HeapBranchAdapterExecutorGeneric<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const READ_SIZE: usize,
> {
    pub pointer_max_bits: usize,
}

#[derive(Clone, derive_new::new)]
pub struct Rv32HeapBranchAdapterFillerGeneric<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const READ_SIZE: usize,
> {
    pub pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const READ_SIZE: usize>
    Rv32HeapBranchAdapterExecutorGeneric<NUM_READS, BLOCKS_PER_READ, READ_SIZE>
{
    pub fn new(pointer_max_bits: usize) -> Self {
        assert!(NUM_READS <= 2);
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
        const READ_SIZE: usize,
    > AdapterTraceExecutor<F>
    for Rv32HeapBranchAdapterExecutorGeneric<NUM_READS, BLOCKS_PER_READ, READ_SIZE>
{
    const WIDTH: usize =
        Rv32HeapBranchAdapterColsGeneric::<F, NUM_READS, BLOCKS_PER_READ, READ_SIZE>::width();
    type ReadData = [[[u8; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type WriteData = ();
    type RecordMut<'a> = &'a mut Rv32HeapBranchAdapterRecordGeneric<NUM_READS, BLOCKS_PER_READ>;

    fn start(pc: u32, memory: &TracingMemory, adapter_record: &mut Self::RecordMut<'_>) {
        adapter_record.from_pc = pc;
        adapter_record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let Instruction { a, b, d, e, .. } = *instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        // Read register values
        record.rs_vals = from_fn(|i| {
            record.rs_ptr[i] = if i == 0 { a } else { b }.as_canonical_u32();
            u32::from_le_bytes(tracing_read(
                memory,
                RV32_REGISTER_AS,
                record.rs_ptr[i],
                &mut record.rs_read_aux[i].prev_timestamp,
            ))
        });

        // Read memory values in 4-byte chunks
        from_fn(|i| {
            debug_assert!(
                record.rs_vals[i] as usize + READ_SIZE * BLOCKS_PER_READ - 1
                    < (1 << self.pointer_max_bits)
            );
            from_fn(|j| {
                tracing_read(
                    memory,
                    RV32_MEMORY_AS,
                    record.rs_vals[i] + (j * READ_SIZE) as u32,
                    &mut record.heap_read_aux[i][j].prev_timestamp,
                )
            })
        })
    }

    fn write(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _data: Self::WriteData,
        _record: &mut Self::RecordMut<'_>,
    ) {
        // This adapter doesn't write anything
    }
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const READ_SIZE: usize,
    > AdapterTraceFiller<F>
    for Rv32HeapBranchAdapterFillerGeneric<NUM_READS, BLOCKS_PER_READ, READ_SIZE>
{
    const WIDTH: usize =
        Rv32HeapBranchAdapterColsGeneric::<F, NUM_READS, BLOCKS_PER_READ, READ_SIZE>::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv32HeapBranchAdapterRecordGeneric<NUM_READS, BLOCKS_PER_READ> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let cols: &mut Rv32HeapBranchAdapterColsGeneric<F, NUM_READS, BLOCKS_PER_READ, READ_SIZE> =
            adapter_row.borrow_mut();

        // Range checks:
        // **NOTE**: Must do the range checks before overwriting the records
        debug_assert!(self.pointer_max_bits <= RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS);
        let limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits;
        const MSL_SHIFT: usize = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);
        self.bitwise_lookup_chip.request_range(
            (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
            if NUM_READS > 1 {
                (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits
            } else {
                0
            },
        );

        // **NOTE**: Must iterate everything in reverse order to avoid overwriting the records
        let heap_ts_start = record.from_timestamp + NUM_READS as u32;
        for read_idx in (0..NUM_READS).rev() {
            for block_idx in (0..BLOCKS_PER_READ).rev() {
                let ts_offset = read_idx * BLOCKS_PER_READ + block_idx;
                mem_helper.fill(
                    record.heap_read_aux[read_idx][block_idx].prev_timestamp,
                    heap_ts_start + ts_offset as u32,
                    cols.heap_read_aux[read_idx][block_idx].as_mut(),
                );
            }
        }

        for i in (0..NUM_READS).rev() {
            mem_helper.fill(
                record.rs_read_aux[i].prev_timestamp,
                record.from_timestamp + i as u32,
                cols.rs_read_aux[i].as_mut(),
            );
        }

        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(col, record)| {
                *col = record.to_le_bytes().map(F::from_canonical_u8);
            });

        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptr.iter().rev())
            .for_each(|(col, record)| {
                *col = F::from_canonical_u32(*record);
            });

        cols.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
