use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::{once, zip},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir, CONST_BLOCK_SIZE,
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
    abstract_compose, tracing_read, tracing_write, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
};

/// The fixed block size for memory bus interactions (4 bytes).
pub const VEC_HEAP_BLOCK_SIZE: usize = CONST_BLOCK_SIZE;

/// This adapter reads from R (R <= 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive reads of size `READ_SIZE` from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive writes of size `WRITE_SIZE` to the
///   heap, starting from the address in `rd`.
/// * Memory bus interactions are sent in `VEC_HEAP_BLOCK_SIZE` (4-byte) chunks.
/// * `READ_CHUNKS` must equal `READ_SIZE / VEC_HEAP_BLOCK_SIZE`.
/// * `WRITE_CHUNKS` must equal `WRITE_SIZE / VEC_HEAP_BLOCK_SIZE`.
#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32VecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CHUNKS: usize,
    const WRITE_CHUNKS: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    pub rs_val: [[T; RV32_REGISTER_NUM_LIMBS]; NUM_READS],
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    // Now nested by chunks: [NUM_READS][BLOCKS_PER_READ][READ_CHUNKS] aux cols
    pub reads_aux: [[[MemoryReadAuxCols<T>; READ_CHUNKS]; BLOCKS_PER_READ]; NUM_READS],
    // Now nested by chunks: [BLOCKS_PER_WRITE][WRITE_CHUNKS] aux cols, each 4 bytes
    pub writes_aux: [[MemoryWriteAuxCols<T, VEC_HEAP_BLOCK_SIZE>; WRITE_CHUNKS]; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32VecHeapAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CHUNKS: usize,
    const WRITE_CHUNKS: usize,
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
        const READ_CHUNKS: usize,
        const WRITE_CHUNKS: usize,
    > BaseAir<F>
    for Rv32VecHeapAdapterAir<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
    >
{
    fn width(&self) -> usize {
        Rv32VecHeapAdapterCols::<
            F,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
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
        const READ_CHUNKS: usize,
        const WRITE_CHUNKS: usize,
    > VmAdapterAir<AB>
    for Rv32VecHeapAdapterAir<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
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
        // Consistency checks (debug only)
        debug_assert_eq!(READ_CHUNKS, READ_SIZE / VEC_HEAP_BLOCK_SIZE);
        debug_assert_eq!(WRITE_CHUNKS, WRITE_SIZE / VEC_HEAP_BLOCK_SIZE);

        let cols: &Rv32VecHeapAdapterCols<
            _,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        > = local.borrow();
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

        // We constrain the highest limbs of heap pointers to be less than 2^(addr_bits -
        // (RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1))). This ensures that no overflow
        // occurs when computing memory pointers. Since the number of cells accessed with each
        // address will be small enough, and combined with the memory argument, it ensures
        // that all the cells accessed in the memory are less than 2^addr_bits.
        let need_range_check: Vec<AB::Var> = cols
            .rs_val
            .iter()
            .chain(std::iter::repeat_n(&cols.rd_val, 2))
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
        for pair in need_range_check.chunks_exact(2) {
            self.bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the u32 register value into single field element, with `abstract_compose`
        let rd_val_f: AB::Expr = abstract_compose(cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(abstract_compose);

        let e = AB::F::from_canonical_u32(RV32_MEMORY_AS);

        // Reads from heap - send READ_CHUNKS messages of VEC_HEAP_BLOCK_SIZE bytes per block
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux) {
            for (block_idx, (read_block, block_aux)) in zip(reads, reads_aux).enumerate() {
                for (chunk_idx, aux) in block_aux.iter().enumerate() {
                    let chunk_offset = block_idx * READ_SIZE + chunk_idx * VEC_HEAP_BLOCK_SIZE;
                    let chunk_data: [AB::Expr; VEC_HEAP_BLOCK_SIZE] =
                        from_fn(|i| read_block[chunk_idx * VEC_HEAP_BLOCK_SIZE + i].clone());
                    self.memory_bridge
                        .read(
                            MemoryAddress::new(
                                e,
                                address.clone() + AB::Expr::from_canonical_usize(chunk_offset),
                            ),
                            chunk_data,
                            timestamp_pp(),
                            aux,
                        )
                        .eval(builder, ctx.instruction.is_valid.clone());
                }
            }
        }

        // Writes to heap - send WRITE_CHUNKS messages of VEC_HEAP_BLOCK_SIZE bytes per block
        for (block_idx, (write_block, block_aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            for (chunk_idx, aux) in block_aux.iter().enumerate() {
                let chunk_offset = block_idx * WRITE_SIZE + chunk_idx * VEC_HEAP_BLOCK_SIZE;
                let chunk_data: [AB::Expr; VEC_HEAP_BLOCK_SIZE] =
                    from_fn(|i| write_block[chunk_idx * VEC_HEAP_BLOCK_SIZE + i].clone());
                self.memory_bridge
                    .write(
                        MemoryAddress::new(
                            e,
                            rd_val_f.clone() + AB::Expr::from_canonical_usize(chunk_offset),
                        ),
                        chunk_data,
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
        let cols: &Rv32VecHeapAdapterCols<
            _,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        > = local.borrow();
        cols.from_state.pc
    }
}

// Intermediate type that should not be copied or cloned and should be directly written to
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32VecHeapAdapterRecord<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CHUNKS: usize,
    const WRITE_CHUNKS: usize,
> {
    pub from_pc: u32,
    pub from_timestamp: u32,

    pub rs_ptrs: [u32; NUM_READS],
    pub rd_ptr: u32,

    pub rs_vals: [u32; NUM_READS],
    pub rd_val: u32,

    pub rs_read_aux: [MemoryReadAuxRecord; NUM_READS],
    pub rd_read_aux: MemoryReadAuxRecord,

    // Now nested by chunks: [NUM_READS][BLOCKS_PER_READ][READ_CHUNKS] aux records
    pub reads_aux: [[[MemoryReadAuxRecord; READ_CHUNKS]; BLOCKS_PER_READ]; NUM_READS],
    // Now nested by chunks: [BLOCKS_PER_WRITE][WRITE_CHUNKS] aux records, each 4 bytes
    pub writes_aux:
        [[MemoryWriteBytesAuxRecord<VEC_HEAP_BLOCK_SIZE>; WRITE_CHUNKS]; BLOCKS_PER_WRITE],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv32VecHeapAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CHUNKS: usize,
    const WRITE_CHUNKS: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv32VecHeapAdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
    const READ_CHUNKS: usize,
    const WRITE_CHUNKS: usize,
> {
    pointer_max_bits: usize,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<
        F: PrimeField32,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
        const READ_SIZE: usize,
        const WRITE_SIZE: usize,
        const READ_CHUNKS: usize,
        const WRITE_CHUNKS: usize,
    > AdapterTraceExecutor<F>
    for Rv32VecHeapAdapterExecutor<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
    >
{
    const WIDTH: usize = Rv32VecHeapAdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
    >::width();
    type ReadData = [[[u8; READ_SIZE]; BLOCKS_PER_READ]; NUM_READS];
    type WriteData = [[u8; WRITE_SIZE]; BLOCKS_PER_WRITE];
    type RecordMut<'a> = &'a mut Rv32VecHeapAdapterRecord<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
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
        record: &mut &mut Rv32VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        >,
    ) -> Self::ReadData {
        debug_assert_eq!(READ_CHUNKS, READ_SIZE / VEC_HEAP_BLOCK_SIZE);

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

        // Read memory values - now in VEC_HEAP_BLOCK_SIZE chunks
        from_fn(|read_idx| {
            debug_assert!(
                (record.rs_vals[read_idx] + (READ_SIZE * BLOCKS_PER_READ - 1) as u32)
                    < (1 << self.pointer_max_bits) as u32
            );
            from_fn(|block_idx| {
                let mut block_data = [0u8; READ_SIZE];
                for chunk_idx in 0..READ_CHUNKS {
                    let chunk_ptr = record.rs_vals[read_idx]
                        + (block_idx * READ_SIZE + chunk_idx * VEC_HEAP_BLOCK_SIZE) as u32;
                    let chunk: [u8; VEC_HEAP_BLOCK_SIZE] = tracing_read(
                        memory,
                        RV32_MEMORY_AS,
                        chunk_ptr,
                        &mut record.reads_aux[read_idx][block_idx][chunk_idx].prev_timestamp,
                    );
                    block_data
                        [chunk_idx * VEC_HEAP_BLOCK_SIZE..(chunk_idx + 1) * VEC_HEAP_BLOCK_SIZE]
                        .copy_from_slice(&chunk);
                }
                block_data
            })
        })
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut &mut Rv32VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        >,
    ) {
        debug_assert_eq!(WRITE_CHUNKS, WRITE_SIZE / VEC_HEAP_BLOCK_SIZE);
        debug_assert_eq!(instruction.e.as_canonical_u32(), RV32_MEMORY_AS);

        debug_assert!(
            record.rd_val as usize + WRITE_SIZE * BLOCKS_PER_WRITE - 1
                < (1 << self.pointer_max_bits)
        );

        // Write in VEC_HEAP_BLOCK_SIZE chunks
        for block_idx in 0..BLOCKS_PER_WRITE {
            for chunk_idx in 0..WRITE_CHUNKS {
                let chunk_ptr = record.rd_val
                    + (block_idx * WRITE_SIZE + chunk_idx * VEC_HEAP_BLOCK_SIZE) as u32;
                let chunk: [u8; VEC_HEAP_BLOCK_SIZE] = data[block_idx]
                    [chunk_idx * VEC_HEAP_BLOCK_SIZE..(chunk_idx + 1) * VEC_HEAP_BLOCK_SIZE]
                    .try_into()
                    .unwrap();
                tracing_write(
                    memory,
                    RV32_MEMORY_AS,
                    chunk_ptr,
                    chunk,
                    &mut record.writes_aux[block_idx][chunk_idx].prev_timestamp,
                    &mut record.writes_aux[block_idx][chunk_idx].prev_data,
                );
            }
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
        const READ_CHUNKS: usize,
        const WRITE_CHUNKS: usize,
    > AdapterTraceFiller<F>
    for Rv32VecHeapAdapterFiller<
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
    >
{
    const WIDTH: usize = Rv32VecHeapAdapterCols::<
        F,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE,
        READ_CHUNKS,
        WRITE_CHUNKS,
    >::width();

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY:
        // - caller ensures `adapter_row` contains a valid record representation that was previously
        //   written by the executor
        let record: &Rv32VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        > = unsafe { get_record_from_slice(&mut adapter_row, ()) };

        let cols: &mut Rv32VecHeapAdapterCols<
            F,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE,
            READ_CHUNKS,
            WRITE_CHUNKS,
        > = adapter_row.borrow_mut();

        // Range checks:
        // **NOTE**: Must do the range checks before overwriting the records
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

        // Total timestamp delta: register reads + heap reads (chunks) + heap writes (chunks)
        let timestamp_delta = NUM_READS
            + 1
            + NUM_READS * BLOCKS_PER_READ * READ_CHUNKS
            + BLOCKS_PER_WRITE * WRITE_CHUNKS;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // **NOTE**: Must iterate everything in reverse order to avoid overwriting the records
        // Writes - iterate blocks then chunks in reverse
        for (block_aux, cols_block) in record
            .writes_aux
            .iter()
            .zip(cols.writes_aux.iter_mut())
            .rev()
        {
            for (chunk_aux, cols_chunk) in block_aux.iter().zip(cols_block.iter_mut()).rev() {
                cols_chunk.set_prev_data(chunk_aux.prev_data.map(F::from_canonical_u8));
                mem_helper.fill(
                    chunk_aux.prev_timestamp,
                    timestamp_mm(),
                    cols_chunk.as_mut(),
                );
            }
        }

        // Reads - iterate reads, blocks, chunks in reverse
        for (reads_aux, cols_reads) in record.reads_aux.iter().zip(cols.reads_aux.iter_mut()).rev()
        {
            for (block_aux, cols_block) in reads_aux.iter().zip(cols_reads.iter_mut()).rev() {
                for (chunk_aux, cols_chunk) in block_aux.iter().zip(cols_block.iter_mut()).rev() {
                    mem_helper.fill(
                        chunk_aux.prev_timestamp,
                        timestamp_mm(),
                        cols_chunk.as_mut(),
                    );
                }
            }
        }

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

        cols.rd_val = record.rd_val.to_le_bytes().map(F::from_canonical_u8);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(record.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = val.to_le_bytes().map(F::from_canonical_u8);
            });
        cols.rd_ptr = F::from_canonical_u32(record.rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(record.rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_canonical_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_canonical_u32(record.from_timestamp);
        cols.from_state.pc = F::from_canonical_u32(record.from_pc);
    }
}
