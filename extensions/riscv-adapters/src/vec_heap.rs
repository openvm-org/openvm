use std::{
    borrow::{Borrow, BorrowMut},
    iter::{once, zip},
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir,
        BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{
        offline_checker::{
            pack_u8_block, pack_u8_block_bytes, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols,
        },
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr, expand_to_rv64_block, ptr_bound_from_high_u16_expr, ptr_bound_from_ptr,
    ptr_to_field_u16_limbs, u16_limbs_to_ptr, RV64_PTR_U16_LIMBS, U16_BITS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

/// This adapter reads from R (R <= 2) pointers and writes to 1 pointer.
/// * The data is read from the heap (address space 2), and the pointers are read from registers
///   (address space 1).
/// * Reads take the form of `BLOCKS_PER_READ` consecutive `MEMORY_BLOCK_BYTES` reads from the heap,
///   starting from the addresses in `rs[0]` (and `rs[1]` if `R = 2`).
/// * Writes take the form of `BLOCKS_PER_WRITE` consecutive `MEMORY_BLOCK_BYTES` writes to the
///   heap, starting from the address in `rd`.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct Rv64VecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    pub rs_val: [[T; RV64_PTR_U16_LIMBS]; NUM_READS],
    pub rd_val: [T; RV64_PTR_U16_LIMBS],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64VecHeapAdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>)]
pub struct Rv64VecHeapAdapterAir<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    /// Maximum bit width of guest byte pointers.
    pointer_max_bits: usize,
}

impl<
        F: Field,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > BaseAir<F> for Rv64VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    fn width(&self) -> usize {
        Rv64VecHeapAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width()
    }
}

impl<const NUM_READS: usize, const BLOCKS: usize>
    Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>
{
    pub fn fill_trace_row_from_projection<F: PrimeField32>(
        &self,
        range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut [F],
        input: &VecHeapTraceInput<NUM_READS, BLOCKS>,
    ) {
        self.fill_trace_row_from_values(
            range_checker,
            mem_helper,
            adapter_row,
            VecHeapTraceValues {
                from_pc: input.from_pc,
                from_timestamp: input.from_timestamp,
                rs_ptrs: &input.rs_ptrs,
                rd_ptr: input.rd_ptr,
                rs_vals: &input.rs_vals,
                rd_val: input.rd_val,
                rs_prev_timestamps: &input.rs_prev_timestamps,
                rd_prev_timestamp: input.rd_prev_timestamp,
                heap_prev_timestamps: &input.heap_prev_timestamps,
                write_prev_timestamps: &input.write_prev_timestamps,
                write_predecessors: &input.write_predecessors,
            },
        );
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > VmAdapterAir<AB> for Rv64VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    type Interface = VecHeapAdapterInterface<
        AB::Expr,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        MEMORY_BLOCK_BYTES,
        MEMORY_BLOCK_BYTES,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &Rv64VecHeapAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(RV64_REGISTER_AS),
                        byte_ptr_to_u16_ptr::<AB>(ptr),
                    ),
                    expand_to_rv64_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Each materialized pointer is stored as two u16 cells. Bound the high
        // cell against the guest byte-pointer limit.
        for val in cols.rs_val.iter().chain(once(&cols.rd_val)) {
            self.range_bus
                .range_check(
                    ptr_bound_from_high_u16_expr(val[1], self.pointer_max_bits),
                    U16_BITS,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        // Compose the two u16 cells into low 32-bit heap/register pointers.
        let rd_val_f: AB::Expr = u16_limbs_to_ptr(&cols.rd_val);
        let rs_val_f: [AB::Expr; NUM_READS] = cols.rs_val.map(|limbs| u16_limbs_to_ptr(&limbs));

        let e = AB::F::from_u32(RV64_MEMORY_AS);
        // Reads from heap
        for (address, reads, reads_aux) in izip!(rs_val_f, ctx.reads, &cols.reads_aux,) {
            for (i, (read, aux)) in zip(reads, reads_aux).enumerate() {
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            e,
                            byte_ptr_to_u16_ptr::<AB>(
                                address.clone() + AB::Expr::from_usize(i * MEMORY_BLOCK_BYTES),
                            ),
                        ),
                        pack_u8_block::<AB>(&read),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (i, (write, aux)) in zip(ctx.writes, &cols.writes_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e,
                        byte_ptr_to_u16_ptr::<AB>(
                            rd_val_f.clone() + AB::Expr::from_usize(i * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    pack_u8_block::<AB>(&write),
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
        let cols: &Rv64VecHeapAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        cols.from_state.pc
    }
}

/// Minimal, record-free input needed to fill one vector-heap adapter trace row.
///
/// Checkpoint replay projects this value directly from the immutable program and
/// chronology-resolved memory log. It intentionally contains neither AIR columns
/// nor arena/layout metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VecHeapTraceInput<const NUM_READS: usize, const BLOCKS: usize> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub local_opcode: u32,
    pub rs_ptrs: [u32; NUM_READS],
    pub rd_ptr: u32,
    pub rs_vals: [u32; NUM_READS],
    pub rd_val: u32,
    pub rs_prev_timestamps: [u32; NUM_READS],
    pub rd_prev_timestamp: u32,
    pub heap_prev_timestamps: [[u32; BLOCKS]; NUM_READS],
    pub write_prev_timestamps: [u32; BLOCKS],
    pub heap_reads: [[[u16; BLOCK_FE_WIDTH]; BLOCKS]; NUM_READS],
    pub writes: [[u16; BLOCK_FE_WIDTH]; BLOCKS],
    pub write_predecessors: [[u16; BLOCK_FE_WIDTH]; BLOCKS],
}

struct VecHeapTraceValues<
    'a,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    from_pc: u32,
    from_timestamp: u32,
    rs_ptrs: &'a [u32; NUM_READS],
    rd_ptr: u32,
    rs_vals: &'a [u32; NUM_READS],
    rd_val: u32,
    rs_prev_timestamps: &'a [u32; NUM_READS],
    rd_prev_timestamp: u32,
    heap_prev_timestamps: &'a [[u32; BLOCKS_PER_READ]; NUM_READS],
    write_prev_timestamps: &'a [u32; BLOCKS_PER_WRITE],
    write_predecessors: &'a [[u16; BLOCK_FE_WIDTH]; BLOCKS_PER_WRITE],
}

/// The layout must match `VecHeapTraceInput` in `vec_heap_replay.cuh`, whose
/// size is `VEC_HEAP_TRACE_INPUT_BYTES` and whose alignment is that of
/// `uint32_t`, for every replay shape the projection launcher instantiates.
const fn vec_heap_trace_input_layout_matches<const NUM_READS: usize, const BLOCKS: usize>() -> bool
{
    size_of::<VecHeapTraceInput<NUM_READS, BLOCKS>>()
        == 24 + 12 * NUM_READS + 12 * NUM_READS * BLOCKS + 20 * BLOCKS
        && align_of::<VecHeapTraceInput<NUM_READS, BLOCKS>>() == align_of::<u32>()
}

const _: () = assert!(vec_heap_trace_input_layout_matches::<2, 4>());
const _: () = assert!(vec_heap_trace_input_layout_matches::<2, 6>());
const _: () = assert!(vec_heap_trace_input_layout_matches::<2, 8>());
const _: () = assert!(vec_heap_trace_input_layout_matches::<2, 12>());
const _: () = assert!(vec_heap_trace_input_layout_matches::<1, 8>());
const _: () = assert!(vec_heap_trace_input_layout_matches::<1, 12>());

pub fn vec_heap_u16_blocks_to_bytes<'a>(limbs: impl IntoIterator<Item = &'a u16>) -> Vec<u8> {
    limbs
        .into_iter()
        .flat_map(|limb| limb.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod projection_tests {
    use super::vec_heap_u16_blocks_to_bytes;

    #[test]
    fn projection_preserves_both_bytes_of_each_memory_cell() {
        let limbs = [0xabcd, 0x0102, 0xff00, 0x0080];
        assert_eq!(
            vec_heap_u16_blocks_to_bytes(&limbs),
            [0xcd, 0xab, 0x02, 0x01, 0x00, 0xff, 0x80, 0x00]
        );
    }
}

#[derive(derive_new::new, Clone, Copy)]
pub struct Rv64VecHeapAdapterExecutor<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct Rv64VecHeapAdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pointer_max_bits: usize,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const BLOCKS_PER_WRITE: usize>
    Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    pub fn pointer_max_bits(&self) -> usize {
        self.pointer_max_bits
    }
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const BLOCKS_PER_WRITE: usize>
    Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    /// Fills adapter columns from semantic replay values rather than a chip record.
    fn fill_trace_row_from_values<F: PrimeField32>(
        &self,
        range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut [F],
        values: VecHeapTraceValues<'_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>,
    ) {
        let VecHeapTraceValues {
            from_pc,
            from_timestamp,
            rs_ptrs,
            rd_ptr,
            rs_vals,
            rd_val,
            rs_prev_timestamps,
            rd_prev_timestamp,
            heap_prev_timestamps,
            write_prev_timestamps,
            write_predecessors,
        } = values;
        let cols: &mut Rv64VecHeapAdapterCols<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            adapter_row.borrow_mut();

        for &ptr in rs_vals.iter().chain(once(&rd_val)) {
            range_checker.add_count(ptr_bound_from_ptr(ptr, self.pointer_max_bits), U16_BITS);
        }

        let timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;
        let mut timestamp = from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        write_prev_timestamps
            .iter()
            .rev()
            .zip(write_predecessors.iter().rev())
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|((prev_timestamp, predecessor), cols_write)| {
                let mut predecessor_bytes = [0u8; MEMORY_BLOCK_BYTES];
                for (bytes, &limb) in predecessor_bytes.chunks_exact_mut(2).zip(predecessor) {
                    bytes.copy_from_slice(&limb.to_le_bytes());
                }
                cols_write.set_prev_data(pack_u8_block_bytes(&predecessor_bytes));
                mem_helper.fill(*prev_timestamp, timestamp_mm(), cols_write.as_mut());
            });

        heap_prev_timestamps
            .iter()
            .zip(cols.reads_aux.iter_mut())
            .rev()
            .for_each(|(reads, cols_reads)| {
                reads.iter().zip(cols_reads.iter_mut()).rev().for_each(
                    |(prev_timestamp, cols_read)| {
                        mem_helper.fill(*prev_timestamp, timestamp_mm(), cols_read.as_mut());
                    },
                );
            });

        mem_helper.fill(rd_prev_timestamp, timestamp_mm(), cols.rd_read_aux.as_mut());

        rs_prev_timestamps
            .iter()
            .zip(cols.rs_read_aux.iter_mut())
            .rev()
            .for_each(|(prev_timestamp, cols_aux)| {
                mem_helper.fill(*prev_timestamp, timestamp_mm(), cols_aux.as_mut());
            });

        cols.rd_val = ptr_to_field_u16_limbs(rd_val);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = ptr_to_field_u16_limbs(*val);
            });
        cols.rd_ptr = F::from_u32(rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_u32(from_timestamp);
        cols.from_state.pc = F::from_u32(from_pc);
    }
}
