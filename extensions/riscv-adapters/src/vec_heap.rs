use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    iter::once,
};

use itertools::izip;
use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, ExecutionState, VecHeapAdapterInterface, VmAdapterAir,
        BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE,
    },
    system::memory::{
        offline_checker::{
            pack_u8_block, pack_u8_block_bytes, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols,
        },
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_AS},
};
use openvm_riscv_circuit::adapters::{
    add_const_u16_limbs_value, byte_ptr_limbs_to_cell_ptr_limbs_value, cell_ptr_hi_bits,
    eval_add_const_u16_limbs, eval_byte_ptr_limbs_to_u16_cell_ptr_limbs, expand_to_block,
    ptr_to_field_u16_limbs, reg_byte_ptr_to_cell_ptr_limbs, u32_to_ptr_limbs, PTR_U16_LIMBS,
    U16_BITS,
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
pub struct VecHeapAdapterCols<
    T,
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pub from_state: ExecutionState<T>,

    pub rs_ptr: [T; NUM_READS],
    pub rd_ptr: T,

    /// Low 32 bits of rs registers as little-endian 16-bit *byte*-pointer limbs.
    pub rs_val: [[T; PTR_U16_LIMBS]; NUM_READS],
    /// Low 32 bits of rd register as little-endian 16-bit *byte*-pointer limbs.
    pub rd_val: [T; PTR_U16_LIMBS],

    /// Carry for converting each base byte pointer to AS-native u16 *cell* pointer limbs.
    pub rs_cell_carry: [T; NUM_READS],
    pub rd_cell_carry: T,
    /// Per-block carry for adding the cell offset `j * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to
    /// each base cell pointer (block `j`'s carry into the high cell limb).
    pub reads_add_carry: [[T; BLOCKS_PER_READ]; NUM_READS],
    pub writes_add_carry: [T; BLOCKS_PER_WRITE],

    pub rs_read_aux: [MemoryReadAuxCols<T>; NUM_READS],
    pub rd_read_aux: MemoryReadAuxCols<T>,

    pub reads_aux: [[MemoryReadAuxCols<T>; BLOCKS_PER_READ]; NUM_READS],
    pub writes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; BLOCKS_PER_WRITE],
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(VecHeapAdapterCols<u8, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>)]
pub struct VecHeapAdapterAir<
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
    > BaseAir<F> for VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    fn width(&self) -> usize {
        VecHeapAdapterCols::<F, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>::width()
    }
}

impl<const NUM_READS: usize, const BLOCKS: usize> VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS> {
    pub fn fill_trace_row_from_projection<F: PrimeField32>(
        &self,
        range_checker: &openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut [F],
        input: &VecHeapTraceInput<NUM_READS, BLOCKS>,
    ) {
        let cols: &mut VecHeapAdapterCols<F, NUM_READS, BLOCKS, BLOCKS> = adapter_row.borrow_mut();

        // Byte -> cell pointer conversion carry and per-block cell-offset carry columns, plus
        // the matching range-check counts, for each base pointer.
        let cell_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;
        let cell_hi_bits = cell_ptr_hi_bits(self.pointer_max_bits);
        for (&byte_ptr, conv_col, add_cols) in izip!(
            input.rs_vals.iter().chain(once(&input.rd_val)),
            cols.rs_cell_carry
                .iter_mut()
                .chain(once(&mut cols.rd_cell_carry)),
            cols.reads_add_carry
                .iter_mut()
                .chain(once(&mut cols.writes_add_carry)),
        ) {
            let (conv_carry, base_cell) =
                byte_ptr_limbs_to_cell_ptr_limbs_value(u32_to_ptr_limbs(byte_ptr));
            range_checker.add_count(base_cell[1], cell_hi_bits);
            *conv_col = F::from_u32(conv_carry);
            for (j, add_col) in add_cols.iter_mut().enumerate() {
                let (add_carry, block_cell_ptr) =
                    add_const_u16_limbs_value(base_cell, j as u32 * cell_stride);
                range_checker.add_count(block_cell_ptr[0], U16_BITS);
                *add_col = F::from_u32(add_carry);
            }
        }

        let timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS + BLOCKS;
        let mut timestamp = input.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        input
            .write_prev_timestamps
            .iter()
            .rev()
            .zip(input.write_predecessors.iter().rev())
            .zip(cols.writes_aux.iter_mut().rev())
            .for_each(|((prev_timestamp, predecessor), cols_write)| {
                let mut predecessor_bytes = [0u8; MEMORY_BLOCK_BYTES];
                for (bytes, &limb) in predecessor_bytes.chunks_exact_mut(2).zip(predecessor) {
                    bytes.copy_from_slice(&limb.to_le_bytes());
                }
                cols_write.set_prev_data(pack_u8_block_bytes(&predecessor_bytes));
                mem_helper.fill(*prev_timestamp, timestamp_mm(), cols_write.as_mut());
            });

        input
            .heap_prev_timestamps
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

        mem_helper.fill(
            input.rd_prev_timestamp,
            timestamp_mm(),
            cols.rd_read_aux.as_mut(),
        );

        input
            .rs_prev_timestamps
            .iter()
            .zip(cols.rs_read_aux.iter_mut())
            .rev()
            .for_each(|(prev_timestamp, cols_aux)| {
                mem_helper.fill(*prev_timestamp, timestamp_mm(), cols_aux.as_mut());
            });

        cols.rd_val = ptr_to_field_u16_limbs(input.rd_val);
        cols.rs_val
            .iter_mut()
            .rev()
            .zip(input.rs_vals.iter().rev())
            .for_each(|(cols_val, val)| {
                *cols_val = ptr_to_field_u16_limbs(*val);
            });
        cols.rd_ptr = F::from_u32(input.rd_ptr);
        cols.rs_ptr
            .iter_mut()
            .rev()
            .zip(input.rs_ptrs.iter().rev())
            .for_each(|(cols_ptr, ptr)| {
                *cols_ptr = F::from_u32(*ptr);
            });
        cols.from_state.timestamp = F::from_u32(input.from_timestamp);
        cols.from_state.pc = F::from_u32(input.from_pc);
    }
}

impl<
        AB: InteractionBuilder,
        const NUM_READS: usize,
        const BLOCKS_PER_READ: usize,
        const BLOCKS_PER_WRITE: usize,
    > VmAdapterAir<AB> for VecHeapAdapterAir<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
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
        let cols: &VecHeapAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
            local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Read register values for rs, rd (register pointers are small).
        for (ptr, val, aux) in izip!(cols.rs_ptr, cols.rs_val, &cols.rs_read_aux).chain(once((
            cols.rd_ptr,
            cols.rd_val,
            &cols.rd_read_aux,
        ))) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::F::from_u32(REGISTER_AS),
                        reg_byte_ptr_to_cell_ptr_limbs::<AB>(ptr),
                    ),
                    expand_to_block(&val),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        let byte_ptr_max_bits = self.pointer_max_bits;
        let e = AB::F::from_u32(MEMORY_AS);
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
        let rd_base_cell = eval_byte_ptr_limbs_to_u16_cell_ptr_limbs::<AB>(
            builder,
            self.range_bus,
            cols.rd_val.map(Into::into),
            cols.rd_cell_carry,
            byte_ptr_max_bits,
            ctx.instruction.is_valid.clone(),
        );

        // Reads from heap: block `j` is at base cell pointer + `j * cell_ptr_block_stride`.
        for (base_cell, reads, reads_aux, add_carry) in izip!(
            rs_base_cell,
            ctx.reads,
            &cols.reads_aux,
            &cols.reads_add_carry
        ) {
            for (j, (read, aux, carry)) in izip!(reads, reads_aux, add_carry).enumerate() {
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
                        MemoryAddress::from_cell_pointer_limbs(
                            e,
                            block_cell_ptr,
                            AB::F::from_u32(BLOCK_FE_WIDTH as u32).inverse(),
                        ),
                        pack_u8_block::<AB>(&read),
                        timestamp_pp(),
                        aux,
                    )
                    .eval(builder, ctx.instruction.is_valid.clone());
            }
        }

        // Writes to heap
        for (j, (write, aux, carry)) in
            izip!(ctx.writes, &cols.writes_aux, &cols.writes_add_carry).enumerate()
        {
            let block_cell_ptr = eval_add_const_u16_limbs::<AB>(
                builder,
                self.range_bus,
                rd_base_cell.clone(),
                j as u32 * cell_ptr_block_stride,
                *carry,
                ctx.instruction.is_valid.clone(),
            );
            self.memory_bridge
                .write(
                    MemoryAddress::from_cell_pointer_limbs(
                        e,
                        block_cell_ptr,
                        AB::F::from_u32(BLOCK_FE_WIDTH as u32).inverse(),
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
                    AB::Expr::from_u32(REGISTER_AS),
                    e.into(),
                ],
                cols.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid.clone());
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &VecHeapAdapterCols<_, NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE> =
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

#[derive(derive_new::new)]
pub struct VecHeapAdapterFiller<
    const NUM_READS: usize,
    const BLOCKS_PER_READ: usize,
    const BLOCKS_PER_WRITE: usize,
> {
    pointer_max_bits: usize,
}

impl<const NUM_READS: usize, const BLOCKS_PER_READ: usize, const BLOCKS_PER_WRITE: usize>
    VecHeapAdapterFiller<NUM_READS, BLOCKS_PER_READ, BLOCKS_PER_WRITE>
{
    pub fn pointer_max_bits(&self) -> usize {
        self.pointer_max_bits
    }
}
