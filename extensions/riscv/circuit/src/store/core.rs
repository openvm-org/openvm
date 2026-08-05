use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{Postflight, PostflightError, VmChipWrapper, *},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::*,
    BaseAirWithPublicValues,
};

use crate::adapters::{
    is_multi_byte_access_width, shift_encoder, u16_cell_byte, StoreMultiByteAdapterCols,
    StoreMultiByteAdapterFiller, StoreInstruction, BYTE_SHIFT_SELECTOR_WIDTH,
    DOUBLEWORD_ACCESS_WIDTH, HALFWORD_ACCESS_WIDTH, NUM_BYTE_SHIFTS, BYTE_BITS,
    WORD_ACCESS_WIDTH,
};

/// The single opcode handled by the store chip of the given width.
pub(crate) fn store_opcode<const STORE_WIDTH: usize>() -> LoadStoreOpcode {
    match STORE_WIDTH {
        DOUBLEWORD_ACCESS_WIDTH => STORED,
        WORD_ACCESS_WIDTH => STOREW,
        HALFWORD_ACCESS_WIDTH => STOREH,
        _ => unreachable!("unsupported width for store"),
    }
}

/// Handles halfword, word, and doubleword stores at any byte offset.
///
/// Even offsets replace whole u16 cells. Odd offsets decompose the source and the two boundary
/// cells into bytes so bytes outside the stored range remain unchanged. Byte columns and lookups
/// are unused on even offsets.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreCoreCols<T, const NUM_VALUE_CELLS: usize> {
    pub selector: [T; BYTE_SHIFT_SELECTOR_WIDTH],
    pub read_data: [T; BLOCK_FE_WIDTH],
    /// Previous contents of the two consecutive memory blocks; the second is used only when the
    /// access crosses a block boundary.
    pub prev_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Low bytes of the low source register cells covered by the store width. Constrained and
    /// used only on odd shifts.
    pub value_lo_bytes: [T; NUM_VALUE_CELLS],
    /// The bytes preserved by an odd-shift store: the low byte of the first overlapped memory
    /// cell and the high byte of the last. Constrained and used only on odd shifts.
    pub prev_bound_bytes: [T; 2],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreCoreCols<u8, NUM_VALUE_CELLS>)]
pub struct StoreCoreAir<const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize> {
    pub offset: usize,
    local_opcode: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreCoreAir<STORE_WIDTH, NUM_VALUE_CELLS>
{
    // First byte offset at which the store reaches the next memory block.
    const FIRST_CROSSING_SHIFT: usize = MEMORY_BLOCK_BYTES - STORE_WIDTH + 1;

    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        const {
            assert!(is_multi_byte_access_width(STORE_WIDTH));
            assert!(NUM_VALUE_CELLS == STORE_WIDTH / U16_CELL_SIZE);
        }
        Self {
            offset,
            local_opcode: store_opcode::<STORE_WIDTH>() as usize,
            encoder: shift_encoder(),
            bitwise_lookup_bus,
        }
    }

    pub(crate) fn new_with_local_opcode(
        offset: usize,
        local_opcode: usize,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
    ) -> Self {
        const {
            assert!(is_multi_byte_access_width(STORE_WIDTH));
            assert!(NUM_VALUE_CELLS == STORE_WIDTH / U16_CELL_SIZE);
        }
        Self {
            offset,
            local_opcode,
            encoder: shift_encoder(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field, const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize> BaseAir<F>
    for StoreCoreAir<STORE_WIDTH, NUM_VALUE_CELLS>
{
    fn width(&self) -> usize {
        StoreCoreCols::<F, NUM_VALUE_CELLS>::width()
    }
}

impl<F: Field, const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize> BaseAirWithPublicValues<F>
    for StoreCoreAir<STORE_WIDTH, NUM_VALUE_CELLS>
{
}

impl<AB, I, const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize> VmCoreAir<AB, I>
    for StoreCoreAir<STORE_WIDTH, NUM_VALUE_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([[AB::Expr; BLOCK_FE_WIDTH]; 2], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 2]>,
    I::ProcessedInstruction: From<StoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &StoreCoreCols<AB::Var, NUM_VALUE_CELLS> = (*local_core).borrow();
        let width = STORE_WIDTH / U16_CELL_SIZE;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        // cross = Σ flag[s] over shifts `s` where `s + STORE_WIDTH > 8`.
        let cross = flags[Self::FIRST_CROSSING_SHIFT..]
            .iter()
            .fold(AB::Expr::ZERO, |acc, flag| acc + flag.clone());

        // odd_shift = Σᵢ flag[2i + 1].
        let odd_shift = flags
            .iter()
            .skip(1)
            .step_by(2)
            .fold(AB::Expr::ZERO, |acc, flag| acc + flag.clone());

        // Cell `k` of the two consecutive previous memory blocks.
        let prev_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.prev_data[0][cell]
            } else {
                cols.prev_data[1][cell - BLOCK_FE_WIDTH]
            }
        };

        let inv_2_pow_8 = AB::F::from_u32(1 << BYTE_BITS).inverse();
        // read_data[i] = value_lo_bytes[i] + 2^8 * value_hi_bytes[i] on odd shifts.
        let value_hi_bytes: [AB::Expr; NUM_VALUE_CELLS] =
            std::array::from_fn(|i| (cols.read_data[i] - cols.value_lo_bytes[i]) * inv_2_pow_8);
        for (&lo, hi) in cols.value_lo_bytes.iter().zip(value_hi_bytes.iter()) {
            self.bitwise_lookup_bus
                .send_range(lo, hi.clone())
                .eval(builder, odd_shift.clone());
        }
        // prev_bound_cells[b] = Σᵢ flag[2i + 1] * prev_full(i + b * width).
        let prev_bound_cells: [AB::Expr; 2] = std::array::from_fn(|which| {
            flags.iter().skip(1).step_by(2).enumerate().fold(
                AB::Expr::ZERO,
                |acc, (cell_offset, flag)| {
                    acc + flag.clone() * prev_full(cell_offset + which * width)
                },
            )
        });
        // prev_bound_cells[0] = preserved_lo + 2^8 * overwritten_hi.
        // prev_bound_cells[1] = overwritten_lo + 2^8 * preserved_hi.
        let first_cell_hi = (prev_bound_cells[0].clone() - cols.prev_bound_bytes[0]) * inv_2_pow_8;
        let last_cell_lo = prev_bound_cells[1].clone()
            - cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS);
        self.bitwise_lookup_bus
            .send_range(cols.prev_bound_bytes[0], first_cell_hi)
            .eval(builder, odd_shift.clone());
        self.bitwise_lookup_bus
            .send_range(last_cell_lo, cols.prev_bound_bytes[1])
            .eval(builder, odd_shift.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_usize(self.local_opcode),
        );
        // shift_amount = Σₛ s * flag[s].
        let shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(byte_shift)
            });

        // write_data[cell] = Σₛ flag[s] * candidate(s, cell).
        // The branches below define `candidate`: even shifts splice whole cells; odd shifts
        // splice bytes and preserve the two boundary bytes.
        let write_data: [[AB::Expr; BLOCK_FE_WIDTH]; 2] = std::array::from_fn(|block| {
            std::array::from_fn(|k| {
                let cell = block * BLOCK_FE_WIDTH + k;
                flags
                    .iter()
                    .enumerate()
                    .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                        let first = byte_shift / 2;
                        let term = if byte_shift % 2 == 0 {
                            if cell >= first && cell < first + width {
                                cols.read_data[cell - first].into()
                            } else {
                                prev_full(cell).into()
                            }
                        } else if cell < first || cell > first + width {
                            prev_full(cell).into()
                        } else if cell == first {
                            cols.prev_bound_bytes[0]
                                + cols.value_lo_bytes[0] * AB::Expr::from_u32(1 << BYTE_BITS)
                        } else if cell == first + width {
                            value_hi_bytes[width - 1].clone()
                                + cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS)
                        } else {
                            value_hi_bytes[cell - first - 1].clone()
                                + cols.value_lo_bytes[cell - first]
                                    * AB::Expr::from_u32(1 << BYTE_BITS)
                        };
                        acc + flag.clone() * term
                    })
            })
        });
        AdapterAirContext {
            to_pc: None,
            reads: (
                cols.prev_data.map(|block| block.map(Into::into)),
                cols.read_data.map(Into::into),
            )
                .into(),
            writes: write_data.into(),
            instruction: StoreInstruction {
                is_valid,
                opcode: expected_opcode,
                shift_amount,
                store_cross: cross,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct StoreFiller<
    A = StoreMultiByteAdapterFiller,
    const STORE_WIDTH: usize = WORD_ACCESS_WIDTH,
    const NUM_VALUE_CELLS: usize = 2,
> {
    pub(crate) adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
}

impl<const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreFiller<StoreMultiByteAdapterFiller, STORE_WIDTH, NUM_VALUE_CELLS>
{
    pub fn new(
        adapter: StoreMultiByteAdapterFiller,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ) -> Self {
        Self::new_with_adapter(adapter, offset, bitwise_lookup_chip)
    }
}

impl<A, const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreFiller<A, STORE_WIDTH, NUM_VALUE_CELLS>
{
    pub(crate) fn new_with_adapter(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ) -> Self {
        const {
            assert!(is_multi_byte_access_width(STORE_WIDTH));
            assert!(NUM_VALUE_CELLS == STORE_WIDTH / U16_CELL_SIZE);
        }
        Self {
            adapter,
            offset,
            encoder: shift_encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl<A, const STORE_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreFiller<A, STORE_WIDTH, NUM_VALUE_CELLS>
{
    pub(crate) fn fill_core_row<F: PrimeField32>(
        &self,
        shift: usize,
        read_data: [u16; BLOCK_FE_WIDTH],
        prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
        core_row: &mut StoreCoreCols<F, NUM_VALUE_CELLS>,
    ) {
        debug_assert!(shift < NUM_BYTE_SHIFTS, "invalid store shift {shift}");

        let width = STORE_WIDTH / U16_CELL_SIZE;
        let prev_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        // The byte range checks are gated on the odd-shift indicator, so even shifts request no
        // lookups and leave the byte columns zero.
        let (value_lo_bytes, prev_bound_cells): ([u16; NUM_VALUE_CELLS], [[u16; 2]; 2]) =
            if shift % 2 == 1 {
                let lo_bytes = std::array::from_fn(|i| u16_cell_byte(read_data[i], 0));
                let bound_cells = std::array::from_fn(|which| {
                    let cell = prev_full[shift / 2 + which * width];
                    [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
                });
                for (i, lo) in lo_bytes.iter().enumerate() {
                    self.bitwise_lookup_chip
                        .request_range(*lo as u32, u16_cell_byte(read_data[i], 1) as u32);
                }
                for cell_bytes in &bound_cells {
                    self.bitwise_lookup_chip
                        .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
                }
                (lo_bytes, bound_cells)
            } else {
                ([0; NUM_VALUE_CELLS], [[0; 2]; 2])
            };

        core_row.value_lo_bytes = value_lo_bytes.map(F::from_u16);
        core_row.prev_bound_bytes =
            [prev_bound_cells[0][0], prev_bound_cells[1][1]].map(F::from_u16);
        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(|block| block.map(F::from_u16));
        let flag_point: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] =
            self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*flag_point).map(F::from_u32);
    }
}

pub(crate) fn generate_trace_from_postflight<
    F: PrimeField32,
    const STORE_WIDTH: usize,
    const NUM_VALUE_CELLS: usize,
>(
    chip: &VmChipWrapper<
        F,
        StoreFiller<StoreMultiByteAdapterFiller, STORE_WIDTH, NUM_VALUE_CELLS>,
    >,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(store_opcode::<STORE_WIDTH>().global_opcode());
    let adapter_width = StoreMultiByteAdapterCols::<F>::width();
    let width = adapter_width + StoreCoreCols::<F, NUM_VALUE_CELLS>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let (read_data, prev_data, shift) = chip.inner.adapter.replay::<F, STORE_WIDTH>(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |read_data, prev_data, shift| {
                crate::store::common::store_write_data(
                    store_opcode::<STORE_WIDTH>(),
                    read_data,
                    prev_data,
                    shift,
                )
            },
        )?;
        chip.inner
            .fill_core_row(shift, read_data, prev_data, core_row.borrow_mut());
        Ok(())
    })?;
    trace.values[steps.len() * width..]
        .par_chunks_exact_mut(width)
        .for_each(fill_padding_row);
    Ok(trace)
}

pub(crate) fn fill_padding_row<F: PrimeField32>(row: &mut [F]) {
    let _ = row;
}
