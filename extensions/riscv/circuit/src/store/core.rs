use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        u16_cell_byte, Rv64StoreAdapterCols, Rv64StoreAdapterFiller, Rv64StoreAdapterRecord,
        StoreInstruction, RV64_BYTE_BITS, STORE_WIDTH_DOUBLEWORD, STORE_WIDTH_HALFWORD,
        STORE_WIDTH_WORD,
    },
    store::common::StoreRecord,
};

const SELECTOR_MAX_DEGREE: u32 = 2;

/// Static description of a store chip: the single opcode it handles and the byte shifts it
/// supports, each shift encoded as a separate selector case.
#[derive(Clone, Copy)]
pub(crate) struct StoreInfo {
    opcode: Rv64LoadStoreOpcode,
    byte_shifts: &'static [usize],
}

fn encoder<const SELECTOR_WIDTH: usize>(byte_shifts: &[usize]) -> Encoder {
    let encoder = Encoder::new(byte_shifts.len(), SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

const STORE_DOUBLEWORD_INFO: StoreInfo = StoreInfo {
    opcode: STORED,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
const STORE_WORD_INFO: StoreInfo = StoreInfo {
    opcode: STOREW,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
const STORE_HALFWORD_INFO: StoreInfo = StoreInfo {
    opcode: STOREH,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
pub(crate) fn store_info<const STORE_WIDTH: usize>() -> StoreInfo {
    match STORE_WIDTH {
        STORE_WIDTH_DOUBLEWORD => STORE_DOUBLEWORD_INFO,
        STORE_WIDTH_WORD => STORE_WORD_INFO,
        STORE_WIDTH_HALFWORD => STORE_HALFWORD_INFO,
        _ => unreachable!("unsupported width for store"),
    }
}

/// Handles halfword, word, and doubleword stores. The core combines source register data with the
/// previous contents of the touched memory blocks so bytes outside the store width stay unchanged.
///
/// Even shifts move whole u16 cells. Odd shifts additionally use byte decompositions: the low
/// bytes of all source value cells (`value_lo_bytes`, high bytes derived in the AIR) plus the two
/// preserved bytes of the first and last overlapped memory cells (`prev_bound_bytes`, overwritten
/// bytes derived in the AIR), from which every touched cell of the written blocks is a linear
/// recombination that splices the value bytes between the preserved boundary bytes. All byte
/// range checks are gated on the odd-shift selector sum, so on even shifts the byte columns are
/// unconstrained and unused.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_VALUE_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    pub read_data: [T; BLOCK_FE_WIDTH],
    /// Previous contents of the two consecutive memory blocks; the second is used only when the
    /// access crosses a block boundary.
    pub prev_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Low bytes of the low `STORE_WIDTH / 2` source register cells. Constrained and used only
    /// on odd shifts.
    pub value_lo_bytes: [T; NUM_VALUE_CELLS],
    /// The bytes preserved by an odd-shift store: the low byte of the first overlapped memory
    /// cell and the high byte of the last. Constrained and used only on odd shifts.
    pub prev_bound_bytes: [T; 2],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreCoreCols<u8, SELECTOR_WIDTH, NUM_VALUE_CELLS>)]
pub struct StoreCoreAir<
    const STORE_WIDTH: usize,
    const SELECTOR_WIDTH: usize,
    const NUM_VALUE_CELLS: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
{
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        debug_assert_eq!(NUM_VALUE_CELLS, STORE_WIDTH / 2);
        Self {
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(store_info::<STORE_WIDTH>().byte_shifts),
            bitwise_lookup_bus,
        }
    }
}

impl<
        F: Field,
        const STORE_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_VALUE_CELLS: usize,
    > BaseAir<F> for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
{
    fn width(&self) -> usize {
        StoreCoreCols::<F, SELECTOR_WIDTH, NUM_VALUE_CELLS>::width()
    }
}

impl<
        F: Field,
        const STORE_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_VALUE_CELLS: usize,
    > BaseAirWithPublicValues<F> for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
{
}

impl<
        AB,
        I,
        const STORE_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_VALUE_CELLS: usize,
    > VmCoreAir<AB, I> for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
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
        let cols: &StoreCoreCols<AB::Var, SELECTOR_WIDTH, NUM_VALUE_CELLS> = (*local_core).borrow();
        let info = store_info::<STORE_WIDTH>();
        let width = STORE_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        let cross = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift + STORE_WIDTH > 2 * BLOCK_FE_WIDTH {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );

        let odd_shift = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift % 2 == 1 {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );

        // Cell `k` of the two consecutive previous memory blocks.
        let prev_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.prev_data[0][cell]
            } else {
                cols.prev_data[1][cell - BLOCK_FE_WIDTH]
            }
        };

        let inv_2_pow_8 = AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        // High byte of source value cell `i`, derived from its materialized low byte. The range
        // checks are gated on the odd-shift indicator, so on even shifts the low-byte columns
        // are unconstrained; they only feed odd-shift selector terms, which are then zero.
        let value_hi_byte = |i: usize| (cols.read_data[i] - cols.value_lo_bytes[i]) * inv_2_pow_8;
        for i in 0..width {
            self.bitwise_lookup_bus
                .send_range(cols.value_lo_bytes[i], value_hi_byte(i))
                .eval(builder, odd_shift.clone());
        }
        // The first and last overlapped memory cells, flag-selected per odd shift; zero on even
        // shifts and invalid rows.
        let prev_bound_cell = |which: usize| {
            info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 1 {
                        acc + flags[case_idx].clone() * prev_full(byte_shift / 2 + which * width)
                    } else {
                        acc
                    }
                },
            )
        };
        // The overwritten boundary-cell bytes are derived from the overlapped cells; range
        // checking them completes the decompositions of both boundary cells.
        let first_cell_hi = (prev_bound_cell(0) - cols.prev_bound_bytes[0]) * inv_2_pow_8;
        let last_cell_lo =
            prev_bound_cell(1) - cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        self.bitwise_lookup_bus
            .send_range(cols.prev_bound_bytes[0], first_cell_hi)
            .eval(builder, odd_shift.clone());
        self.bitwise_lookup_bus
            .send_range(last_cell_lo, cols.prev_bound_bytes[1])
            .eval(builder, odd_shift.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(info.opcode as u8),
        );
        let shift_amount = info
            .byte_shifts
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &byte_shift)| {
                acc + flags[i].clone() * AB::Expr::from_usize(byte_shift)
            });

        // Contents of both written blocks. Even shifts splice whole value cells between
        // preserved cells; odd shifts splice value bytes, with the two boundary cells mixing a
        // preserved byte and a value byte.
        let write_data: [[AB::Expr; BLOCK_FE_WIDTH]; 2] = std::array::from_fn(|block| {
            std::array::from_fn(|k| {
                let cell = block * BLOCK_FE_WIDTH + k;
                info.byte_shifts.iter().enumerate().fold(
                    AB::Expr::ZERO,
                    |acc, (case_idx, &byte_shift)| {
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
                                + cols.value_lo_bytes[0] * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        } else if cell == first + width {
                            value_hi_byte(width - 1)
                                + cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        } else {
                            value_hi_byte(cell - first - 1)
                                + cols.value_lo_bytes[cell - first]
                                    * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        };
                        acc + flags[case_idx].clone() * term
                    },
                )
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
    A = Rv64StoreAdapterFiller,
    const STORE_WIDTH: usize = STORE_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = 3,
    const NUM_VALUE_CELLS: usize = 2,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    StoreFiller<A, STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
{
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(store_info::<STORE_WIDTH>().byte_shifts),
            bitwise_lookup_chip,
        }
    }
}

impl<F, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_VALUE_CELLS: usize>
    TraceFiller<F>
    for StoreFiller<Rv64StoreAdapterFiller, STORE_WIDTH, SELECTOR_WIDTH, NUM_VALUE_CELLS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // StoreCoreCols::width() elements.
        let (mut adapter_row, mut core_row) = unsafe {
            row_slice
                .split_at_mut_unchecked(<Rv64StoreAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        let adapter_record: &Rv64StoreAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid StoreRecord written by the executor during trace
        // generation.
        let record: &StoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data;
        let prev_data = record.prev_data;
        let core_row: &mut StoreCoreCols<F, SELECTOR_WIDTH, NUM_VALUE_CELLS> =
            core_row.borrow_mut();
        let case_idx = store_info::<STORE_WIDTH>()
            .byte_shifts
            .iter()
            .position(|&byte_shift| byte_shift == shift)
            .expect("invalid store shift");

        let width = STORE_WIDTH / 2;
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
        // Only the preserved bytes are materialized: the low byte of the first overlapped cell
        // and the high byte of the last.
        core_row.prev_bound_bytes =
            [prev_bound_cells[0][0], prev_bound_cells[1][1]].map(F::from_u16);
        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(|block| block.map(F::from_u16));
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }

    fn fill_dummy_trace_row(&self, row_slice: &mut [F]) {
        let (adapter_row, _) = unsafe {
            row_slice
                .split_at_mut_unchecked(<Rv64StoreAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        let adapter_row: &mut Rv64StoreAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.mem_as = F::from_u32(2);
    }
}
