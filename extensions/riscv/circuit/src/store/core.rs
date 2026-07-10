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
/// Even shifts move whole u16 cells. Odd shifts additionally use byte decompositions: all source
/// value cells (`value_bytes`) plus the first and last overlapped memory cells
/// (`prev_bound_bytes`), from which every touched cell of the written blocks is a linear
/// recombination that splices the value bytes between the preserved boundary bytes.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_VALUE_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    /// Kept as a degree-1 copy of the sum of block-crossing selector flags.
    pub cross: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
    /// Previous contents of the two consecutive memory blocks; the second is used only when the
    /// access crosses a block boundary.
    pub prev_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Byte decompositions `[lo, hi]` of the low `STORE_WIDTH / 2` source register cells.
    /// All-zero on even shifts.
    pub value_bytes: [[T; 2]; NUM_VALUE_CELLS],
    /// Byte decompositions `[lo, hi]` of the first and last memory cells overlapped by an
    /// odd-shift store. All-zero on even shifts.
    pub prev_bound_bytes: [[T; 2]; 2],
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
        builder.assert_eq(cols.is_valid, is_valid.clone());

        let cross_flags = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift + STORE_WIDTH > 2 * BLOCK_FE_WIDTH {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );
        builder.assert_eq(cols.cross, cross_flags);

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

        // Source value cells are shift-independent, so their decompositions are gated on the
        // odd-shift indicator instead of being flag-selected. On even shifts the slots are
        // forced to zero, and (0, 0) passes the byte range check.
        for (i, cell_bytes) in cols.value_bytes.iter().enumerate() {
            self.bitwise_lookup_bus
                .send_range(cell_bytes[0], cell_bytes[1])
                .eval(builder, is_valid.clone());
            builder.assert_eq(
                cell_bytes[0] + cell_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS),
                odd_shift.clone() * cols.read_data[i],
            );
        }
        // The first and last overlapped memory cells are flag-selected per odd shift.
        for (which, cell_bytes) in cols.prev_bound_bytes.iter().enumerate() {
            self.bitwise_lookup_bus
                .send_range(cell_bytes[0], cell_bytes[1])
                .eval(builder, is_valid.clone());
            let expected_cell = info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 1 {
                        acc + flags[case_idx].clone() * prev_full(byte_shift / 2 + which * width)
                    } else {
                        acc
                    }
                },
            );
            builder.assert_eq(
                cell_bytes[0] + cell_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS),
                expected_cell,
            );
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.is_valid * AB::Expr::from_u8(info.opcode as u8),
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
                            cols.prev_bound_bytes[0][0]
                                + cols.value_bytes[0][0] * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        } else if cell == first + width {
                            cols.value_bytes[width - 1][1]
                                + cols.prev_bound_bytes[1][1]
                                    * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        } else {
                            cols.value_bytes[cell - first - 1][1]
                                + cols.value_bytes[cell - first][0]
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
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                shift_amount,
                store_cross: cols.cross.into(),
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
        let (value_bytes, prev_bound_bytes) = if shift % 2 == 1 {
            (
                std::array::from_fn(|i| {
                    [
                        u16_cell_byte(read_data[i], 0),
                        u16_cell_byte(read_data[i], 1),
                    ]
                }),
                std::array::from_fn(|which| {
                    let cell = prev_full[shift / 2 + which * width];
                    [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
                }),
            )
        } else {
            ([[0; 2]; NUM_VALUE_CELLS], [[0; 2]; 2])
        };
        for cell_bytes in value_bytes.iter().chain(prev_bound_bytes.iter()) {
            self.bitwise_lookup_chip
                .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
        }

        core_row.value_bytes = value_bytes.map(|bytes| bytes.map(F::from_u16));
        core_row.prev_bound_bytes = prev_bound_bytes.map(|bytes| bytes.map(F::from_u16));
        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(|block| block.map(F::from_u16));
        core_row.cross = F::from_bool(shift + STORE_WIDTH > 2 * BLOCK_FE_WIDTH);
        core_row.is_valid = F::ONE;
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
