use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
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
        u16_cell_byte, LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord,
        LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD, RV64_BYTE_BITS, RV64_BYTE_SIGN_BIT,
        RV64_U16_SIGN_BIT, U16_BITS,
    },
    load::LoadRecord,
};

const SELECTOR_MAX_DEGREE: u32 = 2;

/// Static description of a signed load chip: the single opcode it handles and the byte shifts it
/// supports, each shift encoded as a separate selector case.
#[derive(Clone, Copy)]
pub(crate) struct LoadSignExtendInfo {
    opcode: Rv64LoadStoreOpcode,
    byte_shifts: &'static [usize],
}

const LOAD_SIGN_EXTEND_WORD_INFO: LoadSignExtendInfo = LoadSignExtendInfo {
    opcode: LOADW,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
const LOAD_SIGN_EXTEND_HALFWORD_INFO: LoadSignExtendInfo = LoadSignExtendInfo {
    opcode: LOADH,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
pub(crate) fn load_sign_extend_info<const LOAD_WIDTH: usize>() -> LoadSignExtendInfo {
    match LOAD_WIDTH {
        LOAD_WIDTH_WORD => LOAD_SIGN_EXTEND_WORD_INFO,
        LOAD_WIDTH_HALFWORD => LOAD_SIGN_EXTEND_HALFWORD_INFO,
        _ => unreachable!("unsupported width for signed load"),
    }
}

fn encoder<const SELECTOR_WIDTH: usize>(byte_shifts: &[usize]) -> Encoder {
    let encoder = Encoder::new(byte_shifts.len(), SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

/// Handles signed halfword and word loads. Each supported byte shift is encoded as a separate
/// selector case.
///
/// Even shifts move whole u16 cells and take the sign from the high bit of the top cell. Odd
/// shifts additionally use `touched_cell_bytes`: byte decompositions of the `LOAD_WIDTH / 2 + 1`
/// consecutive cells overlapped by the load, from which every loaded cell is the linear
/// recombination `hi_i + 2^8 * lo_{i+1}` and the sign comes from the high bit of the last slot's
/// low byte.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_TOUCHED_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    /// Kept as a degree-1 copy of the sum of block-crossing selector flags.
    pub cross: T,
    /// The sign bit that is extended to the remaining cells.
    pub data_most_sig_bit: T,
    /// Two consecutive 8-byte memory blocks; the second is used only when the access crosses a
    /// block boundary.
    pub read_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Byte decompositions `[lo, hi]` of the cells overlapped by an odd-shift load, starting at
    /// the cell containing the first loaded byte. All-zero on even shifts.
    pub touched_cell_bytes: [[T; 2]; NUM_TOUCHED_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendCoreCols<u8, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>)]
pub struct LoadSignExtendCoreAir<
    const LOAD_WIDTH: usize,
    const SELECTOR_WIDTH: usize,
    const NUM_TOUCHED_CELLS: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    range_bus: VariableRangeCheckerBus,
}

impl<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_TOUCHED_CELLS: usize>
    LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
{
    pub fn new(
        offset: usize,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        debug_assert_eq!(NUM_TOUCHED_CELLS, LOAD_WIDTH / 2 + 1);
        Self {
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_sign_extend_info::<LOAD_WIDTH>().byte_shifts),
            bitwise_lookup_bus,
            range_bus,
        }
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_TOUCHED_CELLS: usize,
    > BaseAir<F> for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
{
    fn width(&self) -> usize {
        LoadSignExtendCoreCols::<F, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>::width()
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_TOUCHED_CELLS: usize,
    > BaseAirWithPublicValues<F>
    for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
{
}

impl<
        AB,
        I,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_TOUCHED_CELLS: usize,
    > VmCoreAir<AB, I> for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BLOCK_FE_WIDTH]; 2]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<LoadInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadSignExtendCoreCols<AB::Var, SELECTOR_WIDTH, NUM_TOUCHED_CELLS> =
            (*local_core).borrow();
        let info = load_sign_extend_info::<LOAD_WIDTH>();
        let width = LOAD_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        builder.assert_eq(cols.is_valid, is_valid.clone());
        builder.assert_bool(cols.data_most_sig_bit);

        let cross_flags = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift + LOAD_WIDTH > 2 * BLOCK_FE_WIDTH {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );
        builder.assert_eq(cols.cross, cross_flags);

        // Cell `k` of the two consecutive memory blocks.
        let read_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.read_data[0][cell]
            } else {
                cols.read_data[1][cell - BLOCK_FE_WIDTH]
            }
        };

        for (j, cell_bytes) in cols.touched_cell_bytes.iter().enumerate() {
            self.bitwise_lookup_bus
                .send_range(cell_bytes[0], cell_bytes[1])
                .eval(builder, is_valid.clone());
            // On an odd shift the j-th slot recomposes to the j-th overlapped cell; the byte
            // range checks above make the decomposition unique. On even shifts every term of
            // the sum is zero, which forces the slot bytes to zero.
            let expected_cell = info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 1 {
                        acc + flags[case_idx].clone() * read_full(byte_shift / 2 + j)
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

        let (even_shift, odd_shift) = info.byte_shifts.iter().enumerate().fold(
            (AB::Expr::ZERO, AB::Expr::ZERO),
            |(even, odd), (case_idx, &byte_shift)| {
                if byte_shift % 2 == 0 {
                    (even + flags[case_idx].clone(), odd)
                } else {
                    (even, odd + flags[case_idx].clone())
                }
            },
        );

        // On even shifts the top loaded byte is the high byte of the selected top cell:
        // constrain the sign bit at u16 granularity.
        let even_sign_cell = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift % 2 == 0 {
                    acc + flags[case_idx].clone() * read_full(byte_shift / 2 + width - 1)
                } else {
                    acc
                }
            },
        );
        self.range_bus
            .range_check(
                even_sign_cell
                    - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_U16_SIGN_BIT as u32),
                U16_BITS - 1,
            )
            .eval(builder, even_shift);
        // On odd shifts the top loaded byte is the low byte of the last touched cell: constrain
        // the sign bit at byte granularity.
        self.range_bus
            .range_check(
                cols.touched_cell_bytes[NUM_TOUCHED_CELLS - 1][0]
                    - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_BYTE_SIGN_BIT as u32),
                RV64_BYTE_BITS - 1,
            )
            .eval(builder, odd_shift.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.is_valid * AB::Expr::from_u8(info.opcode as u8),
        );
        let load_shift_amount = info
            .byte_shifts
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &byte_shift)| {
                acc + flags[i].clone() * AB::Expr::from_usize(byte_shift)
            });

        let sign_extend = cols.data_most_sig_bit * AB::Expr::from_u32(u16::MAX as u32);
        let write_data = std::array::from_fn(|i| {
            if i >= width {
                return cols.is_valid * sign_extend.clone();
            }
            // Even shifts move whole cells. All odd shifts share one slot-indexed term: loaded
            // cell `i` is the high byte of overlapped cell `i` next to the low byte of
            // overlapped cell `i + 1`.
            let even_term = info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 0 {
                        acc + flags[case_idx].clone() * read_full(byte_shift / 2 + i)
                    } else {
                        acc
                    }
                },
            );
            even_term
                + odd_shift.clone()
                    * (cols.touched_cell_bytes[i][1]
                        + cols.touched_cell_bytes[i + 1][0]
                            * AB::Expr::from_u32(1 << RV64_BYTE_BITS))
        });
        AdapterAirContext {
            to_pc: None,
            reads: cols.read_data.map(|block| block.map(Into::into)).into(),
            writes: [write_data].into(),
            instruction: LoadInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                shift_amount: load_shift_amount,
                load_cross: cols.cross.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct LoadSignExtendFiller<
    A = Rv64LoadAdapterFiller,
    const LOAD_WIDTH: usize = LOAD_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = 3,
    const NUM_TOUCHED_CELLS: usize = 3,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_TOUCHED_CELLS: usize>
    LoadSignExtendFiller<A, LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
{
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_sign_extend_info::<LOAD_WIDTH>().byte_shifts),
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_TOUCHED_CELLS: usize>
    TraceFiller<F>
    for LoadSignExtendFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, SELECTOR_WIDTH, NUM_TOUCHED_CELLS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadSignExtendCoreCols::width() elements.
        let (mut adapter_row, mut core_row) = unsafe {
            row_slice
                .split_at_mut_unchecked(<Rv64LoadAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        let adapter_record: &Rv64LoadAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid LoadRecord written by the executor during trace
        // generation.
        let record: &LoadRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data;
        let core_row: &mut LoadSignExtendCoreCols<F, SELECTOR_WIDTH, NUM_TOUCHED_CELLS> =
            core_row.borrow_mut();
        let case_idx = load_sign_extend_info::<LOAD_WIDTH>()
            .byte_shifts
            .iter()
            .position(|&byte_shift| byte_shift == shift)
            .expect("invalid signed load shift");

        let width = LOAD_WIDTH / 2;
        let read_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        let touched_cell_bytes: [[u16; 2]; NUM_TOUCHED_CELLS] = if shift % 2 == 1 {
            std::array::from_fn(|j| {
                let cell = read_full[shift / 2 + j];
                [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
            })
        } else {
            [[0; 2]; NUM_TOUCHED_CELLS]
        };
        for cell_bytes in &touched_cell_bytes {
            self.bitwise_lookup_chip
                .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
        }

        let sign_bit = if shift.is_multiple_of(2) {
            let sign_cell = read_full[shift / 2 + width - 1];
            let bit = sign_cell & RV64_U16_SIGN_BIT;
            self.range_checker_chip
                .add_count((sign_cell - bit) as u32, U16_BITS - 1);
            bit != 0
        } else {
            let sign_byte = touched_cell_bytes[NUM_TOUCHED_CELLS - 1][0];
            let bit = sign_byte & RV64_BYTE_SIGN_BIT;
            self.range_checker_chip
                .add_count((sign_byte - bit) as u32, RV64_BYTE_BITS - 1);
            bit != 0
        };

        core_row.touched_cell_bytes = touched_cell_bytes.map(|bytes| bytes.map(F::from_u16));
        core_row.read_data = read_data.map(|block| block.map(F::from_u16));
        core_row.data_most_sig_bit = F::from_bool(sign_bit);
        core_row.cross = F::from_bool(shift + LOAD_WIDTH > 2 * BLOCK_FE_WIDTH);
        core_row.is_valid = F::ONE;
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
