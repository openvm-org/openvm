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
        shift_encoder, u16_cell_byte, LoadInstruction, Rv64LoadAdapterFiller,
        Rv64LoadAdapterRecord, LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD, NUM_BYTE_SHIFTS,
        RV64_BYTE_BITS, RV64_BYTE_SIGN_BIT, RV64_U16_SIGN_BIT, U16_BITS,
    },
    load::LoadRecord,
};

/// The single opcode handled by the signed load chip of the given width.
pub(crate) fn load_sign_extend_opcode<const LOAD_WIDTH: usize>() -> Rv64LoadStoreOpcode {
    match LOAD_WIDTH {
        LOAD_WIDTH_WORD => LOADW,
        LOAD_WIDTH_HALFWORD => LOADH,
        _ => unreachable!("unsupported width for signed load"),
    }
}

/// Handles signed halfword and word loads. Each supported byte shift is encoded as a separate
/// selector case.
///
/// Even shifts move whole u16 cells and take the sign from the high bit of the top cell. Odd
/// shifts additionally use `overlap_lo_bytes`: the low bytes of the `LOAD_WIDTH / 2 + 1`
/// consecutive cells overlapped by the load. Each overlapped cell's high byte is derived in the
/// AIR as `(cell - lo) * 2^-8` and range checked together with the low byte, which makes every
/// overlapped-cell decomposition unique; result cell `i` then recomposes as `hi_i + 2^8 *
/// lo_{i+1}`, and the sign comes from the high bit of the last low byte (the top loaded byte).
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// The sign bit that is extended to the remaining cells.
    pub data_most_sig_bit: T,
    /// Two consecutive 8-byte memory blocks; the second is used only when the access crosses a
    /// block boundary.
    pub read_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Low bytes of the `LOAD_WIDTH / 2 + 1` cells overlapped by an odd-shift load. All-zero on
    /// even shifts.
    pub overlap_lo_bytes: [T; NUM_OVERLAP_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendCoreCols<u8, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>)]
pub struct LoadSignExtendCoreAir<
    const LOAD_WIDTH: usize,
    const SELECTOR_WIDTH: usize,
    const NUM_OVERLAP_CELLS: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    range_bus: VariableRangeCheckerBus,
}

impl<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
    pub fn new(
        offset: usize,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        debug_assert_eq!(NUM_OVERLAP_CELLS, LOAD_WIDTH / 2 + 1);
        Self {
            offset,
            encoder: shift_encoder::<SELECTOR_WIDTH>(),
            bitwise_lookup_bus,
            range_bus,
        }
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > BaseAir<F> for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
    fn width(&self) -> usize {
        LoadSignExtendCoreCols::<F, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>::width()
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > BaseAirWithPublicValues<F>
    for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
}

impl<
        AB,
        I,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > VmCoreAir<AB, I> for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
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
        let cols: &LoadSignExtendCoreCols<AB::Var, SELECTOR_WIDTH, NUM_OVERLAP_CELLS> =
            (*local_core).borrow();
        let width = LOAD_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        builder.assert_bool(cols.data_most_sig_bit);

        let cross = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                if byte_shift + LOAD_WIDTH > 2 * BLOCK_FE_WIDTH {
                    acc + flag.clone()
                } else {
                    acc
                }
            });

        // Cell `k` of the two consecutive memory blocks.
        let read_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.read_data[0][cell]
            } else {
                cols.read_data[1][cell - BLOCK_FE_WIDTH]
            }
        };
        // The j-th cell overlapped by an odd-shift load; zero on even shifts and invalid rows.
        let odd_cell = |j: usize| {
            flags
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                    if byte_shift % 2 == 1 {
                        acc + flag.clone() * read_full(byte_shift / 2 + j)
                    } else {
                        acc
                    }
                })
        };

        // High byte of overlapped cell `j`, derived from its materialized low byte; range
        // checking the `(lo, hi)` pair makes the decomposition of every overlapped cell unique.
        // On even shifts the overlapped-cell sums are zero, which forces the low bytes to zero.
        let inv_2_pow_8 = AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        let overlap_hi_byte = |j: usize| (odd_cell(j) - cols.overlap_lo_bytes[j]) * inv_2_pow_8;
        for j in 0..=width {
            self.bitwise_lookup_bus
                .send_range(cols.overlap_lo_bytes[j], overlap_hi_byte(j))
                .eval(builder, is_valid.clone());
        }

        let (even_shift, odd_shift) = flags.iter().enumerate().fold(
            (AB::Expr::ZERO, AB::Expr::ZERO),
            |(even, odd), (byte_shift, flag)| {
                if byte_shift % 2 == 0 {
                    (even + flag.clone(), odd)
                } else {
                    (even, odd + flag.clone())
                }
            },
        );

        // On even shifts the top loaded byte is the high byte of the selected top cell:
        // constrain the sign bit at u16 granularity.
        let even_sign_cell =
            flags
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                    if byte_shift % 2 == 0 {
                        acc + flag.clone() * read_full(byte_shift / 2 + width - 1)
                    } else {
                        acc
                    }
                });
        self.range_bus
            .range_check(
                even_sign_cell
                    - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_U16_SIGN_BIT as u32),
                U16_BITS - 1,
            )
            .eval(builder, even_shift);
        // On odd shifts the top loaded byte is the last overlapped cell's low byte: constrain
        // the sign bit at byte granularity.
        self.range_bus
            .range_check(
                cols.overlap_lo_bytes[NUM_OVERLAP_CELLS - 1]
                    - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_BYTE_SIGN_BIT as u32),
                RV64_BYTE_BITS - 1,
            )
            .eval(builder, odd_shift.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(load_sign_extend_opcode::<LOAD_WIDTH>() as u8),
        );
        let load_shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(byte_shift)
            });

        let sign_extend = cols.data_most_sig_bit * AB::Expr::from_u32(u16::MAX as u32);
        let write_data = std::array::from_fn(|i| {
            if i >= width {
                return is_valid.clone() * sign_extend.clone();
            }
            // Even shifts move whole cells. All odd shifts share one slot-indexed term: result
            // cell `i` recomposes from the high byte of overlapped cell `i` and the low byte of
            // overlapped cell `i + 1`; both vanish on even shifts.
            let even_term =
                flags
                    .iter()
                    .enumerate()
                    .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                        if byte_shift % 2 == 0 {
                            acc + flag.clone() * read_full(byte_shift / 2 + i)
                        } else {
                            acc
                        }
                    });
            even_term
                + overlap_hi_byte(i)
                + cols.overlap_lo_bytes[i + 1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
        });
        AdapterAirContext {
            to_pc: None,
            reads: cols.read_data.map(|block| block.map(Into::into)).into(),
            writes: [write_data].into(),
            instruction: LoadInstruction {
                is_valid,
                opcode: expected_opcode,
                shift_amount: load_shift_amount,
                load_cross: cross,
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
    const NUM_OVERLAP_CELLS: usize = 3,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    LoadSignExtendFiller<A, LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
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
            encoder: shift_encoder::<SELECTOR_WIDTH>(),
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    TraceFiller<F>
    for LoadSignExtendFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
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
        let core_row: &mut LoadSignExtendCoreCols<F, SELECTOR_WIDTH, NUM_OVERLAP_CELLS> =
            core_row.borrow_mut();
        debug_assert!(shift < NUM_BYTE_SHIFTS, "invalid signed load shift {shift}");

        let width = LOAD_WIDTH / 2;
        let read_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        // The high bytes are derived in the AIR and only range checked here.
        let (overlap_lo_bytes, overlap_hi_bytes): (
            [u16; NUM_OVERLAP_CELLS],
            [u16; NUM_OVERLAP_CELLS],
        ) = if shift % 2 == 1 {
            (
                std::array::from_fn(|j| u16_cell_byte(read_full[shift / 2 + j], 0)),
                std::array::from_fn(|j| u16_cell_byte(read_full[shift / 2 + j], 1)),
            )
        } else {
            ([0; NUM_OVERLAP_CELLS], [0; NUM_OVERLAP_CELLS])
        };
        for (lo, hi) in overlap_lo_bytes.iter().zip(overlap_hi_bytes.iter()) {
            self.bitwise_lookup_chip
                .request_range(*lo as u32, *hi as u32);
        }

        let sign_bit = if shift.is_multiple_of(2) {
            let sign_cell = read_full[shift / 2 + width - 1];
            let bit = sign_cell & RV64_U16_SIGN_BIT;
            self.range_checker_chip
                .add_count((sign_cell - bit) as u32, U16_BITS - 1);
            bit != 0
        } else {
            let sign_byte = overlap_lo_bytes[NUM_OVERLAP_CELLS - 1];
            let bit = sign_byte & RV64_BYTE_SIGN_BIT;
            self.range_checker_chip
                .add_count((sign_byte - bit) as u32, RV64_BYTE_BITS - 1);
            bit != 0
        };

        core_row.overlap_lo_bytes = overlap_lo_bytes.map(F::from_u16);
        core_row.read_data = read_data.map(|block| block.map(F::from_u16));
        core_row.data_most_sig_bit = F::from_bool(sign_bit);
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(shift).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
