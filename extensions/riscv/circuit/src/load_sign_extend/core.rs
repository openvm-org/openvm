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
        shift_encoder, u16_cell_byte, LoadInstruction, Rv64LoadMultiByteAdapterFiller,
        Rv64LoadMultiByteAdapterRecord, BYTE_SHIFT_SELECTOR_WIDTH, LOAD_WIDTH_HALFWORD,
        LOAD_WIDTH_WORD, NUM_BYTE_SHIFTS, RV64_BYTE_BITS, RV64_BYTE_SIGN_BIT, RV64_U16_SIGN_BIT,
        U16_BITS,
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

/// Handles signed halfword and word loads at any byte offset.
///
/// Even offsets select whole u16 cells. Odd offsets decompose the overlapped cells and recombine
/// adjacent bytes. The last loaded byte supplies the sign.
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
    /// even shifts. The corresponding high bytes are derived in the AIR.
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
    // First byte offset at which the load reaches the next memory block.
    const FIRST_CROSSING_SHIFT: usize = MEMORY_BLOCK_BYTES - LOAD_WIDTH + 1;

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

        // cross = Σ flag[s] over shifts `s` where `s + LOAD_WIDTH > 8`.
        let cross = flags[Self::FIRST_CROSSING_SHIFT..]
            .iter()
            .fold(AB::Expr::ZERO, |acc, flag| acc + flag.clone());

        // Cell `k` of the two consecutive memory blocks.
        let read_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.read_data[0][cell]
            } else {
                cols.read_data[1][cell - BLOCK_FE_WIDTH]
            }
        };
        // odd_cells[j] = Σᵢ flag[2i + 1] * read_full(i + j).
        let odd_cells: [AB::Expr; NUM_OVERLAP_CELLS] = std::array::from_fn(|j| {
            flags
                .iter()
                .skip(1)
                .step_by(2)
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (cell_offset, flag)| {
                    acc + flag.clone() * read_full(cell_offset + j)
                })
        });

        // odd_cells[j] = overlap_lo_bytes[j] + 2^8 * overlap_hi_bytes[j].
        // Both sides are zero on even shifts.
        let inv_2_pow_8 = AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        let overlap_hi_bytes: [AB::Expr; NUM_OVERLAP_CELLS] = std::array::from_fn(|j| {
            (odd_cells[j].clone() - cols.overlap_lo_bytes[j]) * inv_2_pow_8
        });
        for (&lo, hi) in cols.overlap_lo_bytes.iter().zip(overlap_hi_bytes.iter()) {
            self.bitwise_lookup_bus
                .send_range(lo, hi.clone())
                .eval(builder, is_valid.clone());
        }

        // even_sign_cell = Σᵢ flag[2i] * read_full(i + width - 1).
        let even_sign_cell = flags
            .iter()
            .step_by(2)
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (cell_offset, flag)| {
                acc + flag.clone() * read_full(cell_offset + width - 1)
            });
        // sign_cell = even_sign_cell + 2^8 * overlap_lo_bytes[last].
        let sign_cell = even_sign_cell
            + cols.overlap_lo_bytes[NUM_OVERLAP_CELLS - 1]
                * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        self.range_bus
            .range_check(
                sign_cell - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_U16_SIGN_BIT as u32),
                U16_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(load_sign_extend_opcode::<LOAD_WIDTH>() as u8),
        );
        // load_shift_amount = Σₛ s * flag[s].
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
            // even_term[i] = Σₖ flag[2k] * read_full(k + i).
            let even_term = flags
                .iter()
                .step_by(2)
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (cell_offset, flag)| {
                    acc + flag.clone() * read_full(cell_offset + i)
                });
            // result[i] = even_term[i] + overlap_hi_bytes[i]
            //             + 2^8 * overlap_lo_bytes[i + 1].
            even_term
                + overlap_hi_bytes[i].clone()
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
    A = Rv64LoadMultiByteAdapterFiller,
    const LOAD_WIDTH: usize = LOAD_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = BYTE_SHIFT_SELECTOR_WIDTH,
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
    for LoadSignExtendFiller<
        Rv64LoadMultiByteAdapterFiller,
        LOAD_WIDTH,
        SELECTOR_WIDTH,
        NUM_OVERLAP_CELLS,
    >
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadSignExtendCoreCols::width() elements.
        let (mut adapter_row, mut core_row) = unsafe {
            row_slice.split_at_mut_unchecked(
                <Rv64LoadMultiByteAdapterFiller as AdapterTraceFiller<F>>::WIDTH,
            )
        };
        let adapter_record: &Rv64LoadMultiByteAdapterRecord =
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
                .add_count(((sign_byte - bit) as u32) << RV64_BYTE_BITS, U16_BITS - 1);
            bit != 0
        };

        core_row.overlap_lo_bytes = overlap_lo_bytes.map(F::from_u16);
        core_row.read_data = read_data.map(|block| block.map(F::from_u16));
        core_row.data_most_sig_bit = F::from_bool(sign_bit);
        let pt: &[u32; SELECTOR_WIDTH] = self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*pt).map(F::from_u32);
    }
}
