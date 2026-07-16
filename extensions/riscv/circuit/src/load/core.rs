use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
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
        Rv64LoadAdapterRecord, LOAD_WIDTH_DOUBLEWORD, LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD,
        NUM_BYTE_SHIFTS, RV64_BYTE_BITS,
    },
    load::common::LoadRecord,
};

/// The single opcode handled by the load chip of the given width.
pub(crate) fn load_opcode<const LOAD_WIDTH: usize>() -> Rv64LoadStoreOpcode {
    match LOAD_WIDTH {
        LOAD_WIDTH_DOUBLEWORD => LOADD,
        LOAD_WIDTH_WORD => LOADWU,
        LOAD_WIDTH_HALFWORD => LOADHU,
        _ => unreachable!("unsupported width for load"),
    }
}

/// Handles unsigned halfword, word, and doubleword loads. Each supported byte shift is encoded
/// as a separate selector case.
///
/// Even shifts move whole u16 cells. Odd shifts additionally use `overlap_lo_bytes`: the low
/// bytes of the `LOAD_WIDTH / 2 + 1` consecutive cells overlapped by the load. Each overlapped
/// cell's high byte is derived in the AIR as `(cell - lo) * 2^-8` and range checked together
/// with the low byte, which makes every overlapped-cell decomposition unique; result cell `i`
/// then recomposes as `hi_i + 2^8 * lo_{i+1}`.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Two consecutive 8-byte memory blocks; the second is used only when the access crosses a
    /// block boundary.
    pub read_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Low bytes of the `LOAD_WIDTH / 2 + 1` cells overlapped by an odd-shift load. All-zero on
    /// even shifts.
    pub overlap_lo_bytes: [T; NUM_OVERLAP_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadCoreCols<u8, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>)]
pub struct LoadCoreAir<
    const LOAD_WIDTH: usize,
    const SELECTOR_WIDTH: usize,
    const NUM_OVERLAP_CELLS: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        debug_assert_eq!(NUM_OVERLAP_CELLS, LOAD_WIDTH / 2 + 1);
        Self {
            offset,
            encoder: shift_encoder::<SELECTOR_WIDTH>(),
            bitwise_lookup_bus,
        }
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > BaseAir<F> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
    fn width(&self) -> usize {
        LoadCoreCols::<F, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>::width()
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > BaseAirWithPublicValues<F> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
}

impl<
        AB,
        I,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_OVERLAP_CELLS: usize,
    > VmCoreAir<AB, I> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
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
        let cols: &LoadCoreCols<AB::Var, SELECTOR_WIDTH, NUM_OVERLAP_CELLS> =
            (*local_core).borrow();
        let width = LOAD_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

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

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(load_opcode::<LOAD_WIDTH>() as u8),
        );
        let shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(byte_shift)
            });

        let write_data = std::array::from_fn(|i| {
            if i >= width {
                return AB::Expr::ZERO;
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
                shift_amount,
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
pub struct LoadFiller<
    A = Rv64LoadAdapterFiller,
    const LOAD_WIDTH: usize = LOAD_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = 3,
    const NUM_OVERLAP_CELLS: usize = 3,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    LoadFiller<A, LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
{
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder::<SELECTOR_WIDTH>(),
            bitwise_lookup_chip,
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_OVERLAP_CELLS: usize>
    TraceFiller<F>
    for LoadFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, SELECTOR_WIDTH, NUM_OVERLAP_CELLS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadCoreCols::width() elements.
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
        let core_row: &mut LoadCoreCols<F, SELECTOR_WIDTH, NUM_OVERLAP_CELLS> =
            core_row.borrow_mut();
        debug_assert!(shift < NUM_BYTE_SHIFTS, "invalid load shift {shift}");

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

        core_row.overlap_lo_bytes = overlap_lo_bytes.map(F::from_u16);
        core_row.read_data = read_data.map(|block| block.map(F::from_u16));
        let pt: &[u32; SELECTOR_WIDTH] = self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*pt).map(F::from_u32);
    }
}
