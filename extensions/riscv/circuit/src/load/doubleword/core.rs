use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::LOADD;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        u16_cell_byte, LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord,
        RV64_BYTE_BITS,
    },
    load::common::LoadRecord,
};

pub const LOAD_DOUBLEWORD_SELECTOR_WIDTH: usize = 3;
const NUM_CASES: usize = 8;
const SELECTOR_MAX_DEGREE: u32 = 2;
/// A doubleword access spans `NUM_SLOTS = 5` cells at an odd shift (`8 / 2 + 1`).
const NUM_SLOTS: usize = 5;
/// Cells read but never overlapped by the value: `8 - NUM_SLOTS`.
const NUM_NONOVERLAP: usize = 3;

fn encoder() -> Encoder {
    let encoder = Encoder::new(NUM_CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), LOAD_DOUBLEWORD_SELECTOR_WIDTH);
    encoder
}

/// First cell of the `NUM_SLOTS`-cell window for byte shift `shift`.
const fn window_start(shift: usize) -> usize {
    shift / 2
}

/// For byte shift `shift` and block cell position `p`, either the slot index that decomposes it
/// (`Ok`), or the `read_nonoverlap` index that holds it (`Err`). A cell is in the window iff
/// `window_start <= p <= window_start + 4`; the remaining cells are ordered by position into
/// `read_nonoverlap`.
const fn slot_or_nonoverlap(shift: usize, p: usize) -> Result<usize, usize> {
    let c0 = window_start(shift);
    if c0 <= p && p < c0 + NUM_SLOTS {
        Ok(p - c0)
    } else if p < c0 {
        Err(p)
    } else {
        Err(p - NUM_SLOTS)
    }
}

/// Doubleword-load core with a column-reduced trace layout.
///
/// A doubleword load always reads the whole containing 8-byte block (both blocks when it crosses),
/// so the read is *entirely* the loaded value plus a few spectator cells. Instead of committing the
/// two full blocks (`read_data`, 8 cells) alongside the value's byte decomposition, this core
/// commits only:
/// - `cell_bytes`: whole-cell `[lo, hi]` decompositions of the `NUM_SLOTS = 5` cells in the window
///   `[shift/2, shift/2 + 5)`. On every crossing shift all 5 slots hold real read cells (5
///   overlapped on odd shifts; 4 overlapped + 1 spectator on even shifts), so the value's bytes are
///   always available for the register write and every windowed read cell is `lo + 2^8 * hi`.
/// - `read_nonoverlap`: the `NUM_NONOVERLAP = 3` read cells outside the window, kept as plain
///   cells.
///
/// The two 8-byte blocks are reconstructed for the memory bus from these columns, saving 3 columns
/// versus committing `read_data` and the value bytes separately.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadDoublewordCoreCols<T> {
    pub selector: [T; LOAD_DOUBLEWORD_SELECTOR_WIDTH],
    /// Read cells outside the window, ordered by block position.
    pub read_nonoverlap: [T; NUM_NONOVERLAP],
    /// Whole-cell `[lo, hi]` decompositions of the window cells `[shift/2, shift/2 + 5)`.
    pub cell_bytes: [[T; 2]; NUM_SLOTS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadDoublewordCoreCols<u8>)]
pub struct LoadDoublewordCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl LoadDoublewordCoreAir {
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            offset,
            encoder: encoder(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for LoadDoublewordCoreAir {
    fn width(&self) -> usize {
        LoadDoublewordCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for LoadDoublewordCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for LoadDoublewordCoreAir
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
        let cols: &LoadDoublewordCoreCols<AB::Var> = (*local_core).borrow();

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        let shift_flag = |shift: usize| flags[shift].clone();
        let byte_scale = AB::Expr::from_u32(1 << RV64_BYTE_BITS);

        // Range-check every slot's byte pair; this makes each windowed cell's decomposition
        // unique (and its recomposition below a valid u16).
        for slot in cols.cell_bytes.iter() {
            self.bitwise_lookup_bus
                .send_range(slot[0], slot[1])
                .eval(builder, is_valid.clone());
        }

        // Recompose window cell `j` from its byte pair.
        let slot_cell =
            |j: usize| cols.cell_bytes[j][0] + cols.cell_bytes[j][1] * byte_scale.clone();

        // Reconstruct block cell `p` from the slots (windowed cells) and `read_nonoverlap` (the
        // rest), selected by the shift.
        let recon = |p: usize| -> AB::Expr {
            (0..NUM_CASES).fold(AB::Expr::ZERO, |acc, shift| {
                let term = match slot_or_nonoverlap(shift, p) {
                    Ok(j) => slot_cell(j),
                    Err(k) => cols.read_nonoverlap[k].into(),
                };
                acc + shift_flag(shift) * term
            })
        };
        let block0 = std::array::from_fn(&recon);
        let block1 = std::array::from_fn(|p| recon(BLOCK_FE_WIDTH + p));

        // Register write: the four result cells. On even shifts each result cell is a whole window
        // cell; on odd shifts it straddles two adjacent window cells.
        let even_shift = (0..NUM_CASES)
            .step_by(2)
            .fold(AB::Expr::ZERO, |acc, s| acc + shift_flag(s));
        let odd_shift = (1..NUM_CASES)
            .step_by(2)
            .fold(AB::Expr::ZERO, |acc, s| acc + shift_flag(s));
        let write_data = std::array::from_fn(|r| {
            let even_term = slot_cell(r);
            let odd_term = cols.cell_bytes[r][1] + cols.cell_bytes[r + 1][0] * byte_scale.clone();
            even_shift.clone() * even_term + odd_shift.clone() * odd_term
        });

        // A doubleword access crosses the block boundary at every nonzero shift.
        let cross = (1..NUM_CASES).fold(AB::Expr::ZERO, |acc, s| acc + shift_flag(s));
        let shift_amount = (0..NUM_CASES).fold(AB::Expr::ZERO, |acc, s| {
            acc + shift_flag(s) * AB::Expr::from_usize(s)
        });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(LOADD as u8),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [block0, block1].into(),
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
pub struct LoadDoublewordFiller<A = Rv64LoadAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A> LoadDoublewordFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl<F> TraceFiller<F> for LoadDoublewordFiller<Rv64LoadAdapterFiller>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadDoublewordCoreCols::width() elements.
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
        let core_row: &mut LoadDoublewordCoreCols<F> = core_row.borrow_mut();

        let read_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        let c0 = window_start(shift);

        let cell_bytes: [[u16; 2]; NUM_SLOTS] = std::array::from_fn(|j| {
            let cell = read_full[c0 + j];
            [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
        });
        for slot in &cell_bytes {
            self.bitwise_lookup_chip
                .request_range(slot[0] as u32, slot[1] as u32);
        }
        let read_nonoverlap: [u16; NUM_NONOVERLAP] = std::array::from_fn(|k| {
            let p = if k < c0 { k } else { k + NUM_SLOTS };
            read_full[p]
        });

        core_row.cell_bytes = cell_bytes.map(|bytes| bytes.map(F::from_u16));
        core_row.read_nonoverlap = read_nonoverlap.map(F::from_u16);
        let pt: [u32; LOAD_DOUBLEWORD_SELECTOR_WIDTH] =
            self.encoder.get_flag_pt(shift).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
