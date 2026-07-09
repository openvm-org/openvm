use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    encoder::Encoder, var_range::SharedVariableRangeCheckerChip, AlignedBorrow, ColumnsAir,
    StructReflection, StructReflectionHelper, SubAir,
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
        Rv64StoreAdapterCols, Rv64StoreAdapterFiller, Rv64StoreAdapterRecord, StoreInstruction,
        STORE_WIDTH_DOUBLEWORD, STORE_WIDTH_HALFWORD, STORE_WIDTH_WORD,
    },
    store::common::StoreRecord,
};

const SELECTOR_MAX_DEGREE: u32 = 2;

#[derive(Clone, Copy)]
pub(crate) struct WidthAlignedCase {
    opcode: Rv64LoadStoreOpcode,
    byte_shift: usize,
}

impl WidthAlignedCase {
    fn cell_shift(self) -> usize {
        self.byte_shift / 2
    }
}

fn encoder<const NUM_CASES: usize, const SELECTOR_WIDTH: usize>() -> Encoder {
    let encoder = Encoder::new(NUM_CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

const STORE_DOUBLEWORD_CASES: [WidthAlignedCase; 1] = [WidthAlignedCase {
    opcode: STORED,
    byte_shift: 0,
}];

const STORE_WORD_CASES: [WidthAlignedCase; 2] = [
    WidthAlignedCase {
        opcode: STOREW,
        byte_shift: 0,
    },
    WidthAlignedCase {
        opcode: STOREW,
        byte_shift: 4,
    },
];

const STORE_HALFWORD_CASES: [WidthAlignedCase; 4] = [
    WidthAlignedCase {
        opcode: STOREH,
        byte_shift: 0,
    },
    WidthAlignedCase {
        opcode: STOREH,
        byte_shift: 2,
    },
    WidthAlignedCase {
        opcode: STOREH,
        byte_shift: 4,
    },
    WidthAlignedCase {
        opcode: STOREH,
        byte_shift: 6,
    },
];

pub(crate) fn store_width_aligned_cases<const STORE_WIDTH: usize>() -> &'static [WidthAlignedCase] {
    match STORE_WIDTH {
        STORE_WIDTH_DOUBLEWORD => &STORE_DOUBLEWORD_CASES,
        STORE_WIDTH_WORD => &STORE_WORD_CASES,
        STORE_WIDTH_HALFWORD => &STORE_HALFWORD_CASES,
        _ => unreachable!("unsupported width for width-aligned store"),
    }
}

/// Handles halfword, word, and doubleword stores. The core combines source register data with the
/// previous memory block so bytes outside the store width stay unchanged.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreWidthAlignedCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreWidthAlignedCoreCols<u8, SELECTOR_WIDTH>)]
pub struct StoreWidthAlignedCoreAir<
    const STORE_WIDTH: usize,
    const NUM_CASES: usize,
    const SELECTOR_WIDTH: usize,
> {
    pub offset: usize,
    encoder: Encoder,
}

impl<const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    StoreWidthAlignedCoreAir<STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
{
    pub fn new(offset: usize) -> Self {
        debug_assert_eq!(store_width_aligned_cases::<STORE_WIDTH>().len(), NUM_CASES);
        Self {
            offset,
            encoder: encoder::<NUM_CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F: Field, const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    BaseAir<F> for StoreWidthAlignedCoreAir<STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        StoreWidthAlignedCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    BaseAirWithPublicValues<F>
    for StoreWidthAlignedCoreAir<STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
{
}

impl<AB, I, const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    VmCoreAir<AB, I> for StoreWidthAlignedCoreAir<STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([AB::Var; BLOCK_FE_WIDTH], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<StoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &StoreWidthAlignedCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let cases = store_width_aligned_cases::<STORE_WIDTH>();
        let width = STORE_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        builder.assert_eq(cols.is_valid, is_valid.clone());

        let expected_opcode = cases
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * AB::Expr::from_u8(case.opcode as u8)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);
        let shift_amount = cases
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * AB::Expr::from_usize(case.byte_shift)
            });

        let write_data = std::array::from_fn(|i| {
            cases
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (case_idx, case)| {
                    let shift = case.cell_shift();
                    let term = if i >= shift && i < shift + width {
                        cols.read_data[i - shift].into()
                    } else {
                        cols.prev_data[i].into()
                    };
                    acc + flags[case_idx].clone() * term
                })
        });
        AdapterAirContext {
            to_pc: None,
            reads: (cols.prev_data, cols.read_data.map(Into::into)).into(),
            writes: [write_data].into(),
            instruction: StoreInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                shift_amount,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct StoreWidthAlignedFiller<
    A = Rv64StoreAdapterFiller,
    const STORE_WIDTH: usize = STORE_WIDTH_WORD,
    const NUM_CASES: usize = 2,
    const SELECTOR_WIDTH: usize = 1,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
}

impl<A, const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    StoreWidthAlignedFiller<A, STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        debug_assert_eq!(store_width_aligned_cases::<STORE_WIDTH>().len(), NUM_CASES);
        Self {
            adapter,
            offset,
            encoder: encoder::<NUM_CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F, const STORE_WIDTH: usize, const NUM_CASES: usize, const SELECTOR_WIDTH: usize>
    TraceFiller<F>
    for StoreWidthAlignedFiller<Rv64StoreAdapterFiller, STORE_WIDTH, NUM_CASES, SELECTOR_WIDTH>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // StoreWidthAlignedCoreCols::width() elements.
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
        let core_row: &mut StoreWidthAlignedCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let cases = store_width_aligned_cases::<STORE_WIDTH>();
        let case_idx = cases
            .iter()
            .position(|case| case.byte_shift == shift)
            .expect("invalid width-aligned store shift");

        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
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
