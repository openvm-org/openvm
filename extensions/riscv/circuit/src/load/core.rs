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
        LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord, LOAD_WIDTH_DOUBLEWORD,
        LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD,
    },
    load::common::LoadRecord,
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

fn encoder<const CASES: usize, const SELECTOR_WIDTH: usize>() -> Encoder {
    let encoder = Encoder::new(CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

const DOUBLEWORD_LOAD_CASES: [WidthAlignedCase; 1] = [WidthAlignedCase {
    opcode: LOADD,
    byte_shift: 0,
}];
const WORD_LOAD_CASES: [WidthAlignedCase; 2] = [
    WidthAlignedCase {
        opcode: LOADWU,
        byte_shift: 0,
    },
    WidthAlignedCase {
        opcode: LOADWU,
        byte_shift: 4,
    },
];
const HALFWORD_LOAD_CASES: [WidthAlignedCase; 4] = [
    WidthAlignedCase {
        opcode: LOADHU,
        byte_shift: 0,
    },
    WidthAlignedCase {
        opcode: LOADHU,
        byte_shift: 2,
    },
    WidthAlignedCase {
        opcode: LOADHU,
        byte_shift: 4,
    },
    WidthAlignedCase {
        opcode: LOADHU,
        byte_shift: 6,
    },
];
pub(crate) fn load_width_aligned_cases<const LOAD_WIDTH: usize>() -> &'static [WidthAlignedCase] {
    match LOAD_WIDTH {
        LOAD_WIDTH_DOUBLEWORD => &DOUBLEWORD_LOAD_CASES,
        LOAD_WIDTH_WORD => &WORD_LOAD_CASES,
        LOAD_WIDTH_HALFWORD => &HALFWORD_LOAD_CASES,
        _ => unreachable!("unsupported width for width-aligned load"),
    }
}

/// Handles unsigned halfword, word, and doubleword loads. Each `(opcode, shift)` pair is encoded
/// as a separate selector case.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadWidthAlignedCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    /// The 8-byte memory block containing the effective load address.
    pub read_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadWidthAlignedCoreCols<u8, SELECTOR_WIDTH>)]
pub struct LoadWidthAlignedCoreAir<
    const LOAD_WIDTH: usize,
    const CASES: usize,
    const SELECTOR_WIDTH: usize,
> {
    pub offset: usize,
    encoder: Encoder,
}

impl<const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    LoadWidthAlignedCoreAir<LOAD_WIDTH, CASES, SELECTOR_WIDTH>
{
    pub fn new(offset: usize) -> Self {
        debug_assert_eq!(load_width_aligned_cases::<LOAD_WIDTH>().len(), CASES);
        Self {
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F: Field, const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize> BaseAir<F>
    for LoadWidthAlignedCoreAir<LOAD_WIDTH, CASES, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        LoadWidthAlignedCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    BaseAirWithPublicValues<F> for LoadWidthAlignedCoreAir<LOAD_WIDTH, CASES, SELECTOR_WIDTH>
{
}

impl<AB, I, const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    VmCoreAir<AB, I> for LoadWidthAlignedCoreAir<LOAD_WIDTH, CASES, SELECTOR_WIDTH>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[AB::Expr; BLOCK_FE_WIDTH]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<LoadInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadWidthAlignedCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let cases = load_width_aligned_cases::<LOAD_WIDTH>();
        let width = LOAD_WIDTH / 2;

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
                    let term = if i < width {
                        cols.read_data[i + shift].into()
                    } else {
                        AB::Expr::ZERO
                    };
                    acc + flags[case_idx].clone() * term
                })
        });
        AdapterAirContext {
            to_pc: None,
            reads: cols.read_data.map(Into::into).into(),
            writes: [write_data].into(),
            instruction: LoadInstruction {
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
pub struct LoadWidthAlignedFiller<
    A = Rv64LoadAdapterFiller,
    const LOAD_WIDTH: usize = LOAD_WIDTH_WORD,
    const CASES: usize = 2,
    const SELECTOR_WIDTH: usize = 1,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
}

impl<A, const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    LoadWidthAlignedFiller<A, LOAD_WIDTH, CASES, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        debug_assert_eq!(load_width_aligned_cases::<LOAD_WIDTH>().len(), CASES);
        Self {
            adapter,
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const CASES: usize, const SELECTOR_WIDTH: usize> TraceFiller<F>
    for LoadWidthAlignedFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, CASES, SELECTOR_WIDTH>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadWidthAlignedCoreCols::width() elements.
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
        let core_row: &mut LoadWidthAlignedCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let cases = load_width_aligned_cases::<LOAD_WIDTH>();
        let case_idx = cases
            .iter()
            .position(|case| case.byte_shift == shift)
            .expect("invalid width-aligned load shift");

        core_row.read_data = read_data.map(F::from_u16);
        core_row.is_valid = F::ONE;
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
