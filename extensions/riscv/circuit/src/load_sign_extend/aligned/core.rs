use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        load_adapter_context, LoadInstruction, Rv64LoadAdapterFiller, RV64_U16_SIGN_BIT, U16_BITS,
    },
    load::LoadRecord,
    load_sign_extend::common::{KIND_HALFWORD, KIND_WORD},
};

const SELECTOR_MAX_DEGREE: u32 = 2;

#[derive(Clone, Copy)]
pub(crate) struct SignedAlignedCase {
    opcode: Rv64LoadStoreOpcode,
    byte_shift: usize,
}

impl SignedAlignedCase {
    fn cell_shift(self) -> usize {
        self.byte_shift / 2
    }
}

const WORD_CASES: [SignedAlignedCase; 2] = [
    SignedAlignedCase {
        opcode: LOADW,
        byte_shift: 0,
    },
    SignedAlignedCase {
        opcode: LOADW,
        byte_shift: 4,
    },
];

const HALFWORD_CASES: [SignedAlignedCase; 4] = [
    SignedAlignedCase {
        opcode: LOADH,
        byte_shift: 0,
    },
    SignedAlignedCase {
        opcode: LOADH,
        byte_shift: 2,
    },
    SignedAlignedCase {
        opcode: LOADH,
        byte_shift: 4,
    },
    SignedAlignedCase {
        opcode: LOADH,
        byte_shift: 6,
    },
];

pub(crate) fn signed_aligned_cases<const KIND: usize>() -> &'static [SignedAlignedCase] {
    match KIND {
        KIND_WORD => &WORD_CASES,
        KIND_HALFWORD => &HALFWORD_CASES,
        _ => unreachable!("unsupported signed aligned load kind"),
    }
}

fn access_cells<const KIND: usize>() -> usize {
    match KIND {
        KIND_WORD => 2,
        KIND_HALFWORD => 1,
        _ => unreachable!("unsupported signed aligned load kind"),
    }
}

fn encoder<const CASES: usize, const SELECTOR_WIDTH: usize>() -> Encoder {
    let encoder = Encoder::new(CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendAlignedCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    pub is_valid: T,
    pub data_most_sig_bit: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendAlignedCoreCols<u8, SELECTOR_WIDTH>)]
pub struct LoadSignExtendAlignedCoreAir<
    const KIND: usize,
    const CASES: usize,
    const SELECTOR_WIDTH: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    range_bus: VariableRangeCheckerBus,
}

impl<const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    LoadSignExtendAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
    pub fn new(offset: usize, range_bus: VariableRangeCheckerBus) -> Self {
        debug_assert_eq!(signed_aligned_cases::<KIND>().len(), CASES);
        Self {
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
            range_bus,
        }
    }
}

impl<F: Field, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> BaseAir<F>
    for LoadSignExtendAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        LoadSignExtendAlignedCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    BaseAirWithPublicValues<F> for LoadSignExtendAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
}

impl<AB, I, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> VmCoreAir<AB, I>
    for LoadSignExtendAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
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
        let cols: &LoadSignExtendAlignedCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let cases = signed_aligned_cases::<KIND>();
        let width = access_cells::<KIND>();

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        builder.assert_eq(cols.is_valid, is_valid.clone());
        builder.assert_bool(cols.data_most_sig_bit);

        let sign_cell = cases
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * cols.read_data[case.cell_shift() + width - 1]
            });
        self.range_bus
            .range_check(
                sign_cell - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_U16_SIGN_BIT as u32),
                U16_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = cases
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * AB::Expr::from_u8(case.opcode as u8)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);
        let load_shift_amount = cases
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * AB::Expr::from_usize(case.byte_shift)
            });

        let sign_extend = cols.data_most_sig_bit * AB::Expr::from_u32(u16::MAX as u32);
        let write_data = std::array::from_fn(|i| {
            cases
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (case_idx, case)| {
                    let shift = case.cell_shift();
                    let term = if i < width {
                        cols.read_data[i + shift].into()
                    } else {
                        sign_extend.clone()
                    };
                    acc + flags[case_idx].clone() * term
                })
        });
        load_adapter_context::<AB, I>(
            cols.is_valid.into(),
            expected_opcode,
            load_shift_amount,
            cols.read_data,
            write_data,
        )
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct LoadSignExtendAlignedFiller<
    A = Rv64LoadAdapterFiller,
    const KIND: usize = KIND_WORD,
    const CASES: usize = 2,
    const SELECTOR_WIDTH: usize = 2,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    LoadSignExtendAlignedFiller<A, KIND, CASES, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        debug_assert_eq!(signed_aligned_cases::<KIND>().len(), CASES);
        Self {
            adapter,
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
            range_checker_chip,
        }
    }
}

impl<F, A, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> TraceFiller<F>
    for LoadSignExtendAlignedFiller<A, KIND, CASES, SELECTOR_WIDTH>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &LoadRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let opcode = Rv64LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let shift = record.shift_amount as usize;
        let read_data = record.read_data;
        let core_row: &mut LoadSignExtendAlignedCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let cases = signed_aligned_cases::<KIND>();
        let case_idx = cases
            .iter()
            .position(|case| case.opcode == opcode && case.byte_shift == shift)
            .expect("invalid signed aligned load opcode/shift");

        let width = access_cells::<KIND>();
        let sign_cell = read_data[shift / 2 + width - 1];
        let sign_bit = sign_cell & RV64_U16_SIGN_BIT;
        self.range_checker_chip
            .add_count((sign_cell - sign_bit) as u32, U16_BITS - 1);
        core_row.data_most_sig_bit = F::from_bool(sign_bit != 0);
        core_row.read_data = read_data.map(F::from_u16);
        core_row.is_valid = F::ONE;
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
