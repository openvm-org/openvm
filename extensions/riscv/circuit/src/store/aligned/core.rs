use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    encoder::Encoder, var_range::SharedVariableRangeCheckerChip, AlignedBorrow, ColumnsAir,
    StructReflection, StructReflectionHelper, SubAir,
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
        store_adapter_context, Rv64StoreAdapterCols, Rv64StoreAdapterFiller, StoreInstruction,
    },
    store::common::{StoreRecord, KIND_DOUBLEWORD, KIND_HALFWORD, KIND_WORD},
};

const SELECTOR_MAX_DEGREE: u32 = 2;

#[derive(Clone, Copy)]
pub(crate) struct AlignedCase {
    opcode: Rv64LoadStoreOpcode,
    byte_shift: usize,
}

impl AlignedCase {
    fn cell_shift(self) -> usize {
        self.byte_shift / 2
    }
}

fn access_cells<const KIND: usize>() -> usize {
    match KIND {
        KIND_DOUBLEWORD => 4,
        KIND_WORD => 2,
        KIND_HALFWORD => 1,
        _ => unreachable!("unsupported aligned store kind"),
    }
}

fn encoder<const CASES: usize, const SELECTOR_WIDTH: usize>() -> Encoder {
    let encoder = Encoder::new(CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

const DOUBLEWORD_STORE_CASES: [AlignedCase; 1] = [AlignedCase {
    opcode: STORED,
    byte_shift: 0,
}];

const WORD_STORE_CASES: [AlignedCase; 2] = [
    AlignedCase {
        opcode: STOREW,
        byte_shift: 0,
    },
    AlignedCase {
        opcode: STOREW,
        byte_shift: 4,
    },
];

const HALFWORD_STORE_CASES: [AlignedCase; 4] = [
    AlignedCase {
        opcode: STOREH,
        byte_shift: 0,
    },
    AlignedCase {
        opcode: STOREH,
        byte_shift: 2,
    },
    AlignedCase {
        opcode: STOREH,
        byte_shift: 4,
    },
    AlignedCase {
        opcode: STOREH,
        byte_shift: 6,
    },
];

pub(crate) fn store_aligned_cases<const KIND: usize>() -> &'static [AlignedCase] {
    match KIND {
        KIND_DOUBLEWORD => &DOUBLEWORD_STORE_CASES,
        KIND_WORD => &WORD_STORE_CASES,
        KIND_HALFWORD => &HALFWORD_STORE_CASES,
        _ => unreachable!("unsupported aligned store kind"),
    }
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreAlignedCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    pub is_valid: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreAlignedCoreCols<u8, SELECTOR_WIDTH>)]
pub struct StoreAlignedCoreAir<const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> {
    pub offset: usize,
    encoder: Encoder,
}

impl<const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    StoreAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
    pub fn new(offset: usize) -> Self {
        debug_assert_eq!(store_aligned_cases::<KIND>().len(), CASES);
        Self {
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F: Field, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> BaseAir<F>
    for StoreAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        StoreAlignedCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    BaseAirWithPublicValues<F> for StoreAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
{
}

impl<AB, I, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> VmCoreAir<AB, I>
    for StoreAlignedCoreAir<KIND, CASES, SELECTOR_WIDTH>
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
        let cols: &StoreAlignedCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let cases = store_aligned_cases::<KIND>();
        let width = access_cells::<KIND>();

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
        store_adapter_context::<AB, I>(
            cols.is_valid.into(),
            expected_opcode,
            shift_amount,
            cols.read_data,
            cols.prev_data,
            write_data,
        )
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct StoreAlignedFiller<
    A = Rv64StoreAdapterFiller,
    const KIND: usize = KIND_WORD,
    const CASES: usize = 2,
    const SELECTOR_WIDTH: usize = 1,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
}

impl<A, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize>
    StoreAlignedFiller<A, KIND, CASES, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        debug_assert_eq!(store_aligned_cases::<KIND>().len(), CASES);
        Self {
            adapter,
            offset,
            encoder: encoder::<CASES, SELECTOR_WIDTH>(),
        }
    }
}

impl<F, A, const KIND: usize, const CASES: usize, const SELECTOR_WIDTH: usize> TraceFiller<F>
    for StoreAlignedFiller<A, KIND, CASES, SELECTOR_WIDTH>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &StoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let opcode = Rv64LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let shift = record.shift_amount as usize;
        let read_data = record.read_data;
        let prev_data = record.prev_data;
        let core_row: &mut StoreAlignedCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let cases = store_aligned_cases::<KIND>();
        let case_idx = cases
            .iter()
            .position(|case| case.opcode == opcode && case.byte_shift == shift)
            .expect("invalid aligned store opcode/shift");

        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
        core_row.is_valid = F::ONE;
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }

    fn fill_dummy_trace_row(&self, row_slice: &mut [F]) {
        let (adapter_row, _) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        let adapter_row: &mut Rv64StoreAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.mem_as = F::from_u32(2);
    }
}
