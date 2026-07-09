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

/// Static description of a width-aligned store chip: the single opcode it handles and the byte
/// shifts it supports, each shift encoded as a separate selector case.
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
    byte_shifts: &[0],
};
const STORE_WORD_INFO: StoreInfo = StoreInfo {
    opcode: STOREW,
    byte_shifts: &[0, 4],
};
const STORE_HALFWORD_INFO: StoreInfo = StoreInfo {
    opcode: STOREH,
    byte_shifts: &[0, 2, 4, 6],
};
pub(crate) fn store_info<const STORE_WIDTH: usize>() -> StoreInfo {
    match STORE_WIDTH {
        STORE_WIDTH_DOUBLEWORD => STORE_DOUBLEWORD_INFO,
        STORE_WIDTH_WORD => STORE_WORD_INFO,
        STORE_WIDTH_HALFWORD => STORE_HALFWORD_INFO,
        _ => unreachable!("unsupported width for width-aligned store"),
    }
}

/// Handles halfword, word, and doubleword stores. The core combines source register data with the
/// previous memory block so bytes outside the store width stay unchanged.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreCoreCols<u8, SELECTOR_WIDTH>)]
pub struct StoreCoreAir<const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize> {
    pub offset: usize,
    encoder: Encoder,
}

impl<const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize>
    StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH>
{
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(store_info::<STORE_WIDTH>().byte_shifts),
        }
    }
}

impl<F: Field, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize> BaseAir<F>
    for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        StoreCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize> BaseAirWithPublicValues<F>
    for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH>
{
}

impl<AB, I, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize> VmCoreAir<AB, I>
    for StoreCoreAir<STORE_WIDTH, SELECTOR_WIDTH>
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
        let cols: &StoreCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let info = store_info::<STORE_WIDTH>();
        let width = STORE_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        builder.assert_eq(cols.is_valid, is_valid.clone());

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

        let write_data = std::array::from_fn(|i| {
            info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    let shift = byte_shift / 2;
                    let term = if i >= shift && i < shift + width {
                        cols.read_data[i - shift].into()
                    } else {
                        cols.prev_data[i].into()
                    };
                    acc + flags[case_idx].clone() * term
                },
            )
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
pub struct StoreFiller<
    A = Rv64StoreAdapterFiller,
    const STORE_WIDTH: usize = STORE_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = 1,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
}

impl<A, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize>
    StoreFiller<A, STORE_WIDTH, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(store_info::<STORE_WIDTH>().byte_shifts),
        }
    }
}

impl<F, const STORE_WIDTH: usize, const SELECTOR_WIDTH: usize> TraceFiller<F>
    for StoreFiller<Rv64StoreAdapterFiller, STORE_WIDTH, SELECTOR_WIDTH>
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
        let core_row: &mut StoreCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let case_idx = store_info::<STORE_WIDTH>()
            .byte_shifts
            .iter()
            .position(|&byte_shift| byte_shift == shift)
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
