use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
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
        LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord, LOAD_WIDTH_HALFWORD,
        LOAD_WIDTH_WORD, RV64_U16_SIGN_BIT, U16_BITS,
    },
    load::LoadRecord,
};

const SELECTOR_MAX_DEGREE: u32 = 2;

/// Static description of a signed width-aligned load chip: the single opcode it handles and the
/// byte shifts it supports, each shift encoded as a separate selector case.
#[derive(Clone, Copy)]
pub(crate) struct LoadSignExtendInfo {
    opcode: Rv64LoadStoreOpcode,
    byte_shifts: &'static [usize],
}

const LOAD_SIGN_EXTEND_WORD_INFO: LoadSignExtendInfo = LoadSignExtendInfo {
    opcode: LOADW,
    byte_shifts: &[0, 4],
};
const LOAD_SIGN_EXTEND_HALFWORD_INFO: LoadSignExtendInfo = LoadSignExtendInfo {
    opcode: LOADH,
    byte_shifts: &[0, 2, 4, 6],
};
pub(crate) fn load_sign_extend_info<const LOAD_WIDTH: usize>() -> LoadSignExtendInfo {
    match LOAD_WIDTH {
        LOAD_WIDTH_WORD => LOAD_SIGN_EXTEND_WORD_INFO,
        LOAD_WIDTH_HALFWORD => LOAD_SIGN_EXTEND_HALFWORD_INFO,
        _ => unreachable!("unsupported width for signed width-aligned load"),
    }
}

fn encoder<const SELECTOR_WIDTH: usize>(byte_shifts: &[usize]) -> Encoder {
    let encoder = Encoder::new(byte_shifts.len(), SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

/// Handles signed halfword and word loads. The core builds the register write from the selected
/// read cells and sign-extends the remaining cells.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendCoreCols<T, const SELECTOR_WIDTH: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    /// The sign bit that is extended to the remaining cells.
    pub data_most_sig_bit: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendCoreCols<u8, SELECTOR_WIDTH>)]
pub struct LoadSignExtendCoreAir<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize> {
    pub offset: usize,
    encoder: Encoder,
    range_bus: VariableRangeCheckerBus,
}

impl<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize>
    LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH>
{
    pub fn new(offset: usize, range_bus: VariableRangeCheckerBus) -> Self {
        Self {
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_sign_extend_info::<LOAD_WIDTH>().byte_shifts),
            range_bus,
        }
    }
}

impl<F: Field, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize> BaseAir<F>
    for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH>
{
    fn width(&self) -> usize {
        LoadSignExtendCoreCols::<F, SELECTOR_WIDTH>::width()
    }
}

impl<F: Field, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize> BaseAirWithPublicValues<F>
    for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH>
{
}

impl<AB, I, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize> VmCoreAir<AB, I>
    for LoadSignExtendCoreAir<LOAD_WIDTH, SELECTOR_WIDTH>
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
        let cols: &LoadSignExtendCoreCols<AB::Var, SELECTOR_WIDTH> = (*local_core).borrow();
        let info = load_sign_extend_info::<LOAD_WIDTH>();
        let width = LOAD_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        builder.assert_eq(cols.is_valid, is_valid.clone());
        builder.assert_bool(cols.data_most_sig_bit);

        // Constrain that data_most_sig_bit matches the selected source sign cell.
        let sign_cell =
            info.byte_shifts
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (i, &byte_shift)| {
                    acc + flags[i].clone() * cols.read_data[byte_shift / 2 + width - 1]
                });
        self.range_bus
            .range_check(
                sign_cell - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_U16_SIGN_BIT as u32),
                U16_BITS - 1,
            )
            .eval(builder, is_valid.clone());

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
            info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    let shift = byte_shift / 2;
                    let term = if i < width {
                        cols.read_data[i + shift].into()
                    } else {
                        sign_extend.clone()
                    };
                    acc + flags[case_idx].clone() * term
                },
            )
        });
        AdapterAirContext {
            to_pc: None,
            // Signed loads only support in-block shifts so far, so the second block is never
            // read.
            reads: [
                cols.read_data.map(Into::into),
                std::array::from_fn(|_| AB::Expr::ZERO),
            ]
            .into(),
            writes: [write_data].into(),
            instruction: LoadInstruction {
                is_valid: cols.is_valid.into(),
                opcode: expected_opcode,
                shift_amount: load_shift_amount,
                load_cross: AB::Expr::ZERO,
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
    const SELECTOR_WIDTH: usize = 2,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize>
    LoadSignExtendFiller<A, LOAD_WIDTH, SELECTOR_WIDTH>
{
    pub fn new(
        adapter: A,
        offset: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_sign_extend_info::<LOAD_WIDTH>().byte_shifts),
            range_checker_chip,
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize> TraceFiller<F>
    for LoadSignExtendFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, SELECTOR_WIDTH>
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
        let read_data = record.read_data[0];
        let core_row: &mut LoadSignExtendCoreCols<F, SELECTOR_WIDTH> = core_row.borrow_mut();
        let case_idx = load_sign_extend_info::<LOAD_WIDTH>()
            .byte_shifts
            .iter()
            .position(|&byte_shift| byte_shift == shift)
            .expect("invalid signed width-aligned load shift");

        let width = LOAD_WIDTH / 2;
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
