use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::*;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        u16_cell_byte, LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord,
        RV64_BYTE_BITS, RV64_BYTE_SIGN_BIT,
    },
    load::LoadRecord,
};

const LOAD_SIGN_EXTEND_BYTE_NUM_CASES: usize = 8;
const LOAD_SIGN_EXTEND_BYTE_SELECTOR_MAX_DEGREE: u32 = 2;
pub(crate) const LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH: usize = 3;

fn encoder() -> Encoder {
    let encoder = Encoder::new(
        LOAD_SIGN_EXTEND_BYTE_NUM_CASES,
        LOAD_SIGN_EXTEND_BYTE_SELECTOR_MAX_DEGREE,
        true,
    );
    debug_assert_eq!(encoder.width(), LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH);
    encoder
}

/// Handles signed byte loads by decomposing the selected u16 cell and sign-extending the chosen
/// byte.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadSignExtendByteCoreCols<T> {
    pub selector: [T; LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH],
    /// Kept as a degree-1 copy of the selector validity.
    pub is_valid: T,
    /// The sign bit that is extended to the remaining cells.
    pub data_most_sig_bit: T,
    pub read_cell_bytes: [T; 2],
    pub read_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadSignExtendByteCoreCols<u8>)]
pub struct LoadSignExtendByteCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
    range_bus: VariableRangeCheckerBus,
}

impl LoadSignExtendByteCoreAir {
    pub fn new(
        offset: usize,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        range_bus: VariableRangeCheckerBus,
    ) -> Self {
        Self {
            offset,
            encoder: encoder(),
            bitwise_lookup_bus,
            range_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for LoadSignExtendByteCoreAir {
    fn width(&self) -> usize {
        LoadSignExtendByteCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for LoadSignExtendByteCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for LoadSignExtendByteCoreAir
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
        let cols: &LoadSignExtendByteCoreCols<AB::Var> = (*local_core).borrow();
        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        builder.assert_eq(cols.is_valid, is_valid.clone());
        builder.assert_bool(cols.data_most_sig_bit);

        self.bitwise_lookup_bus
            .send_range(cols.read_cell_bytes[0], cols.read_cell_bytes[1])
            .eval(builder, is_valid.clone());

        let read_cell = cols.read_cell_bytes[0]
            + cols.read_cell_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        let expected_read_cell = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                acc + flag.clone() * cols.read_data[shift / 2]
            });
        builder.assert_eq(read_cell, expected_read_cell);

        let selected_byte = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                let byte = if shift % 2 == 0 {
                    cols.read_cell_bytes[0].into()
                } else {
                    cols.read_cell_bytes[1].into()
                };
                acc + flag.clone() * byte
            });
        // Constrain that data_most_sig_bit matches the selected source byte.
        self.range_bus
            .range_check(
                selected_byte.clone()
                    - cols.data_most_sig_bit * AB::Expr::from_u32(RV64_BYTE_SIGN_BIT as u32),
                RV64_BYTE_BITS - 1,
            )
            .eval(builder, is_valid.clone());

        let sign_cell = cols.data_most_sig_bit * AB::Expr::from_u32(u16::MAX as u32);
        let write_data = std::array::from_fn(|i| {
            if i == 0 {
                selected_byte.clone() + cols.data_most_sig_bit * AB::Expr::from_u32(0xff00)
            } else {
                sign_cell.clone()
            }
        });
        let load_shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(shift)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(LOADB as u8),
        );

        AdapterAirContext {
            to_pc: None,
            // A byte load never crosses a block boundary, so the second block is never read.
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
pub struct LoadSignExtendByteFiller<A = Rv64LoadAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<A> LoadSignExtendByteFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder(),
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
}

impl<F> TraceFiller<F> for LoadSignExtendByteFiller<Rv64LoadAdapterFiller>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadSignExtendByteCoreCols::width() elements.
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
        let core_row: &mut LoadSignExtendByteCoreCols<F> = core_row.borrow_mut();

        let read_cell = read_data[shift / 2];
        let read_cell_bytes = [u16_cell_byte(read_cell, 0), u16_cell_byte(read_cell, 1)];
        self.bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_bytes = read_cell_bytes.map(F::from_u16);

        let byte = read_cell_bytes[shift % 2];
        let sign_bit = byte & RV64_BYTE_SIGN_BIT;
        self.range_checker_chip
            .add_count((byte - sign_bit) as u32, RV64_BYTE_BITS - 1);
        core_row.data_most_sig_bit = F::from_bool(sign_bit != 0);
        core_row.read_data = read_data.map(F::from_u16);
        core_row.is_valid = F::ONE;
        let pt: [u32; LOAD_SIGN_EXTEND_BYTE_SELECTOR_WIDTH] =
            self.encoder.get_flag_pt(shift).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
