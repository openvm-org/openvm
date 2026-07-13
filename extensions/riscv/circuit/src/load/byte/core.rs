use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
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
        shift_encoder, u16_cell_byte, LoadInstruction, Rv64LoadAdapterRecord,
        Rv64LoadByteAdapterFiller, RV64_BYTE_BITS,
    },
    load::common::LoadRecord,
};

pub(crate) const LOAD_BYTE_SELECTOR_WIDTH: usize = 3;

/// Handles unsigned byte loads by decomposing the selected u16 cell and zero-extending the chosen
/// byte.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadByteCoreCols<T> {
    pub selector: [T; LOAD_BYTE_SELECTOR_WIDTH],
    pub read_cell_bytes: [T; 2],
    pub read_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadByteCoreCols<u8>)]
pub struct LoadByteCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl LoadByteCoreAir {
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            offset,
            encoder: shift_encoder::<LOAD_BYTE_SELECTOR_WIDTH>(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for LoadByteCoreAir {
    fn width(&self) -> usize {
        LoadByteCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for LoadByteCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for LoadByteCoreAir
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
        let cols: &LoadByteCoreCols<AB::Var> = (*local_core).borrow();
        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

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
        let write_data = std::array::from_fn(|i| {
            if i == 0 {
                selected_byte.clone()
            } else {
                AB::Expr::ZERO
            }
        });
        let shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(shift)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(LOADBU as u8),
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
                is_valid,
                opcode: expected_opcode,
                shift_amount,
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
pub struct LoadByteFiller<A = Rv64LoadByteAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A> LoadByteFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder::<LOAD_BYTE_SELECTOR_WIDTH>(),
            bitwise_lookup_chip,
        }
    }
}

impl<F> TraceFiller<F> for LoadByteFiller<Rv64LoadByteAdapterFiller>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadByteCoreCols::width() elements.
        let (mut adapter_row, mut core_row) = unsafe {
            row_slice
                .split_at_mut_unchecked(<Rv64LoadByteAdapterFiller as AdapterTraceFiller<F>>::WIDTH)
        };
        let adapter_record: &Rv64LoadAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid LoadRecord written by the executor during trace
        // generation.
        let record: &LoadRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data[0];
        let core_row: &mut LoadByteCoreCols<F> = core_row.borrow_mut();

        let read_cell = read_data[shift / 2];
        let read_cell_bytes = [u16_cell_byte(read_cell, 0), u16_cell_byte(read_cell, 1)];
        self.bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_bytes = read_cell_bytes.map(F::from_u16);
        core_row.read_data = read_data.map(F::from_u16);
        let pt: [u32; LOAD_BYTE_SELECTOR_WIDTH] =
            self.encoder.get_flag_pt(shift).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
