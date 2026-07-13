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
        set_u16_cell_byte, shift_encoder, u16_cell_byte, Rv64StoreAdapterRecord,
        Rv64StoreByteAdapterCols, Rv64StoreByteAdapterFiller, StoreInstruction, RV64_BYTE_BITS,
    },
    store::common::{store_write_data, StoreRecord},
};

pub(crate) const STORE_BYTE_SELECTOR_WIDTH: usize = 3;

/// Handles byte stores by replacing one byte in the previous memory block and preserving all other
/// bytes.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreByteCoreCols<T> {
    pub selector: [T; STORE_BYTE_SELECTOR_WIDTH],
    /// Low byte of the first source register cell — the stored byte. The cell's high byte is
    /// derived in the AIR.
    pub read_lo_byte: T,
    pub prev_cell_bytes: [T; 2],
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreByteCoreCols<u8>)]
pub struct StoreByteCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl StoreByteCoreAir {
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            offset,
            encoder: shift_encoder::<STORE_BYTE_SELECTOR_WIDTH>(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for StoreByteCoreAir {
    fn width(&self) -> usize {
        StoreByteCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for StoreByteCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for StoreByteCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([[AB::Expr; BLOCK_FE_WIDTH]; 2], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 2]>,
    I::ProcessedInstruction: From<StoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &StoreByteCoreCols<AB::Var> = (*local_core).borrow();
        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        // The stored byte is the low byte of the first source cell; the cell's high byte is
        // derived from it, and range checking the pair makes the decomposition unique.
        let read_hi_byte = (cols.read_data[0] - cols.read_lo_byte)
            * AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        self.bitwise_lookup_bus
            .send_range(cols.read_lo_byte, read_hi_byte)
            .eval(builder, is_valid.clone());
        self.bitwise_lookup_bus
            .send_range(cols.prev_cell_bytes[0], cols.prev_cell_bytes[1])
            .eval(builder, is_valid.clone());

        let prev_cell = cols.prev_cell_bytes[0]
            + cols.prev_cell_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        let expected_prev_cell = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                acc + flag.clone() * cols.prev_data[shift / 2]
            });
        builder.assert_eq(prev_cell, expected_prev_cell);

        let write_data = std::array::from_fn(|i| {
            flags
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                    let cell_shift = shift / 2;
                    let byte_idx = shift % 2;
                    let term = if i == cell_shift {
                        if byte_idx == 0 {
                            cols.read_lo_byte
                                + cols.prev_cell_bytes[1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        } else {
                            cols.prev_cell_bytes[0]
                                + cols.read_lo_byte * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        }
                    } else {
                        cols.prev_data[i].into()
                    };
                    acc + flag.clone() * term
                })
        });
        let shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(shift)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(STOREB as u8),
        );

        AdapterAirContext {
            to_pc: None,
            // A byte store never crosses a block boundary, so the second block is never
            // touched.
            reads: (
                [
                    cols.prev_data.map(Into::into),
                    std::array::from_fn(|_| AB::Expr::ZERO),
                ],
                cols.read_data.map(Into::into),
            )
                .into(),
            writes: [write_data, std::array::from_fn(|_| AB::Expr::ZERO)].into(),
            instruction: StoreInstruction {
                is_valid,
                opcode: expected_opcode,
                shift_amount,
                store_cross: AB::Expr::ZERO,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct StoreByteFiller<A = Rv64StoreByteAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A> StoreByteFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder::<STORE_BYTE_SELECTOR_WIDTH>(),
            bitwise_lookup_chip,
        }
    }
}

impl<F> TraceFiller<F> for StoreByteFiller<Rv64StoreByteAdapterFiller>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // StoreByteCoreCols::width() elements.
        let (mut adapter_row, mut core_row) = unsafe {
            row_slice.split_at_mut_unchecked(
                <Rv64StoreByteAdapterFiller as AdapterTraceFiller<F>>::WIDTH,
            )
        };
        let adapter_record: &Rv64StoreAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid StoreRecord written by the executor during trace
        // generation.
        let record: &StoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data;
        let prev_data = record.prev_data[0];
        let core_row: &mut StoreByteCoreCols<F> = core_row.borrow_mut();
        let cell_shift = shift / 2;
        let byte_idx = shift % 2;

        // The cell's high byte is derived in the AIR and only range checked here.
        let read_lo_byte = u16_cell_byte(read_data[0], 0);
        self.bitwise_lookup_chip
            .request_range(read_lo_byte as u32, u16_cell_byte(read_data[0], 1) as u32);
        core_row.read_lo_byte = F::from_u16(read_lo_byte);

        let prev_cell_bytes = [
            u16_cell_byte(prev_data[cell_shift], 0),
            u16_cell_byte(prev_data[cell_shift], 1),
        ];
        self.bitwise_lookup_chip
            .request_range(prev_cell_bytes[0] as u32, prev_cell_bytes[1] as u32);
        core_row.prev_cell_bytes = prev_cell_bytes.map(F::from_u16);
        debug_assert_eq!(
            store_write_data(STOREB, read_data, [prev_data, [0; BLOCK_FE_WIDTH]], shift)[0]
                [cell_shift],
            set_u16_cell_byte(prev_data[cell_shift], byte_idx, read_lo_byte)
        );

        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
        let pt: [u32; STORE_BYTE_SELECTOR_WIDTH] =
            self.encoder.get_flag_pt(shift).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }

    fn fill_dummy_trace_row(&self, row_slice: &mut [F]) {
        let (adapter_row, _) = unsafe {
            row_slice.split_at_mut_unchecked(
                <Rv64StoreByteAdapterFiller as AdapterTraceFiller<F>>::WIDTH,
            )
        };
        let adapter_row: &mut Rv64StoreByteAdapterCols<F> = adapter_row.borrow_mut();
        adapter_row.mem_as = F::from_u32(2);
    }
}
