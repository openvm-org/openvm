use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceFiller, TraceFiller,
        VmAdapterInterface, VmCoreAir, BLOCK_FE_WIDTH,
    },
    system::memory::MemoryAuxColsFactory,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::STOREB;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        set_u16_cell_byte, shift_encoder, u16_cell_byte, Rv64StoreByteAdapterCols,
        Rv64StoreByteAdapterFiller, Rv64StoreByteAdapterRecord, StoreByteInstruction,
        BYTE_SHIFT_SELECTOR_WIDTH, RV64_BYTE_BITS,
    },
    store::common::{store_write_data, StoreByteRecord},
};

/// Handles byte stores by replacing one byte in the previous memory block and preserving all other
/// bytes.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreByteCoreCols<T> {
    pub selector: [T; BYTE_SHIFT_SELECTOR_WIDTH],
    /// Low byte of the first source register cell — the stored byte. The cell's high byte is
    /// derived in the AIR.
    pub read_lo_byte: T,
    /// Low byte of the selected previous memory cell. The high byte is derived in the AIR.
    pub prev_cell_lo_byte: T,
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
            encoder: shift_encoder(),
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
    I::Reads: From<([AB::Expr; BLOCK_FE_WIDTH], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[AB::Expr; BLOCK_FE_WIDTH]>,
    I::ProcessedInstruction: From<StoreByteInstruction<AB::Expr>>,
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

        // read_data[0] = read_lo_byte + 2^8 * read_hi_byte.
        let inv_2_pow_8 = AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        let read_hi_byte = (cols.read_data[0] - cols.read_lo_byte) * inv_2_pow_8;
        self.bitwise_lookup_bus
            .send_range(cols.read_lo_byte, read_hi_byte)
            .eval(builder, is_valid.clone());
        // selected_prev_cell = Σᵢ (flag[2i] + flag[2i + 1]) * prev_data[i].
        let selected_prev_cell = flags
            .chunks_exact(2)
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (cell, flags)| {
                acc + (flags[0].clone() + flags[1].clone()) * cols.prev_data[cell]
            });
        // selected_prev_cell = prev_cell_lo_byte + 2^8 * prev_cell_hi_byte.
        let prev_cell_hi_byte = (selected_prev_cell - cols.prev_cell_lo_byte) * inv_2_pow_8;
        self.bitwise_lookup_bus
            .send_range(cols.prev_cell_lo_byte, prev_cell_hi_byte)
            .eval(builder, is_valid.clone());

        let write_data = std::array::from_fn(|i| {
            is_valid.clone() * cols.prev_data[i]
                + flags[2 * i].clone() * (cols.read_lo_byte - cols.prev_cell_lo_byte)
                + flags[2 * i + 1].clone()
                    * (cols.read_lo_byte * AB::Expr::from_u32(1 << RV64_BYTE_BITS)
                        - cols.prev_data[i]
                        + cols.prev_cell_lo_byte)
        });
        // shift_amount = Σₛ s * flag[s].
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
            reads: (
                cols.prev_data.map(Into::into),
                cols.read_data.map(Into::into),
            )
                .into(),
            writes: write_data.into(),
            instruction: StoreByteInstruction {
                is_valid,
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
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder(),
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
        let adapter_record: &Rv64StoreByteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid StoreByteRecord written by the executor during trace
        // generation.
        let record: &StoreByteRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data;
        let prev_data = record.prev_data;
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
        core_row.prev_cell_lo_byte = F::from_u16(prev_cell_bytes[0]);
        debug_assert_eq!(
            store_write_data(STOREB, read_data, [prev_data, [0; BLOCK_FE_WIDTH]], shift)[0]
                [cell_shift],
            set_u16_cell_byte(prev_data[cell_shift], byte_idx, read_lo_byte)
        );

        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
        let pt: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] = self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*pt).map(F::from_u32);
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
