use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, Postflight, PostflightError, PostflightStep, VmAdapterInterface,
        VmCoreAir, BLOCK_FE_WIDTH,
    },
    system::memory::MemoryAuxColsFactory,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::LoadStoreOpcode::STOREB;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        shift_encoder, u16_cell_byte, StoreByteAdapterCols, StoreByteAdapterFiller,
        StoreByteInstruction, BYTE_BITS, BYTE_SHIFT_SELECTOR_WIDTH,
    },
    store::common::store_write_data,
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
        _from_pc_idx: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &StoreByteCoreCols<AB::Var> = (*local_core).borrow();
        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        // read_data[0] = read_lo_byte + 2^8 * read_hi_byte.
        let inv_2_pow_8 = AB::F::from_u32(1 << BYTE_BITS).inverse();
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
                    * (cols.read_lo_byte * AB::Expr::from_u32(1 << BYTE_BITS) - cols.prev_data[i]
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
            to_pc_idx: None,
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
pub struct StoreByteFiller<A = StoreByteAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
}

impl<A> StoreByteFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl StoreByteFiller<StoreByteAdapterFiller> {
    pub(super) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut StoreByteAdapterCols<F>,
        core_row: &mut StoreByteCoreCols<F>,
    ) -> Result<(), PostflightError> {
        let (read_data, prev_data, shift) = self.adapter.replay(
            postflight,
            step,
            mem_helper,
            adapter_row,
            |read_data, prev_data, shift| {
                store_write_data(STOREB, read_data, [prev_data, [0; BLOCK_FE_WIDTH]], shift)[0]
            },
        )?;
        let cell_shift = shift / 2;

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
        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
        let flag_point: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] =
            self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*flag_point).map(F::from_u32);
        Ok(())
    }
}
