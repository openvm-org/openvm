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
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::LOADBU;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        shift_encoder, u16_cell_byte, LoadByteInstruction, Rv64LoadByteAdapterFiller,
        Rv64LoadByteAdapterRecord, BYTE_SHIFT_SELECTOR_WIDTH, RV64_BYTE_BITS,
    },
    load::common::LoadByteRecord,
};

/// Handles unsigned byte loads by decomposing the selected u16 cell and zero-extending the chosen
/// byte.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadByteCoreCols<T> {
    pub selector: [T; BYTE_SHIFT_SELECTOR_WIDTH],
    /// Low byte of the selected memory cell. The high byte is derived in the AIR.
    pub read_cell_lo_byte: T,
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
            encoder: shift_encoder(),
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
    I::Reads: From<[AB::Expr; BLOCK_FE_WIDTH]>,
    I::Writes: From<[AB::Expr; BLOCK_FE_WIDTH]>,
    I::ProcessedInstruction: From<LoadByteInstruction<AB::Expr>>,
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

        // For cell `i`, flags `2i` and `2i + 1` select its low and high byte, respectively.
        // even_shift_selector = Σᵢ flag[2i].
        // odd_shift_selector = Σᵢ flag[2i + 1].
        // even_selected_cell = Σᵢ flag[2i] * read_data[i].
        // odd_selected_cell = Σᵢ flag[2i + 1] * read_data[i].
        // Keeping the selections separate makes every flag (degree 2) * cell term degree 3.
        let (even_shift_selector, odd_shift_selector, even_selected_cell, odd_selected_cell) =
            flags.chunks_exact(2).enumerate().fold(
                (
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ),
                |(even, odd, even_cell, odd_cell), (cell, flags)| {
                    let even_flag = flags[0].clone();
                    let odd_flag = flags[1].clone();
                    let read_cell = cols.read_data[cell];
                    (
                        even + even_flag.clone(),
                        odd + odd_flag.clone(),
                        even_cell + even_flag * read_cell,
                        odd_cell + odd_flag * read_cell,
                    )
                },
            );
        let inv_2_pow_8 = AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        // selected_cell = lo + 2^8 * hi.
        let selected_cell = even_selected_cell + odd_selected_cell.clone();
        let read_cell_hi_byte = (selected_cell - cols.read_cell_lo_byte) * inv_2_pow_8;
        self.bitwise_lookup_bus
            .send_range(cols.read_cell_lo_byte, read_cell_hi_byte)
            .eval(builder, is_valid.clone());

        // selected_byte = even_shift_selector * lo
        //               + (odd_selected_cell - odd_shift_selector * lo) / 2^8.
        let odd_selected_hi_byte =
            (odd_selected_cell - odd_shift_selector * cols.read_cell_lo_byte) * inv_2_pow_8;
        let selected_byte = even_shift_selector * cols.read_cell_lo_byte + odd_selected_hi_byte;
        let write_data = std::array::from_fn(|i| {
            if i == 0 {
                selected_byte.clone()
            } else {
                AB::Expr::ZERO
            }
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
            is_valid.clone() * AB::Expr::from_u8(LOADBU as u8),
        );

        AdapterAirContext {
            to_pc: None,
            reads: cols.read_data.map(Into::into).into(),
            writes: write_data.into(),
            instruction: LoadByteInstruction {
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
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: shift_encoder(),
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
        let adapter_record: &Rv64LoadByteAdapterRecord =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let shift = adapter_record.shift_amount();
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid LoadByteRecord written by the executor during trace
        // generation.
        let record: &LoadByteRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let read_data = record.read_data;
        let core_row: &mut LoadByteCoreCols<F> = core_row.borrow_mut();

        let read_cell = read_data[shift / 2];
        let read_cell_bytes = [u16_cell_byte(read_cell, 0), u16_cell_byte(read_cell, 1)];
        self.bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_lo_byte = F::from_u16(read_cell_bytes[0]);
        core_row.read_data = read_data.map(F::from_u16);
        let pt: &[u32; BYTE_SHIFT_SELECTOR_WIDTH] = self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*pt).map(F::from_u32);
    }
}
