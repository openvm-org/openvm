use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::STORED;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{
        u16_cell_byte, Rv64StoreAdapterCols, Rv64StoreAdapterFiller, Rv64StoreAdapterRecord,
        StoreInstruction, RV64_BYTE_BITS,
    },
    store::common::StoreRecord,
};

pub const STORE_DOUBLEWORD_SELECTOR_WIDTH: usize = 3;
const NUM_CASES: usize = 8;
const SELECTOR_MAX_DEGREE: u32 = 2;
/// A doubleword store writes `WIDTH_CELLS = 4` value cells (the whole source register).
const WIDTH_CELLS: usize = 4;

fn encoder() -> Encoder {
    let encoder = Encoder::new(NUM_CASES, SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), STORE_DOUBLEWORD_SELECTOR_WIDTH);
    encoder
}

/// Doubleword-store core with a column-reduced trace layout.
///
/// A doubleword store writes the entire 8-byte source register, so the value *is* the whole `rs2`
/// read. Rather than committing `rs2` both as cells (`read_data`) and as bytes (`value_bytes`),
/// this core commits only `value_bytes` (the byte decomposition of all four `rs2` cells,
/// unconditionally) and reconstructs `rs2` for the register read as `lo + 2^8 * hi` per cell — a
/// fixed-position, degree-1 recomposition, since `rs2` cells are not shifted. That saves the 4
/// `read_data` columns.
///
/// `prev_data` (the previous contents of both written blocks) stays: the offline checker needs the
/// previous value of every written cell. `prev_bound_bytes` holds the two preserved boundary bytes;
/// the overwritten boundary bytes are derived in the AIR.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct StoreDoublewordCoreCols<T> {
    pub selector: [T; STORE_DOUBLEWORD_SELECTOR_WIDTH],
    /// Previous contents of the two written memory blocks.
    pub prev_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Byte decompositions `[lo, hi]` of the four `rs2` value cells.
    pub value_bytes: [[T; 2]; WIDTH_CELLS],
    /// The bytes preserved by an odd-shift store: the low byte of the first overlapped memory cell
    /// and the high byte of the last. All-zero on even shifts.
    pub prev_bound_bytes: [T; 2],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(StoreDoublewordCoreCols<u8>)]
pub struct StoreDoublewordCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl StoreDoublewordCoreAir {
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            offset,
            encoder: encoder(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for StoreDoublewordCoreAir {
    fn width(&self) -> usize {
        StoreDoublewordCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for StoreDoublewordCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for StoreDoublewordCoreAir
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
        let cols: &StoreDoublewordCoreCols<AB::Var> = (*local_core).borrow();

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        let shift_flag = |shift: usize| flags[shift].clone();
        let byte_scale = AB::Expr::from_u32(1 << RV64_BYTE_BITS);

        let cross = (1..NUM_CASES).fold(AB::Expr::ZERO, |acc, s| acc + shift_flag(s));

        // `rs2` value cell `i`, reconstructed from its committed byte decomposition (fixed
        // position, no shift). The register read below pins these to the true `rs2` cells, and the
        // range checks make the decomposition unique.
        let value_cell =
            |i: usize| cols.value_bytes[i][0] + cols.value_bytes[i][1] * byte_scale.clone();
        for cell_bytes in cols.value_bytes.iter() {
            self.bitwise_lookup_bus
                .send_range(cell_bytes[0], cell_bytes[1])
                .eval(builder, is_valid.clone());
        }

        // Cell `k` of the two consecutive previous memory blocks.
        let prev_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.prev_data[0][cell]
            } else {
                cols.prev_data[1][cell - BLOCK_FE_WIDTH]
            }
        };
        // The first and last overlapped memory cells, flag-selected per odd shift; zero on even
        // shifts and invalid rows.
        let prev_bound_cell = |which: usize| {
            (0..NUM_CASES).fold(AB::Expr::ZERO, |acc, byte_shift| {
                if byte_shift % 2 == 1 {
                    acc + shift_flag(byte_shift) * prev_full(byte_shift / 2 + which * WIDTH_CELLS)
                } else {
                    acc
                }
            })
        };
        // The overwritten boundary-cell bytes are derived from the overlapped cells; range checking
        // them completes the decompositions of both boundary cells (and forces the preserved bytes
        // to zero on even shifts).
        let first_cell_hi = (prev_bound_cell(0) - cols.prev_bound_bytes[0])
            * AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        let last_cell_lo = prev_bound_cell(1) - cols.prev_bound_bytes[1] * byte_scale.clone();
        self.bitwise_lookup_bus
            .send_range(cols.prev_bound_bytes[0], first_cell_hi)
            .eval(builder, is_valid.clone());
        self.bitwise_lookup_bus
            .send_range(last_cell_lo, cols.prev_bound_bytes[1])
            .eval(builder, is_valid.clone());

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(STORED as u8),
        );
        let shift_amount = (0..NUM_CASES).fold(AB::Expr::ZERO, |acc, s| {
            acc + shift_flag(s) * AB::Expr::from_usize(s)
        });

        // Contents of both written blocks. Even shifts splice whole value cells between preserved
        // cells; odd shifts splice value bytes, with the two boundary cells mixing a preserved byte
        // and a value byte.
        let write_data: [[AB::Expr; BLOCK_FE_WIDTH]; 2] = std::array::from_fn(|block| {
            std::array::from_fn(|k| {
                let cell = block * BLOCK_FE_WIDTH + k;
                (0..NUM_CASES).fold(AB::Expr::ZERO, |acc, byte_shift| {
                    let first = byte_shift / 2;
                    let term = if byte_shift % 2 == 0 {
                        if cell >= first && cell < first + WIDTH_CELLS {
                            value_cell(cell - first)
                        } else {
                            prev_full(cell).into()
                        }
                    } else if cell < first || cell > first + WIDTH_CELLS {
                        prev_full(cell).into()
                    } else if cell == first {
                        cols.prev_bound_bytes[0] + cols.value_bytes[0][0] * byte_scale.clone()
                    } else if cell == first + WIDTH_CELLS {
                        cols.value_bytes[WIDTH_CELLS - 1][1]
                            + cols.prev_bound_bytes[1] * byte_scale.clone()
                    } else {
                        cols.value_bytes[cell - first - 1][1]
                            + cols.value_bytes[cell - first][0] * byte_scale.clone()
                    };
                    acc + shift_flag(byte_shift) * term
                })
            })
        });

        let read_data: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(value_cell);
        AdapterAirContext {
            to_pc: None,
            reads: (cols.prev_data.map(|block| block.map(Into::into)), read_data).into(),
            writes: write_data.into(),
            instruction: StoreInstruction {
                is_valid,
                opcode: expected_opcode,
                shift_amount,
                store_cross: cross,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct StoreDoublewordFiller<A = Rv64StoreAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A> StoreDoublewordFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl<F> TraceFiller<F> for StoreDoublewordFiller<Rv64StoreAdapterFiller>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // StoreDoublewordCoreCols::width() elements.
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
        let core_row: &mut StoreDoublewordCoreCols<F> = core_row.borrow_mut();

        let prev_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        // The value bytes are the unconditional decomposition of the four rs2 cells; they feed both
        // the register-read reconstruction and (on odd shifts) the write splice.
        let value_bytes: [[u16; 2]; WIDTH_CELLS] = std::array::from_fn(|i| {
            [
                u16_cell_byte(read_data[i], 0),
                u16_cell_byte(read_data[i], 1),
            ]
        });
        let prev_bound_cells: [[u16; 2]; 2] = if shift % 2 == 1 {
            std::array::from_fn(|which| {
                let cell = prev_full[shift / 2 + which * WIDTH_CELLS];
                [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
            })
        } else {
            [[0; 2]; 2]
        };
        for cell_bytes in value_bytes.iter().chain(prev_bound_cells.iter()) {
            self.bitwise_lookup_chip
                .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
        }

        core_row.value_bytes = value_bytes.map(|bytes| bytes.map(F::from_u16));
        // Only the preserved bytes are materialized: the low byte of the first overlapped cell and
        // the high byte of the last.
        core_row.prev_bound_bytes =
            [prev_bound_cells[0][0], prev_bound_cells[1][1]].map(F::from_u16);
        core_row.prev_data = prev_data.map(|block| block.map(F::from_u16));
        let pt: [u32; STORE_DOUBLEWORD_SELECTOR_WIDTH] =
            self.encoder.get_flag_pt(shift).try_into().unwrap();
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
