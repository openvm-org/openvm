use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
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
        u16_cell_byte, LoadInstruction, Rv64LoadAdapterFiller, Rv64LoadAdapterRecord,
        LOAD_WIDTH_DOUBLEWORD, LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD, RV64_BYTE_BITS,
    },
    load::common::LoadRecord,
};

const SELECTOR_MAX_DEGREE: u32 = 2;

/// Static description of a load chip: the single opcode it handles and the byte shifts it
/// supports, each shift encoded as a separate selector case.
#[derive(Clone, Copy)]
pub(crate) struct LoadInfo {
    opcode: Rv64LoadStoreOpcode,
    byte_shifts: &'static [usize],
}

fn encoder<const SELECTOR_WIDTH: usize>(byte_shifts: &[usize]) -> Encoder {
    let encoder = Encoder::new(byte_shifts.len(), SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}

const LOAD_DOUBLEWORD_INFO: LoadInfo = LoadInfo {
    opcode: LOADD,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
const LOAD_WORD_INFO: LoadInfo = LoadInfo {
    opcode: LOADWU,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
const LOAD_HALFWORD_INFO: LoadInfo = LoadInfo {
    opcode: LOADHU,
    byte_shifts: &[0, 1, 2, 3, 4, 5, 6, 7],
};
pub(crate) fn load_info<const LOAD_WIDTH: usize>() -> LoadInfo {
    match LOAD_WIDTH {
        LOAD_WIDTH_DOUBLEWORD => LOAD_DOUBLEWORD_INFO,
        LOAD_WIDTH_WORD => LOAD_WORD_INFO,
        LOAD_WIDTH_HALFWORD => LOAD_HALFWORD_INFO,
        _ => unreachable!("unsupported width for load"),
    }
}

/// Handles unsigned halfword, word, and doubleword loads. Each supported byte shift is encoded
/// as a separate selector case.
///
/// Even shifts move whole u16 cells. Odd shifts additionally use `loaded_cell_bytes`: byte
/// decompositions of the `LOAD_WIDTH / 2` result cells, whose bytes are also the interior bytes
/// of the `LOAD_WIDTH / 2 + 1` consecutive cells overlapped by the load. The two overlapped-cell
/// bytes outside the loaded range are derived in the AIR instead of being materialized.
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadCoreCols<T, const SELECTOR_WIDTH: usize, const NUM_LOADED_CELLS: usize> {
    pub selector: [T; SELECTOR_WIDTH],
    /// Two consecutive 8-byte memory blocks; the second is used only when the access crosses a
    /// block boundary.
    pub read_data: [[T; BLOCK_FE_WIDTH]; 2],
    /// Byte decompositions `[lo, hi]` of the result cells of an odd-shift load: `[i][0]` is the
    /// high byte of overlapped cell `i` and `[i][1]` is the low byte of overlapped cell `i + 1`,
    /// so result cell `i` is `[i][0] + 2^8 * [i][1]`. All-zero on even shifts.
    pub loaded_cell_bytes: [[T; 2]; NUM_LOADED_CELLS],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadCoreCols<u8, SELECTOR_WIDTH, NUM_LOADED_CELLS>)]
pub struct LoadCoreAir<
    const LOAD_WIDTH: usize,
    const SELECTOR_WIDTH: usize,
    const NUM_LOADED_CELLS: usize,
> {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_LOADED_CELLS: usize>
    LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
{
    pub fn new(offset: usize, bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        debug_assert_eq!(NUM_LOADED_CELLS, LOAD_WIDTH / 2);
        Self {
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_info::<LOAD_WIDTH>().byte_shifts),
            bitwise_lookup_bus,
        }
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_LOADED_CELLS: usize,
    > BaseAir<F> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
{
    fn width(&self) -> usize {
        LoadCoreCols::<F, SELECTOR_WIDTH, NUM_LOADED_CELLS>::width()
    }
}

impl<
        F: Field,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_LOADED_CELLS: usize,
    > BaseAirWithPublicValues<F> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
{
}

impl<
        AB,
        I,
        const LOAD_WIDTH: usize,
        const SELECTOR_WIDTH: usize,
        const NUM_LOADED_CELLS: usize,
    > VmCoreAir<AB, I> for LoadCoreAir<LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
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
        let cols: &LoadCoreCols<AB::Var, SELECTOR_WIDTH, NUM_LOADED_CELLS> = (*local_core).borrow();
        let info = load_info::<LOAD_WIDTH>();
        let width = LOAD_WIDTH / 2;

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);

        let cross = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift + LOAD_WIDTH > 2 * BLOCK_FE_WIDTH {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );

        // Cell `k` of the two consecutive memory blocks.
        let read_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.read_data[0][cell]
            } else {
                cols.read_data[1][cell - BLOCK_FE_WIDTH]
            }
        };
        // The j-th cell overlapped by an odd-shift load; zero on even shifts and invalid rows.
        let odd_cell = |j: usize| {
            info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 1 {
                        acc + flags[case_idx].clone() * read_full(byte_shift / 2 + j)
                    } else {
                        acc
                    }
                },
            )
        };

        for cell_bytes in cols.loaded_cell_bytes.iter() {
            self.bitwise_lookup_bus
                .send_range(cell_bytes[0], cell_bytes[1])
                .eval(builder, is_valid.clone());
        }
        // On an odd shift, interior overlapped cell `j` recomposes from adjacent result-cell
        // bytes; the byte range checks above make the decomposition unique. On even shifts the
        // overlapped-cell sums are zero, which forces the bytes to zero.
        for j in 1..width {
            builder.assert_eq(
                cols.loaded_cell_bytes[j - 1][1]
                    + cols.loaded_cell_bytes[j][0] * AB::Expr::from_u32(1 << RV64_BYTE_BITS),
                odd_cell(j),
            );
        }
        // The two overlapped-cell bytes outside the loaded range are derived from the boundary
        // cells; range checking them completes the decompositions of both boundary cells (and
        // forces the remaining loaded bytes to zero on even shifts).
        let first_cell_lo =
            odd_cell(0) - cols.loaded_cell_bytes[0][0] * AB::Expr::from_u32(1 << RV64_BYTE_BITS);
        let last_cell_hi = (odd_cell(width) - cols.loaded_cell_bytes[width - 1][1])
            * AB::F::from_u32(1 << RV64_BYTE_BITS).inverse();
        self.bitwise_lookup_bus
            .send_range(first_cell_lo, last_cell_hi)
            .eval(builder, is_valid.clone());

        let odd_shift = info.byte_shifts.iter().enumerate().fold(
            AB::Expr::ZERO,
            |acc, (case_idx, &byte_shift)| {
                if byte_shift % 2 == 1 {
                    acc + flags[case_idx].clone()
                } else {
                    acc
                }
            },
        );

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(info.opcode as u8),
        );
        let shift_amount = info
            .byte_shifts
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, &byte_shift)| {
                acc + flags[i].clone() * AB::Expr::from_usize(byte_shift)
            });

        let write_data = std::array::from_fn(|i| {
            if i >= width {
                return AB::Expr::ZERO;
            }
            // Even shifts move whole cells. All odd shifts share one slot-indexed term: result
            // cell `i` recomposes from its own byte decomposition.
            let even_term = info.byte_shifts.iter().enumerate().fold(
                AB::Expr::ZERO,
                |acc, (case_idx, &byte_shift)| {
                    if byte_shift % 2 == 0 {
                        acc + flags[case_idx].clone() * read_full(byte_shift / 2 + i)
                    } else {
                        acc
                    }
                },
            );
            even_term
                + odd_shift.clone()
                    * (cols.loaded_cell_bytes[i][0]
                        + cols.loaded_cell_bytes[i][1] * AB::Expr::from_u32(1 << RV64_BYTE_BITS))
        });
        AdapterAirContext {
            to_pc: None,
            reads: cols.read_data.map(|block| block.map(Into::into)).into(),
            writes: [write_data].into(),
            instruction: LoadInstruction {
                is_valid,
                opcode: expected_opcode,
                shift_amount,
                load_cross: cross,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct LoadFiller<
    A = Rv64LoadAdapterFiller,
    const LOAD_WIDTH: usize = LOAD_WIDTH_WORD,
    const SELECTOR_WIDTH: usize = 3,
    const NUM_LOADED_CELLS: usize = 2,
> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_LOADED_CELLS: usize>
    LoadFiller<A, LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
{
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder::<SELECTOR_WIDTH>(load_info::<LOAD_WIDTH>().byte_shifts),
            bitwise_lookup_chip,
        }
    }
}

impl<F, const LOAD_WIDTH: usize, const SELECTOR_WIDTH: usize, const NUM_LOADED_CELLS: usize>
    TraceFiller<F>
    for LoadFiller<Rv64LoadAdapterFiller, LOAD_WIDTH, SELECTOR_WIDTH, NUM_LOADED_CELLS>
where
    F: PrimeField32,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least the adapter width plus
        // LoadCoreCols::width() elements.
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
        let read_data = record.read_data;
        let core_row: &mut LoadCoreCols<F, SELECTOR_WIDTH, NUM_LOADED_CELLS> =
            core_row.borrow_mut();
        let case_idx = load_info::<LOAD_WIDTH>()
            .byte_shifts
            .iter()
            .position(|&byte_shift| byte_shift == shift)
            .expect("invalid load shift");

        let read_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| read_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        let (loaded_cell_bytes, bound_bytes): ([[u16; 2]; NUM_LOADED_CELLS], [u16; 2]) =
            if shift % 2 == 1 {
                (
                    std::array::from_fn(|i| {
                        [
                            u16_cell_byte(read_full[shift / 2 + i], 1),
                            u16_cell_byte(read_full[shift / 2 + i + 1], 0),
                        ]
                    }),
                    [
                        u16_cell_byte(read_full[shift / 2], 0),
                        u16_cell_byte(read_full[shift / 2 + NUM_LOADED_CELLS], 1),
                    ],
                )
            } else {
                ([[0; 2]; NUM_LOADED_CELLS], [0; 2])
            };
        for cell_bytes in &loaded_cell_bytes {
            self.bitwise_lookup_chip
                .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
        }
        self.bitwise_lookup_chip
            .request_range(bound_bytes[0] as u32, bound_bytes[1] as u32);

        core_row.loaded_cell_bytes = loaded_cell_bytes.map(|bytes| bytes.map(F::from_u16));
        core_row.read_data = read_data.map(|block| block.map(F::from_u16));
        let pt: [u32; SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
