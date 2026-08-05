use std::borrow::Borrow;

use openvm_circuit::arch::{
    AdapterAirContext, VmCoreAir, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES, U16_CELL_SIZE,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::SharedVariableRangeCheckerChip,
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::{
    RevealInstruction, RevealAdapterAirInterface, RevealAdapterFiller, REVEAL_ACCESS_WIDTH,
    REVEAL_VALUE_CELLS,
};
use crate::adapters::{u16_cell_byte, BYTE_BITS};

const NUM_REVEAL_SHIFTS: usize = MEMORY_BLOCK_BYTES;
const REVEAL_SHIFT_SELECTOR_WIDTH: usize = 3;
const REVEAL_SHIFT_SELECTOR_MAX_DEGREE: u32 = 2;

fn reveal_shift_encoder() -> Encoder {
    Encoder::new(NUM_REVEAL_SHIFTS, REVEAL_SHIFT_SELECTOR_MAX_DEGREE, true)
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RevealCoreCols<T> {
    pub selector: [T; REVEAL_SHIFT_SELECTOR_WIDTH],
    pub src_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [[T; BLOCK_FE_WIDTH]; 2],
    pub src_lo_bytes: [T; REVEAL_VALUE_CELLS],
    pub prev_bound_bytes: [T; 2],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(RevealCoreCols<u8>)]
pub struct RevealCoreAir {
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

#[derive(Clone)]
pub struct RevealFiller {
    pub(crate) adapter: RevealAdapterFiller,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
}

impl RevealCoreAir {
    const FIRST_CROSSING_SHIFT: usize = MEMORY_BLOCK_BYTES - REVEAL_ACCESS_WIDTH + 1;

    pub fn new(bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            encoder: reveal_shift_encoder(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for RevealCoreAir {
    fn width(&self) -> usize {
        RevealCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RevealCoreAir {}

impl<AB: InteractionBuilder> VmCoreAir<AB, RevealAdapterAirInterface> for RevealCoreAir {
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, RevealAdapterAirInterface> {
        let cols: &RevealCoreCols<AB::Var> = local_core.borrow();

        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        let crosses_block = flags[Self::FIRST_CROSSING_SHIFT..]
            .iter()
            .fold(AB::Expr::ZERO, |acc, flag| acc + flag.clone());
        let odd_shift = flags
            .iter()
            .skip(1)
            .step_by(2)
            .fold(AB::Expr::ZERO, |acc, flag| acc + flag.clone());

        let prev_full = |cell: usize| {
            if cell < BLOCK_FE_WIDTH {
                cols.prev_data[0][cell]
            } else {
                cols.prev_data[1][cell - BLOCK_FE_WIDTH]
            }
        };

        let inv_2_pow_8 = AB::F::from_u32(1 << BYTE_BITS).inverse();
        let src_hi_bytes: [AB::Expr; REVEAL_VALUE_CELLS] =
            std::array::from_fn(|i| (cols.src_data[i] - cols.src_lo_bytes[i]) * inv_2_pow_8);
        for (&lo, hi) in cols.src_lo_bytes.iter().zip(src_hi_bytes.iter()) {
            self.bitwise_lookup_bus
                .send_range(lo, hi.clone())
                .eval(builder, odd_shift.clone());
        }

        let prev_bound_cells: [AB::Expr; 2] = std::array::from_fn(|which| {
            flags.iter().skip(1).step_by(2).enumerate().fold(
                AB::Expr::ZERO,
                |acc, (cell_offset, flag)| {
                    acc + flag.clone() * prev_full(cell_offset + which * REVEAL_VALUE_CELLS)
                },
            )
        });
        let first_cell_hi = (prev_bound_cells[0].clone() - cols.prev_bound_bytes[0]) * inv_2_pow_8;
        let last_cell_lo = prev_bound_cells[1].clone()
            - cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS);
        self.bitwise_lookup_bus
            .send_range(cols.prev_bound_bytes[0], first_cell_hi)
            .eval(builder, odd_shift.clone());
        self.bitwise_lookup_bus
            .send_range(last_cell_lo, cols.prev_bound_bytes[1])
            .eval(builder, odd_shift.clone());

        let opcode = VmCoreAir::<AB, RevealAdapterAirInterface>::expr_to_global_expr(
            self,
            is_valid.clone() * AB::Expr::from_u8(RevealOpcode::REVEAL as u8),
        );
        let shift_amount = flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                acc + flag.clone() * AB::Expr::from_usize(byte_shift)
            });

        let write_data: [[AB::Expr; BLOCK_FE_WIDTH]; 2] = std::array::from_fn(|block| {
            std::array::from_fn(|cell_in_block| {
                let cell = block * BLOCK_FE_WIDTH + cell_in_block;
                flags
                    .iter()
                    .enumerate()
                    .fold(AB::Expr::ZERO, |acc, (byte_shift, flag)| {
                        let first = byte_shift / U16_CELL_SIZE;
                        let term = if byte_shift.is_multiple_of(U16_CELL_SIZE) {
                            if cell >= first && cell < first + REVEAL_VALUE_CELLS {
                                cols.src_data[cell - first].into()
                            } else {
                                prev_full(cell).into()
                            }
                        } else if cell < first || cell > first + REVEAL_VALUE_CELLS {
                            prev_full(cell).into()
                        } else if cell == first {
                            cols.prev_bound_bytes[0]
                                + cols.src_lo_bytes[0] * AB::Expr::from_u32(1 << BYTE_BITS)
                        } else if cell == first + REVEAL_VALUE_CELLS {
                            src_hi_bytes[REVEAL_VALUE_CELLS - 1].clone()
                                + cols.prev_bound_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS)
                        } else {
                            src_hi_bytes[cell - first - 1].clone()
                                + cols.src_lo_bytes[cell - first]
                                    * AB::Expr::from_u32(1 << BYTE_BITS)
                        };
                        acc + flag.clone() * term
                    })
            })
        });

        AdapterAirContext {
            to_pc: None,
            reads: (
                cols.prev_data.map(|block| block.map(Into::into)),
                cols.src_data.map(Into::into),
            ),
            writes: write_data,
            instruction: RevealInstruction {
                is_valid,
                opcode,
                shift_amount,
                crosses_block,
            },
        }
    }

    fn start_offset(&self) -> usize {
        RevealOpcode::CLASS_OFFSET
    }
}

impl RevealFiller {
    pub fn new(
        pointer_max_bits: usize,
        range_checker: SharedVariableRangeCheckerChip,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ) -> Self {
        Self {
            adapter: RevealAdapterFiller::new(pointer_max_bits, range_checker),
            encoder: reveal_shift_encoder(),
            bitwise_lookup_chip,
        }
    }

    pub(crate) fn fill_core_row<F: PrimeField32>(
        &self,
        shift: usize,
        src_data: [u16; BLOCK_FE_WIDTH],
        prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
        core_row: &mut RevealCoreCols<F>,
    ) {
        debug_assert!(shift < NUM_REVEAL_SHIFTS, "invalid reveal shift {shift}");
        let prev_full: [u16; 2 * BLOCK_FE_WIDTH] =
            std::array::from_fn(|cell| prev_data[cell / BLOCK_FE_WIDTH][cell % BLOCK_FE_WIDTH]);
        let (src_lo_bytes, prev_bound_cells) = if shift % U16_CELL_SIZE == 1 {
            let lo_bytes = std::array::from_fn(|i| u16_cell_byte(src_data[i], 0));
            let bound_cells = std::array::from_fn(|which| {
                let cell = prev_full[shift / U16_CELL_SIZE + which * REVEAL_VALUE_CELLS];
                [u16_cell_byte(cell, 0), u16_cell_byte(cell, 1)]
            });
            for (i, lo) in lo_bytes.iter().enumerate() {
                self.bitwise_lookup_chip
                    .request_range(*lo as u32, u16_cell_byte(src_data[i], 1) as u32);
            }
            for cell_bytes in &bound_cells {
                self.bitwise_lookup_chip
                    .request_range(cell_bytes[0] as u32, cell_bytes[1] as u32);
            }
            (lo_bytes, bound_cells)
        } else {
            ([0; REVEAL_VALUE_CELLS], [[0; 2]; 2])
        };

        core_row.src_lo_bytes = src_lo_bytes.map(F::from_u16);
        core_row.prev_bound_bytes =
            [prev_bound_cells[0][0], prev_bound_cells[1][1]].map(F::from_u16);
        core_row.src_data = src_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(|block| block.map(F::from_u16));
        let flag_point: &[u32; REVEAL_SHIFT_SELECTOR_WIDTH] =
            self.encoder.flag_pt(shift).try_into().unwrap();
        core_row.selector = (*flag_point).map(F::from_u32);
    }
}
