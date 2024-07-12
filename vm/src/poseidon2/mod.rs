use p3_field::Field;
use p3_field::PrimeField32;

use afs_chips::is_zero::IsZeroAir;
use afs_chips::sub_chip::LocalTraceInstructions;
use columns::{Poseidon2ChipCols, Poseidon2ChipIoCols};
use poseidon2_air::poseidon2::Poseidon2Air;
use poseidon2_air::poseidon2::Poseidon2Config;

use crate::cpu::trace::Instruction;
use crate::cpu::OpCode;
use crate::cpu::OpCode::*;
use crate::poseidon2::columns::Poseidon2ChipAuxCols;
use crate::vm::VirtualMachine;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Poseidon2 chip.
///
/// Carries the requested rows and the underlying subair for subtrace generation.
/// Poseidon2Chip implements its own constraints and interactions.
/// Cached rows are represented as `Poseidon2ChipCols` structs, not flat vectors.
pub struct Poseidon2Chip<const WIDTH: usize, F: Clone> {
    pub air: Poseidon2Air<WIDTH, F>,
    pub rows: Vec<Poseidon2ChipCols<WIDTH, F>>,
}

/// Map VM instructions to Poseidon2IO columns.
fn make_io_cols<F: Field>(
    start_timestamp: usize,
    instruction: Instruction<F>,
) -> Poseidon2ChipIoCols<F> {
    let Instruction {
        opcode,
        op_a,
        op_b,
        op_c,
        d,
        e,
    } = instruction;
    Poseidon2ChipIoCols::<F> {
        is_alloc: F::one(),
        clk: F::from_canonical_usize(start_timestamp),
        a: op_a,
        b: op_b,
        c: op_c,
        d,
        e,
        cmp: F::from_bool(opcode == COMP_POS2),
    }
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    pub fn from_poseidon2_config(config: Poseidon2Config<WIDTH, F>, bus_index: usize) -> Self {
        let air = Poseidon2Air::<WIDTH, F>::from_config(config, bus_index);
        Self { air, rows: vec![] }
    }

    pub fn interaction_width() -> usize {
        7
    }

    pub fn max_accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        3 + (2 * WIDTH)
    }
}

const WIDTH: usize = 16;
impl<F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    /// Key method of Poseidon2Chip.
    ///
    /// Called using `vm` and not `&self`. Reads two chunks from memory and generates a trace row for
    /// the given instruction using the subair, storing it in `rows`. Then, writes output to memory,
    /// truncating if the instruction is a compression.
    pub fn poseidon2_perm<const WORD_SIZE: usize>(
        vm: &mut VirtualMachine<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
        } = instruction;
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);

        let mut timestamp = start_timestamp;

        let addresses = [op_a, op_b, op_c].map(|operand| {
            timestamp += 1;
            vm.memory_chip.read_elem(timestamp - 1, d, operand)
        });

        let data_1: Vec<F> = (0..WIDTH / 2)
            .map(|i| {
                timestamp += 1;
                vm.memory_chip.read_elem(
                    timestamp - 1,
                    e,
                    addresses[0] + F::from_canonical_usize(i),
                )
            })
            .collect();
        let data_2: Vec<F> = (0..WIDTH / 2)
            .map(|i| {
                timestamp += 1;
                vm.memory_chip.read_elem(
                    timestamp - 1,
                    e,
                    addresses[1] + F::from_canonical_usize(i),
                )
            })
            .collect();

        // SAFETY: only allowed because WIDTH constrained to 16 above
        let input_state: [F; WIDTH] = [data_1, data_2].concat().try_into().unwrap();
        let internal = vm.poseidon2_chip.air.generate_trace_row(input_state);
        let output = internal.io.output;
        let is_zero_row = IsZeroAir {}.generate_trace_row(d);
        vm.poseidon2_chip.rows.push(Poseidon2ChipCols {
            io: make_io_cols(start_timestamp, instruction),
            aux: Poseidon2ChipAuxCols {
                addresses,
                d_is_zero: is_zero_row.io.is_zero,
                is_zero_inv: is_zero_row.inv,
                internal,
            },
        });

        let iter_range = if opcode == PERM_POS2 {
            output.iter().enumerate().take(WIDTH)
        } else {
            output.iter().enumerate().take(WIDTH / 2)
        };

        for (i, &output_elem) in iter_range {
            vm.memory_chip.write_elem(
                timestamp,
                e,
                addresses[2] + F::from_canonical_usize(i),
                output_elem,
            );
            timestamp += 1;
        }
    }
}
