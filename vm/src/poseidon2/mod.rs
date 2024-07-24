use p3_field::PrimeField32;

use columns::{Poseidon2VmCols, Poseidon2VmIoCols};
use poseidon2_air::poseidon2::Poseidon2Air;
use poseidon2_air::poseidon2::Poseidon2Config;

use crate::cpu::trace::Instruction;
use crate::cpu::OpCode;
use crate::cpu::OpCode::*;
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
pub struct Poseidon2VmAir<const WIDTH: usize, F: Clone> {
    pub inner: Poseidon2Air<WIDTH, F>,
}

pub struct Poseidon2Chip<const WIDTH: usize, F: PrimeField32> {
    pub air: Poseidon2VmAir<WIDTH, F>,
    pub rows: Vec<Poseidon2VmCols<WIDTH, F>>,
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
    pub fn from_poseidon2_config(config: Poseidon2Config<WIDTH, F>, bus_index: usize) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, bus_index);
        Self { inner }
    }

    pub fn interaction_width() -> usize {
        7
    }

    /// Map VM instructions to Poseidon2IO columns.
    fn make_io_cols(start_timestamp: usize, instruction: Instruction<F>) -> Poseidon2VmIoCols<F> {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
        } = instruction;
        Poseidon2VmIoCols::<F> {
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
}

const WIDTH: usize = 16;
impl<F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    pub fn from_poseidon2_config(config: Poseidon2Config<WIDTH, F>, bus_index: usize) -> Self {
        let air = Poseidon2VmAir::<WIDTH, F>::from_poseidon2_config(config, bus_index);
        Self { air, rows: vec![] }
    }
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
        let new_row = vm.poseidon2_chip.air.generate_row(
            start_timestamp,
            instruction,
            addresses,
            input_state,
        );
        let output = new_row.aux.internal.io.output;
        vm.poseidon2_chip.rows.push(new_row);

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

    pub fn max_accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        3 + (2 * WIDTH)
    }
}
