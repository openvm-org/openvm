use crate::vm::VirtualMachine;
use afs_chips::sub_chip::LocalTraceInstructions;
use columns::{Poseidon2ChipCols, Poseidon2ChipIoCols};
use p3_field::Field;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::Poseidon2Air;
use poseidon2_air::poseidon2::Poseidon2Config;
use crate::cpu::OpCode;
use crate::cpu::OpCode::*;
use crate::cpu::trace::Instruction;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Poseidon2 chip.
///
/// Carries the requested rows and the underlying air.

pub struct Poseidon2Chip<const WIDTH: usize, F: Clone> {
    pub air: Poseidon2Air<WIDTH, F>,
    pub rows: Vec<Poseidon2ChipCols<WIDTH, F>>,
}

fn make_io_cols<F: Field>(start_timestamp: usize, instruction: Instruction<F>) -> Poseidon2ChipIoCols<F> {
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
        cmp: F::from_bool(opcode == COMPRESS_POSEIDON2)
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
        assert!(opcode == COMPRESS_POSEIDON2 || opcode == PERM_POSEIDON2);
        40
    }
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
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
        assert!(opcode == COMPRESS_POSEIDON2 || opcode == PERM_POSEIDON2);

        let data_1: [F; 8] = core::array::from_fn(|i| {
            vm.memory_chip
                .read_elem(start_timestamp + i, d, op_a + F::from_canonical_usize(i))
        });
        let data_2: [F; 8] = core::array::from_fn(|i| {
            vm.memory_chip
                .read_elem(start_timestamp + 8 + i, d, op_b + F::from_canonical_usize(i))
        });
        let input_state: [F; 16] = [data_1, data_2].concat().try_into().unwrap();
        let aux = vm.poseidon2_chip.air.generate_trace_row(input_state);
        let output = aux.io.output;
        vm.poseidon2_chip.rows.push(Poseidon2ChipCols {
            io: make_io_cols(start_timestamp, instruction),
            aux,
        });
        // TODO adjust for compression
        let iter_range = if opcode == PERM_POSEIDON2 {
            output.iter().enumerate().take(16)
        } else {
            output.iter().enumerate().take(8)
        };

        for (i, &output_elem) in iter_range {
            vm.memory_chip.write_word(
                start_timestamp + 16 + i,
                e,
                op_c + F::from_canonical_usize(i),
                [output_elem; WORD_SIZE],
            );
        }
    }
}

