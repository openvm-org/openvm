use p3_field::PrimeField32;

use crate::memory::tree::Hasher;
use columns::*;
use poseidon2_air::poseidon2::Poseidon2Air;
use poseidon2_air::poseidon2::Poseidon2Config;

use crate::cpu::trace::Instruction;
use crate::cpu::OpCode;
use crate::cpu::OpCode::*;
use crate::vm::VirtualMachine;
use afs_primitives::{is_zero::IsZeroAir, sub_chip::LocalTraceInstructions};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Poseidon2 Air, VM version.
///
/// Carries the subair for subtrace generation. Sticking to the conventions, this struct carries no state.
/// `direct` determines whether direct interactions are enabled. By default they are on.
pub struct Poseidon2VmAir<const WIDTH: usize, F: Clone> {
    pub inner: Poseidon2Air<WIDTH, F>,
    direct: bool, // Whether direct interactions are enabled.
}

/// Poseidon2 Chip.
///
/// Carries the Poseidon2VmAir for constraints, and cached state for trace generation.
pub struct Poseidon2Chip<const WIDTH: usize, F: PrimeField32> {
    pub air: Poseidon2VmAir<WIDTH, F>,
    pub rows: Vec<Poseidon2VmCols<WIDTH, F>>,
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(config: Poseidon2Config<WIDTH, F>, bus_index: usize) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, bus_index);
        Self {
            inner,
            direct: true,
        }
    }

    /// By default direct bus is on. If `continuations = OFF`, this should be called.
    pub fn set_direct(&mut self, direct: bool) {
        self.direct = direct;
    }

    /// By default direct bus is on. If `continuations = OFF`, this should be called.
    pub fn disable_direct(&mut self) {
        self.direct = false;
    }

    /// Number of interactions through opcode bus.
    pub fn opcode_interaction_width() -> usize {
        7
    }

    /// Number of interactions through direct bus.
    pub fn direct_interaction_width() -> usize {
        WIDTH + WIDTH / 2
    }

    /// Map VM instructions to Poseidon2IO columns, for opcodes.
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
            is_opcode: F::one(),
            is_direct: F::zero(),
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
    /// Construct from Poseidon2 config and bus index.
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

const CHUNK: usize = 8;
impl<const WIDTH: usize, F: PrimeField32> Hasher<CHUNK, F> for Poseidon2Chip<WIDTH, F> {
    /// Key method for Hasher trait.
    ///
    /// Takes two chunks, hashes them, and returns the result. Total with 3 * CHUNK, exposed in `direct_interaction_width()`.
    ///
    /// No interactions with other chips.
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
        let mut input_state = [F::zero(); WIDTH];
        input_state[..8].copy_from_slice(&left);
        input_state[8..16].copy_from_slice(&right);
        let internal = self.air.inner.generate_trace_row(input_state);
        let output = internal.io.output;
        let io_row = Poseidon2VmIoCols::direct_io_cols();
        let is_zero_row = IsZeroAir {}.generate_trace_row(io_row.d);
        self.rows.push(Poseidon2VmCols {
            io: io_row,
            aux: Poseidon2VmAuxCols {
                addresses: [F::zero(); 3],
                d_is_zero: is_zero_row.io.is_zero,
                is_zero_inv: is_zero_row.inv,
                internal,
            },
        });
        output[..8].try_into().unwrap()
    }
}
