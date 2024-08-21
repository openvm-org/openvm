use std::{array, cell::RefCell, rc::Rc};

use p3_field::PrimeField32;

use afs_primitives::sub_chip::LocalTraceInstructions;
use columns::*;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::OpCodeExecutor,
        columns::ExecutionState,
        instructions::{OpCode, OpCode::*},
    },
    cpu::trace::Instruction,
    memory::{offline_checker::MemoryChip, tree::Hasher},
};

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
    pub execution_bus: ExecutionBus,
}

/// Poseidon2 Chip.
///
/// Carries the Poseidon2VmAir for constraints, and cached state for trace generation.
pub struct Poseidon2Chip<const WIDTH: usize, F: PrimeField32> {
    pub air: Poseidon2VmAir<WIDTH, F>,
    pub rows: Vec<Poseidon2VmCols<WIDTH, F>>,
    pub memory_chip: Rc<RefCell<MemoryChip<1, F>>>,
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        config: Poseidon2Config<WIDTH, F>,
        bus_index: usize,
        execution_bus: ExecutionBus,
    ) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, bus_index);
        Self {
            inner,
            direct: true,
            execution_bus,
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
    fn make_io_cols(
        execution_state: ExecutionState<usize>,
        instruction: &Instruction<F>,
    ) -> Poseidon2VmIoCols<F> {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            ..
        } = instruction;
        Poseidon2VmIoCols::<F> {
            is_opcode: F::one(),
            is_direct: F::zero(),
            pc: F::from_canonical_usize(execution_state.pc),
            start_timestamp: F::from_canonical_usize(execution_state.timestamp),
            a: *op_a,
            b: *op_b,
            c: *op_c,
            d: *d,
            e: *e,
            cmp: F::from_bool(*opcode == COMP_POS2),
        }
    }
}

const WIDTH: usize = 16;
impl<F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        config: Poseidon2Config<WIDTH, F>,
        bus_index: usize,
        execution_bus: ExecutionBus,
        memory_chip: Rc<RefCell<MemoryChip<1, F>>>,
    ) -> Self {
        let air =
            Poseidon2VmAir::<WIDTH, F>::from_poseidon2_config(config, bus_index, execution_bus);
        Self {
            air,
            rows: vec![],
            memory_chip,
        }
    }

    pub fn max_accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        3 + (2 * WIDTH)
    }

    pub fn current_height(&self) -> usize {
        self.rows.len()
    }
}

impl<const WIDTH: usize, F: PrimeField32> OpCodeExecutor<F> for Poseidon2Chip<WIDTH, F> {
    /// Key method of Poseidon2Chip.
    ///
    ///  Reads two chunks from memory and generates a trace row for
    /// the given instruction using the subair, storing it in `rows`. Then, writes output to memory,
    /// truncating if the instruction is a compression.
    ///
    /// Used for both compression and permutation.
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            op_f: _f,
            op_g: _g,
            debug: _debug,
        } = instruction.clone();
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        debug_assert_eq!(crate::poseidon2::WIDTH, CHUNK * 2);

        let mut timestamp = prev_state.timestamp;
        let mut read = |address_space, addr, ts: &mut usize| {
            *ts += 1;
            self.memory_chip
                .borrow_mut()
                .read_elem(*ts - 1, address_space, addr)
        };

        let dst = read(d, op_a, &mut timestamp);
        let lhs = read(d, op_b, &mut timestamp);
        let rhs = if opcode == COMP_POS2 {
            read(d, op_c, &mut timestamp)
        } else {
            // We still need to advance the timestamp to match the interaction constraints.
            timestamp += 1;
            lhs + F::from_canonical_usize(CHUNK)
        };

        let input_state: [F; WIDTH] = array::from_fn(|i| {
            if i < CHUNK {
                read(e, lhs + F::from_canonical_usize(i), &mut timestamp)
            } else {
                read(e, rhs + F::from_canonical_usize(i - CHUNK), &mut timestamp)
            }
        });

        let new_row = self
            .air
            .generate_row(prev_state, instruction, dst, lhs, rhs, input_state);

        let output = new_row.aux.internal.io.output;
        let len = if opcode == PERM_POS2 { WIDTH } else { CHUNK };

        for (i, &output_elem) in output.iter().enumerate().take(len) {
            self.memory_chip.borrow_mut().write_elem(
                timestamp + i,
                e,
                dst + F::from_canonical_usize(i),
                output_elem,
            );
        }
        timestamp += WIDTH;

        self.rows.push(new_row);

        ExecutionState {
            pc: prev_state.pc + 1,
            timestamp,
        }
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
        self.rows.push(Poseidon2VmCols {
            io: io_row,
            aux: Poseidon2VmAuxCols {
                dst: F::zero(),
                lhs: F::zero(),
                rhs: F::zero(),
                internal,
            },
        });
        output[..8].try_into().unwrap()
    }
}
