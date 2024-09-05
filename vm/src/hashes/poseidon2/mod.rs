use std::array;

use afs_primitives::sub_chip::LocalTraceInstructions;
use columns::*;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};

use self::air::Poseidon2VmAir;
use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode::*,
    },
    cpu::trace::Instruction,
    memory::{manager::MemoryChipRef, offline_checker::bridge::MemoryOfflineChecker, tree::Hasher},
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const WIDTH: usize = 16;
pub const CHUNK: usize = 8;

/// Poseidon2 Chip.
///
/// Carries the Poseidon2VmAir for constraints, and cached state for trace generation.
#[derive(Debug)]
pub struct Poseidon2Chip<F: PrimeField32> {
    pub air: Poseidon2VmAir<F>,
    pub rows: Vec<Poseidon2VmCols<F>>,
    pub memory_chip: MemoryChipRef<F>,
}

impl<F: PrimeField32> Poseidon2VmAir<F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        config: Poseidon2Config<WIDTH, F>,
        execution_bus: ExecutionBus,
        mem_oc: MemoryOfflineChecker,
    ) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, 0);
        Self {
            inner,
            execution_bus,
            mem_oc,
            direct: true,
        }
    }

    pub fn timestamp_delta(&self) -> usize {
        3 + (2 * WIDTH)
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
}

impl<F: PrimeField32> Poseidon2Chip<F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        p2_config: Poseidon2Config<WIDTH, F>,
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        let air = Poseidon2VmAir::<F>::from_poseidon2_config(
            p2_config,
            execution_bus,
            memory_chip.borrow().make_offline_checker(),
        );
        Self {
            air,
            rows: vec![],
            memory_chip,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for Poseidon2Chip<F> {
    /// Reads two chunks from memory and generates a trace row for
    /// the given instruction using the subair, storing it in `rows`. Then, writes output to memory,
    /// truncating if the instruction is a compression.
    ///
    /// Used for both compression and permutation.
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let mut memory_chip = self.memory_chip.borrow_mut();

        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            ..
        } = instruction;

        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        debug_assert_eq!(WIDTH, CHUNK * 2);

        let dst_read = memory_chip.read_cell(d, op_a);
        let lhs_ptr_read = memory_chip.read_cell(d, op_b);
        let rhs_ptr_read = if opcode == COMP_POS2 {
            Some(memory_chip.read_cell(d, op_c))
        } else {
            memory_chip.increment_timestamp();
            None
        };

        let dst = dst_read.value();
        let lhs_ptr = lhs_ptr_read.value();
        let rhs_ptr = rhs_ptr_read
            .clone()
            .map(|rhs_ptr_read| rhs_ptr_read.value())
            .unwrap_or(lhs_ptr + F::from_canonical_usize(CHUNK));

        let input_1 = memory_chip.read(e, lhs_ptr);
        let input_2 = memory_chip.read(e, rhs_ptr);
        let input_state: [F; WIDTH] = array::from_fn(|i| {
            if i < CHUNK {
                input_1.data[i]
            } else {
                input_2.data[i - CHUNK]
            }
        });

        let internal = self.air.inner.generate_trace_row(input_state);
        let output = internal.io.output;

        let output_1 = memory_chip.write(e, dst, output[..CHUNK].try_into().unwrap());
        let output_2 = if opcode == PERM_POS2 {
            Some(memory_chip.write(e, dst, output[CHUNK..].try_into().unwrap()))
        } else {
            memory_chip.increment_timestamp_by(F::from_canonical_usize(CHUNK));
            None
        };

        let ptr_aux_cols = [Some(dst_read), Some(lhs_ptr_read), rhs_ptr_read].map(|maybe_read| {
            maybe_read
                .map(|read| memory_chip.make_read_aux_cols(read))
                .unwrap_or_else(|| memory_chip.make_disabled_read_aux_cols())
        });
        let input_aux_cols = [input_1, input_2].map(|x| memory_chip.make_read_aux_cols(x));
        let output_aux_cols = [Some(output_1), output_2].map(|maybe_write| {
            maybe_write
                .map(|write| memory_chip.make_write_aux_cols(write))
                .unwrap_or(memory_chip.make_disabled_write_aux_cols())
        });

        let row = Poseidon2VmCols {
            io: Poseidon2VmIoCols {
                is_opcode: F::one(),
                is_direct: F::zero(),
                pc: F::from_canonical_usize(from_state.pc),
                timestamp: F::from_canonical_usize(from_state.timestamp),
                a: op_a,
                b: op_b,
                c: op_c,
                d,
                e,
                cmp: F::from_bool(opcode == COMP_POS2),
            },
            aux: Poseidon2VmAuxCols {
                dst,
                lhs_ptr,
                rhs_ptr,
                internal,
                ptr_aux_cols,
                input_aux_cols,
                output_aux_cols,
            },
        };

        self.rows.push(row);

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + self.air.timestamp_delta(),
        }
    }
}

impl<F: PrimeField32> Hasher<CHUNK, F> for Poseidon2Chip<F> {
    /// Key method for Hasher trait.
    ///
    /// Takes two chunks, hashes them, and returns the result. Total width 3 * CHUNK, exposed in `direct_interaction_width()`.
    ///
    /// No interactions with other chips.
    fn hash(&mut self, left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
        let mut input_state = [F::zero(); WIDTH];
        input_state[..8].copy_from_slice(&left);
        input_state[8..16].copy_from_slice(&right);

        // This is not currently supported
        todo!();

        // self.calculate(Instruction::default(), true);
        // self.rows.last().unwrap().aux.internal.io.output[..8]
        //     .try_into()
        //     .unwrap()
    }
}
