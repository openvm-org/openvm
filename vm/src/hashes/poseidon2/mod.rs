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
    memory::{
        manager::{trace_builder::MemoryTraceBuilder, MemoryChipRef},
        offline_checker::bridge::MemoryOfflineChecker,
        tree::Hasher,
    },
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Poseidon2 Chip.
///
/// Carries the Poseidon2VmAir for constraints, and cached state for trace generation.
#[derive(Debug)]
pub struct Poseidon2Chip<const WIDTH: usize, F: PrimeField32> {
    pub air: Poseidon2VmAir<WIDTH, F>,
    pub rows: Vec<Poseidon2VmCols<WIDTH, F>>,
    pub memory_chip: MemoryChipRef<F>,
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2VmAir<WIDTH, F> {
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

    /// Map VM instructions to Poseidon2IO columns, for opcodes.
    fn make_io_cols(
        ExecutionState { pc, timestamp }: ExecutionState<F>,
        instruction: Instruction<F>,
    ) -> Poseidon2VmIoCols<F> {
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
        } = instruction;
        Poseidon2VmIoCols {
            is_opcode: F::one(),
            is_direct: F::zero(),
            pc,
            timestamp,
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
    pub fn from_poseidon2_config(
        p2_config: Poseidon2Config<WIDTH, F>,
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        let air = Poseidon2VmAir::<WIDTH, F>::from_poseidon2_config(
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

impl<F: PrimeField32> InstructionExecutor<F> for Poseidon2Chip<WIDTH, F> {
    /// Reads two chunks from memory and generates a trace row for
    /// the given instruction using the subair, storing it in `rows`. Then, writes output to memory,
    /// truncating if the instruction is a compression.
    ///
    /// Used for both compression and permutation.
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let mut mem_trace_builder = MemoryTraceBuilder::new(self.memory_chip.clone());

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
        debug_assert_eq!(WIDTH, CHUNK * 2);

        let dst = mem_trace_builder.read_elem(d, op_a);
        let lhs = mem_trace_builder.read_elem(d, op_b);
        let rhs = if opcode == COMP_POS2 {
            mem_trace_builder.read_elem(d, op_c)
        } else {
            mem_trace_builder.disabled_op();
            mem_trace_builder.increment_clk();
            lhs + F::from_canonical_usize(CHUNK)
        };

        let input_state: [F; WIDTH] = array::from_fn(|i| {
            if i < CHUNK {
                mem_trace_builder.read_elem(e, lhs + F::from_canonical_usize(i))
            } else {
                mem_trace_builder.read_elem(e, rhs + F::from_canonical_usize(i - CHUNK))
            }
        });

        let internal = self.air.inner.generate_trace_row(input_state);
        let output = internal.io.output;
        let len = if opcode == PERM_POS2 { WIDTH } else { CHUNK };

        for (i, &output_elem) in output.iter().enumerate().take(len) {
            mem_trace_builder.write_cell(e, dst + F::from_canonical_usize(i), output_elem);
        }

        // Generate disabled MemoryOfflineCheckerAuxCols in case len != WIDTH
        for _ in len..WIDTH {
            mem_trace_builder.disabled_op();
            mem_trace_builder.increment_clk();
        }

        let io = Poseidon2VmAir::<WIDTH, F>::make_io_cols(
            from_state.map(F::from_canonical_usize),
            instruction.clone(),
        );

        let row = Poseidon2VmCols {
            io,
            aux: Poseidon2VmAuxCols::<WIDTH, F> {
                dst,
                lhs,
                rhs,
                internal,
                mem_oc_aux_cols: mem_trace_builder.take_accesses_buffer(),
            },
        };

        self.rows.push(row);

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + self.air.timestamp_delta(),
        }
    }
}

const CHUNK: usize = 8;
impl<F: PrimeField32> Hasher<CHUNK, F> for Poseidon2Chip<WIDTH, F> {
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
