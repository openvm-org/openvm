use std::{array, cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use columns::*;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Config};

use self::air::Poseidon2VmAir;
use crate::{
    cpu::{
        trace::Instruction,
        OpCode::{self, *},
    },
    memory::{
        manager::{trace_builder::MemoryTraceBuilder, MemoryManager},
        offline_checker::bridge::MemoryOfflineChecker,
        tree::Hasher,
    },
    vm::config::MemoryConfig,
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
pub struct Poseidon2Chip<
    const WIDTH: usize,
    const NUM_WORDS: usize,
    const WORD_SIZE: usize,
    F: PrimeField32,
> {
    pub air: Poseidon2VmAir<WIDTH, WORD_SIZE, F>,
    pub rows: Vec<Poseidon2VmCols<WIDTH, WORD_SIZE, F>>,
    pub memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<const WIDTH: usize, const WORD_SIZE: usize, F: PrimeField32>
    Poseidon2VmAir<WIDTH, WORD_SIZE, F>
{
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        config: Poseidon2Config<WIDTH, F>,
        mem_config: MemoryConfig,
        bus_index: usize,
    ) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, bus_index);
        Self {
            inner,
            mem_oc: MemoryOfflineChecker::new(mem_config.clk_max_bits, mem_config.decomp),
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
    fn make_io_cols(start_clk: F, instruction: Instruction<F>) -> Poseidon2VmIoCols<F> {
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
        Poseidon2VmIoCols::<F> {
            is_opcode: F::one(),
            is_direct: F::zero(),
            clk: start_clk,
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
impl<const WORD_SIZE: usize, const NUM_WORDS: usize, F: PrimeField32>
    Poseidon2Chip<WIDTH, NUM_WORDS, WORD_SIZE, F>
{
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        p2_config: Poseidon2Config<WIDTH, F>,
        mem_config: MemoryConfig,
        memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
        range_checker: Arc<RangeCheckerGateChip>,
        bus_index: usize,
    ) -> Self {
        let air = Poseidon2VmAir::<WIDTH, WORD_SIZE, F>::from_poseidon2_config(
            p2_config, mem_config, bus_index,
        );
        Self {
            air,
            rows: vec![],
            memory_manager,
            range_checker,
        }
    }

    /// Key method of Poseidon2Chip.
    ///
    /// Called using `vm` and not `&self`. Reads two chunks from memory and generates a trace row for
    /// the given instruction using the subair, storing it in `rows`. Then, writes output to memory,
    /// truncating if the instruction is a compression.
    ///
    /// Used for both compression and permutation.
    ///
    /// TODO[osama]: finish this comment
    /// When is_direct is true, this function does not do any
    pub fn calculate(&mut self, instruction: Instruction<F>, is_direct: bool) {
        assert!(!is_direct);

        let start_clk = self.memory_manager.borrow().get_clk();
        let mut mem_trace_builder = MemoryTraceBuilder::<NUM_WORDS, WORD_SIZE, F>::new(
            self.memory_manager.clone(),
            self.range_checker.clone(),
            self.air.mem_oc,
        );

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
            mem_trace_builder.disabled_read(d);
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
            mem_trace_builder.write_elem(e, dst + F::from_canonical_usize(i), output_elem);
        }

        // Generate disabled MemoryOfflineCheckerAuxCols in case len != WIDTH
        for _ in len..WIDTH {
            mem_trace_builder.disabled_write(e);
            mem_trace_builder.increment_clk();
        }

        let io = if is_direct {
            Poseidon2VmIoCols::direct_io_cols(start_clk)
        } else {
            Poseidon2VmAir::<WIDTH, WORD_SIZE, F>::make_io_cols(start_clk, instruction)
        };

        let row = Poseidon2VmCols {
            io,
            aux: Poseidon2VmAuxCols::<WIDTH, WORD_SIZE, F> {
                dst,
                lhs,
                rhs,
                internal,
                mem_oc_aux_cols: mem_trace_builder.take_accesses_buffer(),
            },
        };

        self.rows.push(row);
    }

    pub fn max_accesses_per_instruction(opcode: OpCode) -> usize {
        assert!(opcode == COMP_POS2 || opcode == PERM_POS2);
        3 + 2 * WIDTH
    }

    pub fn current_height(&self) -> usize {
        self.rows.len()
    }
}

const CHUNK: usize = 8;
impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> Hasher<CHUNK, F>
    for Poseidon2Chip<WIDTH, NUM_WORDS, WORD_SIZE, F>
{
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
