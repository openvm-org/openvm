use std::{array, cell::Ref};

use afs_primitives::sub_chip::LocalTraceInstructions;
use columns::*;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Cols, Poseidon2Config};

use self::air::Poseidon2VmAir;
use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode::*,
    },
    cpu::trace::Instruction,
    memory::{
        manager::{MemoryChip, MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
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

pub const WIDTH: usize = 16;
pub const CHUNK: usize = 8;

/// Poseidon2 Chip.
///
/// Carries the Poseidon2VmAir for constraints, and cached state for trace generation.
#[derive(Debug)]
pub struct Poseidon2Chip<F: PrimeField32> {
    pub air: Poseidon2VmAir<F>,
    pub memory_chip: MemoryChipRef<F>,

    records: Vec<Poseidon2Record<F>>,
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
            records: vec![],
            memory_chip,
        }
    }

    fn record_to_cols(
        memory_chip: &Ref<MemoryChip<F>>,
        record: Poseidon2Record<F>,
    ) -> Poseidon2VmCols<F> {
        let dst_ptr = record.dst_ptr_read.value();
        let lhs_ptr = record.lhs_ptr_read.value();

        let ptr_aux_cols = [
            Some(record.dst_ptr_read),
            Some(record.lhs_ptr_read),
            record.rhs_ptr_read,
        ]
        .map(|maybe_read| {
            maybe_read.map_or_else(
                || memory_chip.make_disabled_read_aux_cols(),
                |read| memory_chip.make_read_aux_cols(read),
            )
        });

        let input_aux_cols =
            [record.lhs_read, record.rhs_read].map(|read| memory_chip.make_read_aux_cols(read));

        let output_aux_cols =
            [Some(record.output1_write), record.output2_write].map(|maybe_write| {
                maybe_write.map_or_else(
                    || memory_chip.make_disabled_write_aux_cols(),
                    |write| memory_chip.make_write_aux_cols(write),
                )
            });

        Poseidon2VmCols {
            io: Poseidon2VmIoCols {
                is_opcode: F::one(),
                is_direct: F::zero(),
                pc: F::from_canonical_usize(record.from_state.pc),
                timestamp: F::from_canonical_usize(record.from_state.timestamp),
                a: record.instruction.op_a,
                b: record.instruction.op_b,
                c: record.instruction.op_c,
                d: record.instruction.d,
                e: record.instruction.e,
                cmp: F::from_bool(record.instruction.opcode == COMP_POS2),
            },
            aux: Poseidon2VmAuxCols {
                dst_ptr,
                lhs_ptr,
                rhs_ptr: record.rhs_ptr,
                internal: record.internal_cols,
                ptr_aux_cols,
                input_aux_cols,
                output_aux_cols,
            },
        }
    }
}

#[derive(Debug)]
struct Poseidon2Record<F> {
    instruction: Instruction<F>,
    from_state: ExecutionState<usize>,
    internal_cols: Poseidon2Cols<WIDTH, F>,
    dst_ptr_read: MemoryReadRecord<1, F>,
    lhs_ptr_read: MemoryReadRecord<1, F>,
    // None for permute (since rhs_ptr is computed from lhs_ptr).
    rhs_ptr_read: Option<MemoryReadRecord<1, F>>,
    rhs_ptr: F,
    lhs_read: MemoryReadRecord<CHUNK, F>,
    rhs_read: MemoryReadRecord<CHUNK, F>,
    output1_write: MemoryWriteRecord<CHUNK, F>,
    // None for compress (since output is of size CHUNK).
    output2_write: Option<MemoryWriteRecord<CHUNK, F>>,
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

        assert!(matches!(opcode, COMP_POS2 | PERM_POS2));
        debug_assert_eq!(WIDTH, CHUNK * 2);

        let chunk_f = F::from_canonical_usize(CHUNK);

        let dst_ptr_read = memory_chip.read_cell(d, op_a);
        let dst_ptr = dst_ptr_read.value();

        let lhs_ptr_read = memory_chip.read_cell(d, op_b);
        let lhs_ptr = lhs_ptr_read.value();

        let (rhs_ptr, rhs_ptr_read) = match opcode {
            COMP_POS2 => {
                let rhs_ptr_read = memory_chip.read_cell(d, op_c);
                (rhs_ptr_read.value(), Some(rhs_ptr_read))
            }
            PERM_POS2 => {
                memory_chip.increment_timestamp();
                (lhs_ptr + chunk_f, None)
            }
            _ => panic!("unrecognized Poseidon2Chip opcode"),
        };

        let lhs_read = memory_chip.read(e, lhs_ptr);
        let rhs_read = memory_chip.read(e, rhs_ptr);
        let input_state: [F; WIDTH] = array::from_fn(|i| {
            if i < CHUNK {
                lhs_read.data[i]
            } else {
                rhs_read.data[i - CHUNK]
            }
        });

        let internal_cols = self.air.inner.generate_trace_row(input_state);
        let output = internal_cols.io.output;

        let output1: [F; CHUNK] = array::from_fn(|i| output[i]);
        let output2: [F; CHUNK] = array::from_fn(|i| output[CHUNK + i]);

        let output1_write = memory_chip.write(e, dst_ptr, output1);
        let output2_write = match opcode {
            COMP_POS2 => {
                memory_chip.increment_timestamp();
                None
            }
            PERM_POS2 => Some(memory_chip.write(e, dst_ptr + chunk_f, output2)),
            _ => unreachable!(),
        };

        self.records.push(Poseidon2Record {
            instruction,
            from_state,
            internal_cols,
            dst_ptr_read,
            lhs_ptr_read,
            rhs_ptr_read,
            rhs_ptr,
            lhs_read,
            rhs_read,
            output1_write,
            output2_write,
        });

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
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
