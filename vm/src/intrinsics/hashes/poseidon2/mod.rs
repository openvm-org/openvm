use std::array;

use afs_primitives::sub_chip::LocalTraceInstructions;
use columns::*;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::{Poseidon2Air, Poseidon2Cols, Poseidon2Config};

use self::air::Poseidon2VmAir;
use crate::{
    arch::{
        instructions::{
            Poseidon2Opcode::{self, *},
            UsizeOpcode,
        },
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            tree::HasherChip,
            MemoryAuxColsFactory, MemoryControllerRef, MemoryReadRecord, MemoryWriteRecord,
        },
        program::{bridge::ProgramBus, ExecutionError, Instruction},
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
    pub memory_controller: MemoryControllerRef<F>,

    records: Vec<Poseidon2Record<F>>,

    offset: usize,
}

impl<F: PrimeField32> Poseidon2VmAir<F> {
    /// Construct from Poseidon2 config and bus index.
    pub fn from_poseidon2_config(
        config: Poseidon2Config<WIDTH, F>,
        max_constraint_degree: usize,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        offset: usize,
    ) -> Self {
        let inner = Poseidon2Air::<WIDTH, F>::from_config(config, max_constraint_degree, 0);
        Self {
            inner,
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            direct: true,
            offset,
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
        max_constraint_degree: usize,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        offset: usize,
    ) -> Self {
        let air = Poseidon2VmAir::<F>::from_poseidon2_config(
            p2_config,
            max_constraint_degree,
            execution_bus,
            program_bus,
            memory_controller.borrow().memory_bridge(),
            offset,
        );
        Self {
            air,
            records: vec![],
            memory_controller,
            offset,
        }
    }

    fn record_to_cols(
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        record: Poseidon2Record<F>,
    ) -> Poseidon2VmCols<F> {
        match record {
            Poseidon2Record::FromInstruction {
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
            } => {
                let dst_ptr = dst_ptr_read.value();
                let lhs_ptr = lhs_ptr_read.value();

                let ptr_aux_cols =
                    [Some(dst_ptr_read), Some(lhs_ptr_read), rhs_ptr_read].map(|maybe_read| {
                        maybe_read.map_or_else(MemoryReadAuxCols::disabled, |read| {
                            aux_cols_factory.make_read_aux_cols(read)
                        })
                    });

                let input_aux_cols =
                    [lhs_read, rhs_read].map(|read| aux_cols_factory.make_read_aux_cols(read));

                let output_aux_cols = [Some(output1_write), output2_write].map(|maybe_write| {
                    maybe_write.map_or_else(MemoryWriteAuxCols::disabled, |write| {
                        aux_cols_factory.make_write_aux_cols(write)
                    })
                });

                Poseidon2VmCols {
                    io: Poseidon2VmIoCols {
                        is_opcode: F::one(),
                        is_compress_direct: F::zero(),
                        pc: F::from_canonical_u32(from_state.pc),
                        timestamp: F::from_canonical_u32(from_state.timestamp),
                        a: instruction.a,
                        b: instruction.b,
                        c: instruction.c,
                        d: instruction.d,
                        e: instruction.e,
                        is_compress_opcode: F::from_bool(instruction.opcode == COMP_POS2 as usize),
                    },
                    aux: Poseidon2VmAuxCols {
                        dst_ptr,
                        lhs_ptr,
                        rhs_ptr,
                        internal: internal_cols,
                        ptr_aux_cols,
                        input_aux_cols,
                        output_aux_cols,
                    },
                }
            }
            Poseidon2Record::DirectCompress { inner_cols } => Poseidon2VmCols {
                io: Poseidon2VmIoCols {
                    is_opcode: F::zero(),
                    is_compress_direct: F::one(),
                    pc: F::zero(),
                    timestamp: F::zero(),
                    a: F::zero(),
                    b: F::zero(),
                    c: F::zero(),
                    d: F::zero(),
                    e: F::zero(),
                    is_compress_opcode: F::zero(),
                },
                aux: Poseidon2VmAuxCols {
                    dst_ptr: F::zero(),
                    lhs_ptr: F::zero(),
                    rhs_ptr: F::zero(),
                    internal: inner_cols,
                    ptr_aux_cols: array::from_fn(|_| MemoryReadAuxCols::disabled()),
                    input_aux_cols: array::from_fn(|_| MemoryReadAuxCols::disabled()),
                    output_aux_cols: array::from_fn(|_| MemoryWriteAuxCols::disabled()),
                },
            },
        }
    }
}

#[derive(Debug)]
enum Poseidon2Record<F> {
    FromInstruction {
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
        internal_cols: Poseidon2Cols<WIDTH, F>,
        dst_ptr_read: MemoryReadRecord<F, 1>,
        lhs_ptr_read: MemoryReadRecord<F, 1>,
        // None for permute (since rhs_ptr is computed from lhs_ptr).
        rhs_ptr_read: Option<MemoryReadRecord<F, 1>>,
        rhs_ptr: F,
        lhs_read: MemoryReadRecord<F, CHUNK>,
        rhs_read: MemoryReadRecord<F, CHUNK>,
        output1_write: MemoryWriteRecord<F, CHUNK>,
        // None for compress (since output is of size CHUNK).
        output2_write: Option<MemoryWriteRecord<F, CHUNK>>,
    },
    DirectCompress {
        inner_cols: Poseidon2Cols<WIDTH, F>,
    },
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
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let mut memory_controller = self.memory_controller.borrow_mut();

        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = instruction;
        let local_opcode_index = opcode - self.offset;

        let local_opcode_index = Poseidon2Opcode::from_usize(local_opcode_index);

        assert!(matches!(local_opcode_index, COMP_POS2 | PERM_POS2));
        debug_assert_eq!(WIDTH, CHUNK * 2);

        let chunk_f = F::from_canonical_usize(CHUNK);

        let dst_ptr_read = memory_controller.read_cell(d, a);
        let dst_ptr = dst_ptr_read.value();

        let lhs_ptr_read = memory_controller.read_cell(d, b);
        let lhs_ptr = lhs_ptr_read.value();

        let (rhs_ptr, rhs_ptr_read) = match local_opcode_index {
            COMP_POS2 => {
                let rhs_ptr_read = memory_controller.read_cell(d, c);
                (rhs_ptr_read.value(), Some(rhs_ptr_read))
            }
            PERM_POS2 => {
                memory_controller.increment_timestamp();
                (lhs_ptr + chunk_f, None)
            }
        };

        let lhs_read = memory_controller.read(e, lhs_ptr);
        let rhs_read = memory_controller.read(e, rhs_ptr);
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

        let output1_write = memory_controller.write(e, dst_ptr, output1);
        let output2_write = match local_opcode_index {
            COMP_POS2 => {
                memory_controller.increment_timestamp();
                None
            }
            PERM_POS2 => Some(memory_controller.write(e, dst_ptr + chunk_f, output2)),
        };

        self.records.push(Poseidon2Record::FromInstruction {
            instruction: Instruction {
                opcode: local_opcode_index as usize,
                ..instruction
            },
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

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_controller.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = Poseidon2Opcode::from_usize(opcode - self.offset);
        format!("{local_opcode_index:?}")
    }
}

impl<F: PrimeField32> HasherChip<CHUNK, F> for Poseidon2Chip<F> {
    /// Key method for Hasher trait.
    ///
    /// Takes two chunks, hashes them, and returns the result. Total width 3 * CHUNK, exposed in `direct_interaction_width()`.
    ///
    /// No interactions with other chips.
    fn compress_and_record(&mut self, lhs: &[F; CHUNK], rhs: &[F; CHUNK]) -> [F; CHUNK] {
        let mut input_state = [F::zero(); WIDTH];
        input_state[..CHUNK].copy_from_slice(lhs);
        input_state[CHUNK..].copy_from_slice(rhs);

        let inner_cols = self.air.inner.generate_trace_row(input_state);
        let output = array::from_fn(|i| inner_cols.io.output[i]);

        self.records
            .push(Poseidon2Record::DirectCompress { inner_cols });

        output
    }

    fn compress(&self, lhs: &[F; CHUNK], rhs: &[F; CHUNK]) -> [F; CHUNK] {
        let mut input_state = [F::zero(); WIDTH];
        input_state[..CHUNK].copy_from_slice(lhs);
        input_state[CHUNK..].copy_from_slice(rhs);

        let inner_cols = self.air.inner.generate_trace_row(input_state);
        array::from_fn(|i| inner_cols.io.output[i])
    }
}
