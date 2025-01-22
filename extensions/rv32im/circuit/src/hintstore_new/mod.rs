use std::{
    borrow::{Borrow, BorrowMut},
    sync::{Arc, Mutex, OnceLock},
};

use openvm_circuit::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor, Streams,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::next_power_of_two_or_zero,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32HintStoreOpcode::{HINT_BUFFER, HINT_STOREW};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter, Stateful,
};
use serde::{Deserialize, Serialize};

use crate::adapters::{compose, decompose};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct HintStoreNewCols<T> {
    // common
    pub is_single: T,
    pub is_buffer: T,
    // should be 1 for single
    pub rem_words_limbs: [T; RV32_REGISTER_NUM_LIMBS],

    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_data: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs1_aux_cols: MemoryReadAuxCols<T>,

    pub imm: T,
    pub imm_sign: T,
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    pub mem_ptr_limbs: [T; 2],
    pub write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub data: [T; RV32_REGISTER_NUM_LIMBS],

    // only buffer
    pub is_buffer_start: T,
    pub num_words_ptr: T,
    pub num_words_aux_cols: MemoryReadAuxCols<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct HintStoreNewAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub bitwise_operation_lookup_bus: BitwiseOperationLookupBus,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for HintStoreNewAir {
    fn width(&self) -> usize {
        HintStoreNewCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for HintStoreNewAir {}
impl<F: Field> PartitionedBaseAir<F> for HintStoreNewAir {}

impl<AB: InteractionBuilder> Air<AB> for HintStoreNewAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_cols: &HintStoreNewCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next_cols: &HintStoreNewCols<AB::Var> = (*next).borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        builder.assert_bool(local_cols.is_single);
        builder.assert_bool(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_buffer_start);
        builder
            .when(local_cols.is_buffer_start)
            .assert_one(local_cols.is_buffer);
        builder.assert_bool(local_cols.is_single + local_cols.is_buffer);


        let is_valid = local_cols.is_single + local_cols.is_buffer;
        let is_start = local_cols.is_single + local_cols.is_buffer_start;
        // should only be used when is_buffer is truer
        let is_end = AB::Expr::ONE - next_cols.is_buffer + next_cols.is_buffer_start;

        let mut rem_words = AB::Expr::ZERO;
        let mut next_rem_words = AB::Expr::ZERO;
        let mut mem_ptr = AB::Expr::ZERO;
        let mut next_mem_ptr = AB::Expr::ZERO;
        for i in (0..RV32_REGISTER_NUM_LIMBS).rev() {
            rem_words = rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + local_cols.rem_words_limbs[i];
            next_rem_words = next_rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + next_cols.rem_words_limbs[i];
        }
        for i in (0..RV32_REGISTER_NUM_LIMBS / 2).rev() {
            mem_ptr = mem_ptr * AB::F::from_canonical_u32(1 << (2 * RV32_CELL_BITS))
                + local_cols.mem_ptr_limbs[i];
            next_mem_ptr = next_mem_ptr * AB::F::from_canonical_u32(1 << (2 * RV32_CELL_BITS))
                + next_cols.mem_ptr_limbs[i];
        }

        // read rs1
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.rs1_ptr,
                ),
                local_cols.rs1_data,
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, is_start.clone());

        // read num_words
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.num_words_ptr,
                ),
                local_cols.rem_words_limbs,
                timestamp_pp(),
                &local_cols.num_words_aux_cols,
            )
            .eval(builder, local_cols.is_buffer_start.clone());

        builder
            .when(local_cols.is_single)
            .assert_one(rem_words.clone());

        // constrain mem_ptr = rs1 + imm as a u32 addition with 2 limbs
        let limbs_01 = local_cols.rs1_data[0]
            + local_cols.rs1_data[1] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);
        let limbs_23 = local_cols.rs1_data[2]
            + local_cols.rs1_data[3] * AB::F::from_canonical_u32(1 << RV32_CELL_BITS);

        let inv = AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2)).inverse();
        let carry = (limbs_01 + local_cols.imm - local_cols.mem_ptr_limbs[0]) * inv;

        builder.when(is_start.clone()).assert_bool(carry.clone());

        builder.when(is_start.clone()).assert_bool(local_cols.imm_sign);
        let imm_extend_limb =
            local_cols.imm_sign * AB::F::from_canonical_u32((1 << (RV32_CELL_BITS * 2)) - 1);
        let carry = (limbs_23 + imm_extend_limb + carry - local_cols.mem_ptr_limbs[1]) * inv;
        builder.when(is_start.clone()).assert_bool(carry.clone());

        // preventing mem_ptr overflow
        self.range_bus
            .range_check(local_cols.mem_ptr_limbs[0], RV32_CELL_BITS * 2)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(
                local_cols.mem_ptr_limbs[1],
                self.pointer_max_bits - RV32_CELL_BITS * 2,
            )
            .eval(builder, is_valid.clone());

        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.bitwise_operation_lookup_bus
                .send_range(local_cols.data[i * 2], local_cols.data[i * 2 + 1])
                .eval(builder, is_valid.clone());
        }

        let mem_ptr = local_cols.mem_ptr_limbs[0]
            + local_cols.mem_ptr_limbs[1] * AB::F::from_canonical_u32(1 << (RV32_CELL_BITS * 2));

        // write hint
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_MEMORY_AS), mem_ptr.clone()),
                local_cols.data,
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, is_valid.clone());

        let to_pc = local_cols.from_state.pc + AB::F::from_canonical_u32(DEFAULT_PC_STEP);
        self.execution_bridge
            .execute(
                (local_cols.is_single
                    * AB::F::from_canonical_usize(HINT_STOREW.global_opcode().as_usize()))
                    + (local_cols.is_buffer
                        * AB::F::from_canonical_usize(HINT_BUFFER.global_opcode().as_usize())),
                [
                    local_cols.is_buffer * (local_cols.num_words_ptr),
                    local_cols.rs1_ptr.into(),
                    local_cols.imm.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp
                        + (rem_words.clone() * AB::F::from_canonical_usize(timestamp_delta)),
                },
            )
            .eval(builder, is_start.clone());

        // buffer transition

        builder
            .when(local_cols.is_buffer)
            .when(is_end.clone())
            .assert_one(rem_words.clone());
        builder
            .when(local_cols.is_buffer)
            .when(AB::Expr::ONE - is_end.clone())
            .assert_one(rem_words.clone() - next_rem_words.clone());
        builder
            .when(local_cols.is_buffer)
            .when(AB::Expr::ONE - is_end.clone())
            .assert_eq(
                next_mem_ptr.clone() - mem_ptr.clone(),
                AB::F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS),
            );
        builder
            .when(local_cols.is_buffer)
            .when(AB::Expr::ONE - is_end.clone())
            .assert_eq(
                timestamp + AB::F::from_canonical_usize(timestamp_delta),
                next_cols.from_state.timestamp,
            );
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct HintStoreRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub instruction: Instruction<F>,
    pub rs1: [F; RV32_REGISTER_NUM_LIMBS],
    pub rs1_read: RecordId,
    pub offset: u32,
    pub offset_sign: bool,
    pub mem_ptr: u32,
    pub num_words: u32,

    pub num_words_read: Option<RecordId>,
    pub hints: Vec<([F; RV32_REGISTER_NUM_LIMBS], RecordId)>,
}

pub struct NewHintStoreChip<F: Field> {
    air: HintStoreNewAir,
    records: Vec<HintStoreRecord<F>>,
    height: usize,
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    pub streams: OnceLock<Arc<Mutex<Streams<F>>>>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F: PrimeField32> NewHintStoreChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        pointer_max_bits: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        memory_bridge: MemoryBridge,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    ) -> Self {
        let air = HintStoreNewAir {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            range_bus: range_checker_chip.bus(),
            bitwise_operation_lookup_bus: bitwise_lookup_chip.bus(),
            pointer_max_bits,
        };
        Self {
            records: vec![],
            air,
            height: 0,
            offline_memory,
            streams: OnceLock::new(),
            bitwise_lookup_chip,
            range_checker_chip,
        }
    }
    pub fn set_streams(&mut self, streams: Arc<Mutex<Streams<F>>>) {
        self.streams.set(streams).unwrap();
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for NewHintStoreChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let &Instruction {
            opcode,
            a: num_words_ptr,
            b: rs1_ptr,
            c: offset,
            d,
            e,
            ..
        } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let (rs1_read, rs1) = memory.read::<RV32_REGISTER_NUM_LIMBS>(d, rs1_ptr);
        let (num_words, num_words_read) = if opcode == HINT_STOREW.global_opcode() {
            memory.increment_timestamp();
            (1, None)
        } else {
            let (num_words_read, num_words_limbs) =
                memory.read::<RV32_REGISTER_NUM_LIMBS>(d, num_words_ptr);
            (compose(num_words_limbs), Some(num_words_read))
        };
        let rs1_val = compose(rs1);
        let offset = offset.as_canonical_u32();
        let offset_sign = (offset & 0x8000) >> 15;
        let offset_extended = offset + offset_sign * 0xffff0000;

        let ptr_val = rs1_val.wrapping_add(offset_extended);
        assert!(ptr_val < (1 << self.air.pointer_max_bits));
        let mem_ptr_limbs: [u32; 2] =
            std::array::from_fn(|i| (ptr_val >> (i * (RV32_CELL_BITS * 2))) & 0xffff);

        let mut streams = self.streams.get().unwrap().lock().unwrap();
        if streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS {
            return Err(ExecutionError::HintOutOfBounds { pc: from_state.pc });
        }

        let ptr = mem_ptr_limbs[0] + mem_ptr_limbs[1] * (1 << (RV32_CELL_BITS * 2));

        let mut record = HintStoreRecord {
            from_state,
            instruction: instruction.clone(),
            rs1,
            rs1_read,
            offset,
            offset_sign: offset_sign == 1,
            mem_ptr: ptr_val,
            num_words,
            num_words_read,
            hints: vec![],
        };

        for word_index in 0..num_words {
            if word_index != 0 {
                memory.increment_timestamp();
                memory.increment_timestamp();
            }

            let data: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| streams.hint_stream.pop_front().unwrap());
            let (write, _) = memory.write(
                e,
                F::from_canonical_u32(ptr + (RV32_REGISTER_NUM_LIMBS as u32 * word_index)),
                data,
            );
            record.hints.push((data, write));

            for i in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
                self.bitwise_lookup_chip.request_range(
                    data[2 * i].as_canonical_u32(),
                    data[2 * i + 1].as_canonical_u32(),
                );
            }
        }

        self.height += record.hints.len();
        self.records.push(record);

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STOREW.global_opcode().as_usize() {
            String::from("HINT_STOREW")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {}", opcode)
        }
    }
}

impl<F: Field> ChipUsageGetter for NewHintStoreChip<F> {
    fn air_name(&self) -> String {
        "FriReducedOpeningAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.height
    }

    fn trace_width(&self) -> usize {
        HintStoreNewCols::<F>::width()
    }
}

impl<F: PrimeField32> NewHintStoreChip<F> {
    // returns number of used u32s
    fn record_to_rows(
        record: HintStoreRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
        memory: &OfflineMemory<F>,
        range_checker_chip: SharedVariableRangeCheckerChip,
        pointer_max_bits: usize,
    ) -> usize {
        let width = HintStoreNewCols::<F>::width();
        let cols: &mut HintStoreNewCols<F> = slice[..width].borrow_mut();

        cols.is_single = F::from_bool(record.num_words_read.is_none());
        cols.is_buffer = F::from_bool(record.num_words_read.is_some());
        cols.is_buffer_start = cols.is_buffer;

        cols.from_state = record.from_state.map(F::from_canonical_u32);
        cols.rs1_ptr = record.instruction.b;
        cols.rs1_data = record.rs1;
        cols.rs1_aux_cols =
            aux_cols_factory.make_read_aux_cols(memory.record_by_id(record.rs1_read));

        cols.imm = record.instruction.c;
        cols.imm_sign = F::from_bool(record.offset_sign);

        cols.num_words_ptr = record.instruction.a;
        if let Some(num_words_read) = record.num_words_read {
            cols.num_words_aux_cols =
                aux_cols_factory.make_read_aux_cols(memory.record_by_id(num_words_read));
        }

        let mut mem_ptr = record.mem_ptr;
        let mut rem_words = record.num_words;
        let mut used_u32s = 0;
        for (i, &(data, write)) in record.hints.iter().enumerate() {
            let cols: &mut HintStoreNewCols<F> = slice[used_u32s..used_u32s + width].borrow_mut();
            cols.from_state.timestamp = F::from_canonical_u32(record.from_state.timestamp + (3 * i as u32));
            cols.data = data;
            cols.write_aux = aux_cols_factory.make_write_aux_cols(memory.record_by_id(write));
            cols.rem_words_limbs = decompose(rem_words);
            let mem_ptr_limbs = std::array::from_fn(|i| {
                (mem_ptr >> (i * (RV32_CELL_BITS * 2))) & 0xffff
            });
            cols.mem_ptr_limbs = mem_ptr_limbs.map(F::from_canonical_u32);
            range_checker_chip
                .add_count(mem_ptr_limbs[0], RV32_CELL_BITS * 2);
            range_checker_chip.add_count(
                mem_ptr_limbs[1],
                pointer_max_bits - RV32_CELL_BITS * 2,
            );
            if i != 0 {
                cols.is_buffer = F::ONE;
            }
            used_u32s += width;
            mem_ptr += RV32_REGISTER_NUM_LIMBS as u32;
            rem_words -= 1;
        }

        used_u32s
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let width = self.trace_width();
        let height = next_power_of_two_or_zero(self.height);
        let mut flat_trace = F::zero_vec(width * height);

        let memory = self.offline_memory.lock().unwrap();

        let aux_cols_factory = memory.aux_cols_factory();

        let mut used_u32s = 0;
        for record in self.records {
            used_u32s += Self::record_to_rows(
                record,
                &aux_cols_factory,
                &mut flat_trace[used_u32s..],
                &memory,
                self.range_checker_chip.clone(),
                self.air.pointer_max_bits,
            );
        }
        // padding rows can just be all zeros
        RowMajorMatrix::new(flat_trace, width)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for NewHintStoreChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        AirProofInput::simple_no_pis(self.air(), self.generate_trace())
    }
}

impl<F: PrimeField32> Stateful<Vec<u8>> for NewHintStoreChip<F> {
    fn load_state(&mut self, state: Vec<u8>) {
        self.records = bitcode::deserialize(&state).unwrap();
        self.height = self.records.iter().map(|record| record.hints.len()).sum();
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.records).unwrap()
    }
}
