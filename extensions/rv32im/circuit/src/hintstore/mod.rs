use std::{
    borrow::{Borrow, BorrowMut},
    sync::{Arc, Mutex, OnceLock},
};

use openvm_circuit::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InsExecutorE1,
        InstructionExecutor, NewVmChipWrapper, Result, StepExecutorE1, Streams, TraceStep,
        VmStateMut,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            online::{GuestMemory, TracingMemory},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::{next_power_of_two_or_zero, not},
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::{
    Rv32HintStoreOpcode,
    Rv32HintStoreOpcode::{HINT_BUFFER, HINT_STOREW},
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use rand::distributions::weighted;
use serde::{Deserialize, Serialize};

use crate::adapters::{
    decompose, memory_read, memory_write, tmp_convert_to_u8s, tracing_read, tracing_write,
};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct Rv32HintStoreCols<T> {
    // common
    pub is_single: T,
    pub is_buffer: T,
    // should be 1 for single
    pub rem_words_limbs: [T; RV32_REGISTER_NUM_LIMBS],

    pub from_state: ExecutionState<T>,
    pub mem_ptr_ptr: T,
    pub mem_ptr_limbs: [T; RV32_REGISTER_NUM_LIMBS],
    pub mem_ptr_aux_cols: MemoryReadAuxCols<T>,

    pub write_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
    pub data: [T; RV32_REGISTER_NUM_LIMBS],

    // only buffer
    pub is_buffer_start: T,
    pub num_words_ptr: T,
    pub num_words_aux_cols: MemoryReadAuxCols<T>,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct Rv32HintStoreAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_operation_lookup_bus: BitwiseOperationLookupBus,
    pub offset: usize,
    pointer_max_bits: usize,
}

impl<F: Field> BaseAir<F> for Rv32HintStoreAir {
    fn width(&self) -> usize {
        Rv32HintStoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32HintStoreAir {}
impl<F: Field> PartitionedBaseAir<F> for Rv32HintStoreAir {}

impl<AB: InteractionBuilder> Air<AB> for Rv32HintStoreAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_cols: &Rv32HintStoreCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next_cols: &Rv32HintStoreCols<AB::Var> = (*next).borrow();

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
        // `is_end` is false iff the next row is a buffer row that is not buffer start
        // This is boolean because is_buffer_start == 1 => is_buffer == 1
        // Note: every non-valid row has `is_end == 1`
        let is_end = not::<AB::Expr>(next_cols.is_buffer) + next_cols.is_buffer_start;

        let mut rem_words = AB::Expr::ZERO;
        let mut next_rem_words = AB::Expr::ZERO;
        let mut mem_ptr = AB::Expr::ZERO;
        let mut next_mem_ptr = AB::Expr::ZERO;
        for i in (0..RV32_REGISTER_NUM_LIMBS).rev() {
            rem_words = rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + local_cols.rem_words_limbs[i];
            next_rem_words = next_rem_words * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + next_cols.rem_words_limbs[i];
            mem_ptr = mem_ptr * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + local_cols.mem_ptr_limbs[i];
            next_mem_ptr = next_mem_ptr * AB::F::from_canonical_u32(1 << RV32_CELL_BITS)
                + next_cols.mem_ptr_limbs[i];
        }

        // Constrain that if local is invalid, then the next state is invalid as well
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_valid.clone()))
            .assert_zero(next_cols.is_single + next_cols.is_buffer);

        // Constrain that when we start a buffer, the is_buffer_start is set to 1
        builder
            .when(local_cols.is_single)
            .assert_one(is_end.clone());
        builder
            .when_first_row()
            .assert_one(not::<AB::Expr>(local_cols.is_buffer) + local_cols.is_buffer_start);

        // read mem_ptr
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_canonical_u32(RV32_REGISTER_AS),
                    local_cols.mem_ptr_ptr,
                ),
                local_cols.mem_ptr_limbs,
                timestamp_pp(),
                &local_cols.mem_ptr_aux_cols,
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
            .eval(builder, local_cols.is_buffer_start);

        // write hint
        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_MEMORY_AS), mem_ptr.clone()),
                local_cols.data,
                timestamp_pp(),
                &local_cols.write_aux,
            )
            .eval(builder, is_valid.clone());

        let expected_opcode = (local_cols.is_single
            * AB::F::from_canonical_usize(HINT_STOREW as usize + self.offset))
            + (local_cols.is_buffer
                * AB::F::from_canonical_usize(HINT_BUFFER as usize + self.offset));

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode,
                [
                    local_cols.is_buffer * (local_cols.num_words_ptr),
                    local_cols.mem_ptr_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                local_cols.from_state,
                rem_words.clone() * AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, is_start.clone());

        // Preventing mem_ptr and rem_words overflow
        // Constraining mem_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1] < 2^(pointer_max_bits -
        // (RV32_REGISTER_NUM_LIMBS - 1)*RV32_CELL_BITS) which implies mem_ptr <=
        // 2^pointer_max_bits Similarly for rem_words <= 2^pointer_max_bits
        self.bitwise_operation_lookup_bus
            .send_range(
                local_cols.mem_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1]
                    * AB::F::from_canonical_usize(
                        1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
                    ),
                local_cols.rem_words_limbs[RV32_REGISTER_NUM_LIMBS - 1]
                    * AB::F::from_canonical_usize(
                        1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
                    ),
            )
            .eval(builder, is_start.clone());

        // Checking that hint is bytes
        for i in 0..RV32_REGISTER_NUM_LIMBS / 2 {
            self.bitwise_operation_lookup_bus
                .send_range(local_cols.data[2 * i], local_cols.data[(2 * i) + 1])
                .eval(builder, is_valid.clone());
        }

        // buffer transition
        // `is_end` implies that the next row belongs to a new instruction,
        // which could be one of empty, hint_single, or hint_buffer
        // Constrains that when the current row is not empty and `is_end == 1`, then `rem_words` is
        // 1
        builder
            .when(is_valid)
            .when(is_end.clone())
            .assert_one(rem_words.clone());

        let mut when_buffer_transition = builder.when(not::<AB::Expr>(is_end.clone()));
        // Notes on `rem_words`: we constrain that `rem_words` doesn't overflow when we first read
        // it and that on each row it decreases by one (below). We also constrain that when
        // the current instruction ends then `rem_words` is 1. However, we don't constrain
        // that when `rem_words` is 1 then we have to end the current instruction.
        // The only way to exploit this if we to do some multiple of `p` number of additional
        // illegal `buffer` rows where `p` is the modulus of `F`. However, when doing `p`
        // additional `buffer` rows we will always increment `mem_ptr` to an illegal memory address
        // at some point, which prevents this exploit.
        when_buffer_transition.assert_one(rem_words.clone() - next_rem_words.clone());
        // Note: we only care about the `next_mem_ptr = compose(next_mem_ptr_limb)` and not the
        // individual limbs: the limbs do not need to be in the range, they can be anything
        // to make `next_mem_ptr` correct -- this is just a way to not have to have another
        // column for `mem_ptr`. The constraint we care about is `next.mem_ptr ==
        // local.mem_ptr + 4`. Finally, since we increment by `4` each time, any out of
        // bounds memory access will be rejected by the memory bus before we overflow the field.
        when_buffer_transition.assert_eq(
            next_mem_ptr.clone() - mem_ptr.clone(),
            AB::F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS),
        );
        when_buffer_transition.assert_eq(
            timestamp + AB::F::from_canonical_usize(timestamp_delta),
            next_cols.from_state.timestamp,
        );
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct Rv32HintStoreRecord<F: Field> {
    pub from_state: ExecutionState<u32>,
    pub instruction: Instruction<F>,
    pub mem_ptr_read: RecordId,
    pub mem_ptr: u32,
    pub num_words: u32,

    pub num_words_read: Option<RecordId>,
    pub hints: Vec<([F; RV32_REGISTER_NUM_LIMBS], RecordId)>,
}

pub struct Rv32HintStoreStep<F: Field> {
    pointer_max_bits: usize,
    offset: usize,
    pub streams: OnceLock<Arc<Mutex<Streams<F>>>>,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<F: PrimeField32> Rv32HintStoreStep<F> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
        pointer_max_bits: usize,
        offset: usize,
    ) -> Self {
        Self {
            pointer_max_bits,
            offset,
            streams: OnceLock::new(),
            bitwise_lookup_chip,
        }
    }

    pub fn set_streams(&mut self, streams: Arc<Mutex<Streams<F>>>) {
        self.streams.set(streams).unwrap();
    }
}

impl<F, CTX> TraceStep<F, CTX> for Rv32HintStoreStep<F>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        if opcode == HINT_STOREW.global_opcode().as_usize() {
            String::from("HINT_STOREW")
        } else if opcode == HINT_BUFFER.global_opcode().as_usize() {
            String::from("HINT_BUFFER")
        } else {
            unreachable!("unsupported opcode: {}", opcode)
        }
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let &Instruction {
            opcode,
            a: num_words_ptr,
            b: mem_ptr_ptr,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let local_opcode = Rv32HintStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let mut row: &mut Rv32HintStoreCols<F> =
            trace[*trace_offset..*trace_offset + width].borrow_mut();

        row.from_state.pc = F::from_canonical_u32(*state.pc);
        row.from_state.timestamp = F::from_canonical_u32(state.memory.timestamp);

        row.mem_ptr_ptr = mem_ptr_ptr;
        let mem_ptr_limbs: [u8; RV32_REGISTER_NUM_LIMBS] = tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            mem_ptr_ptr.as_canonical_u32(),
            &mut row.mem_ptr_aux_cols,
        );
        let mem_ptr = u32::from_le_bytes(mem_ptr_limbs);
        debug_assert!(mem_ptr <= (1 << self.pointer_max_bits));

        row.num_words_ptr = num_words_ptr;
        let num_words = if local_opcode == HINT_STOREW {
            row.is_single = F::ONE;
            state.memory.increment_timestamp();
            1
        } else {
            row.is_buffer_start = F::ONE;
            row.is_buffer = F::ONE;
            let num_words_limbs: [u8; RV32_REGISTER_NUM_LIMBS] = tracing_read(
                state.memory,
                RV32_REGISTER_AS,
                num_words_ptr.as_canonical_u32(),
                &mut row.num_words_aux_cols,
            );
            u32::from_le_bytes(num_words_limbs)
        };
        debug_assert_ne!(num_words, 0);
        debug_assert!(num_words <= (1 << self.pointer_max_bits));

        let mut streams = self.streams.get().unwrap().lock().unwrap();
        if streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
        }

        let mem_ptr_msl = mem_ptr >> ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS);
        let num_words_msl = num_words >> ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS);
        // TODO(ayush): see if this can be moved to fill_trace_row
        self.bitwise_lookup_chip.request_range(
            mem_ptr_msl << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
            num_words_msl << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits),
        );

        for word_index in 0..(num_words as usize) {
            let offset = *trace_offset + word_index * width;
            let row: &mut Rv32HintStoreCols<F> = trace[offset..offset + width].borrow_mut();

            if word_index != 0 {
                row.is_buffer = F::ONE;
                row.from_state.timestamp = F::from_canonical_u32(state.memory.timestamp);

                state.memory.increment_timestamp();
                state.memory.increment_timestamp();
            }

            let data_f: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| streams.hint_stream.pop_front().unwrap());
            let data: [u8; RV32_REGISTER_NUM_LIMBS] =
                data_f.map(|byte| byte.as_canonical_u32() as u8);

            let mem_ptr_word = mem_ptr + (RV32_REGISTER_NUM_LIMBS * word_index) as u32;

            row.data = data_f;
            tracing_write(
                state.memory,
                RV32_MEMORY_AS,
                mem_ptr_word,
                &data,
                &mut row.write_aux,
            );

            row.rem_words_limbs = decompose(num_words - word_index as u32);
            row.mem_ptr_limbs = decompose(mem_ptr_word);
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += (num_words as usize) * width;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let row: &mut Rv32HintStoreCols<F> = row_slice.borrow_mut();

        let mut timestamp = row.from_state.timestamp.as_canonical_u32();

        if row.is_single.is_one() || row.is_buffer_start.is_one() {
            mem_helper.fill_from_prev(timestamp, row.mem_ptr_aux_cols.as_mut());
        }
        timestamp += 1;

        if row.is_buffer_start.is_one() {
            mem_helper.fill_from_prev(timestamp, row.num_words_aux_cols.as_mut());
        }
        timestamp += 1;

        mem_helper.fill_from_prev(timestamp, row.write_aux.as_mut());

        for half in 0..(RV32_REGISTER_NUM_LIMBS / 2) {
            self.bitwise_lookup_chip.request_range(
                row.data[2 * half].as_canonical_u32(),
                row.data[2 * half + 1].as_canonical_u32(),
            );
        }
    }
}

impl<F> StepExecutorE1<F> for Rv32HintStoreStep<F>
where
    F: PrimeField32,
{
    fn execute_e1<Ctx>(
        &mut self,
        state: VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()> {
        let &Instruction {
            opcode,
            a: num_words_ptr,
            b: mem_ptr_ptr,
            d,
            e,
            ..
        } = instruction;

        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let local_opcode = Rv32HintStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        let mem_ptr_limbs = memory_read(
            state.memory,
            RV32_REGISTER_AS,
            mem_ptr_ptr.as_canonical_u32(),
        );
        let mem_ptr = u32::from_le_bytes(mem_ptr_limbs);
        debug_assert!(mem_ptr <= (1 << self.pointer_max_bits));

        let num_words = if local_opcode == HINT_STOREW {
            1
        } else {
            let num_words_limbs = memory_read(
                state.memory,
                RV32_REGISTER_AS,
                num_words_ptr.as_canonical_u32(),
            );
            u32::from_le_bytes(num_words_limbs)
        };
        debug_assert_ne!(num_words, 0);
        debug_assert!(num_words <= (1 << self.pointer_max_bits));

        let mut streams = self.streams.get().unwrap().lock().unwrap();
        if streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
        }

        for word_index in 0..num_words {
            let data: [u8; RV32_REGISTER_NUM_LIMBS] = std::array::from_fn(|_| {
                streams.hint_stream.pop_front().unwrap().as_canonical_u32() as u8
            });
            memory_write(
                state.memory,
                RV32_MEMORY_AS,
                mem_ptr + (RV32_REGISTER_NUM_LIMBS as u32 * word_index),
                &data,
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

pub type Rv32HintStoreChip<F> = NewVmChipWrapper<F, Rv32HintStoreAir, Rv32HintStoreStep<F>>;
