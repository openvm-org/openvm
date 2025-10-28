use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{
            MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
            MemoryWriteBytesAuxRecord,
        },
        online::TracingMemory,
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::{AlignedBorrow, AlignedBytesBorrow};
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
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::adapters::{read_rv32_register, tracing_read, tracing_write};

mod execution;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

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

#[derive(Copy, Clone, Debug)]
pub struct Rv32HintStoreMetadata {
    num_words: usize,
}

impl MultiRowMetadata for Rv32HintStoreMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_words
    }
}

pub type Rv32HintStoreLayout = MultiRowLayout<Rv32HintStoreMetadata>;

// This is the part of the record that we keep only once per instruction
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32HintStoreRecordHeader {
    pub num_words: u32,

    pub from_pc: u32,
    pub timestamp: u32,

    pub mem_ptr_ptr: u32,
    pub mem_ptr: u32,
    pub mem_ptr_aux_record: MemoryReadAuxRecord,

    // will set `num_words_ptr` to `u32::MAX` in case of single hint
    pub num_words_ptr: u32,
    pub num_words_read: MemoryReadAuxRecord,
}

// This is the part of the record that we keep `num_words` times per instruction
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct Rv32HintStoreVar {
    pub data_write_aux: MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS>,
    pub data: [u8; RV32_REGISTER_NUM_LIMBS],
}

/// **SAFETY**: the order of the fields in `Rv32HintStoreRecord` and `Rv32HintStoreVar` is
/// important. The chip also assumes that the offset of the fields `write_aux` and `data` in
/// `Rv32HintStoreCols` is bigger than `size_of::<Rv32HintStoreRecord>()`
#[derive(Debug)]
pub struct Rv32HintStoreRecordMut<'a> {
    pub inner: &'a mut Rv32HintStoreRecordHeader,
    pub var: &'a mut [Rv32HintStoreVar],
}

/// Custom borrowing that splits the buffer into a fixed `Rv32HintStoreRecord` header
/// followed by a slice of `Rv32HintStoreVar`'s of length `num_words` provided at runtime.
/// Uses `align_to_mut()` to make sure the slice is properly aligned to `Rv32HintStoreVar`.
/// Has debug assertions to make sure the above works as expected.
impl<'a> CustomBorrow<'a, Rv32HintStoreRecordMut<'a>, Rv32HintStoreLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Rv32HintStoreLayout) -> Rv32HintStoreRecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits
        // - size_of::<Rv32HintStoreRecordHeader>() is guaranteed <= self.len() by layout
        //   precondition
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Rv32HintStoreRecordHeader>()) };

        // SAFETY:
        // - rest contains bytes that will be interpreted as Rv32HintStoreVar records
        // - align_to_mut ensures proper alignment for Rv32HintStoreVar type
        // - The layout guarantees sufficient space for layout.metadata.num_words records
        let (_, vars, _) = unsafe { rest.align_to_mut::<Rv32HintStoreVar>() };
        Rv32HintStoreRecordMut {
            inner: header_buf.borrow_mut(),
            var: &mut vars[..layout.metadata.num_words],
        }
    }

    unsafe fn extract_layout(&self) -> Rv32HintStoreLayout {
        let header: &Rv32HintStoreRecordHeader = self.borrow();
        MultiRowLayout::new(Rv32HintStoreMetadata {
            num_words: header.num_words as usize,
        })
    }
}

impl SizedRecord<Rv32HintStoreLayout> for Rv32HintStoreRecordMut<'_> {
    fn size(layout: &Rv32HintStoreLayout) -> usize {
        let mut total_len = size_of::<Rv32HintStoreRecordHeader>();
        // Align the pointer to the alignment of `Rv32HintStoreVar`
        total_len = total_len.next_multiple_of(align_of::<Rv32HintStoreVar>());
        total_len += size_of::<Rv32HintStoreVar>() * layout.metadata.num_words;
        total_len
    }

    fn alignment(_layout: &Rv32HintStoreLayout) -> usize {
        align_of::<Rv32HintStoreRecordHeader>()
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32HintStoreExecutor {
    pub pointer_max_bits: usize,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct Rv32HintStoreFiller {
    pointer_max_bits: usize,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
}

impl<F, RA> PreflightExecutor<F, RA> for Rv32HintStoreExecutor
where
    F: PrimeField32,
    for<'buf> RA:
        RecordArena<'buf, MultiRowLayout<Rv32HintStoreMetadata>, Rv32HintStoreRecordMut<'buf>>,
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
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {
            opcode, a, b, d, e, ..
        } = instruction;

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let local_opcode = Rv32HintStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));

        // We do untraced read of `num_words` in order to allocate the record first
        let num_words = if local_opcode == HINT_STOREW {
            1
        } else {
            read_rv32_register(state.memory.data(), a)
        };

        let record = state.ctx.alloc(MultiRowLayout::new(Rv32HintStoreMetadata {
            num_words: num_words as usize,
        }));

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp;
        record.inner.mem_ptr_ptr = b;

        record.inner.mem_ptr = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            b,
            &mut record.inner.mem_ptr_aux_record.prev_timestamp,
        ));

        debug_assert!(record.inner.mem_ptr <= (1 << self.pointer_max_bits));
        debug_assert_ne!(num_words, 0);
        debug_assert!(num_words <= (1 << self.pointer_max_bits));

        record.inner.num_words = num_words;
        if local_opcode == HINT_STOREW {
            state.memory.increment_timestamp();
            record.inner.num_words_ptr = u32::MAX;
        } else {
            record.inner.num_words_ptr = a;
            tracing_read::<RV32_REGISTER_NUM_LIMBS>(
                state.memory,
                RV32_REGISTER_AS,
                record.inner.num_words_ptr,
                &mut record.inner.num_words_read.prev_timestamp,
            );
        };

        if state.streams.hint_stream.len() < RV32_REGISTER_NUM_LIMBS * num_words as usize {
            return Err(ExecutionError::HintOutOfBounds { pc: *state.pc });
        }

        for idx in 0..(num_words as usize) {
            if idx != 0 {
                state.memory.increment_timestamp();
                state.memory.increment_timestamp();
            }

            let data_f: [F; RV32_REGISTER_NUM_LIMBS] =
                std::array::from_fn(|_| state.streams.hint_stream.pop_front().unwrap());
            let data: [u8; RV32_REGISTER_NUM_LIMBS] =
                data_f.map(|byte| byte.as_canonical_u32() as u8);

            record.var[idx].data = data;

            tracing_write(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.mem_ptr + (RV32_REGISTER_NUM_LIMBS * idx) as u32,
                data,
                &mut record.var[idx].data_write_aux.prev_timestamp,
                &mut record.var[idx].data_write_aux.prev_data,
            );
        }
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for Rv32HintStoreFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace.width;
        debug_assert_eq!(width, size_of::<Rv32HintStoreCols<u8>>());
        let mut trace = &mut trace.values[..width * rows_used];
        let mut sizes = Vec::with_capacity(rows_used);
        let mut chunks = Vec::with_capacity(rows_used);

        while !trace.is_empty() {
            // SAFETY:
            // - caller ensures `trace` contains a valid record representation that was previously
            //   written by the executor
            // - header is the first element of the record
            let record: &Rv32HintStoreRecordHeader =
                unsafe { get_record_from_slice(&mut trace, ()) };
            let (chunk, rest) = trace.split_at_mut(width * record.num_words as usize);
            sizes.push(record.num_words);
            chunks.push(chunk);
            trace = rest;
        }

        let msl_rshift: u32 = ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS) as u32;
        let msl_lshift: u32 =
            (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.pointer_max_bits) as u32;

        chunks
            .par_iter_mut()
            .zip(sizes.par_iter())
            .for_each(|(chunk, &num_words)| {
                // SAFETY:
                // - caller ensures `trace` contains a valid record representation that was
                //   previously written by the executor
                // - chunk contains a valid Rv32HintStoreRecordMut with the exact layout specified
                // - get_record_from_slice will correctly split the buffer into header and variable
                //   components based on this layout
                let record: Rv32HintStoreRecordMut = unsafe {
                    get_record_from_slice(
                        chunk,
                        MultiRowLayout::new(Rv32HintStoreMetadata {
                            num_words: num_words as usize,
                        }),
                    )
                };
                self.bitwise_lookup_chip.request_range(
                    (record.inner.mem_ptr >> msl_rshift) << msl_lshift,
                    (num_words >> msl_rshift) << msl_lshift,
                );

                let mut timestamp = record.inner.timestamp + num_words * 3;
                let mut mem_ptr = record.inner.mem_ptr + num_words * RV32_REGISTER_NUM_LIMBS as u32;

                // Assuming that `num_words` is usually small (e.g. 1 for `HINT_STOREW`)
                // it is better to do a serial pass of the rows per instruction (going from the last
                // row to the first row) instead of a parallel pass, since need to
                // copy the record to a new buffer in parallel case.
                chunk
                    .rchunks_exact_mut(width)
                    .zip(record.var.iter().enumerate().rev())
                    .for_each(|(row, (idx, var))| {
                        for pair in var.data.chunks_exact(2) {
                            self.bitwise_lookup_chip
                                .request_range(pair[0] as u32, pair[1] as u32);
                        }

                        let cols: &mut Rv32HintStoreCols<F> = row.borrow_mut();
                        let is_single = record.inner.num_words_ptr == u32::MAX;
                        timestamp -= 3;
                        if idx == 0 && !is_single {
                            mem_helper.fill(
                                record.inner.num_words_read.prev_timestamp,
                                timestamp + 1,
                                cols.num_words_aux_cols.as_mut(),
                            );
                            cols.num_words_ptr = F::from_canonical_u32(record.inner.num_words_ptr);
                        } else {
                            mem_helper.fill_zero(cols.num_words_aux_cols.as_mut());
                            cols.num_words_ptr = F::ZERO;
                        }

                        cols.is_buffer_start = F::from_bool(idx == 0 && !is_single);

                        // Note: writing in reverse
                        cols.data = var.data.map(|x| F::from_canonical_u8(x));

                        cols.write_aux.set_prev_data(
                            var.data_write_aux
                                .prev_data
                                .map(|x| F::from_canonical_u8(x)),
                        );
                        mem_helper.fill(
                            var.data_write_aux.prev_timestamp,
                            timestamp + 2,
                            cols.write_aux.as_mut(),
                        );

                        if idx == 0 {
                            mem_helper.fill(
                                record.inner.mem_ptr_aux_record.prev_timestamp,
                                timestamp,
                                cols.mem_ptr_aux_cols.as_mut(),
                            );
                        } else {
                            mem_helper.fill_zero(cols.mem_ptr_aux_cols.as_mut());
                        }

                        mem_ptr -= RV32_REGISTER_NUM_LIMBS as u32;
                        cols.mem_ptr_limbs = mem_ptr.to_le_bytes().map(|x| F::from_canonical_u8(x));
                        cols.mem_ptr_ptr = F::from_canonical_u32(record.inner.mem_ptr_ptr);

                        cols.from_state.timestamp = F::from_canonical_u32(timestamp);
                        cols.from_state.pc = F::from_canonical_u32(record.inner.from_pc);

                        cols.rem_words_limbs = (num_words - idx as u32)
                            .to_le_bytes()
                            .map(|x| F::from_canonical_u8(x));
                        cols.is_buffer = F::from_bool(!is_single);
                        cols.is_single = F::from_bool(is_single);
                    });
            })
    }
}

pub type Rv32HintStoreChip<F> = VmChipWrapper<F, Rv32HintStoreFiller>;
