use std::{array::from_fn, borrow::Borrow};

use itertools::izip;
use openvm_blake3_transpiler::Rv32Blake3Opcode;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, utils::not};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_rv32im_circuit::adapters::abstract_compose;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_blake3_air::{Blake3Air, NUM_BLAKE3_COLS as NUM_BLAKE3_COMPRESS_COLS};

use super::{
    columns::{Blake3VmCols, NUM_BLAKE3_VM_COLS},
    BLAKE3_BLOCK_BYTES, BLAKE3_DIGEST_BYTES, BLAKE3_DIGEST_WRITES, BLAKE3_INPUT_READS,
    BLAKE3_REGISTER_READS, BLAKE3_WORD_SIZE,
};

/// AIR for BLAKE3 VM extension.
///
/// This AIR wraps the p3-blake3-air compression function and adds VM-specific
/// constraints for memory access, instruction execution, and multi-block hashing.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Blake3VmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send 8-bit XOR/range check requests to.
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Maximum number of bits allowed for an address pointer.
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for Blake3VmAir {}
impl<F> PartitionedBaseAir<F> for Blake3VmAir {}
impl<F> BaseAir<F> for Blake3VmAir {
    fn width(&self) -> usize {
        NUM_BLAKE3_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for Blake3VmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &Blake3VmCols<AB::Var> = (*local).borrow();
        let next: &Blake3VmCols<AB::Var> = (*next).borrow();

        // 1. Evaluate the p3-blake3-air compression constraints
        self.eval_blake3_compress(builder);

        // 2. Constrain boolean flags
        self.constrain_flags(builder, local);

        // 3. Constrain chaining value flow between blocks
        self.constrain_chaining_values(builder, local, next);

        // 4. Constrain block transitions for multi-block hashing
        self.constrain_block_transition(builder, local, next);

        // 5. Instruction execution and register reads
        let start_read_timestamp =
            self.eval_instruction(builder, local, &local.mem_oc.register_aux);

        // 6. Input memory reads
        let start_write_timestamp = self.constrain_input_read(builder, local, start_read_timestamp);

        // 7. Output memory writes (only on last block)
        self.constrain_output_write(builder, local, start_write_timestamp);
    }
}

impl Blake3VmAir {
    /// Evaluate the blake3 compression function constraints.
    ///
    /// WARNING: The blake3 AIR columns **must** be the first columns in the main AIR.
    #[inline]
    pub fn eval_blake3_compress<AB: AirBuilder>(&self, builder: &mut AB) {
        let blake3_air = Blake3Air {};
        let mut sub_builder =
            SubAirBuilder::<AB, Blake3Air, AB::Var>::new(builder, 0..NUM_BLAKE3_COMPRESS_COLS);
        blake3_air.eval(&mut sub_builder);
    }

    /// Constrain boolean flags and their relationships.
    fn constrain_flags<AB: AirBuilder>(&self, builder: &mut AB, local: &Blake3VmCols<AB::Var>) {
        let instruction = &local.instruction;

        // All flags must be boolean
        builder.assert_bool(instruction.is_enabled);
        builder.assert_bool(instruction.is_new_start);
        builder.assert_bool(instruction.is_last_block);

        // is_enabled_first_block = is_enabled * is_new_start
        builder.assert_eq(
            instruction.is_enabled_first_block,
            instruction.is_enabled * instruction.is_new_start,
        );

        // On first row, if enabled, must be a new start
        builder
            .when_first_row()
            .when(instruction.is_enabled)
            .assert_one(instruction.is_new_start);

        // is_new_start implies is_enabled (can't have new start on dummy row)
        builder
            .when(instruction.is_new_start)
            .assert_one(instruction.is_enabled);
    }

    /// Constrain chaining value flow between consecutive blocks.
    ///
    /// For BLAKE3:
    /// - First block uses IV as chaining value
    /// - Subsequent blocks use truncated output of previous compression
    fn constrain_chaining_values<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Blake3VmCols<AB::Var>,
        next: &Blake3VmCols<AB::Var>,
    ) {
        let instruction = &local.instruction;

        // When this is NOT the last block and next row continues the same hash,
        // the next row's chaining value should equal this row's output (truncated).
        //
        // p3-blake3-air structure:
        // - inner.chaining_values: [[[T; 32]; 4]; 2] = 8 words × 32 bits
        // - inner.outputs: [[[T; 32]; 4]; 4] = 16 words × 32 bits
        //
        // BLAKE3 truncation: new_cv = outputs[0..8] XOR outputs[8..16]
        // But for simplicity in the AIR, we can constrain that next.chaining_values
        // equals the appropriately truncated outputs.

        let is_continuing = instruction.is_enabled * not(instruction.is_last_block);

        // When continuing, next row must not be a new start
        builder
            .when_transition()
            .when(is_continuing.clone())
            .assert_zero(next.instruction.is_new_start);

        // Constrain chaining value transition:
        // new_cv = outputs[0..8] (first 8 output words, NOT XORed with anything)
        //
        // The chaining_values are stored as [[[T; 32]; 4]; 2] = 2 groups of 4 words
        // outputs[0] = first 4 words of output = new_cv[0..4]
        // outputs[1] = next 4 words of output = new_cv[4..8]
        for group in 0..2 {
            for word in 0..4 {
                for bit in 0..32 {
                    let output_bit = local.inner.outputs[group][word][bit];
                    let next_cv_bit = next.inner.chaining_values[group][word][bit];

                    builder
                        .when_transition()
                        .when(is_continuing.clone())
                        .assert_eq(next_cv_bit, output_bit);
                }
            }
        }

        // When is_new_start, chaining values should be IV
        // IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        //       0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
        let iv: [u32; 8] = [
            0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
            0x5BE0CD19,
        ];

        let is_first_block = instruction.is_enabled_first_block;
        for (idx, &iv_word) in iv.iter().enumerate() {
            let group = idx / 4;
            let word = idx % 4;
            for bit in 0..32 {
                let expected_bit = AB::F::from_canonical_u32((iv_word >> bit) & 1);
                builder
                    .when(is_first_block)
                    .assert_eq(local.inner.chaining_values[group][word][bit], expected_bit);
            }
        }
    }

    /// Constrain consistency between consecutive blocks of the same hash.
    fn constrain_block_transition<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        local: &Blake3VmCols<AB::Var>,
        next: &Blake3VmCols<AB::Var>,
    ) {
        let instruction = &local.instruction;

        // When transitioning between blocks of the same hash
        let is_continuing = instruction.is_enabled * not(instruction.is_last_block);

        let mut transition_builder = builder.when_transition();
        let mut block_transition = transition_builder.when(is_continuing);

        // Instruction metadata stays the same
        block_transition.assert_eq(instruction.pc, next.instruction.pc);
        block_transition.assert_eq(instruction.is_enabled, next.instruction.is_enabled);
        block_transition.assert_eq(instruction.dst_ptr, next.instruction.dst_ptr);
        block_transition.assert_eq(instruction.src_ptr, next.instruction.src_ptr);
        block_transition.assert_eq(instruction.len_ptr, next.instruction.len_ptr);

        // dst stays the same (only used on last block)
        for i in 0..RV32_REGISTER_NUM_LIMBS {
            block_transition.assert_eq(instruction.dst[i], next.instruction.dst[i]);
        }

        // src pointer advances by BLAKE3_BLOCK_BYTES
        block_transition.assert_eq(
            next.instruction.src,
            instruction.src + AB::F::from_canonical_usize(BLAKE3_BLOCK_BYTES),
        );

        // remaining_len decreases by BLAKE3_BLOCK_BYTES
        block_transition.assert_eq(
            next.instruction.remaining_len,
            instruction.remaining_len - AB::F::from_canonical_usize(BLAKE3_BLOCK_BYTES),
        );

        // Timestamp advances by register reads + input reads
        block_transition.assert_eq(
            next.instruction.start_timestamp,
            instruction.start_timestamp
                + AB::F::from_canonical_usize(BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS),
        );

        // After last block, next must be a new start (or dummy)
        builder
            .when_transition()
            .when(instruction.is_last_block * instruction.is_enabled)
            .assert_one(next.instruction.is_new_start + not(next.instruction.is_enabled));
    }

    /// Evaluate instruction execution: program bus, execution bus, register reads.
    /// Returns the timestamp after register reads.
    fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Blake3VmCols<AB::Var>,
        register_aux: &[openvm_circuit::system::memory::offline_checker::MemoryReadAuxCols<AB::Var>;
             BLAKE3_REGISTER_READS],
    ) -> AB::Expr {
        let instruction = &local.instruction;

        // Only receive instruction on first block of a new hash
        let should_receive = instruction.is_enabled_first_block;

        let timestamp_change = Self::timestamp_change::<AB::Expr>(instruction.remaining_len.into());

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(Rv32Blake3Opcode::BLAKE3 as usize + self.offset),
                [
                    instruction.dst_ptr.into(),
                    instruction.src_ptr.into(),
                    instruction.len_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change,
            )
            .eval(builder, should_receive.clone());

        // Read registers: dst, src, len
        let mut timestamp: AB::Expr = instruction.start_timestamp.into();
        let reg_addr_space = AB::F::from_canonical_u32(RV32_REGISTER_AS);

        // Helper to recover 4-byte register value from limbs representation
        let recover_limbs = |limbs: [AB::Var; RV32_REGISTER_NUM_LIMBS - 1],
                             val: AB::Var|
         -> [AB::Expr; RV32_REGISTER_NUM_LIMBS] {
            from_fn(|i| {
                if i == 0 {
                    // byte 0 = val - sum(limbs[j] << ((j+1)*8))
                    limbs
                        .into_iter()
                        .enumerate()
                        .fold(val.into(), |acc, (j, limb)| {
                            acc - limb
                                * AB::Expr::from_canonical_usize(1 << ((j + 1) * RV32_CELL_BITS))
                        })
                } else {
                    limbs[i - 1].into()
                }
            })
        };

        let dst_data = instruction.dst.map(Into::into);
        let src_data = recover_limbs(instruction.src_limbs, instruction.src);
        let len_data = recover_limbs(instruction.len_limbs, instruction.remaining_len);

        for (ptr, value, aux) in izip!(
            [
                instruction.dst_ptr,
                instruction.src_ptr,
                instruction.len_ptr
            ],
            [dst_data, src_data, len_data],
            register_aux,
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(reg_addr_space, ptr),
                    value,
                    timestamp.clone(),
                    aux,
                )
                .eval(builder, should_receive.clone());

            timestamp += AB::Expr::ONE;
        }

        // Range check the most significant limbs to ensure pointers fit in ptr_max_bits
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        let need_range_check = [
            *instruction.dst.last().unwrap(),
            *instruction.src_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(), // repeat for even count
        ];
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, should_receive.clone());
        }

        timestamp
    }

    /// Constrain reading input block from memory.
    /// Returns timestamp after input reads.
    fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Blake3VmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
    ) -> AB::Expr {
        let instruction = &local.instruction;
        let mem = &local.mem_oc;

        // Only read input when enabled
        let is_enabled = instruction.is_enabled;

        let mut timestamp = start_read_timestamp;

        // Read 64 bytes (16 words of 4 bytes each)
        // p3-blake3-air has inner.inputs: [[T; 32]; 16] - 16 words, 32 bits each
        for i in 0..BLAKE3_INPUT_READS {
            let ptr = instruction.src + AB::F::from_canonical_usize(i * BLAKE3_WORD_SIZE);

            // Convert bits to bytes for memory read
            // inner.inputs[i] has 32 bits, we need 4 bytes (little-endian)
            let word_bits = &local.inner.inputs[i];
            let word_bytes: [AB::Expr; BLAKE3_WORD_SIZE] = from_fn(|byte_idx| {
                // Each byte is 8 bits starting at bit byte_idx*8
                (0..8).fold(AB::Expr::ZERO, |acc, bit| {
                    acc + word_bits[byte_idx * 8 + bit] * AB::F::from_canonical_usize(1 << bit)
                })
            });

            let word: [AB::Expr; BLAKE3_WORD_SIZE] = from_fn(|j| {
                if j == 0 {
                    word_bytes[0].clone()
                } else {
                    // Select between partial_block and word_bytes based on whether
                    // this is a partial read at the end
                    word_bytes[j].clone()
                }
            });

            // Determine if we should read this word
            // We should read if i * BLAKE3_WORD_SIZE < remaining_len (on first block)
            // or if i * BLAKE3_WORD_SIZE < remaining_len (on subsequent blocks)
            // For simplicity, always read when enabled - trace filler zero-pads

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::F::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    word,
                    timestamp.clone(),
                    &mem.input_reads[i],
                )
                .eval(builder, is_enabled);

            timestamp += AB::Expr::ONE;
        }

        timestamp
    }

    /// Constrain writing digest output to memory (only on last block).
    fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Blake3VmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
    ) {
        let instruction = &local.instruction;
        let mem = &local.mem_oc;

        // Only write output on last block of enabled hash
        let is_final = instruction.is_enabled * instruction.is_last_block;

        // p3-blake3-air outputs structure:
        // outputs[0][i] = state_row0[i] ^ state_row2[i] = word[i] ^ word[i+8]
        // outputs[1][i] = state_row1[i] ^ state_row3[i] = word[i+4] ^ word[i+12]
        //
        // So outputs[0..2] already contains the truncated digest (XOR of state halves).
        // We just need to convert bits to bytes and write.

        let dst = abstract_compose::<AB::Expr, _>(instruction.dst.map(Into::into));

        for i in 0..(BLAKE3_DIGEST_BYTES / BLAKE3_WORD_SIZE) {
            let group = i / 4; // 0 for words 0-3, 1 for words 4-7
            let word_idx = i % 4;

            // Convert output bits to bytes (no additional XOR needed)
            let digest_bytes: [AB::Expr; BLAKE3_WORD_SIZE] = from_fn(|byte_idx| {
                (0..8).fold(AB::Expr::ZERO, |acc, bit| {
                    let bit_idx = byte_idx * 8 + bit;
                    let out_bit = local.inner.outputs[group][word_idx][bit_idx];
                    acc + out_bit * AB::F::from_canonical_usize(1 << bit)
                })
            });

            let timestamp = start_write_timestamp.clone() + AB::F::from_canonical_usize(i);

            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::F::from_canonical_u32(RV32_MEMORY_AS),
                        dst.clone() + AB::F::from_canonical_usize(i * BLAKE3_WORD_SIZE),
                    ),
                    digest_bytes,
                    timestamp,
                    &mem.digest_writes[i],
                )
                .eval(builder, is_final.clone());
        }
    }

    /// Calculate timestamp change for one complete hash operation.
    /// This is an upper bound based on input length.
    pub fn timestamp_change<T: FieldAlgebra>(len: T) -> T {
        // Per block: BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS
        // Final block adds: BLAKE3_DIGEST_WRITES
        // Number of blocks: ceil(len / 64)
        // Upper bound: len + REGISTER_READS + INPUT_READS + DIGEST_WRITES
        len + T::from_canonical_usize(
            BLAKE3_REGISTER_READS + BLAKE3_INPUT_READS + BLAKE3_DIGEST_WRITES,
        )
    }
}
