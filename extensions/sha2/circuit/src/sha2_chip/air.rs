use std::{cmp::min, convert::TryInto};

use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, encoder::Encoder, utils::not, SubAir,
};
use openvm_instructions::{
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_sha2_air::{compose, Sha256Config, Sha2Air, Sha512Config};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{Sha2Variant, Sha2VmDigestColsRef, Sha2VmRoundColsRef, ShaChipConfig};

/// Sha2VmAir does all constraints related to message padding and
/// the Sha2Air subair constrains the actual hash
#[derive(Clone, Debug, derive_new::new)]
pub struct Sha2VmAir<C: ShaChipConfig> {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send byte checks to
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Maximum number of bits allowed for an address pointer
    /// Must be at least 24
    pub ptr_max_bits: usize,
    pub(super) sha_subair: Sha2Air<C>,
    pub(super) padding_encoder: Encoder,
}

impl<F: Field, C: ShaChipConfig> BaseAirWithPublicValues<F> for Sha2VmAir<C> {}
impl<F: Field, C: ShaChipConfig> PartitionedBaseAir<F> for Sha2VmAir<C> {}
impl<F: Field, C: ShaChipConfig> BaseAir<F> for Sha2VmAir<C> {
    fn width(&self) -> usize {
        C::VM_WIDTH
    }
}

impl<AB: InteractionBuilder, C: ShaChipConfig> Air<AB> for Sha2VmAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.eval_padding(builder);
        self.eval_transitions(builder);
        self.eval_reads(builder);
        self.eval_last_row(builder);

        self.sha_subair.eval(builder, C::VM_CONTROL_WIDTH);
    }
}

#[allow(dead_code, non_camel_case_types)]
pub(super) enum PaddingFlags {
    /// Not considered for padding - W's are not constrained
    NotConsidered,
    /// Not padding - W's should be equal to the message
    NotPadding,
    /// FIRST_PADDING_i: it is the first row with padding and there are i cells of non-padding
    FirstPadding0,
    FirstPadding1,
    FirstPadding2,
    FirstPadding3,
    FirstPadding4,
    FirstPadding5,
    FirstPadding6,
    FirstPadding7,
    FirstPadding8,
    FirstPadding9,
    FirstPadding10,
    FirstPadding11,
    FirstPadding12,
    FirstPadding13,
    FirstPadding14,
    FirstPadding15,
    FirstPadding16,
    FirstPadding17,
    FirstPadding18,
    FirstPadding19,
    FirstPadding20,
    FirstPadding21,
    FirstPadding22,
    FirstPadding23,
    FirstPadding24,
    FirstPadding25,
    FirstPadding26,
    FirstPadding27,
    FirstPadding28,
    FirstPadding29,
    FirstPadding30,
    FirstPadding31,
    /// FIRST_PADDING_i_LastRow: it is the first row with padding and there are i cells of non-padding
    ///                          AND it is the last reading row of the message
    /// NOTE: if the Last row has padding it has to be at least 9 cells since the last 8 cells are padded with
    /// the message length
    FirstPadding0_LastRow,
    FirstPadding1_LastRow,
    FirstPadding2_LastRow,
    FirstPadding3_LastRow,
    FirstPadding4_LastRow,
    FirstPadding5_LastRow,
    FirstPadding6_LastRow,
    FirstPadding7_LastRow,
    FirstPadding8_LastRow,
    FirstPadding9_LastRow,
    FirstPadding10_LastRow,
    FirstPadding11_LastRow,
    FirstPadding12_LastRow,
    FirstPadding13_LastRow,
    FirstPadding14_LastRow,
    FirstPadding15_LastRow,

    /// The entire row is padding AND it is not the first row with padding
    /// AND it is the 4th row of the last block of the message
    EntirePaddingLastRow,
    /// The entire row is padding AND it is not the first row with padding
    EntirePadding,
}

impl PaddingFlags {
    /// The number of padding flags (including NotConsidered)
    pub const COUNT: usize = EntirePadding as usize + 1;
}

use PaddingFlags::*;
impl<C: ShaChipConfig> Sha2VmAir<C> {
    /// Implement all necessary constraints for the padding
    fn eval_padding<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local_cols = Sha2VmRoundColsRef::<AB::Var>::from::<C>(&local[..C::VM_ROUND_WIDTH]);
        let next_cols = Sha2VmRoundColsRef::<AB::Var>::from::<C>(&next[..C::VM_ROUND_WIDTH]);

        // Constrain the sanity of the padding flags
        self.padding_encoder
            .eval(builder, local_cols.control.pad_flags.as_slice().unwrap());

        builder.assert_one(self.padding_encoder.contains_flag_range::<AB>(
            local_cols.control.pad_flags.as_slice().unwrap(),
            NotConsidered as usize..=EntirePadding as usize,
        ));

        Self::eval_padding_transitions(self, builder, &local_cols, &next_cols);
        Self::eval_padding_row(self, builder, &local_cols);
    }

    fn eval_padding_transitions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2VmRoundColsRef<AB::Var>,
        next: &Sha2VmRoundColsRef<AB::Var>,
    ) {
        let next_is_last_row = *next.inner.flags.is_digest_row * *next.inner.flags.is_last_block;

        // Constrain that `padding_occured` is 1 on a suffix of rows in each message, excluding the last
        // digest row, and 0 everywhere else. Furthermore, the suffix starts in the first 4 rows of some block.

        builder.assert_bool(*local.control.padding_occurred);
        // Last round row in the last block has padding_occurred = 1
        // This is the end of the suffix
        builder
            .when(next_is_last_row.clone())
            .assert_one(*local.control.padding_occurred);

        // Digest row in the last block has padding_occurred = 0
        builder
            .when(next_is_last_row.clone())
            .assert_zero(*next.control.padding_occurred);

        // If padding_occurred = 1 in the current row, then padding_occurred = 1 in the next row,
        // unless next is the last digest row
        builder
            .when(*local.control.padding_occurred - next_is_last_row.clone())
            .assert_one(*next.control.padding_occurred);

        // If next row is not first 4 rows of a block, then next.padding_occurred = local.padding_occurred.
        // So padding_occurred only changes in the first 4 rows of a block.
        builder
            .when_transition()
            .when(not(*next.inner.flags.is_first_4_rows) - next_is_last_row)
            .assert_eq(
                *next.control.padding_occurred,
                *local.control.padding_occurred,
            );

        // Constrain the that the start of the padding is correct
        let next_is_first_padding_row =
            *next.control.padding_occurred - *local.control.padding_occurred;
        // Row index if its between 0..4, else 0
        let next_row_idx = self.sha_subair.row_idx_encoder.flag_with_val::<AB>(
            next.inner.flags.row_idx.as_slice().unwrap(),
            &(0..C::MESSAGE_ROWS).map(|x| (x, x)).collect::<Vec<_>>(),
        );
        // How many non-padding cells there are in the next row.
        // Will be 0 on non-padding rows.
        let next_padding_offset = self.padding_encoder.flag_with_val::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            &(0..C::MAX_FIRST_PADDING + 1)
                .map(|i| (FirstPadding0 as usize + i, i))
                .collect::<Vec<_>>(),
        ) + self.padding_encoder.flag_with_val::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            &(0..C::MAX_FIRST_PADDING_LAST_ROW + 1)
                .map(|i| (FirstPadding0_LastRow as usize + i, i))
                .collect::<Vec<_>>(),
        );

        // Will be 0 on last digest row since:
        //   - padding_occurred = 0 is constrained above
        //   - next_row_idx = 0 since row_idx is not in 0..4
        //   - and next_padding_offset = 0 since `pad_flags = NotConsidered`
        let expected_len = *next.inner.flags.local_block_idx
            * *next.control.padding_occurred
            * AB::Expr::from_canonical_usize(C::BLOCK_U8S)
            + next_row_idx * AB::Expr::from_canonical_usize(C::READ_SIZE)
            + next_padding_offset;

        // Note: `next_is_first_padding_row` is either -1,0,1
        // If 1, then this constrains the length of message
        // If -1, then `next` must be the last digest row and so this constraint will be 0 == 0
        builder.when(next_is_first_padding_row).assert_eq(
            expected_len,
            *next.control.len * *next.control.padding_occurred,
        );

        // Constrain the padding flags are of correct type (eg is not padding or first padding)
        let is_next_first_padding = self.padding_encoder.contains_flag_range::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            FirstPadding0 as usize..=(FirstPadding15_LastRow as usize),
        );

        let is_next_last_padding = self.padding_encoder.contains_flag_range::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            FirstPadding0_LastRow as usize..=EntirePaddingLastRow as usize,
        );

        let is_next_entire_padding = self.padding_encoder.contains_flag_range::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            EntirePaddingLastRow as usize..=EntirePadding as usize,
        );

        let is_next_not_considered = self.padding_encoder.contains_flag::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            &[NotConsidered as usize],
        );

        let is_next_not_padding = self.padding_encoder.contains_flag::<AB>(
            next.control.pad_flags.as_slice().unwrap(),
            &[NotPadding as usize],
        );

        let is_next_4th_row = self
            .sha_subair
            .row_idx_encoder
            .contains_flag::<AB>(next.inner.flags.row_idx.as_slice().unwrap(), &[3]);

        // `pad_flags` is `NotConsidered` on all rows except the first 4 rows of a block
        builder.assert_eq(
            not(*next.inner.flags.is_first_4_rows),
            is_next_not_considered,
        );

        // `pad_flags` is `EntirePadding` if the previous row is padding
        builder.when(*next.inner.flags.is_first_4_rows).assert_eq(
            *local.control.padding_occurred * *next.control.padding_occurred,
            is_next_entire_padding,
        );

        // `pad_flags` is `FirstPadding*` if current row is padding and the previous row is not padding
        builder.when(*next.inner.flags.is_first_4_rows).assert_eq(
            not(*local.control.padding_occurred) * *next.control.padding_occurred,
            is_next_first_padding,
        );

        // `pad_flags` is `NotPadding` if current row is not padding
        builder
            .when(*next.inner.flags.is_first_4_rows)
            .assert_eq(not(*next.control.padding_occurred), is_next_not_padding);

        // `pad_flags` is `*LastRow` on the row that contsrains the last four words of the message
        builder
            .when(*next.inner.flags.is_last_block)
            .assert_eq(is_next_4th_row, is_next_last_padding);
    }

    fn eval_padding_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2VmRoundColsRef<AB::Var>,
    ) {
        let message = (0..C::READ_SIZE)
            .map(|i| {
                local.inner.message_schedule.carry_or_buffer[[i / (C::WORD_U8S), i % (C::WORD_U8S)]]
            })
            .collect::<Vec<_>>();

        let get_ith_byte = |i: usize| {
            let word_idx = i / C::WORD_U8S;
            let word = local
                .inner
                .message_schedule
                .w
                .row(word_idx)
                .mapv(|x| x.into());
            // Need to reverse the byte order to match the endianness of the memory
            let byte_idx = C::WORD_U8S - i % C::WORD_U8S - 1;
            compose::<AB::Expr>(
                &word.as_slice().unwrap()[byte_idx * 8..(byte_idx + 1) * 8],
                1,
            )
        };

        let is_not_padding = self.padding_encoder.contains_flag::<AB>(
            local.control.pad_flags.as_slice().unwrap(),
            &[NotPadding as usize],
        );

        // Check the `w`s on case by case basis
        for (i, message_byte) in message.iter().enumerate() {
            let w = get_ith_byte(i);
            let should_be_message = is_not_padding.clone()
                + if i < C::MAX_FIRST_PADDING {
                    self.padding_encoder.contains_flag_range::<AB>(
                        local.control.pad_flags.as_slice().unwrap(),
                        FirstPadding0 as usize + i + 1
                            ..=FirstPadding0 as usize + C::MAX_FIRST_PADDING,
                    )
                } else {
                    AB::Expr::ZERO
                }
                + if i < C::MAX_FIRST_PADDING_LAST_ROW {
                    self.padding_encoder.contains_flag_range::<AB>(
                        local.control.pad_flags.as_slice().unwrap(),
                        FirstPadding0_LastRow as usize + i + 1
                            ..=FirstPadding0_LastRow as usize + C::MAX_FIRST_PADDING_LAST_ROW,
                    )
                } else {
                    AB::Expr::ZERO
                };

            builder
                .when(should_be_message)
                .assert_eq(w.clone(), *message_byte);

            let should_be_zero = self.padding_encoder.contains_flag::<AB>(
                local.control.pad_flags.as_slice().unwrap(),
                &[EntirePadding as usize],
            ) +
            // - 4 because the last 4 bytes are the padded length
            if i < C::CELLS_PER_ROW - 4 {
                self.padding_encoder.contains_flag::<AB>(
                    local.control.pad_flags.as_slice().unwrap(),
                    &[EntirePaddingLastRow as usize],
                ) + if i > 0 {
                    self.padding_encoder.contains_flag_range::<AB>(
                        local.control.pad_flags.as_slice().unwrap(),
                        FirstPadding0_LastRow as usize
                            ..=min(
                                FirstPadding0_LastRow as usize + i - 1,
                                FirstPadding0_LastRow as usize + C::MAX_FIRST_PADDING_LAST_ROW,
                            ),
                    )
                } else {
                    AB::Expr::ZERO
                }
            } else {
                AB::Expr::ZERO
            } + if i > 0 {
                self.padding_encoder.contains_flag_range::<AB>(
                    local.control.pad_flags.as_slice().unwrap(),
                    FirstPadding0 as usize..=FirstPadding0 as usize + i - 1,
                )
            } else {
                AB::Expr::ZERO
            };
            builder.when(should_be_zero).assert_zero(w.clone());

            // Assumes bit-length of message is a multiple of 8 (message is bytes)
            // This is true because the message is given as &[u8]
            let should_be_128 = self.padding_encoder.contains_flag::<AB>(
                local.control.pad_flags.as_slice().unwrap(),
                &[FirstPadding0 as usize + i],
            ) + if i < 8 {
                self.padding_encoder.contains_flag::<AB>(
                    local.control.pad_flags.as_slice().unwrap(),
                    &[FirstPadding0_LastRow as usize + i],
                )
            } else {
                AB::Expr::ZERO
            };

            builder
                .when(should_be_128)
                .assert_eq(AB::Expr::from_canonical_u32(1 << 7), w);

            // should be len is handled outside of the loop
        }
        let appended_len = compose::<AB::Expr>(
            &[
                get_ith_byte(C::CELLS_PER_ROW - 1),
                get_ith_byte(C::CELLS_PER_ROW - 2),
                get_ith_byte(C::CELLS_PER_ROW - 3),
                get_ith_byte(C::CELLS_PER_ROW - 4),
            ],
            RV32_CELL_BITS,
        );

        let actual_len = *local.control.len;

        let is_last_padding_row = self.padding_encoder.contains_flag_range::<AB>(
            local.control.pad_flags.as_slice().unwrap(),
            FirstPadding0_LastRow as usize..=EntirePaddingLastRow as usize,
        );

        builder.when(is_last_padding_row.clone()).assert_eq(
            appended_len * AB::F::from_canonical_usize(RV32_CELL_BITS).inverse(), // bit to byte conversion
            actual_len,
        );

        // We constrain that the appended length is in bytes
        builder.when(is_last_padding_row.clone()).assert_zero(
            local.inner.message_schedule.w[[3, 0]]
                + local.inner.message_schedule.w[[3, 1]]
                + local.inner.message_schedule.w[[3, 2]],
        );

        // We can't support messages longer than 2^29 bytes because the length has to fit in a field element.
        // So, constrain that the first 4 bytes of the length are 0.
        // Thus, the bit-length is < 2^32 so the message is < 2^29 bytes.
        for i in 8..12 {
            builder
                .when(is_last_padding_row.clone())
                .assert_zero(get_ith_byte(i));
        }
    }
    /// Implement constraints on `len`, `read_ptr` and `cur_timestamp`
    fn eval_transitions<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local_cols = Sha2VmRoundColsRef::<AB::Var>::from::<C>(&local[..C::VM_ROUND_WIDTH]);
        let next_cols = Sha2VmRoundColsRef::<AB::Var>::from::<C>(&next[..C::VM_ROUND_WIDTH]);

        let is_last_row =
            *local_cols.inner.flags.is_last_block * *local_cols.inner.flags.is_digest_row;
        // Len should be the same for the entire message
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_last_row.clone()))
            .assert_eq(*next_cols.control.len, *local_cols.control.len);

        // Read ptr should increment by [C::READ_SIZE] for the first 4 rows and stay the same otherwise
        let read_ptr_delta =
            *local_cols.inner.flags.is_first_4_rows * AB::Expr::from_canonical_usize(C::READ_SIZE);
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_last_row.clone()))
            .assert_eq(
                *next_cols.control.read_ptr,
                *local_cols.control.read_ptr + read_ptr_delta,
            );

        // Timestamp should increment by 1 for the first 4 rows and stay the same otherwise
        let timestamp_delta = *local_cols.inner.flags.is_first_4_rows * AB::Expr::ONE;
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_last_row.clone()))
            .assert_eq(
                *next_cols.control.cur_timestamp,
                *local_cols.control.cur_timestamp + timestamp_delta,
            );
    }

    /// Implement the reads for the first 4 rows of a block
    fn eval_reads<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_cols = Sha2VmRoundColsRef::<AB::Var>::from::<C>(&local[..C::VM_ROUND_WIDTH]);

        let message: Vec<AB::Var> = (0..C::READ_SIZE)
            .map(|i| {
                local_cols.inner.message_schedule.carry_or_buffer
                    [[i / (C::WORD_U16S * 2), i % (C::WORD_U16S * 2)]]
            })
            .collect();

        match C::VARIANT {
            Sha2Variant::Sha256 => {
                let message: [AB::Var; Sha256Config::READ_SIZE] =
                    message.try_into().unwrap_or_else(|_| {
                        panic!("message is not the correct size");
                    });
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                            *local_cols.control.read_ptr,
                        ),
                        message,
                        *local_cols.control.cur_timestamp,
                        local_cols.read_aux,
                    )
                    .eval(builder, *local_cols.inner.flags.is_first_4_rows);
            }
            // Sha512 and Sha384 have the same read size so we put them together
            Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                let message: [AB::Var; Sha512Config::READ_SIZE] =
                    message.try_into().unwrap_or_else(|_| {
                        panic!("message is not the correct size");
                    });
                self.memory_bridge
                    .read(
                        MemoryAddress::new(
                            AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                            *local_cols.control.read_ptr,
                        ),
                        message,
                        *local_cols.control.cur_timestamp,
                        local_cols.read_aux,
                    )
                    .eval(builder, *local_cols.inner.flags.is_first_4_rows);
            }
        }
    }
    /// Implement the constraints for the last row of a message
    fn eval_last_row<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local_cols = Sha2VmDigestColsRef::<AB::Var>::from::<C>(&local[..C::VM_DIGEST_WIDTH]);

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        let is_last_row =
            *local_cols.inner.flags.is_last_block * *local_cols.inner.flags.is_digest_row;

        let dst_ptr: [AB::Var; RV32_REGISTER_NUM_LIMBS] =
            local_cols.dst_ptr.to_vec().try_into().unwrap_or_else(|_| {
                panic!("dst_ptr is not the correct size");
            });
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    *local_cols.rd_ptr,
                ),
                dst_ptr,
                timestamp_pp(),
                &local_cols.register_reads_aux[0],
            )
            .eval(builder, is_last_row.clone());

        let src_ptr: [AB::Var; RV32_REGISTER_NUM_LIMBS] =
            local_cols.src_ptr.to_vec().try_into().unwrap_or_else(|_| {
                panic!("src_ptr is not the correct size");
            });
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    *local_cols.rs1_ptr,
                ),
                src_ptr,
                timestamp_pp(),
                &local_cols.register_reads_aux[1],
            )
            .eval(builder, is_last_row.clone());

        let len_data: [AB::Var; RV32_REGISTER_NUM_LIMBS] =
            local_cols.len_data.to_vec().try_into().unwrap_or_else(|_| {
                panic!("len_data is not the correct size");
            });
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    *local_cols.rs2_ptr,
                ),
                len_data,
                timestamp_pp(),
                &local_cols.register_reads_aux[2],
            )
            .eval(builder, is_last_row.clone());
        // range check that the memory pointers don't overflow
        // Note: no need to range check the length since we read from memory step by step and
        //       the memory bus will catch any memory accesses beyond ptr_max_bits
        let shift = AB::Expr::from_canonical_usize(
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.ptr_max_bits),
        );
        // This only works if self.ptr_max_bits >= 24 which is typically the case
        self.bitwise_lookup_bus
            .send_range(
                // It is fine to shift like this since we already know that dst_ptr and src_ptr have [RV32_CELL_BITS] bits
                local_cols.dst_ptr[RV32_REGISTER_NUM_LIMBS - 1] * shift.clone(),
                local_cols.src_ptr[RV32_REGISTER_NUM_LIMBS - 1] * shift.clone(),
            )
            .eval(builder, is_last_row.clone());

        // the number of reads that happened to read the entire message: we do 4 reads per block
        let time_delta = (*local_cols.inner.flags.local_block_idx + AB::Expr::ONE)
            * AB::Expr::from_canonical_usize(4);
        // Every time we read the message we increment the read pointer by C::READ_SIZE
        let read_ptr_delta = time_delta.clone() * AB::Expr::from_canonical_usize(C::READ_SIZE);

        let result: Vec<AB::Var> = (0..C::HASH_SIZE)
            .map(|i| {
                // The limbs are written in big endian order to the memory so need to be reversed
                local_cols.inner.final_hash[[i / C::WORD_U8S, C::WORD_U8S - i % C::WORD_U8S - 1]]
            })
            .collect();

        let dst_ptr_val = compose::<AB::Expr>(
            local_cols.dst_ptr.mapv(|x| x.into()).as_slice().unwrap(),
            RV32_CELL_BITS,
        );

        match C::VARIANT {
            Sha2Variant::Sha256 => {
                debug_assert_eq!(C::NUM_WRITES, 1);
                debug_assert_eq!(local_cols.writes_aux_base.len(), 1);
                debug_assert_eq!(local_cols.writes_aux_prev_data.nrows(), 1);
                let prev_data: [AB::Var; Sha256Config::HASH_SIZE] = local_cols
                    .writes_aux_prev_data
                    .row(0)
                    .to_vec()
                    .try_into()
                    .unwrap_or_else(|_| {
                        panic!("writes_aux_prev_data is not the correct size");
                    });
                // Note: revisit in the future to do 2 block writes of 16 cells instead of 1 block write of 32 cells
                //       This could be beneficial as the output is often an input for another hash
                self.memory_bridge
                    .write(
                        MemoryAddress::new(
                            AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                            dst_ptr_val,
                        ),
                        result.try_into().unwrap_or_else(|_| {
                            panic!("result is not the correct size");
                        }),
                        timestamp_pp() + time_delta.clone(),
                        &MemoryWriteAuxCols::from_base(local_cols.writes_aux_base[0], prev_data),
                    )
                    .eval(builder, is_last_row.clone());
            }
            Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
                debug_assert_eq!(C::NUM_WRITES, 2);
                debug_assert_eq!(local_cols.writes_aux_base.len(), 2);
                debug_assert_eq!(local_cols.writes_aux_prev_data.nrows(), 2);

                // For Sha384, set the last 16 cells to 0
                let mut truncated_result: Vec<AB::Expr> =
                    result.iter().map(|x| (*x).into()).collect();
                for x in truncated_result.iter_mut().skip(C::DIGEST_SIZE) {
                    *x = AB::Expr::ZERO;
                }

                // write the digest in two halves because we only support writes up to 32 bytes
                for i in 0..Sha512Config::NUM_WRITES {
                    let prev_data: [AB::Var; Sha512Config::WRITE_SIZE] = local_cols
                        .writes_aux_prev_data
                        .row(i)
                        .to_vec()
                        .try_into()
                        .unwrap_or_else(|_| {
                            panic!("writes_aux_prev_data is not the correct size");
                        });

                    self.memory_bridge
                        .write(
                            MemoryAddress::new(
                                AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                                dst_ptr_val.clone()
                                    + AB::Expr::from_canonical_usize(i * Sha512Config::WRITE_SIZE),
                            ),
                            truncated_result
                                [i * Sha512Config::WRITE_SIZE..(i + 1) * Sha512Config::WRITE_SIZE]
                                .to_vec()
                                .try_into()
                                .unwrap_or_else(|_| {
                                    panic!("result is not the correct size");
                                }),
                            timestamp_pp() + time_delta.clone(),
                            &MemoryWriteAuxCols::from_base(
                                local_cols.writes_aux_base[i],
                                prev_data,
                            ),
                        )
                        .eval(builder, is_last_row.clone());
                }
            }
        }
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(C::OPCODE.global_opcode().as_usize()),
                [
                    <AB::Var as Into<AB::Expr>>::into(*local_cols.rd_ptr),
                    <AB::Var as Into<AB::Expr>>::into(*local_cols.rs1_ptr),
                    <AB::Var as Into<AB::Expr>>::into(*local_cols.rs2_ptr),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                *local_cols.from_state,
                AB::Expr::from_canonical_usize(timestamp_delta) + time_delta.clone(),
            )
            .eval(builder, is_last_row.clone());

        // Assert that we read the correct length of the message
        let len_val = compose::<AB::Expr>(
            local_cols.len_data.mapv(|x| x.into()).as_slice().unwrap(),
            RV32_CELL_BITS,
        );
        builder
            .when(is_last_row.clone())
            .assert_eq(*local_cols.control.len, len_val);
        // Assert that we started reading from the correct pointer initially
        let src_val = compose::<AB::Expr>(
            local_cols.src_ptr.mapv(|x| x.into()).as_slice().unwrap(),
            RV32_CELL_BITS,
        );
        builder
            .when(is_last_row.clone())
            .assert_eq(*local_cols.control.read_ptr, src_val + read_ptr_delta);
        // Assert that we started reading from the correct timestamp
        builder.when(is_last_row.clone()).assert_eq(
            *local_cols.control.cur_timestamp,
            local_cols.from_state.timestamp + AB::Expr::from_canonical_u32(3) + time_delta,
        );
    }
}
