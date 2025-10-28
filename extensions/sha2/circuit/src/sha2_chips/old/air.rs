use std::{cmp::min, convert::TryInto};

use openvm_circuit::{
    arch::{ExecutionBridge, SystemPort},
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
use openvm_sha2_air::{compose, Sha256Config, Sha2Air, Sha2Variant, Sha512Config};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{Sha2BlockHasherDigestColsRef, Sha2BlockHasherRoundColsRef, Sha2ChipConfig};

#[derive(Clone, Debug)]
pub struct Sha2BlockHasherAir<C: Sha2ChipConfig> {
    /// Bus to send byte checks to
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub(super) sha_subair: Sha2Air<C>,
}

impl<C: Sha2ChipConfig> Sha2BlockHasherAir<C> {
    pub fn new(
        SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        }: SystemPort,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        ptr_max_bits: usize,
        self_bus_idx: BusIndex,
    ) -> Self {
        Self {
            bitwise_lookup_bus,
            sha_subair: Sha2Air::<C>::new(bitwise_lookup_bus, self_bus_idx),
        }
    }
}

impl<F: Field, C: Sha2ChipConfig> BaseAirWithPublicValues<F> for Sha2BlockHasherAir<C> {}
impl<F: Field, C: Sha2ChipConfig> PartitionedBaseAir<F> for Sha2BlockHasherAir<C> {}
impl<F: Field, C: Sha2ChipConfig> BaseAir<F> for Sha2BlockHasherAir<C> {
    fn width(&self) -> usize {
        C::VM_WIDTH
    }
}

impl<AB: InteractionBuilder, C: Sha2ChipConfig> Air<AB> for Sha2BlockHasherAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.eval_transitions(builder);
        self.eval_reads(builder);
        self.eval_last_row(builder);

        self.sha_subair.eval(builder, C::BLOCK_HASHER_CONTROL_WIDTH);
    }
}

impl<C: Sha2ChipConfig> Sha2BlockHasherAir<C> {
    /// Implement constraints on `len`, `read_ptr` and `cur_timestamp`
    fn eval_transitions<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local_cols =
            Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(&local[..C::VM_ROUND_WIDTH]);
        let next_cols =
            Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(&next[..C::VM_ROUND_WIDTH]);

        let is_last_row =
            *local_cols.inner.flags.is_last_block * *local_cols.inner.flags.is_digest_row;
        // Len should be the same for the entire message
        builder
            .when_transition()
            .when(not::<AB::Expr>(is_last_row.clone()))
            .assert_eq(*next_cols.control.len, *local_cols.control.len);

        // Read ptr should increment by [C::READ_SIZE] for the first 4 rows and stay the same
        // otherwise
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
        let local_cols =
            Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(&local[..C::VM_ROUND_WIDTH]);

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
        let local_cols =
            Sha2BlockHasherDigestColsRef::<AB::Var>::from::<C>(&local[..C::VM_DIGEST_WIDTH]);

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
                // It is fine to shift like this since we already know that dst_ptr and src_ptr
                // have [RV32_CELL_BITS] bits
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
                // Note: revisit in the future to do 2 block writes of 16 cells instead of 1 block
                // write of 32 cells. This could be beneficial as the output is often an input for
                // another hash
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
