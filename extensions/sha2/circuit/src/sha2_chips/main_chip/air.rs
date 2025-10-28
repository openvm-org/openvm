use std::marker::PhantomData;

use itertools::izip;
use ndarray::s;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::{
        memory::{offline_checker::MemoryBridge, MemoryAddress},
        SystemPort,
    },
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, utils::compose};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_sha2_air::Sha2BlockHasherSubairConfig;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::config::Sha2MainChipConfig;
use crate::{MessageType, Sha2ColsRef, SHA2_READ_SIZE, SHA2_WRITE_SIZE};

#[derive(Clone, Debug)]
pub struct Sha2MainAir<C: Sha2MainChipConfig> {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub sha2_bus: PermutationCheckBus,
    /// Maximum number of bits allowed for an address pointer
    /// Must be at least 24
    pub ptr_max_bits: usize,
    pub offset: usize,
    _phantom: PhantomData<C>,
}

impl<C: Sha2MainChipConfig> Sha2MainAir<C> {
    pub fn new(
        SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        }: SystemPort,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        ptr_max_bits: usize,
        self_bus_idx: BusIndex,
        offset: usize,
    ) -> Self {
        Self {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            bitwise_lookup_bus,
            sha2_bus: PermutationCheckBus::new(self_bus_idx),
            ptr_max_bits,
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<F, C: Sha2MainChipConfig> BaseAirWithPublicValues<F> for Sha2MainAir<C> {}
impl<F, C: Sha2MainChipConfig> PartitionedBaseAir<F> for Sha2MainAir<C> {}
impl<F, C: Sha2MainChipConfig> BaseAir<F> for Sha2MainAir<C> {
    fn width(&self) -> usize {
        C::MAIN_CHIP_WIDTH
    }
}

impl<AB: InteractionBuilder, C: Sha2MainChipConfig + Sha2BlockHasherSubairConfig> Air<AB>
    for Sha2MainAir<C>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: Sha2ColsRef<AB::Var> = Sha2ColsRef::from::<C>(&local[..C::MAIN_CHIP_WIDTH]);
        let next: Sha2ColsRef<AB::Var> = Sha2ColsRef::from::<C>(&next[..C::MAIN_CHIP_WIDTH]);

        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            local.instruction.from_state.timestamp
                + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.eval_block(builder, &local, &next);
        self.eval_instruction(builder, &local, &mut timestamp_pp);
        self.eval_reads(builder, &local, &mut timestamp_pp);
        self.eval_writes(builder, &local, &mut timestamp_pp);
    }
}

impl<C: Sha2MainChipConfig + Sha2BlockHasherSubairConfig> Sha2MainAir<C> {
    pub fn eval_block<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        next: &Sha2ColsRef<AB::Var>,
    ) {
        builder
            .when_first_row()
            .when(*local.instruction.is_enabled)
            .assert_zero(*local.block.request_id);

        builder
            .when_transition()
            .when(*next.instruction.is_enabled)
            .assert_eq(
                *next.block.request_id,
                *local.block.request_id + AB::Expr::ONE,
            );

        let prev_state_as_u16s: Vec<AB::Expr> = local
            .block
            .prev_state
            .exact_chunks(C::WORD_U8S)
            .into_iter()
            .flat_map(|word| {
                word.as_slice()
                    .unwrap()
                    .chunks_exact(2)
                    .rev()
                    .map(|x| x[0] * AB::F::from_canonical_u64(1 << 8) + x[1])
                    .collect::<Vec<_>>()
            })
            .collect();

        // for each word in the new state, byte1, byte2, ..., byteN, reverse the order of the bytes
        // so that it matches what the block hasher chip expects
        let new_state_big_endian: Vec<AB::Var> = local
            .block
            .new_state
            .exact_chunks(C::WORD_U8S)
            .into_iter()
            .flat_map(|word| word.into_iter().rev().copied().collect::<Vec<_>>())
            .collect();

        // Send (STATE, request_id, prev_state_as_u16s, new_state) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::State as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain(prev_state_as_u16s)
            .chain(new_state_big_endian.into_iter().map(|x| x.into())),
            *local.instruction.is_enabled,
        );

        // Send (MESSAGE_1, request_id, first_half_of_message) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::Message1 as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain(
                local
                    .block
                    .message_bytes
                    .iter()
                    .take(C::BLOCK_BYTES / 2)
                    .map(|x| (*x).into()),
            ),
            *local.instruction.is_enabled,
        );

        // Send (MESSAGE_2, request_id, second_half_of_message) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::Message2 as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain(
                local
                    .block
                    .message_bytes
                    .iter()
                    .skip(C::BLOCK_BYTES / 2)
                    .map(|x| (*x).into()),
            ),
            *local.instruction.is_enabled,
        );
    }

    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        timestamp_pp: &mut impl FnMut() -> AB::Expr,
    ) {
        for (&ptr, val, aux) in izip!(
            [
                local.instruction.dst_reg_ptr,
                local.instruction.state_reg_ptr,
                local.instruction.input_reg_ptr
            ],
            [
                local.instruction.dst_ptr_limbs,
                local.instruction.state_ptr_limbs,
                local.instruction.input_ptr_limbs
            ],
            &local.mem.register_aux,
        ) {
            self.memory_bridge
                .read::<_, _, SHA2_READ_SIZE>(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_REGISTER_AS), ptr),
                    val.to_vec().try_into().unwrap_or_else(|_| panic!()), // can't unwrap because AB::Var doesn't impl Debug
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        // range check the memory pointers
        // TODO: do I need to consider the length of the input or state?
        let shift = AB::Expr::from_canonical_usize(
            1 << (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - self.ptr_max_bits),
        );
        let needs_range_check = [
            local.instruction.dst_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            local.instruction.state_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            local.instruction.input_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
            local.instruction.input_ptr_limbs[RV32_REGISTER_NUM_LIMBS - 1],
        ];
        for pair in needs_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * shift.clone(), pair[1] * shift.clone())
                .eval(builder, *local.instruction.is_enabled);
        }

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(C::OPCODE as usize + self.offset),
                [
                    (*local.instruction.dst_reg_ptr).into(),
                    (*local.instruction.state_reg_ptr).into(),
                    (*local.instruction.input_reg_ptr).into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                *local.instruction.from_state,
                AB::F::from_canonical_usize(C::TIMESTAMP_DELTA),
            )
            .eval(builder, *local.instruction.is_enabled);
    }

    pub fn eval_reads<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        timestamp_pp: &mut impl FnMut() -> AB::Expr,
    ) {
        let input_ptr_val = compose(&local.instruction.input_ptr_limbs.to_vec(), RV32_CELL_BITS);
        for i in 0..C::BLOCK_READS {
            self.memory_bridge
                .read::<_, _, SHA2_READ_SIZE>(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        input_ptr_val.clone() + AB::F::from_canonical_usize(i * SHA2_READ_SIZE),
                    ),
                    local
                        .block
                        .message_bytes
                        .slice(s![i * SHA2_READ_SIZE..(i + 1) * SHA2_READ_SIZE])
                        .to_vec()
                        .try_into()
                        .unwrap_or_else(|_| {
                            panic!("message bytes is not the correct size");
                        }),
                    timestamp_pp(),
                    &local.mem.input_reads[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        let state_ptr_val = compose(&local.instruction.state_ptr_limbs.to_vec(), RV32_CELL_BITS);
        for i in 0..C::STATE_READS {
            self.memory_bridge
                .read::<_, _, SHA2_READ_SIZE>(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        state_ptr_val.clone() + AB::F::from_canonical_usize(i * SHA2_READ_SIZE),
                    ),
                    local
                        .block
                        .prev_state
                        .slice(s![i * SHA2_READ_SIZE..(i + 1) * SHA2_READ_SIZE])
                        .to_vec()
                        .try_into()
                        .unwrap_or_else(|_| {
                            panic!("prev state is not the correct size");
                        }),
                    timestamp_pp(),
                    &local.mem.state_reads[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }
    }

    pub fn eval_writes<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        timestamp_pp: &mut impl FnMut() -> AB::Expr,
    ) {
        let dst_ptr_val = compose(&local.instruction.dst_ptr_limbs.to_vec(), RV32_CELL_BITS);
        for i in 0..C::STATE_READS {
            self.memory_bridge
                .write::<_, _, SHA2_WRITE_SIZE>(
                    MemoryAddress::new(
                        AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                        dst_ptr_val.clone() + AB::F::from_canonical_usize(i * SHA2_READ_SIZE),
                    ),
                    local
                        .block
                        .new_state
                        .slice(s![i * SHA2_READ_SIZE..(i + 1) * SHA2_READ_SIZE])
                        .to_vec()
                        .try_into()
                        .unwrap_or_else(|_| {
                            panic!("new state is not the correct size");
                        }),
                    timestamp_pp(),
                    &local.mem.write_aux[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }
    }
}
