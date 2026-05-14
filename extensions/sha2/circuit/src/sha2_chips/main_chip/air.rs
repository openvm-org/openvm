use std::marker::PhantomData;

use itertools::izip;
use ndarray::s;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::{
        memory::{
            offline_checker::{pack_u8_for_bus, MemoryBridge},
            MemoryAddress,
        },
        SystemPort,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, utils::compose, ColumnsAir,
};
use openvm_instructions::riscv::{
    RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS,
};
use openvm_riscv_circuit::adapters::expand_to_rv64_register;
use openvm_sha2_air::Sha2BlockHasherSubairConfig;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
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

// No columns provided: width is the config-dependent `C::MAIN_CHIP_WIDTH` and rows are accessed
// via `Sha2ColsRef` (a slice-borrowing ref struct, no static `Cols`).
impl<C: Sha2MainChipConfig> ColumnsAir for Sha2MainAir<C> {}

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
        let (local, next) = (main.row_slice(0).unwrap(), main.row_slice(1).unwrap());

        let local: Sha2ColsRef<AB::Var> = Sha2ColsRef::from::<C>(&local[..C::MAIN_CHIP_WIDTH]);
        let next: Sha2ColsRef<AB::Var> = Sha2ColsRef::from::<C>(&next[..C::MAIN_CHIP_WIDTH]);

        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            local.instruction.from_state.timestamp + AB::F::from_usize(timestamp_delta - 1)
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
        builder.assert_bool(*local.instruction.is_enabled);
        builder
            .when_transition()
            .when_ne(*local.instruction.is_enabled, AB::Expr::ONE)
            .assert_zero(*next.instruction.is_enabled);

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

        // `prev_state` is already stored as u16 cells (one u16 per pair of bytes), so the bus
        // payload is just the column values; no byte-pairing transformation is needed.
        let prev_state_as_u16s = local
            .block
            .prev_state
            .iter()
            .map(|x| (*x).into())
            .collect::<Vec<AB::Expr>>();

        // Send (STATE, request_id, prev_state_as_u16s, new_state) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_u8(MessageType::State as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain(prev_state_as_u16s)
            .chain(local.block.new_state.into_iter().copied().map(|x| x.into())),
            *local.instruction.is_enabled,
        );

        // Send (MESSAGE_1, request_id, first_half_of_message) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_u8(MessageType::Message1 as u8),
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
                AB::Expr::from_u8(MessageType::Message2 as u8),
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
            let val_arr: [AB::Var; RV64_WORD_NUM_LIMBS] =
                std::array::from_fn(|i| *val.get(i).unwrap());
            let data = expand_to_rv64_register(&val_arr);
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_REGISTER_AS), ptr),
                    pack_u8_for_bus::<AB>(&data),
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        // range check the high byte of each 32-bit effective pointer
        let shift =
            AB::Expr::from_usize(1 << (RV64_WORD_NUM_LIMBS * RV64_CELL_BITS - self.ptr_max_bits));
        let needs_range_check = [
            local.instruction.dst_ptr_limbs[RV64_WORD_NUM_LIMBS - 1],
            local.instruction.state_ptr_limbs[RV64_WORD_NUM_LIMBS - 1],
            local.instruction.input_ptr_limbs[RV64_WORD_NUM_LIMBS - 1],
            local.instruction.input_ptr_limbs[RV64_WORD_NUM_LIMBS - 1], /* needs_range_check must have even length */
        ];
        for pair in needs_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * shift.clone(), pair[1] * shift.clone())
                .eval(builder, *local.instruction.is_enabled);
        }

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_usize(C::OPCODE as usize + self.offset),
                [
                    (*local.instruction.dst_reg_ptr).into(),
                    (*local.instruction.state_reg_ptr).into(),
                    (*local.instruction.input_reg_ptr).into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_MEMORY_AS),
                ],
                *local.instruction.from_state,
                AB::F::from_usize(C::TIMESTAMP_DELTA),
            )
            .eval(builder, *local.instruction.is_enabled);
    }

    pub fn eval_reads<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        timestamp_pp: &mut impl FnMut() -> AB::Expr,
    ) {
        // Upper 4 bytes of each pointer are constrained to zero, so only compose the low 4 bytes
        // to form the 32-bit effective address. Composing all 8 would overflow the field.
        let input_ptr_val = compose(
            &local
                .instruction
                .input_ptr_limbs
                .slice(s![..RV64_WORD_NUM_LIMBS])
                .to_vec(),
            RV64_CELL_BITS,
        );
        for i in 0..C::BLOCK_READS {
            let chunk: [AB::Var; SHA2_READ_SIZE] = local
                .block
                .message_bytes
                .slice(s![i * SHA2_READ_SIZE..(i + 1) * SHA2_READ_SIZE])
                .to_vec()
                .try_into()
                .unwrap_or_else(|_| {
                    panic!("message bytes is not the correct size");
                });
            let chunk_expr: [AB::Expr; SHA2_READ_SIZE] = chunk.map(Into::into);
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(
                        AB::Expr::from_u32(RV64_MEMORY_AS),
                        input_ptr_val.clone() + AB::F::from_usize(i * SHA2_READ_SIZE),
                    ),
                    pack_u8_for_bus::<AB>(&chunk_expr),
                    timestamp_pp(),
                    &local.mem.input_reads[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        let state_ptr_val = compose(
            &local
                .instruction
                .state_ptr_limbs
                .slice(s![..RV64_WORD_NUM_LIMBS])
                .to_vec(),
            RV64_CELL_BITS,
        );
        // `prev_state` is u16-shaped, so each `SHA2_READ_SIZE` (8-byte) memory read corresponds
        // to `BLOCK_FE_WIDTH` u16 cells consumed from the column. The bus payload is just those
        // cells (no byte→u16 packing).
        for i in 0..C::STATE_READS {
            let chunk: [AB::Expr; openvm_circuit::arch::BLOCK_FE_WIDTH] = std::array::from_fn(|j| {
                (*local
                    .block
                    .prev_state
                    .get(i * openvm_circuit::arch::BLOCK_FE_WIDTH + j)
                    .expect("prev_state index out of bounds"))
                .into()
            });
            self.memory_bridge
                .read_4(
                    MemoryAddress::new(
                        AB::Expr::from_u32(RV64_MEMORY_AS),
                        state_ptr_val.clone() + AB::F::from_usize(i * SHA2_READ_SIZE),
                    ),
                    chunk,
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
        let dst_ptr_val = compose(
            &local
                .instruction
                .dst_ptr_limbs
                .slice(s![..RV64_WORD_NUM_LIMBS])
                .to_vec(),
            RV64_CELL_BITS,
        );
        for i in 0..C::STATE_READS {
            let chunk: [AB::Var; SHA2_WRITE_SIZE] = local
                .block
                .new_state
                .slice(s![i * SHA2_READ_SIZE..(i + 1) * SHA2_READ_SIZE])
                .to_vec()
                .try_into()
                .unwrap_or_else(|_| {
                    panic!("new state is not the correct size");
                });
            let chunk_expr: [AB::Expr; SHA2_WRITE_SIZE] = chunk.map(Into::into);
            self.memory_bridge
                .write_4(
                    MemoryAddress::new(
                        AB::Expr::from_u32(RV64_MEMORY_AS),
                        dst_ptr_val.clone() + AB::F::from_usize(i * SHA2_READ_SIZE),
                    ),
                    pack_u8_for_bus::<AB>(&chunk_expr),
                    timestamp_pp(),
                    &local.mem.write_aux[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }
    }
}
