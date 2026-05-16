use std::marker::PhantomData;

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, BLOCK_FE_WIDTH},
    system::{
        memory::{offline_checker::MemoryBridge, MemoryAddress},
        SystemPort,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, utils::compose,
    var_range::VariableRangeCheckerBus, ColumnsAir,
};
use openvm_instructions::riscv::{RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS};
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
    pub range_bus: VariableRangeCheckerBus,
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
        range_bus: VariableRangeCheckerBus,
        ptr_max_bits: usize,
        self_bus_idx: BusIndex,
        offset: usize,
    ) -> Self {
        Self {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
            bitwise_lookup_bus,
            range_bus,
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

        // `prev_state` and `new_state` are u16 cells; pass directly to the bus.
        let prev_state_as_u16s = local
            .block
            .prev_state
            .iter()
            .map(|x| (*x).into())
            .collect::<Vec<AB::Expr>>();
        let new_state_as_u16s = local
            .block
            .new_state
            .iter()
            .map(|x| (*x).into())
            .collect::<Vec<AB::Expr>>();

        // Send (STATE, request_id, prev_state_as_u16s, new_state_as_u16s) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_u8(MessageType::State as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain(prev_state_as_u16s)
            .chain(new_state_as_u16s),
            *local.instruction.is_enabled,
        );

        // `message_u16s` is u16-shaped; the sha2 bus payload uses these cells directly. Split
        // the column in half: MESSAGE_1 sends the first half, MESSAGE_2 the second half.
        let half_u16s = C::BLOCK_U16S / 2;
        let message_cell = |k: usize| -> AB::Expr {
            (*local
                .block
                .message_u16s
                .get(k)
                .expect("message_u16s index out of bounds"))
            .into()
        };

        // Send (MESSAGE_1, request_id, first_half_of_message_as_u16s) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_u8(MessageType::Message1 as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain((0..half_u16s).map(message_cell)),
            *local.instruction.is_enabled,
        );

        // Send (MESSAGE_2, request_id, second_half_of_message_as_u16s) to the sha2 bus
        self.sha2_bus.send(
            builder,
            [
                AB::Expr::from_u8(MessageType::Message2 as u8),
                (*local.block.request_id).into(),
            ]
            .into_iter()
            .chain((0..half_u16s).map(|k| message_cell(half_u16s + k))),
            *local.instruction.is_enabled,
        );
    }

    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &Sha2ColsRef<AB::Var>,
        timestamp_pp: &mut impl FnMut() -> AB::Expr,
    ) {
        // Register reads: low 32 bits as 2 u16 cells, zero-extended to BLOCK_FE_WIDTH=4 cells.
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
            let bus_payload: [AB::Expr; BLOCK_FE_WIDTH] = [
                (*val.get(0).unwrap()).into(),
                (*val.get(1).unwrap()).into(),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ];
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_REGISTER_AS), ptr),
                    bus_payload,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        // Range-check the high u16 of each 32-bit effective pointer.
        let u16_bits = RV64_CELL_BITS * 2;
        let shift = AB::F::from_u32(1 << (u16_bits * 2 - self.ptr_max_bits));
        for limbs in [
            local.instruction.dst_ptr_limbs,
            local.instruction.state_ptr_limbs,
            local.instruction.input_ptr_limbs,
        ] {
            self.range_bus
                .range_check(*limbs.get(1).unwrap() * shift, u16_bits)
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
        // Compose the 2 u16 cells of each pointer (low 32 bits) into a single field element.
        let u16_bits = RV64_CELL_BITS * 2;
        let input_ptr_val = compose(&local.instruction.input_ptr_limbs.to_vec(), u16_bits);
        // `message_u16s` is u16-shaped: each `SHA2_READ_SIZE` (8-byte) memory read consumes
        // `BLOCK_FE_WIDTH` u16 cells. The bus payload is those cells directly (no byte→u16
        // packing helper needed).
        for i in 0..C::BLOCK_READS {
            let chunk: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|j| {
                (*local
                    .block
                    .message_u16s
                    .get(i * BLOCK_FE_WIDTH + j)
                    .expect("message_u16s index out of bounds"))
                .into()
            });
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        AB::Expr::from_u32(RV64_MEMORY_AS),
                        input_ptr_val.clone() + AB::F::from_usize(i * SHA2_READ_SIZE),
                    ),
                    chunk,
                    timestamp_pp(),
                    &local.mem.input_reads[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }

        let state_ptr_val = compose(&local.instruction.state_ptr_limbs.to_vec(), u16_bits);
        // `prev_state` is u16-shaped, so each `SHA2_READ_SIZE` (8-byte) memory read corresponds
        // to `BLOCK_FE_WIDTH` u16 cells consumed from the column. The bus payload is just those
        // cells (no byte→u16 packing).
        for i in 0..C::STATE_READS {
            let chunk: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|j| {
                (*local
                    .block
                    .prev_state
                    .get(i * BLOCK_FE_WIDTH + j)
                    .expect("prev_state index out of bounds"))
                .into()
            });
            self.memory_bridge
                .read(
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
        let u16_bits = RV64_CELL_BITS * 2;
        let dst_ptr_val = compose(&local.instruction.dst_ptr_limbs.to_vec(), u16_bits);
        // `new_state` is u16-shaped, so each `SHA2_WRITE_SIZE` (8-byte) memory write consumes
        // `BLOCK_FE_WIDTH` u16 cells from the column. The bus payload is those cells (no
        // byte→u16 packing).
        for i in 0..C::STATE_WRITES {
            let chunk: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|j| {
                (*local
                    .block
                    .new_state
                    .get(i * BLOCK_FE_WIDTH + j)
                    .expect("new_state index out of bounds"))
                .into()
            });
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        AB::Expr::from_u32(RV64_MEMORY_AS),
                        dst_ptr_val.clone() + AB::F::from_usize(i * SHA2_WRITE_SIZE),
                    ),
                    chunk,
                    timestamp_pp(),
                    &local.mem.write_aux[i],
                )
                .eval(builder, *local.instruction.is_enabled);
        }
    }
}
