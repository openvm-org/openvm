use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, Postflight, PostflightError, PostflightStep, VmAdapterAir,
        BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_BYTE_BITS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{
    byte_ptr_to_u16_ptr, checked_byte_ptr_to_u16_ptr_value, pack_high_u16, pack_rv64_u16_block,
    rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, ReplayComputation, ReplayResult,
    RV64_PTR_U16_LIMBS,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64MultWAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    /// Upper 4 bytes of rs1 register read, packed as two u16 cells.
    /// Kept in the adapter to constrain the full-width memory read.
    pub rs1_high: [T; RV64_PTR_U16_LIMBS],
    /// Upper 4 bytes of rs2 register read, packed as two u16 cells.
    /// Kept in the adapter to constrain the full-width memory read.
    pub rs2_high: [T; RV64_PTR_U16_LIMBS],
    /// Sign bit of the low-word core result used to build full-width sign-extended writes.
    pub result_sign: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Same instruction format as `Rv64MultAdapterAir`, but only exposes the low 32-bit limbs
/// (`RV64_WORD_NUM_LIMBS`) for reads and writes. Full-width RV64 writes are rebuilt in-adapter by
/// sign-extending the low-word result.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64MultWAdapterCols<u8>)]
pub struct Rv64MultWAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv64MultWAdapterAir {
    fn width(&self) -> usize {
        Rv64MultWAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64MultWAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV64_WORD_NUM_LIMBS,
        RV64_WORD_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64MultWAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] =
            pack_rv64_u16_block(&ctx.reads[0], &local.rs1_high);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs1_ptr),
                ),
                rs1_data,
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let rs2_data: [AB::Expr; BLOCK_FE_WIDTH] =
            pack_rv64_u16_block(&ctx.reads[1], &local.rs2_high);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rs2_ptr),
                ),
                rs2_data,
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Sign-extend the 32-bit result to 64 bits.
        builder.assert_bool(local.result_sign);
        let sign_mask = AB::Expr::from_u32(1 << (RV64_BYTE_BITS - 1));
        let result_word_msl = ctx.writes[0][RV64_WORD_NUM_LIMBS - 1].clone();
        self.bitwise_lookup_bus
            .send_xor(
                result_word_msl.clone(),
                sign_mask.clone(),
                result_word_msl + sign_mask.clone()
                    - AB::Expr::from_u32(2) * local.result_sign * sign_mask,
            )
            .eval(builder, ctx.instruction.is_valid.clone());
        let sign_extend_u16 = AB::Expr::from_u32(u16::MAX as u32) * local.result_sign;
        let sign_extend = [sign_extend_u16.clone(), sign_extend_u16.clone()];
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] =
            pack_rv64_u16_block(&ctx.writes[0], &sign_extend);
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local.rd_ptr),
                ),
                write_data,
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64MultWAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, derive_new::new)]
pub struct Rv64MultWAdapterExecutor;

#[derive(derive_new::new)]
pub struct Rv64MultWAdapterFiller {
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl Rv64MultWAdapterFiller {
    pub(crate) fn replay<F: PrimeField32, M>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64MultWAdapterCols<F>,
        compute: impl FnOnce(
            [[u8; RV64_WORD_NUM_LIMBS]; 2],
        ) -> ReplayComputation<RV64_WORD_NUM_LIMBS, M>,
    ) -> Result<ReplayResult<RV64_WORD_NUM_LIMBS, M>, PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != 0
        {
            return Err(PostflightError::new(
                "word multiplication instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rs2_ptr = instruction.c.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs1_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs1_ptr)?;
        let rs2_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rs2_ptr)?;
        let rd_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        let rs2 = replay.read_u16(RV64_REGISTER_AS, rs2_u16_ptr)?;
        let rs1_bytes = rv64_u16_block_to_bytes(rs1.value);
        let rs2_bytes = rv64_u16_block_to_bytes(rs2.value);
        let inputs = [
            rs1_bytes[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
            rs2_bytes[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
        ];
        let computation = compute(inputs);
        let output = computation.output;
        let result_word_msl = output[RV64_WORD_NUM_LIMBS - 1];
        let result_sign = result_word_msl >> (RV64_BYTE_BITS - 1);
        let sign_extend_limb = u8::MAX * result_sign;
        let mut write_bytes = [sign_extend_limb; RV64_REGISTER_NUM_LIMBS];
        write_bytes[..RV64_WORD_NUM_LIMBS].copy_from_slice(&output);
        let write = replay.write_u16(
            RV64_REGISTER_AS,
            rd_u16_ptr,
            rv64_bytes_to_u16_block(write_bytes),
        )?;
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        self.bitwise_lookup_chip
            .request_xor(result_word_msl as u32, 1 << (RV64_BYTE_BITS - 1));
        adapter_row
            .writes_aux
            .set_prev_data(write.previous_value.map(F::from_u16));
        mem_helper.fill(
            write.previous_timestamp,
            write.timestamp,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            rs2.previous_timestamp,
            rs2.timestamp,
            adapter_row.reads_aux[1].as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );
        adapter_row.result_sign = F::from_u8(result_sign);
        let rs2_high = rs2_bytes[RV64_WORD_NUM_LIMBS..].try_into().unwrap();
        let rs1_high = rs1_bytes[RV64_WORD_NUM_LIMBS..].try_into().unwrap();
        adapter_row.rs2_high = pack_high_u16(&rs2_high);
        adapter_row.rs1_high = pack_high_u16(&rs1_high);
        adapter_row.rs2_ptr = F::from_u32(rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok(ReplayResult {
            inputs,
            output,
            metadata: computation.metadata,
        })
    }
}
