use std::{array, borrow::Borrow, mem::size_of};

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction,
        Postflight, PostflightError, PostflightStep, VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{pc_to_limbs, DEFAULT_PC_STEP},
    riscv::{IMM_AS, REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{
    checked_register_u16_pointer, concat_u16_block, is_canonical_i12,
    reg_byte_ptr_to_cell_ptr_limbs, U16_BITS, WORD_U16_LIMBS,
};

/// Adapter columns for RV64 word instructions with an immediate operand.
///
/// The core sees only the low 32-bit word as two u16 limbs. The upper half of the source register
/// is retained solely to authenticate the full-width register read, while the full-width write is
/// rebuilt by sign-extending the core result.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct BaseAluWImmU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs1_high: [T; WORD_U16_LIMBS],
    pub result_sign: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

const _: () = assert!(size_of::<BaseAluWImmU16AdapterCols<u8>>() == 16);

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BaseAluWImmU16AdapterCols<u8>)]
pub struct BaseAluWImmU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for BaseAluWImmU16AdapterAir {
    fn width(&self) -> usize {
        BaseAluWImmU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for BaseAluWImmU16AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        ImmInstruction<AB::Expr>,
        1,
        1,
        WORD_U16_LIMBS,
        WORD_U16_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &BaseAluWImmU16AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] = concat_u16_block(&ctx.reads[0], &local.rs1_high);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rs1_ptr),
                ),
                rs1_data,
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Recover the sign of the 32-bit result from its top u16 limb. This simultaneously proves
        // that the top result limb is a canonical u16 value.
        builder.assert_bool(local.result_sign);
        let result_high = ctx.writes[0][WORD_U16_LIMBS - 1].clone();
        self.range_bus
            .range_check(
                result_high - local.result_sign * AB::Expr::from_u32(1 << (U16_BITS - 1)),
                U16_BITS - 1,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let sign_extend = local.result_sign * AB::Expr::from_u32(u16::MAX as u32);
        let sign_extend_limbs: [AB::Expr; WORD_U16_LIMBS] = array::from_fn(|_| sign_extend.clone());
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] =
            concat_u16_block(&ctx.writes[0], &sign_extend_limbs);
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rd_ptr),
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
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(IMM_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (openvm_instructions::program::DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> [AB::Var; 2] {
        let local: &BaseAluWImmU16AdapterCols<_> = local.borrow();
        local.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct BaseAluWImmU16AdapterFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl BaseAluWImmU16AdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut BaseAluWImmU16AdapterCols<F>,
        compute: impl FnOnce([u16; WORD_U16_LIMBS], u32) -> [u16; WORD_U16_LIMBS],
    ) -> Result<([u16; WORD_U16_LIMBS], [u16; WORD_U16_LIMBS]), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != IMM_AS
        {
            return Err(PostflightError::new(
                "word register-immediate ALU instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let immediate = instruction.c.as_canonical_u32();
        if !is_canonical_i12(immediate) {
            return Err(PostflightError::new(
                "word register-immediate ALU instruction has a non-canonical immediate",
            ));
        }
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(REGISTER_AS, rs1_u16_ptr)?;
        let input = array::from_fn(|i| rs1.value[i]);
        let output = compute(input, immediate);
        let result_high = output[WORD_U16_LIMBS - 1];
        let result_sign = result_high >> (U16_BITS - 1);
        let sign_extend_limb = if result_sign != 0 { u16::MAX } else { 0 };
        let write_value = array::from_fn(|i| {
            if i < WORD_U16_LIMBS {
                output[i]
            } else {
                sign_extend_limb
            }
        });
        let write = replay.write_u16(REGISTER_AS, rd_u16_ptr, write_value)?;
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        self.range_checker_chip.add_count(
            (result_high & ((1 << (U16_BITS - 1)) - 1)) as u32,
            U16_BITS - 1,
        );
        adapter_row
            .writes_aux
            .set_prev_data(write.previous_value.map(F::from_u16));
        mem_helper.fill(
            write.previous_timestamp,
            write.timestamp,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux.as_mut(),
        );
        adapter_row.result_sign = F::from_u16(result_sign);
        adapter_row.rs1_high = array::from_fn(|i| F::from_u16(rs1.value[WORD_U16_LIMBS + i]));
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = pc_to_limbs(from_pc).map(F::from_u32);

        Ok((input, output))
    }
}
