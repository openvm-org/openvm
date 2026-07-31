use std::{array, borrow::Borrow, mem::size_of};

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
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{
    byte_ptr_to_u16_ptr, checked_register_u16_pointer, concat_rv64_u16_block, RV64_WORD_U16_LIMBS,
    U16_BITS,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct Rv64BaseAluWRegU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Upper 32 bits of the rs1 register read, as two u16 cells.
    pub rs1_high: [T; RV64_WORD_U16_LIMBS],
    pub rs2_ptr: T,
    /// Upper 32 bits of the rs2 register read, as two u16 cells.
    pub rs2_high: [T; RV64_WORD_U16_LIMBS],
    /// Sign bit of the low-word core result (bit 15 of the high result limb), used to build the
    /// full-width sign-extended write.
    pub result_sign: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

const _: () = assert!(size_of::<Rv64BaseAluWRegU16AdapterCols<u8>>() == 20);

/// Exposes the low 32-bit words of two register operands to the core and sign-extends the result.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64BaseAluWRegU16AdapterCols<u8>)]
pub struct Rv64BaseAluWRegU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64BaseAluWRegU16AdapterAir {
    fn width(&self) -> usize {
        Rv64BaseAluWRegU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64BaseAluWRegU16AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV64_WORD_U16_LIMBS,
        RV64_WORD_U16_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv64BaseAluWRegU16AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Rebuild the full 64-bit register read from the low-word core limbs and the stashed
        // upper limbs, then send it on the memory bus.
        let rs1_data: [AB::Expr; BLOCK_FE_WIDTH] =
            concat_rv64_u16_block(&ctx.reads[0], &local.rs1_high);
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
            concat_rv64_u16_block(&ctx.reads[1], &local.rs2_high);
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

        // Sign-extend the 32-bit result to 64 bits. Decomposing the top result limb as
        // `low15 + result_sign * 2^15`, with a 15-bit range check on `low15`, both proves that
        // the limb is a canonical u16 and forces `result_sign` to equal its top bit.
        builder.assert_bool(local.result_sign);
        let result_high = ctx.writes[0][RV64_WORD_U16_LIMBS - 1].clone();
        let sign_weight = AB::Expr::from_u32(1 << (U16_BITS - 1));
        self.range_bus
            .range_check(result_high - local.result_sign * sign_weight, U16_BITS - 1)
            .eval(builder, ctx.instruction.is_valid.clone());
        let sign_extend = local.result_sign * AB::Expr::from_u32(u16::MAX as u32);
        let sign_extend_limbs: [AB::Expr; RV64_WORD_U16_LIMBS] =
            array::from_fn(|_| sign_extend.clone());
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] =
            concat_rv64_u16_block(&ctx.writes[0], &sign_extend_limbs);
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
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64BaseAluWRegU16AdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(derive_new::new)]
pub struct Rv64BaseAluWRegU16AdapterFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl Rv64BaseAluWRegU16AdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64BaseAluWRegU16AdapterCols<F>,
        compute: impl FnOnce([[u16; RV64_WORD_U16_LIMBS]; 2]) -> [u16; RV64_WORD_U16_LIMBS],
    ) -> Result<([[u16; RV64_WORD_U16_LIMBS]; 2], [u16; RV64_WORD_U16_LIMBS]), PostflightError>
    {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_REGISTER_AS
        {
            return Err(PostflightError::new(
                "word register-register ALU instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rs2_ptr = instruction.c.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rs2_u16_ptr = checked_register_u16_pointer(rs2_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        let rs2 = replay.read_u16(RV64_REGISTER_AS, rs2_u16_ptr)?;
        let inputs = [
            array::from_fn(|i| rs1.value[i]),
            array::from_fn(|i| rs2.value[i]),
        ];
        let output = compute(inputs);
        let result_high = output[RV64_WORD_U16_LIMBS - 1];
        let result_sign = result_high >> (U16_BITS - 1);
        let sign_extend_limb = if result_sign != 0 { u16::MAX } else { 0 };
        let write_value = array::from_fn(|i| {
            if i < RV64_WORD_U16_LIMBS {
                output[i]
            } else {
                sign_extend_limb
            }
        });
        let write = replay.write_u16(RV64_REGISTER_AS, rd_u16_ptr, write_value)?;
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
            rs2.previous_timestamp,
            rs2.timestamp,
            adapter_row.reads_aux[1].as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );
        adapter_row.result_sign = F::from_u16(result_sign);
        adapter_row.rs2_high = array::from_fn(|i| F::from_u16(rs2.value[RV64_WORD_U16_LIMBS + i]));
        adapter_row.rs2_ptr = F::from_u32(rs2_ptr);
        adapter_row.rs1_high = array::from_fn(|i| F::from_u16(rs1.value[RV64_WORD_U16_LIMBS + i]));
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok((inputs, output))
    }
}
