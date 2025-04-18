use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, utils::not};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV32_REGISTER_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
};

use super::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32BaseAluAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    /// Pointer if rs2 was a read, immediate value otherwise
    pub rs2: T,
    /// 1 if rs2 was a read, 0 if an immediate
    pub rs2_as: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

/// Reads instructions of the form OP a, b, c, d, e where \[a:4\]_d = \[b:4\]_d op \[c:4\]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct Rv32BaseAluAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl<F: Field> BaseAir<F> for Rv32BaseAluAdapterAir {
    fn width(&self) -> usize {
        Rv32BaseAluAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv32BaseAluAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        RV32_REGISTER_NUM_LIMBS,
        RV32_REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &Rv32BaseAluAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // If rs2 is an immediate value, constrain that:
        // 1. It's a 16-bit two's complement integer (stored in rs2_limbs[0] and rs2_limbs[1])
        // 2. It's properly sign-extended to 32-bits (the upper limbs must match the sign bit)
        let rs2_limbs = ctx.reads[1].clone();
        let rs2_sign = rs2_limbs[2].clone();
        let rs2_imm = rs2_limbs[0].clone()
            + rs2_limbs[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs2_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.rs2_as);
        let mut rs2_imm_when = builder.when(not(local.rs2_as));
        rs2_imm_when.assert_eq(local.rs2, rs2_imm);
        rs2_imm_when.assert_eq(rs2_sign.clone(), rs2_limbs[3].clone());
        rs2_imm_when.assert_zero(
            rs2_sign.clone()
                * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs2_sign),
        );
        self.bitwise_lookup_bus
            .send_range(rs2_limbs[0].clone(), rs2_limbs[1].clone())
            .eval(builder, ctx.instruction.is_valid.clone() - local.rs2_as);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rs1_ptr),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // This constraint ensures that the following memory read only occurs when `is_valid == 1`.
        builder
            .when(local.rs2_as)
            .assert_one(ctx.instruction.is_valid.clone());
        self.memory_bridge
            .read(
                MemoryAddress::new(local.rs2_as, local.rs2),
                ctx.reads[1].clone(),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, local.rs2_as);

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.rd_ptr),
                ctx.writes[0].clone(),
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
                    local.rs2.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    local.rs2_as.into(),
                ],
                local.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv32BaseAluAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}
