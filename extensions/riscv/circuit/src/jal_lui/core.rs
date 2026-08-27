use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::InstructionOperand,
    program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC, PC_IDX_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    ptr_to_u16_limbs, u64_to_u16_block, PC_IDX_LOW_BITS, PTR_U16_LIMBS, RV_IS_TYPE_IMM_BITS,
    RV_J_TYPE_IMM_BITS, U16_BITS,
};

pub(super) const LUI_IMM_LOW_BITS: usize = U16_BITS - RV_IS_TYPE_IMM_BITS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JalLuiCoreCols<T> {
    pub imm: T,
    // Low 32 bits of rd as u16 cells.
    pub rd_data: [T; PTR_U16_LIMBS],
    pub imm_low_4: T,
    pub is_jal: T,
    pub is_lui: T,
    pub is_sign_extend: T,
    pub rd_carry: T,
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(JalLuiCoreCols<u8>)]
pub struct JalLuiCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for JalLuiCoreAir {
    fn width(&self) -> usize {
        JalLuiCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JalLuiCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JalLuiCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc_idx: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JalLuiCoreCols<AB::Var> = (*local_core).borrow();
        let JalLuiCoreCols::<AB::Var> {
            imm,
            rd_data: rd,
            imm_low_4,
            is_jal,
            is_lui,
            is_sign_extend,
            rd_carry,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(is_sign_extend);
        builder.assert_bool(rd_carry);

        // LUI: constrain rd = imm << RV_IS_TYPE_IMM_BITS.
        builder
            .when(is_lui)
            .assert_eq(rd[0], imm_low_4 * AB::F::from_u32(1 << RV_IS_TYPE_IMM_BITS));
        builder.when(is_lui).assert_eq(
            imm,
            imm_low_4 + rd[1] * AB::F::from_u32(1 << LUI_IMM_LOW_BITS),
        );
        builder.when(is_jal).assert_zero(imm_low_4);

        // Range-check the low LUI_IMM_LOW_BITS bits of imm.
        self.range_bus
            .range_check(imm_low_4, LUI_IMM_LOW_BITS)
            .eval(builder, is_lui);

        let pc_step_inv = AB::F::from_u32(DEFAULT_PC_STEP).inverse();

        // JAL: constrain all 33 possible return-address bits. Dividing the aligned low limb by
        // DEFAULT_PC_STEP keeps the equality below the field modulus.
        builder.when(is_jal).assert_eq(
            rd[0] * pc_step_inv
                + rd[1] * AB::F::from_u32(1 << PC_IDX_LOW_BITS)
                + rd_carry * AB::F::from_u32(1 << PC_IDX_BITS),
            from_pc_idx + AB::F::ONE,
        );
        // A carry leaves all low-32-bit return-address limbs zero. This prevents field wrap in
        // the decomposition and pins the carry to the final pc index.
        builder.when(rd_carry).assert_zero(rd[0]);
        builder.when(rd_carry).assert_zero(rd[1]);

        // Range-check the low 32-bit rd cells.
        self.range_bus
            .range_check(rd[0], U16_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(rd[1], U16_BITS)
            .eval(builder, is_valid.clone());

        // Tie is_sign_extend to bit 31 of rd for LUI. The two result flags are opcode-specific.
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd[1] - is_sign_extend * AB::Expr::from_u32(1 << U16_BITS),
                U16_BITS,
            )
            .eval(builder, is_lui);
        builder.when(is_jal).assert_zero(is_sign_extend);
        builder.when(is_lui).assert_zero(rd_carry);

        // JAL return addresses are DEFAULT_PC_STEP-aligned: rd[0] = 4 * x with
        // x < 2^PC_IDX_LOW_BITS. Together with rd[1] < 2^16 this makes the decomposition
        // rd = 4 * (from_pc_idx + 1) unique: the composed pc index rd[1] * 2^PC_IDX_LOW_BITS + x
        // is at most 2^PC_IDX_BITS < p, so it must equal from_pc_idx + 1 over the integers.
        self.range_bus
            .range_check(rd[0] * pc_step_inv, PC_IDX_LOW_BITS)
            .eval(builder, is_jal);

        // LUI sign-extends bit 31; JAL writes its carry into bit 32.
        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(u16::MAX as u32);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd[0].into(),
            rd[1].into(),
            sign_extend_cell.clone() + rd_carry,
            sign_extend_cell,
        ];

        // `imm` is a byte offset (a multiple of DEFAULT_PC_STEP, possibly negative as a field
        // element); pc values on the buses are pc indices, so scale it down by DEFAULT_PC_STEP.
        let to_pc_idx = from_pc_idx + is_lui * AB::Expr::ONE + is_jal * imm * pc_step_inv;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_u32(LUI as u32) + is_jal * AB::F::from_u32(JAL as u32),
        );

        AdapterAirContext {
            to_pc: Some(to_pc_idx),
            reads: [].into(),
            writes: [write_data].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        JalLuiOpcode::CLASS_OFFSET
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct JalLuiExecutor;

#[derive(Clone, derive_new::new)]
pub struct JalLuiFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

/// Returns the signed machine representation of a valid JAL or LUI immediate.
pub(super) fn get_signed_imm(is_jal: bool, imm: InstructionOperand) -> Option<i32> {
    if is_jal {
        crate::adapters::decode_signed_instruction_imm(imm, RV_J_TYPE_IMM_BITS)
    } else {
        let imm = u32::try_from(imm.as_i32()).ok()?;
        (imm < (1 << 20)).then_some(imm as i32)
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    if is_jal {
        let rd = u64::from(pc) + u64::from(DEFAULT_PC_STEP);
        let next_pc = (pc as i64).wrapping_add(imm as i64);
        debug_assert!(next_pc >= 0 && next_pc <= MAX_ALLOWED_PC as i64);
        (next_pc as u32, u64_to_u16_block(rd))
    } else {
        let imm = imm as u32;
        let rd_low = imm << RV_IS_TYPE_IMM_BITS;
        let [lo, hi] = ptr_to_u16_limbs(rd_low);
        let sign = if (hi >> (U16_BITS - 1)) & 1 == 1 {
            u16::MAX
        } else {
            0
        };
        (pc + DEFAULT_PC_STEP, [lo, hi, sign, sign])
    }
}
