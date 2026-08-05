use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, PC_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    ptr_to_u16_limbs, rv64_u32_to_u16_block, RV64_PTR_U16_LIMBS, RV_IS_TYPE_IMM_BITS,
    RV_J_TYPE_IMM_BITS, U16_BITS,
};

pub(super) const LUI_IMM_LOW_BITS: usize = U16_BITS - RV_IS_TYPE_IMM_BITS;
pub(super) const PC_HIGH_U16_SHIFT: usize = 2 * U16_BITS - PC_BITS;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64JalLuiCoreCols<T> {
    pub imm: T,
    // Low 32 bits of rd as u16 cells. Upper register cells are sign extension.
    pub rd_data: [T; RV64_PTR_U16_LIMBS],
    pub is_jal: T,
    pub is_lui: T,
    pub is_sign_extend: T,
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(Rv64JalLuiCoreCols<u8>)]
pub struct Rv64JalLuiCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for Rv64JalLuiCoreAir {
    fn width(&self) -> usize {
        Rv64JalLuiCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv64JalLuiCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for Rv64JalLuiCoreAir
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
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &Rv64JalLuiCoreCols<AB::Var> = (*local_core).borrow();
        let Rv64JalLuiCoreCols::<AB::Var> {
            imm,
            rd_data: rd,
            is_jal,
            is_lui,
            is_sign_extend,
        } = *cols;

        builder.assert_bool(is_lui);
        builder.assert_bool(is_jal);
        let is_valid = is_lui + is_jal;
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(is_sign_extend);

        // LUI: constrain rd = imm << RV_IS_TYPE_IMM_BITS.
        //
        // Derive the low LUI_IMM_LOW_BITS bits of imm.
        //   imm_low_4 * 2^12 = rd[0]                    // definition of the low cell
        //   imm * 2^12       = rd[0] + rd[1] * 2^16     // rd[0..2] read as one 32-bit value
        //   imm_low_4 * 2^12 = imm * 2^12 - rd[1] * 2^16
        //   imm_low_4        = imm - rd[1] * 2^4
        let imm_low_4 = imm - rd[1] * AB::F::from_u32(1 << LUI_IMM_LOW_BITS);
        builder.when(is_lui).assert_eq(
            rd[0],
            imm_low_4.clone() * AB::F::from_u32(1 << RV_IS_TYPE_IMM_BITS),
        );

        // Range-check the low LUI_IMM_LOW_BITS bits of imm.
        self.range_bus
            .range_check(imm_low_4, LUI_IMM_LOW_BITS)
            .eval(builder, is_lui);

        let limb_base = AB::F::from_u32(1 << U16_BITS);

        // JAL: constrain rd_low_32 = from_pc + DEFAULT_PC_STEP.
        builder.when(is_jal).assert_eq(
            rd[0],
            from_pc + AB::F::from_u32(DEFAULT_PC_STEP) - rd[1] * limb_base,
        );

        // Range-check the low 32-bit rd cells.
        self.range_bus
            .range_check(rd[0], U16_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(rd[1], U16_BITS)
            .eval(builder, is_valid.clone());

        // Tie is_sign_extend to bit 31 of rd.
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * rd[1] - is_sign_extend * AB::Expr::from_u32(1 << U16_BITS),
                U16_BITS,
            )
            .eval(builder, is_valid.clone());

        // JAL cannot write a return address outside PC_BITS.
        self.range_bus
            .range_check(rd[1] * AB::F::from_u32(1 << PC_HIGH_U16_SHIFT), U16_BITS)
            .eval(builder, is_jal);

        // Sign-extend bit 31 into the upper RV64 register cells.
        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(u16::MAX as u32);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd[0].into(),
            rd[1].into(),
            sign_extend_cell.clone(),
            sign_extend_cell,
        ];

        let to_pc = from_pc + is_lui * AB::F::from_u32(DEFAULT_PC_STEP) + is_jal * imm;

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            is_lui * AB::F::from_u32(LUI as u32) + is_jal * AB::F::from_u32(JAL as u32),
        );

        AdapterAirContext {
            to_pc: Some(to_pc),
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
        Rv64JalLuiOpcode::CLASS_OFFSET
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalLuiExecutor;

#[derive(Clone, derive_new::new)]
pub struct Rv64JalLuiFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

// returns the canonical signed representation of the immediate
// `imm` can be "negative" as a field element
pub(super) fn get_signed_imm<F: PrimeField32>(is_jal: bool, imm: F) -> i32 {
    let imm_f = imm.as_canonical_u32();
    if is_jal {
        if imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)) {
            imm_f as i32
        } else {
            let neg_imm_f = F::ORDER_U32 - imm_f;
            debug_assert!(neg_imm_f < (1 << (RV_J_TYPE_IMM_BITS - 1)));
            -(neg_imm_f as i32)
        }
    } else {
        imm_f as i32
    }
}

// returns (to_pc, rd_data)
#[inline(always)]
pub(super) fn run_jal_lui(is_jal: bool, pc: u32, imm: i32) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    if is_jal {
        let rd_low = pc.wrapping_add(DEFAULT_PC_STEP);
        let next_pc = pc as i32 + imm;
        debug_assert!(next_pc >= 0);
        (next_pc as u32, rv64_u32_to_u16_block(rd_low))
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
