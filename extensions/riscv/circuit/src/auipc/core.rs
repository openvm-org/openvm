use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{program::PC_BITS, LocalOpcode};
use openvm_riscv_transpiler::AuipcOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

use crate::adapters::{ptr_to_u16_limbs, sext32_to_u64, BYTE_BITS, PTR_U16_LIMBS, U16_BITS};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct AuipcCoreCols<T> {
    pub is_valid: T,
    pub is_sign_extend: T,
    // The immediate is split around the byte shift in AUIPC's `imm << 8`.
    pub imm_low_8: T,
    pub imm_high_16: T,
    // High u16 limb of `from_pc`; the low limb is derived from `from_pc`.
    pub pc_high: T,
    pub rd_data: [T; PTR_U16_LIMBS],
}

#[derive(Debug, Clone, Copy, derive_new::new, ColumnsAir)]
#[columns_via(AuipcCoreCols<u8>)]
pub struct AuipcCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for AuipcCoreAir {
    fn width(&self) -> usize {
        AuipcCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for AuipcCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for AuipcCoreAir
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
        let cols: &AuipcCoreCols<AB::Var> = (*local_core).borrow();

        let AuipcCoreCols {
            is_valid,
            is_sign_extend,
            imm_low_8,
            imm_high_16,
            pc_high,
            rd_data,
        } = *cols;
        builder.assert_bool(is_valid);
        builder.assert_bool(is_sign_extend);

        // We want to constrain rd = from_pc + (imm << BYTE_BITS) where:
        // - rd_data represents the low 32 bits of rd as u16 cells
        // - imm_low_8 and imm_high_16 decompose the 24-bit instruction immediate
        let limb_base = AB::F::from_u32(1 << U16_BITS);
        let carry_divide = limb_base.inverse();
        let imm = imm_low_8 + imm_high_16 * AB::Expr::from_u32(1 << BYTE_BITS);
        let pc_low = from_pc - pc_high * limb_base;

        // `from_pc` is bounded to `PC_BITS` by the program bus.
        self.range_bus
            .range_check(pc_low.clone(), U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(pc_high, PC_BITS - U16_BITS)
            .eval(builder, is_valid);

        let carry_low =
            (pc_low + imm_low_8 * AB::F::from_u32(1 << BYTE_BITS) - rd_data[0]) * carry_divide;
        builder.when(is_valid).assert_bool(carry_low.clone());

        let carry_top = (pc_high + imm_high_16 + carry_low - rd_data[1]) * carry_divide;
        builder.when(is_valid).assert_bool(carry_top.clone());

        // Check that the computed sign matches the top bit of `imm_high_16`.
        let imm_sign = is_sign_extend + carry_top;
        self.range_bus
            .range_check(
                AB::Expr::from_u32(2) * imm_high_16 - imm_sign * AB::Expr::from_u32(1 << U16_BITS),
                U16_BITS,
            )
            .eval(builder, is_valid);

        // Range check rd and immediate limbs.
        self.range_bus
            .range_check(rd_data[0], U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data[1], U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(imm_low_8, BYTE_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(imm_high_16, U16_BITS)
            .eval(builder, is_valid);

        let sign_extend_cell = is_sign_extend * AB::Expr::from_u32(u16::MAX as u32);
        let write_data: [AB::Expr; BLOCK_FE_WIDTH] = [
            rd_data[0].into(),
            rd_data[1].into(),
            sign_extend_cell.clone(),
            sign_extend_cell,
        ];
        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, AUIPC);
        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [write_data].into(),
            instruction: ImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        AuipcOpcode::CLASS_OFFSET
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct AuipcExecutor;

#[derive(Clone, derive_new::new)]
pub struct AuipcFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

// returns rd_data
#[inline(always)]
pub(super) fn run_auipc(pc: u32, imm: u32) -> [u16; BLOCK_FE_WIDTH] {
    let offset = imm << BYTE_BITS;
    let auipc = (pc as u64).wrapping_add(sext32_to_u64(offset));
    let auipc_hi = auipc >> 32;
    debug_assert!(auipc_hi == 0 || auipc_hi == u64::from(u32::MAX));
    let auipc_lo = auipc as u32;

    let [lo, hi] = ptr_to_u16_limbs(auipc_lo);
    let sign = if auipc_hi != 0 { u16::MAX } else { 0 };
    [lo, hi, sign, sign]
}
