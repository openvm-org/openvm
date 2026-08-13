use std::borrow::Borrow;

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC},
    LocalOpcode,
};
use openvm_riscv_transpiler::JalrOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::adapters::{
    address_add_imm, expand_to_block, ptr_to_u16_limbs, u32_to_u16_block, PTR_U16_LIMBS, U16_BITS,
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct JalrCoreCols<T> {
    pub imm: T,
    // Low 32 bits of rs1 as u16 cells.
    pub rs1_data: [T; PTR_U16_LIMBS],
    // High u16 limb of low-32 rd; the low limb is derived from from_pc.
    pub rd_high: [T; PTR_U16_LIMBS - 1],
    pub is_valid: T,

    pub to_pc_least_sig_bit: T,
    /// Little-endian u16 limbs of the byte target after clearing bit 0.
    pub to_pc_limbs: [T; 2],
    pub imm_sign: T,
}

#[derive(Debug, Clone, derive_new::new, ColumnsAir)]
#[columns_via(JalrCoreCols<u8>)]
pub struct JalrCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for JalrCoreAir {
    fn width(&self) -> usize {
        JalrCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for JalrCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for JalrCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<SignedImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: [AB::Var; 2],
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &JalrCoreCols<AB::Var> = (*local_core).borrow();
        let JalrCoreCols::<AB::Var> {
            imm,
            rs1_data: rs1,
            rd_high,
            is_valid,
            imm_sign,
            to_pc_least_sig_bit,
            to_pc_limbs,
        } = *cols;

        builder.assert_bool(is_valid);

        // composed is the high u16 limb of low-32 rd.
        let composed = rd_high[0] * AB::F::from_u32(1 << U16_BITS);

        let pc_step = AB::F::from_u32(DEFAULT_PC_STEP);
        let pc_step_inv = pc_step.inverse();

        // The byte return address is `from_pc + 4`.
        let least_sig_limb = compose_pc(from_pc.map(Into::into)) + pc_step - composed;

        // rd_data_low is the low-32-bit decomposition of the byte return address.
        let rd_data_low: [AB::Expr; PTR_U16_LIMBS] = [least_sig_limb.clone(), rd_high[0].into()];

        // The low limb is DEFAULT_PC_STEP-aligned and both return-address limbs are u16.
        self.range_bus
            .range_check(least_sig_limb.clone() * pc_step_inv, U16_BITS - 2)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(rd_data_low[1].clone(), U16_BITS)
            .eval(builder, is_valid);

        builder.assert_bool(imm_sign);

        let inv = AB::F::from_u32(1 << U16_BITS).inverse();

        // Constrain `to_pc_least_sig_bit + to_pc` to rs1 + imm as a low-32-bit addition.
        // RISC-V clears bit 0; requiring the resulting low limb to be divisible by four also
        // rejects targets with bit 1 set.
        builder.assert_bool(to_pc_least_sig_bit);
        let carry = (rs1[0] + imm - to_pc_limbs[0] - to_pc_least_sig_bit) * inv;
        builder.when(is_valid).assert_bool(carry.clone());

        // Sign-extend the 16-bit immediate into the high u16 limb.
        let imm_extend_limb = imm_sign * AB::F::from_u32(u16::MAX as u32);
        let carry = (rs1[1] + imm_extend_limb + carry - to_pc_limbs[1]) * inv;
        builder.when(is_valid).assert_bool(carry.clone());
        builder.when(is_valid).assert_eq(carry, imm_sign);

        // The limb widths bind the target to the 32-bit byte-PC domain.
        self.range_bus
            .range_check(to_pc_limbs[1], U16_BITS)
            .eval(builder, is_valid);
        self.range_bus
            .range_check(to_pc_limbs[0] * pc_step_inv, U16_BITS - 2)
            .eval(builder, is_valid);
        let to_pc = to_pc_limbs[0] + to_pc_limbs[1] * AB::F::from_u32(1 << U16_BITS);

        // Zero-extend low-32 rs1/rd at the adapter interface.
        let rs1_data = expand_to_block(&rs1);
        let rd_data = expand_to_block(&rd_data_low);

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, JALR);

        AdapterAirContext {
            to_pc: Some(to_pc),
            reads: [rs1_data].into(),
            writes: [rd_data].into(),
            instruction: SignedImmInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                immediate: imm.into(),
                imm_sign: imm_sign.into(),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        JalrOpcode::CLASS_OFFSET
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct JalrExecutor;

#[derive(Clone)]
pub struct JalrFiller {
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl JalrFiller {
    pub fn new(range_checker_chip: SharedVariableRangeCheckerChip) -> Self {
        assert!(range_checker_chip.range_max_bits() >= U16_BITS);
        Self { range_checker_chip }
    }
}

impl JalrFiller {
    pub(crate) fn fill_core_row<F: PrimeField32>(
        &self,
        core_row: &mut JalrCoreCols<F>,
        rs1_val: u32,
        imm: u16,
        imm_sign: bool,
        to_pc: u32,
        rd_data: [u16; BLOCK_FE_WIDTH],
    ) {
        debug_assert_eq!(to_pc & 0b10, 0);
        let to_pc_limbs = ptr_to_u16_limbs(to_pc & !1).map(u32::from);
        self.range_checker_chip
            .add_count(to_pc_limbs[0] / DEFAULT_PC_STEP, U16_BITS - 2);
        self.range_checker_chip.add_count(to_pc_limbs[1], U16_BITS);

        let rd_low_u16_lo = rd_data[0];
        let rd_low_u16_hi = rd_data[1];

        self.range_checker_chip
            .add_count(u32::from(rd_low_u16_lo) / DEFAULT_PC_STEP, U16_BITS - 2);
        self.range_checker_chip
            .add_count(rd_low_u16_hi as u32, U16_BITS);

        // Write in reverse order
        core_row.imm_sign = F::from_bool(imm_sign);
        core_row.to_pc_limbs = to_pc_limbs.map(F::from_u32);
        core_row.to_pc_least_sig_bit = F::from_bool(to_pc & 1 == 1);
        // fill_trace_row is called only on valid rows
        core_row.is_valid = F::ONE;
        core_row.rs1_data = ptr_to_u16_limbs(rs1_val).map(F::from_u16);
        core_row.rd_high = [F::from_u16(rd_low_u16_hi)];
        core_row.imm = F::from_u16(imm);
    }
}

// returns (to_pc, rd_data)
#[cfg(test)]
#[inline(always)]
pub(super) fn run_jalr(
    pc: u32,
    rs1: u32,
    imm: u16,
    imm_sign: bool,
) -> (u32, [u16; BLOCK_FE_WIDTH]) {
    try_run_jalr(pc, rs1, imm, imm_sign).expect("JALR target exceeds implemented PC address space")
}

pub(super) fn try_run_jalr(
    pc: u32,
    rs1: u32,
    imm: u16,
    imm_sign: bool,
) -> Option<(u32, [u16; BLOCK_FE_WIDTH])> {
    let imm_extended = imm as u32 + (imm_sign as u32 * ((u16::MAX as u32) << U16_BITS));
    let to_pc = address_add_imm(rs1, imm_extended);
    if to_pc > u64::from(MAX_ALLOWED_PC) {
        return None;
    }
    let to_pc = to_pc as u32;
    // RISC-V clears bit 0 of the target; a set bit 1 leaves the target misaligned, which is
    // rejected (and unprovable in the circuit).
    if to_pc & 0b10 != 0 {
        return None;
    }

    let rd_low_u32 = pc.wrapping_add(DEFAULT_PC_STEP);
    let rd_data = u32_to_u16_block(rd_low_u32);
    Some((to_pc, rd_data))
}
