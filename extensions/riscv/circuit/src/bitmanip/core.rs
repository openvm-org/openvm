use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{
    BitwiseInvOpcode, ByteUnaryOpcode, CountZerosOpcode, CountZerosWOpcode, CpopOpcode,
    CpopWOpcode, MinMaxOpcode, RotateImmOpcode, RotateOpcode, RotateWImmOpcode, RotateWOpcode,
    ShAddOpcode, SingleBitImmOpcode, SingleBitOpcode, SlliUwOpcode,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::{BITMANIP_LIMB_BITS, BITMANIP_NUM_BITS, BITMANIP_NUM_LIMBS};

pub(crate) const BITMANIP_OFFSET: usize = ShAddOpcode::CLASS_OFFSET;

const fn local<const CLASS_OFFSET: usize>(opcode: usize) -> usize {
    CLASS_OFFSET - BITMANIP_OFFSET + opcode
}

pub(crate) const SH1ADD: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH1ADD as usize);
pub(crate) const SH2ADD: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH2ADD as usize);
pub(crate) const SH3ADD: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH3ADD as usize);
pub(crate) const ADD_UW: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::ADD_UW as usize);
pub(crate) const SH1ADD_UW: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH1ADD_UW as usize);
pub(crate) const SH2ADD_UW: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH2ADD_UW as usize);
pub(crate) const SH3ADD_UW: usize =
    local::<{ ShAddOpcode::CLASS_OFFSET }>(ShAddOpcode::SH3ADD_UW as usize);
pub(crate) const SLLI_UW: usize =
    local::<{ SlliUwOpcode::CLASS_OFFSET }>(SlliUwOpcode::SLLI_UW as usize);
pub(crate) const ANDN: usize =
    local::<{ BitwiseInvOpcode::CLASS_OFFSET }>(BitwiseInvOpcode::ANDN as usize);
pub(crate) const ORN: usize =
    local::<{ BitwiseInvOpcode::CLASS_OFFSET }>(BitwiseInvOpcode::ORN as usize);
pub(crate) const XNOR: usize =
    local::<{ BitwiseInvOpcode::CLASS_OFFSET }>(BitwiseInvOpcode::XNOR as usize);
pub(crate) const ROL: usize = local::<{ RotateOpcode::CLASS_OFFSET }>(RotateOpcode::ROL as usize);
pub(crate) const ROR: usize = local::<{ RotateOpcode::CLASS_OFFSET }>(RotateOpcode::ROR as usize);
pub(crate) const RORI: usize =
    local::<{ RotateImmOpcode::CLASS_OFFSET }>(RotateImmOpcode::RORI as usize);
pub(crate) const ROLW: usize =
    local::<{ RotateWOpcode::CLASS_OFFSET }>(RotateWOpcode::ROLW as usize);
pub(crate) const RORW: usize =
    local::<{ RotateWOpcode::CLASS_OFFSET }>(RotateWOpcode::RORW as usize);
pub(crate) const RORIW: usize =
    local::<{ RotateWImmOpcode::CLASS_OFFSET }>(RotateWImmOpcode::RORIW as usize);
pub(crate) const CLZ: usize =
    local::<{ CountZerosOpcode::CLASS_OFFSET }>(CountZerosOpcode::CLZ as usize);
pub(crate) const CTZ: usize =
    local::<{ CountZerosOpcode::CLASS_OFFSET }>(CountZerosOpcode::CTZ as usize);
pub(crate) const CLZW: usize =
    local::<{ CountZerosWOpcode::CLASS_OFFSET }>(CountZerosWOpcode::CLZW as usize);
pub(crate) const CTZW: usize =
    local::<{ CountZerosWOpcode::CLASS_OFFSET }>(CountZerosWOpcode::CTZW as usize);
pub(crate) const CPOP: usize = local::<{ CpopOpcode::CLASS_OFFSET }>(CpopOpcode::CPOP as usize);
pub(crate) const CPOPW: usize = local::<{ CpopWOpcode::CLASS_OFFSET }>(CpopWOpcode::CPOPW as usize);
pub(crate) const MIN: usize = local::<{ MinMaxOpcode::CLASS_OFFSET }>(MinMaxOpcode::MIN as usize);
pub(crate) const MINU: usize = local::<{ MinMaxOpcode::CLASS_OFFSET }>(MinMaxOpcode::MINU as usize);
pub(crate) const MAX: usize = local::<{ MinMaxOpcode::CLASS_OFFSET }>(MinMaxOpcode::MAX as usize);
pub(crate) const MAXU: usize = local::<{ MinMaxOpcode::CLASS_OFFSET }>(MinMaxOpcode::MAXU as usize);
pub(crate) const SEXT_B: usize =
    local::<{ ByteUnaryOpcode::CLASS_OFFSET }>(ByteUnaryOpcode::SEXT_B as usize);
pub(crate) const SEXT_H: usize =
    local::<{ ByteUnaryOpcode::CLASS_OFFSET }>(ByteUnaryOpcode::SEXT_H as usize);
pub(crate) const ZEXT_H: usize =
    local::<{ ByteUnaryOpcode::CLASS_OFFSET }>(ByteUnaryOpcode::ZEXT_H as usize);
pub(crate) const ORC_B: usize =
    local::<{ ByteUnaryOpcode::CLASS_OFFSET }>(ByteUnaryOpcode::ORC_B as usize);
pub(crate) const REV8: usize =
    local::<{ ByteUnaryOpcode::CLASS_OFFSET }>(ByteUnaryOpcode::REV8 as usize);
pub(crate) const BCLR: usize =
    local::<{ SingleBitOpcode::CLASS_OFFSET }>(SingleBitOpcode::BCLR as usize);
pub(crate) const BSET: usize =
    local::<{ SingleBitOpcode::CLASS_OFFSET }>(SingleBitOpcode::BSET as usize);
pub(crate) const BINV: usize =
    local::<{ SingleBitOpcode::CLASS_OFFSET }>(SingleBitOpcode::BINV as usize);
pub(crate) const BEXT: usize =
    local::<{ SingleBitOpcode::CLASS_OFFSET }>(SingleBitOpcode::BEXT as usize);
pub(crate) const BCLRI: usize =
    local::<{ SingleBitImmOpcode::CLASS_OFFSET }>(SingleBitImmOpcode::BCLRI as usize);
pub(crate) const BSETI: usize =
    local::<{ SingleBitImmOpcode::CLASS_OFFSET }>(SingleBitImmOpcode::BSETI as usize);
pub(crate) const BINVI: usize =
    local::<{ SingleBitImmOpcode::CLASS_OFFSET }>(SingleBitImmOpcode::BINVI as usize);
pub(crate) const BEXTI: usize =
    local::<{ SingleBitImmOpcode::CLASS_OFFSET }>(SingleBitImmOpcode::BEXTI as usize);

const SHADD_OPS: [usize; SHADD_OP_COUNT] = [
    SH1ADD, SH2ADD, SH3ADD, ADD_UW, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW,
];
const REG_OPS: [usize; REG_OP_COUNT] = [
    ANDN, ORN, XNOR, ROL, ROR, ROLW, RORW, MIN, MINU, MAX, MAXU, BCLR, BSET, BINV, BEXT,
];
const IMM_OPS: [usize; IMM_OP_COUNT] = [
    RORI, RORIW, CLZ, CTZ, CLZW, CTZW, CPOP, CPOPW, SEXT_B, SEXT_H, ZEXT_H, ORC_B, REV8, BCLRI,
    BSETI, BINVI, BEXTI,
];

pub(crate) const SHADD_OP_COUNT: usize = 7;
pub(crate) const REG_OP_COUNT: usize = 15;
pub(crate) const IMM_OP_COUNT: usize = 17;

fn flag_pos(ops: &[usize], local_opcode: usize) -> usize {
    ops.iter()
        .position(|&op| op == local_opcode)
        .unwrap_or_else(|| panic!("unsupported bitmanip local opcode {local_opcode}"))
}

pub(crate) fn is_reg_opcode(local_opcode: usize) -> bool {
    REG_OPS.contains(&local_opcode)
}

pub(crate) fn is_shadd_opcode(local_opcode: usize) -> bool {
    SHADD_OPS.contains(&local_opcode)
}

pub(crate) fn is_imm_opcode(local_opcode: usize) -> bool {
    IMM_OPS.contains(&local_opcode)
}

pub(crate) fn is_slli_uw_opcode(local_opcode: usize) -> bool {
    local_opcode == SLLI_UW
}

pub(crate) fn limbs_to_u64(limbs: &[u16; BITMANIP_NUM_LIMBS]) -> u64 {
    limbs
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, limb)| acc | ((*limb as u64) << (16 * i)))
}

pub(crate) fn u64_to_limbs(value: u64) -> [u16; BITMANIP_NUM_LIMBS] {
    array::from_fn(|i| ((value >> (16 * i)) & 0xffff) as u16)
}

fn bits_u64(value: u64) -> [u8; BITMANIP_NUM_BITS] {
    array::from_fn(|i| ((value >> i) & 1) as u8)
}

fn highest_diff_bit(x: u64, y: u64) -> Option<usize> {
    let diff = x ^ y;
    if diff == 0 {
        None
    } else {
        Some(63 - diff.leading_zeros() as usize)
    }
}

pub(crate) fn run_bitmanip_reg(local_opcode: usize, rs1: u64, rs2: u64) -> u64 {
    match local_opcode {
        SH1ADD => rs1.wrapping_shl(1).wrapping_add(rs2),
        SH2ADD => rs1.wrapping_shl(2).wrapping_add(rs2),
        SH3ADD => rs1.wrapping_shl(3).wrapping_add(rs2),
        ADD_UW => (rs1 as u32 as u64).wrapping_add(rs2),
        SH1ADD_UW => ((rs1 as u32 as u64) << 1).wrapping_add(rs2),
        SH2ADD_UW => ((rs1 as u32 as u64) << 2).wrapping_add(rs2),
        SH3ADD_UW => ((rs1 as u32 as u64) << 3).wrapping_add(rs2),
        ANDN => rs1 & !rs2,
        ORN => rs1 | !rs2,
        XNOR => !(rs1 ^ rs2),
        ROL => rs1.rotate_left((rs2 & 63) as u32),
        ROR => rs1.rotate_right((rs2 & 63) as u32),
        ROLW => {
            let value = (rs1 as u32).rotate_left((rs2 & 31) as u32);
            (value as i32 as i64) as u64
        }
        RORW => {
            let value = (rs1 as u32).rotate_right((rs2 & 31) as u32);
            (value as i32 as i64) as u64
        }
        MIN => (rs1 as i64).min(rs2 as i64) as u64,
        MINU => rs1.min(rs2),
        MAX => (rs1 as i64).max(rs2 as i64) as u64,
        MAXU => rs1.max(rs2),
        BCLR => rs1 & !(1u64 << (rs2 & 63)),
        BSET => rs1 | (1u64 << (rs2 & 63)),
        BINV => rs1 ^ (1u64 << (rs2 & 63)),
        BEXT => (rs1 >> (rs2 & 63)) & 1,
        _ => unreachable!("unsupported bitmanip register opcode {local_opcode}"),
    }
}

pub(crate) fn run_bitmanip_imm(local_opcode: usize, rs1: u64, imm: u32) -> u64 {
    match local_opcode {
        SLLI_UW => (rs1 as u32 as u64) << imm,
        RORI => rs1.rotate_right(imm),
        RORIW => {
            let value = (rs1 as u32).rotate_right(imm);
            (value as i32 as i64) as u64
        }
        CLZ => rs1.leading_zeros() as u64,
        CTZ => rs1.trailing_zeros() as u64,
        CLZW => (rs1 as u32).leading_zeros() as u64,
        CTZW => (rs1 as u32).trailing_zeros() as u64,
        CPOP => rs1.count_ones() as u64,
        CPOPW => (rs1 as u32).count_ones() as u64,
        SEXT_B => (rs1 as u8 as i8 as i64) as u64,
        SEXT_H => (rs1 as u16 as i16 as i64) as u64,
        ZEXT_H => rs1 as u16 as u64,
        ORC_B => {
            let mut out = 0u64;
            for i in 0..8 {
                let byte = ((rs1 >> (8 * i)) & 0xff) as u8;
                if byte != 0 {
                    out |= 0xffu64 << (8 * i);
                }
            }
            out
        }
        REV8 => rs1.swap_bytes(),
        BCLRI => rs1 & !(1u64 << imm),
        BSETI => rs1 | (1u64 << imm),
        BINVI => rs1 ^ (1u64 << imm),
        BEXTI => (rs1 >> imm) & 1,
        _ => unreachable!("unsupported bitmanip immediate opcode {local_opcode}"),
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipShAddCoreCols<T> {
    pub a: [T; BITMANIP_NUM_LIMBS],
    pub b: [T; BITMANIP_NUM_LIMBS],
    pub c: [T; BITMANIP_NUM_LIMBS],
    pub opcode_flags: [T; SHADD_OP_COUNT],
    pub bit_shift_carry: [T; BITMANIP_NUM_LIMBS],
    pub bit_shift_aux: [T; BITMANIP_NUM_LIMBS],
    pub add_carry: [T; BITMANIP_NUM_LIMBS + 1],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipShAddCoreCols<u8>)]
pub struct BitManipShAddCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for BitManipShAddCoreAir {
    fn width(&self) -> usize {
        BitManipShAddCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipShAddCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipShAddCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipShAddCoreCols<_> = local_core.borrow();
        let mut is_valid = AB::Expr::ZERO;
        let mut expected_local_opcode = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;
        let mut uw_flag = AB::Expr::ZERO;

        for (flag, local_opcode) in cols.opcode_flags.iter().zip(SHADD_OPS) {
            builder.assert_bool(*flag);
            let flag: AB::Expr = (*flag).into();
            let (shift, is_uw) = shadd_shift_uw(local_opcode);
            is_valid += flag.clone();
            expected_local_opcode += flag.clone() * AB::Expr::from_usize(local_opcode);
            bit_shift += flag.clone() * AB::Expr::from_usize(shift);
            bit_multiplier += flag.clone() * AB::Expr::from_usize(1 << shift);
            carry_multiplier +=
                flag.clone() * AB::Expr::from_usize(1 << (BITMANIP_LIMB_BITS - shift));
            uw_flag += flag * if is_uw { AB::Expr::ONE } else { AB::Expr::ZERO };
        }
        builder.assert_bool(is_valid.clone());

        builder
            .when(is_valid.clone())
            .assert_zero(cols.add_carry[0]);
        for carry in &cols.add_carry[1..] {
            let carry: AB::Expr = (*carry).into();
            builder.assert_zero(is_valid.clone() * carry.clone() * (AB::Expr::ONE - carry));
        }

        let aux_bits = AB::Expr::from_usize(BITMANIP_LIMB_BITS) - bit_shift.clone();
        for limb in 0..BITMANIP_NUM_LIMBS {
            let source_active = if limb < 2 {
                is_valid.clone()
            } else {
                is_valid.clone() - uw_flag.clone()
            };
            let source = cols.b[limb] * source_active;
            builder.assert_eq(
                source,
                cols.bit_shift_aux[limb] + cols.bit_shift_carry[limb] * carry_multiplier.clone(),
            );
            self.range_bus
                .send(cols.bit_shift_carry[limb], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[limb], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        for limb in 0..BITMANIP_NUM_LIMBS {
            let carry_in = if limb == 0 {
                AB::Expr::ZERO
            } else {
                cols.bit_shift_carry[limb - 1].into()
            };
            let shifted = cols.bit_shift_aux[limb] * bit_multiplier.clone() + carry_in;
            builder.assert_zero(
                shifted + cols.c[limb] + cols.add_carry[limb]
                    - cols.a[limb]
                    - AB::Expr::from_usize(1 << BITMANIP_LIMB_BITS) * cols.add_carry[limb + 1],
            );
        }

        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipShAddCoreRecord {
    pub b: [u16; BITMANIP_NUM_LIMBS],
    pub c: [u16; BITMANIP_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipShAddExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipShAddFiller<A> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipShAddExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; BITMANIP_NUM_LIMBS]; 2]>,
            WriteData: From<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipShAddCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BShAdd({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_shadd_opcode(local_opcode));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.local_opcode = local_opcode as u8;

        let output = run_bitmanip_reg(
            local_opcode,
            limbs_to_u64(&core_record.b),
            limbs_to_u64(&core_record.c),
        );
        self.adapter.write(
            state.memory,
            instruction,
            [u64_to_limbs(output)].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipShAddFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipShAddCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let c = record.c;
        let local_opcode = record.local_opcode as usize;
        let b_u64 = limbs_to_u64(&b);
        let c_u64 = limbs_to_u64(&c);
        let a_u64 = run_bitmanip_reg(local_opcode, b_u64, c_u64);
        let a = u64_to_limbs(a_u64);

        let core_row: &mut BitManipShAddCoreCols<F> = core_row.borrow_mut();
        core_row.opcode_flags = [F::ZERO; SHADD_OP_COUNT];
        core_row.opcode_flags[flag_pos(&SHADD_OPS, local_opcode)] = F::ONE;
        core_row.a = a.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
        core_row.c = c.map(F::from_u16);
        fill_shadd_aux(core_row, local_opcode, b, c, &self.range_checker_chip);
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipSlliUwCoreCols<T> {
    pub a: [T; BITMANIP_NUM_LIMBS],
    pub b: [T; BITMANIP_NUM_LIMBS],
    pub bit_shift_marker: [T; BITMANIP_LIMB_BITS],
    pub limb_shift_marker: [T; BITMANIP_NUM_LIMBS],
    pub bit_shift_carry: [T; BITMANIP_NUM_LIMBS],
    pub bit_shift_aux: [T; BITMANIP_NUM_LIMBS],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipSlliUwCoreCols<u8>)]
pub struct BitManipSlliUwCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for BitManipSlliUwCoreAir {
    fn width(&self) -> usize {
        BitManipSlliUwCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipSlliUwCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipSlliUwCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipSlliUwCoreCols<_> = local_core.borrow();

        let mut bit_marker_sum = AB::Expr::ZERO;
        let mut bit_shift = AB::Expr::ZERO;
        let mut bit_multiplier = AB::Expr::ZERO;
        let mut carry_multiplier = AB::Expr::ZERO;
        for bit in 0..BITMANIP_LIMB_BITS {
            builder.assert_bool(cols.bit_shift_marker[bit]);
            let marker: AB::Expr = cols.bit_shift_marker[bit].into();
            bit_marker_sum += marker.clone();
            bit_shift += AB::Expr::from_usize(bit) * marker.clone();
            bit_multiplier += AB::Expr::from_usize(1 << bit) * marker.clone();
            carry_multiplier += AB::Expr::from_usize(1 << (BITMANIP_LIMB_BITS - bit)) * marker;
        }
        builder.assert_bool(bit_marker_sum.clone());
        let is_valid = bit_marker_sum;

        let aux_bits = AB::Expr::from_usize(BITMANIP_LIMB_BITS) - bit_shift.clone();
        for limb in 0..BITMANIP_NUM_LIMBS {
            let source = if limb < 2 {
                cols.b[limb].into()
            } else {
                AB::Expr::ZERO
            };
            builder.assert_eq(
                source * is_valid.clone(),
                cols.bit_shift_aux[limb] + cols.bit_shift_carry[limb] * carry_multiplier.clone(),
            );
            self.range_bus
                .send(cols.bit_shift_carry[limb], bit_shift.clone())
                .eval(builder, is_valid.clone());
            self.range_bus
                .send(cols.bit_shift_aux[limb], aux_bits.clone())
                .eval(builder, is_valid.clone());
        }

        let mut limb_marker_sum = AB::Expr::ZERO;
        let mut limb_shift = AB::Expr::ZERO;
        for limb_shift_idx in 0..BITMANIP_NUM_LIMBS {
            builder.assert_bool(cols.limb_shift_marker[limb_shift_idx]);
            limb_marker_sum += cols.limb_shift_marker[limb_shift_idx].into();
            limb_shift +=
                AB::Expr::from_usize(limb_shift_idx) * cols.limb_shift_marker[limb_shift_idx];

            let mut when_limb_shift = builder.when(cols.limb_shift_marker[limb_shift_idx]);
            for out_limb in 0..BITMANIP_NUM_LIMBS {
                if out_limb < limb_shift_idx {
                    when_limb_shift.assert_zero(cols.a[out_limb]);
                } else {
                    let src_limb = out_limb - limb_shift_idx;
                    let carry_in = if src_limb == 0 {
                        AB::Expr::ZERO
                    } else {
                        cols.bit_shift_carry[src_limb - 1].into()
                    };
                    when_limb_shift.assert_eq(
                        cols.a[out_limb],
                        cols.bit_shift_aux[src_limb] * bit_multiplier.clone() + carry_in,
                    );
                }
            }
        }
        builder.assert_eq(limb_marker_sum, is_valid.clone());

        let immediate = limb_shift * AB::Expr::from_usize(BITMANIP_LIMB_BITS) + bit_shift;
        let expected_opcode = AB::Expr::from_usize(BITMANIP_OFFSET + SLLI_UW);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipSlliUwCoreRecord {
    pub b: [u16; BITMANIP_NUM_LIMBS],
    pub imm: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipSlliUwExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipSlliUwFiller<A> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipSlliUwExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
            WriteData: From<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipSlliUwCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BSlliUw({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_slli_uw_opcode(local_opcode));
        let imm = instruction.c.as_canonical_u32();

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.imm = imm as u8;

        let output = run_bitmanip_imm(local_opcode, limbs_to_u64(&core_record.b), imm);
        self.adapter.write(
            state.memory,
            instruction,
            [u64_to_limbs(output)].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipSlliUwFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipSlliUwCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let imm = record.imm as u32;
        let a = u64_to_limbs(run_bitmanip_imm(SLLI_UW, limbs_to_u64(&b), imm));

        let core_row: &mut BitManipSlliUwCoreCols<F> = core_row.borrow_mut();
        core_row.a = a.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
        fill_slli_uw_aux(core_row, b, imm as usize, &self.range_checker_chip);
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipRegCoreCols<T> {
    pub a: [T; BITMANIP_NUM_LIMBS],
    pub b: [T; BITMANIP_NUM_LIMBS],
    pub c: [T; BITMANIP_NUM_LIMBS],
    pub a_bits: [T; BITMANIP_NUM_BITS],
    pub b_bits: [T; BITMANIP_NUM_BITS],
    pub c_bits: [T; BITMANIP_NUM_BITS],
    pub opcode_flags: [T; REG_OP_COUNT],
    pub index_marker: [T; BITMANIP_NUM_BITS],
    pub minmax_lt: T,
    pub minmax_diff_marker: [T; BITMANIP_NUM_BITS],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipRegCoreCols<u8>)]
pub struct BitManipRegCoreAir;

impl<F: Field> BaseAir<F> for BitManipRegCoreAir {
    fn width(&self) -> usize {
        BitManipRegCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipRegCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipRegCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipRegCoreCols<_> = local_core.borrow();
        let mut is_valid = AB::Expr::ZERO;
        let mut expected_local_opcode = AB::Expr::ZERO;
        for (flag, local_opcode) in cols.opcode_flags.iter().zip(REG_OPS) {
            builder.assert_bool(*flag);
            is_valid += (*flag).into();
            expected_local_opcode += (*flag).into() * AB::Expr::from_usize(local_opcode);
        }
        builder.assert_bool(is_valid.clone());

        constrain_bits_and_limbs(builder, is_valid.clone(), &cols.a_bits, &cols.a);
        constrain_bits_and_limbs(builder, is_valid.clone(), &cols.b_bits, &cols.b);
        constrain_bits_and_limbs(builder, is_valid.clone(), &cols.c_bits, &cols.c);

        let flag = |op| cols.opcode_flags[flag_pos(&REG_OPS, op)];
        let bitwise_valid = [ANDN, ORN, XNOR]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let rotate64_valid = [ROL, ROR]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let rotatew_valid = [ROLW, RORW]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let rotate_valid = rotate64_valid.clone() + rotatew_valid.clone();
        let minmax_valid = [MIN, MINU, MAX, MAXU]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let single_valid = [BCLR, BSET, BINV, BEXT]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));

        self.eval_bitwise_inv(builder, cols, bitwise_valid);
        self.eval_index_markers(
            builder,
            cols,
            rotate64_valid.clone() + single_valid.clone(),
            rotatew_valid.clone(),
        );
        self.eval_rotate(builder, cols, rotate_valid);
        self.eval_minmax(builder, cols, minmax_valid);
        self.eval_single_bit(builder, cols, single_valid);

        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

impl BitManipRegCoreAir {
    fn eval_bitwise_inv<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipRegCoreCols<AB::Var>,
        bitwise_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&REG_OPS, op)];
        for i in 0..BITMANIP_NUM_BITS {
            let b: AB::Expr = cols.b_bits[i].into();
            let c: AB::Expr = cols.c_bits[i].into();
            let a: AB::Expr = cols.a_bits[i].into();
            let expected = flag(ANDN) * b.clone() * (AB::Expr::ONE - c.clone())
                + flag(ORN) * (AB::Expr::ONE - (AB::Expr::ONE - b.clone()) * c.clone())
                + flag(XNOR)
                    * (AB::Expr::ONE - b.clone() - c.clone() + AB::Expr::from_u8(2) * b * c);
            builder.assert_zero(bitwise_valid.clone() * a - expected);
        }
    }

    fn eval_index_markers<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipRegCoreCols<AB::Var>,
        full_index_valid: AB::Expr,
        word_index_valid: AB::Expr,
    ) {
        let mut marker_sum = AB::Expr::ZERO;
        let mut index = AB::Expr::ZERO;
        for (i, marker) in cols.index_marker.iter().enumerate() {
            builder.assert_bool(*marker);
            marker_sum += (*marker).into();
            index += (*marker).into() * AB::Expr::from_usize(i);
        }
        builder.assert_eq(
            marker_sum,
            full_index_valid.clone() + word_index_valid.clone(),
        );

        let c_low6 = (0..6).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.c_bits[i] * AB::Expr::from_usize(1 << i)
        });
        let c_low5 = (0..5).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.c_bits[i] * AB::Expr::from_usize(1 << i)
        });
        builder.assert_zero(full_index_valid * (index.clone() - c_low6));
        builder.assert_zero(word_index_valid * (index - c_low5));
    }

    fn eval_rotate<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipRegCoreCols<AB::Var>,
        rotate_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&REG_OPS, op)];
        for i in 0..BITMANIP_NUM_BITS {
            let mut expected64 = AB::Expr::ZERO;
            let mut expectedw = AB::Expr::ZERO;
            for s in 0..BITMANIP_NUM_BITS {
                let marker = cols.index_marker[s];
                expected64 += flag(ROL)
                    * marker
                    * cols.b_bits[(i + BITMANIP_NUM_BITS - s) % BITMANIP_NUM_BITS];
                expected64 += flag(ROR) * marker * cols.b_bits[(i + s) % BITMANIP_NUM_BITS];
                let w_bit = if i < 32 { i } else { 31 };
                expectedw += flag(ROLW) * marker * cols.b_bits[(w_bit + 32 - (s % 32)) % 32];
                expectedw += flag(RORW) * marker * cols.b_bits[(w_bit + s) % 32];
            }
            builder.assert_zero(rotate_valid.clone() * cols.a_bits[i] - expected64 - expectedw);
        }
    }

    fn eval_minmax<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipRegCoreCols<AB::Var>,
        minmax_valid: AB::Expr,
    ) {
        let minmax_lt: AB::Expr = cols.minmax_lt.into();
        builder.assert_zero(
            minmax_valid.clone() * minmax_lt.clone() * (AB::Expr::ONE - minmax_lt.clone()),
        );

        let mut diff_marker_sum = AB::Expr::ZERO;
        let mut prefix_sum = AB::Expr::ZERO;
        let flag = |op| cols.opcode_flags[flag_pos(&REG_OPS, op)];
        let signed_flag = flag(MIN) + flag(MAX);
        let unsigned_flag = flag(MINU) + flag(MAXU);
        for i in (0..BITMANIP_NUM_BITS).rev() {
            let marker = cols.minmax_diff_marker[i];
            builder.assert_bool(marker);
            let marker_expr: AB::Expr = marker.into();
            diff_marker_sum += marker_expr.clone();
            prefix_sum += marker_expr.clone();
            let b: AB::Expr = cols.b_bits[i].into();
            let c: AB::Expr = cols.c_bits[i].into();
            builder.assert_zero(
                minmax_valid.clone()
                    * (AB::Expr::ONE - prefix_sum.clone())
                    * (b.clone() - c.clone()),
            );
            builder.assert_zero(marker_expr.clone() * (b.clone() + c.clone() - AB::Expr::ONE));
            let signed_here = if i == BITMANIP_NUM_BITS - 1 {
                b
            } else {
                c.clone()
            };
            builder.assert_zero(
                marker_expr.clone()
                    * (signed_flag.clone() * (minmax_lt.clone() - signed_here)
                        + unsigned_flag.clone() * (minmax_lt.clone() - c)),
            );
        }
        builder.assert_bool(diff_marker_sum.clone());
        builder.assert_zero((minmax_valid.clone() - diff_marker_sum.clone()) * minmax_lt.clone());
        builder.assert_zero((AB::Expr::ONE - minmax_valid.clone()) * diff_marker_sum);

        let min_flag = flag(MIN) + flag(MINU);
        let max_flag = flag(MAX) + flag(MAXU);
        for i in 0..BITMANIP_NUM_BITS {
            let b: AB::Expr = cols.b_bits[i].into();
            let c: AB::Expr = cols.c_bits[i].into();
            let min_expected = c.clone() + minmax_lt.clone() * (b.clone() - c.clone());
            let max_expected = b.clone() + minmax_lt.clone() * (c - b);
            builder.assert_zero(
                min_flag.clone() * (cols.a_bits[i] - min_expected)
                    + max_flag.clone() * (cols.a_bits[i] - max_expected),
            );
        }
    }

    fn eval_single_bit<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipRegCoreCols<AB::Var>,
        single_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&REG_OPS, op)];
        let selected = (0..BITMANIP_NUM_BITS).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.index_marker[i] * cols.b_bits[i]
        });
        for i in 0..BITMANIP_NUM_BITS {
            let marker: AB::Expr = cols.index_marker[i].into();
            let b: AB::Expr = cols.b_bits[i].into();
            let expected = flag(BCLR) * b.clone() * (AB::Expr::ONE - marker.clone())
                + flag(BSET) * (b.clone() + marker.clone() * (AB::Expr::ONE - b.clone()))
                + flag(BINV) * (b.clone() + marker * (AB::Expr::ONE - AB::Expr::from_u8(2) * b))
                + flag(BEXT)
                    * if i == 0 {
                        selected.clone()
                    } else {
                        AB::Expr::ZERO
                    };
            builder.assert_zero(single_valid.clone() * cols.a_bits[i] - expected);
        }
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipRegCoreRecord {
    pub b: [u16; BITMANIP_NUM_LIMBS],
    pub c: [u16; BITMANIP_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipRegExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipRegFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipRegExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; BITMANIP_NUM_LIMBS]; 2]>,
            WriteData: From<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipRegCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BReg({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_reg_opcode(local_opcode));

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.local_opcode = local_opcode as u8;

        let output = run_bitmanip_reg(
            local_opcode,
            limbs_to_u64(&core_record.b),
            limbs_to_u64(&core_record.c),
        );
        self.adapter.write(
            state.memory,
            instruction,
            [u64_to_limbs(output)].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipRegFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipRegCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let c = record.c;
        let local_opcode = record.local_opcode as usize;
        let b_u64 = limbs_to_u64(&b);
        let c_u64 = limbs_to_u64(&c);
        let a_u64 = run_bitmanip_reg(local_opcode, b_u64, c_u64);
        let a = u64_to_limbs(a_u64);
        let b_bits = bits_u64(b_u64);
        let c_bits = bits_u64(c_u64);
        let a_bits = bits_u64(a_u64);

        let core_row: &mut BitManipRegCoreCols<F> = core_row.borrow_mut();
        core_row.opcode_flags = [F::ZERO; REG_OP_COUNT];
        core_row.opcode_flags[flag_pos(&REG_OPS, local_opcode)] = F::ONE;
        core_row.index_marker = [F::ZERO; BITMANIP_NUM_BITS];
        if matches!(local_opcode, ROL | ROR | BCLR | BSET | BINV | BEXT) {
            core_row.index_marker[(c_u64 & 63) as usize] = F::ONE;
        } else if matches!(local_opcode, ROLW | RORW) {
            core_row.index_marker[(c_u64 & 31) as usize] = F::ONE;
        }
        core_row.minmax_diff_marker = [F::ZERO; BITMANIP_NUM_BITS];
        core_row.minmax_lt = F::ZERO;
        if matches!(local_opcode, MIN | MINU | MAX | MAXU) {
            let signed = matches!(local_opcode, MIN | MAX);
            let lt = if signed {
                (b_u64 as i64) < (c_u64 as i64)
            } else {
                b_u64 < c_u64
            };
            core_row.minmax_lt = F::from_bool(lt);
            if let Some(idx) = highest_diff_bit(b_u64, c_u64) {
                core_row.minmax_diff_marker[idx] = F::ONE;
            }
        }
        core_row.a_bits = a_bits.map(F::from_u8);
        core_row.b_bits = b_bits.map(F::from_u8);
        core_row.c_bits = c_bits.map(F::from_u8);
        core_row.a = a.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
        core_row.c = c.map(F::from_u16);
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipImmCoreCols<T> {
    pub a: [T; BITMANIP_NUM_LIMBS],
    pub b: [T; BITMANIP_NUM_LIMBS],
    pub a_bits: [T; BITMANIP_NUM_BITS],
    pub b_bits: [T; BITMANIP_NUM_BITS],
    pub opcode_flags: [T; IMM_OP_COUNT],
    pub index_marker: [T; BITMANIP_NUM_BITS],
    pub count_marker: [T; BITMANIP_NUM_BITS + 1],
    pub byte_nonzero: [T; 8],
    pub byte_nonzero_inv: [T; 8],
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipImmCoreCols<u8>)]
pub struct BitManipImmCoreAir;

impl<F: Field> BaseAir<F> for BitManipImmCoreAir {
    fn width(&self) -> usize {
        BitManipImmCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipImmCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipImmCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; BITMANIP_NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BitManipImmCoreCols<_> = local_core.borrow();
        let mut is_valid = AB::Expr::ZERO;
        let mut expected_local_opcode = AB::Expr::ZERO;
        for (flag, local_opcode) in cols.opcode_flags.iter().zip(IMM_OPS) {
            builder.assert_bool(*flag);
            is_valid += (*flag).into();
            expected_local_opcode += (*flag).into() * AB::Expr::from_usize(local_opcode);
        }
        builder.assert_bool(is_valid.clone());

        constrain_bits_and_limbs(builder, is_valid.clone(), &cols.a_bits, &cols.a);
        constrain_bits_and_limbs(builder, is_valid.clone(), &cols.b_bits, &cols.b);

        let flag = |op| cols.opcode_flags[flag_pos(&IMM_OPS, op)];
        let index_valid = [RORI, RORIW, BCLRI, BSETI, BINVI, BEXTI]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let count_valid = [CLZ, CTZ, CLZW, CTZW]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let cpop_valid = flag(CPOP) + flag(CPOPW);
        let byte_valid = [SEXT_B, SEXT_H, ZEXT_H, ORC_B, REV8]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));
        let single_valid = [BCLRI, BSETI, BINVI, BEXTI]
            .into_iter()
            .fold(AB::Expr::ZERO, |acc, op| acc + flag(op));

        let immediate = self.eval_index_markers(builder, cols, index_valid.clone());
        self.eval_rori(builder, cols, flag(RORI), flag(RORIW));
        self.eval_count_zeros(builder, cols, count_valid);
        self.eval_cpop(builder, cols, cpop_valid);
        self.eval_byte_unary(builder, cols, byte_valid);
        self.eval_single_bit_imm(builder, cols, single_valid);

        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        BITMANIP_OFFSET
    }
}

impl BitManipImmCoreAir {
    fn eval_index_markers<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        index_valid: AB::Expr,
    ) -> AB::Expr {
        let mut marker_sum = AB::Expr::ZERO;
        let mut index = AB::Expr::ZERO;
        for (i, marker) in cols.index_marker.iter().enumerate() {
            builder.assert_bool(*marker);
            marker_sum += (*marker).into();
            index += (*marker).into() * AB::Expr::from_usize(i);
        }
        builder.assert_eq(marker_sum, index_valid);
        index
    }

    fn eval_rori<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        rori_flag: AB::Var,
        roriw_flag: AB::Var,
    ) {
        let rotate_valid: AB::Expr = rori_flag.into() + roriw_flag.into();
        for i in 0..BITMANIP_NUM_BITS {
            let mut expected64 = AB::Expr::ZERO;
            let mut expectedw = AB::Expr::ZERO;
            for s in 0..BITMANIP_NUM_BITS {
                expected64 += rori_flag * cols.index_marker[s] * cols.b_bits[(i + s) % 64];
                let w_bit = if i < 32 { i } else { 31 };
                expectedw += roriw_flag * cols.index_marker[s] * cols.b_bits[(w_bit + s) % 32];
            }
            builder.assert_zero(rotate_valid.clone() * cols.a_bits[i] - expected64 - expectedw);
        }
    }

    fn eval_count_zeros<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        count_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&IMM_OPS, op)];
        let clz64 = flag(CLZ);
        let ctz64 = flag(CTZ);
        let clz32 = flag(CLZW);
        let ctz32 = flag(CTZW);
        let mut marker_sum = AB::Expr::ZERO;
        let mut count_value = AB::Expr::ZERO;
        for (k, marker) in cols.count_marker.iter().enumerate() {
            builder.assert_bool(*marker);
            let marker_expr: AB::Expr = (*marker).into();
            marker_sum += marker_expr.clone();
            count_value += marker_expr.clone() * AB::Expr::from_usize(k);
            if k > 32 {
                builder.assert_zero((clz32 + ctz32) * marker_expr.clone());
            }
            constrain_clz_marker(builder, &cols.b_bits, marker_expr.clone() * clz64, k, 64);
            constrain_ctz_marker(builder, &cols.b_bits, marker_expr.clone() * ctz64, k, 64);
            if k <= 32 {
                constrain_clz_marker(builder, &cols.b_bits, marker_expr.clone() * clz32, k, 32);
                constrain_ctz_marker(builder, &cols.b_bits, marker_expr * ctz32, k, 32);
            }
        }
        builder.assert_eq(marker_sum, count_valid.clone());
        builder.assert_zero(count_valid.clone() * (cols.a[0] - count_value));
        for limb in cols.a.iter().skip(1) {
            builder.assert_zero(count_valid.clone() * *limb);
        }
    }

    fn eval_cpop<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        cpop_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&IMM_OPS, op)];
        let sum64 = cols
            .b_bits
            .iter()
            .fold(AB::Expr::ZERO, |acc, bit| acc + *bit);
        let sum32 = cols.b_bits[..32]
            .iter()
            .fold(AB::Expr::ZERO, |acc, bit| acc + *bit);
        let value = flag(CPOP) * sum64 + flag(CPOPW) * sum32;
        builder.assert_zero(cpop_valid.clone() * (cols.a[0] - value));
        for limb in cols.a.iter().skip(1) {
            builder.assert_zero(cpop_valid.clone() * *limb);
        }
    }

    fn eval_byte_unary<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        byte_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&IMM_OPS, op)];
        for byte in 0..8 {
            builder.assert_bool(cols.byte_nonzero[byte]);
            let sum = (0..8).fold(AB::Expr::ZERO, |acc, bit| acc + cols.b_bits[byte * 8 + bit]);
            builder.assert_zero(
                flag(ORC_B) * (sum.clone() * cols.byte_nonzero_inv[byte] - cols.byte_nonzero[byte]),
            );
            builder.assert_zero(
                flag(ORC_B) * sum * (AB::Expr::ONE - AB::Expr::from(cols.byte_nonzero[byte])),
            );
        }
        for i in 0..BITMANIP_NUM_BITS {
            let expected = flag(SEXT_B)
                * if i < 8 {
                    cols.b_bits[i].into()
                } else {
                    cols.b_bits[7].into()
                }
                + flag(SEXT_H)
                    * if i < 16 {
                        cols.b_bits[i].into()
                    } else {
                        cols.b_bits[15].into()
                    }
                + flag(ZEXT_H)
                    * if i < 16 {
                        cols.b_bits[i].into()
                    } else {
                        AB::Expr::ZERO
                    }
                + flag(ORC_B) * cols.byte_nonzero[i / 8]
                + flag(REV8) * cols.b_bits[(7 - (i / 8)) * 8 + (i % 8)];
            builder.assert_zero(byte_valid.clone() * cols.a_bits[i] - expected);
        }
    }

    fn eval_single_bit_imm<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: &BitManipImmCoreCols<AB::Var>,
        single_valid: AB::Expr,
    ) {
        let flag = |op| cols.opcode_flags[flag_pos(&IMM_OPS, op)];
        let selected = (0..BITMANIP_NUM_BITS).fold(AB::Expr::ZERO, |acc, i| {
            acc + cols.index_marker[i] * cols.b_bits[i]
        });
        for i in 0..BITMANIP_NUM_BITS {
            let marker: AB::Expr = cols.index_marker[i].into();
            let b: AB::Expr = cols.b_bits[i].into();
            let expected = flag(BCLRI) * b.clone() * (AB::Expr::ONE - marker.clone())
                + flag(BSETI) * (b.clone() + marker.clone() * (AB::Expr::ONE - b.clone()))
                + flag(BINVI) * (b.clone() + marker * (AB::Expr::ONE - AB::Expr::from_u8(2) * b))
                + flag(BEXTI)
                    * if i == 0 {
                        selected.clone()
                    } else {
                        AB::Expr::ZERO
                    };
            builder.assert_zero(single_valid.clone() * cols.a_bits[i] - expected);
        }
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BitManipImmCoreRecord {
    pub b: [u16; BITMANIP_NUM_LIMBS],
    pub imm: u8,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipImmExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipImmFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipImmExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
            WriteData: From<[[u16; BITMANIP_NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BitManipImmCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BImm({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_imm_opcode(local_opcode));
        let imm = instruction.c.as_canonical_u32();

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();
        core_record.imm = imm as u8;
        core_record.local_opcode = local_opcode as u8;

        let output = run_bitmanip_imm(local_opcode, limbs_to_u64(&core_record.b), imm);
        self.adapter.write(
            state.memory,
            instruction,
            [u64_to_limbs(output)].into(),
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for BitManipImmFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipImmCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let imm = record.imm as u32;
        let local_opcode = record.local_opcode as usize;
        let b_u64 = limbs_to_u64(&b);
        let a_u64 = run_bitmanip_imm(local_opcode, b_u64, imm);
        let a = u64_to_limbs(a_u64);
        let b_bits = bits_u64(b_u64);
        let a_bits = bits_u64(a_u64);

        let core_row: &mut BitManipImmCoreCols<F> = core_row.borrow_mut();
        core_row.opcode_flags = [F::ZERO; IMM_OP_COUNT];
        core_row.opcode_flags[flag_pos(&IMM_OPS, local_opcode)] = F::ONE;
        core_row.index_marker = [F::ZERO; BITMANIP_NUM_BITS];
        if matches!(local_opcode, RORI | RORIW | BCLRI | BSETI | BINVI | BEXTI) {
            core_row.index_marker[imm as usize] = F::ONE;
        }
        core_row.count_marker = [F::ZERO; BITMANIP_NUM_BITS + 1];
        if matches!(local_opcode, CLZ | CTZ | CLZW | CTZW) {
            core_row.count_marker[a_u64 as usize] = F::ONE;
        }
        core_row.byte_nonzero = [F::ZERO; 8];
        core_row.byte_nonzero_inv = [F::ZERO; 8];
        if local_opcode == ORC_B {
            for byte in 0..8 {
                let count = b_bits[byte * 8..byte * 8 + 8]
                    .iter()
                    .map(|bit| *bit as u32)
                    .sum::<u32>();
                if count != 0 {
                    core_row.byte_nonzero[byte] = F::ONE;
                    core_row.byte_nonzero_inv[byte] = F::from_u32(count).inverse();
                }
            }
        }
        core_row.a_bits = a_bits.map(F::from_u8);
        core_row.b_bits = b_bits.map(F::from_u8);
        core_row.a = a.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
    }
}

fn constrain_bits_and_limbs<AB: InteractionBuilder>(
    builder: &mut AB,
    is_valid: AB::Expr,
    bits: &[AB::Var; BITMANIP_NUM_BITS],
    limbs: &[AB::Var; BITMANIP_NUM_LIMBS],
) {
    for bit in bits {
        let bit_expr: AB::Expr = (*bit).into();
        builder.assert_zero(is_valid.clone() * bit_expr.clone() * (AB::Expr::ONE - bit_expr));
    }
    for limb in 0..BITMANIP_NUM_LIMBS {
        let value = (0..16).fold(AB::Expr::ZERO, |acc, bit| {
            acc + bits[16 * limb + bit] * AB::Expr::from_usize(1 << bit)
        });
        builder.assert_zero(is_valid.clone() * (limbs[limb] - value));
    }
}

fn shadd_shift_uw(local_opcode: usize) -> (usize, bool) {
    match local_opcode {
        SH1ADD => (1, false),
        SH2ADD => (2, false),
        SH3ADD => (3, false),
        ADD_UW => (0, true),
        SH1ADD_UW => (1, true),
        SH2ADD_UW => (2, true),
        SH3ADD_UW => (3, true),
        _ => unreachable!("unsupported shadd opcode {local_opcode}"),
    }
}

fn fill_shadd_aux<F: PrimeField32>(
    row: &mut BitManipShAddCoreCols<F>,
    local_opcode: usize,
    rs1: [u16; BITMANIP_NUM_LIMBS],
    rs2: [u16; BITMANIP_NUM_LIMBS],
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let (shift, uw) = match local_opcode {
        SH1ADD => (1, false),
        SH2ADD => (2, false),
        SH3ADD => (3, false),
        ADD_UW => (0, true),
        SH1ADD_UW => (1, true),
        SH2ADD_UW => (2, true),
        SH3ADD_UW => (3, true),
        _ => unreachable!(),
    };
    row.bit_shift_carry = [F::ZERO; BITMANIP_NUM_LIMBS];
    row.bit_shift_aux = [F::ZERO; BITMANIP_NUM_LIMBS];
    let carry_mask = (1u32 << shift) - 1;
    let aux_mask = (1u32 << (BITMANIP_LIMB_BITS - shift)) - 1;
    for (limb, &rs1_limb) in rs1.iter().enumerate() {
        let source = if uw && limb >= 2 { 0 } else { rs1_limb as u32 };
        let aux = source & aux_mask;
        let carry = (source >> (BITMANIP_LIMB_BITS - shift)) & carry_mask;
        row.bit_shift_aux[limb] = F::from_u32(aux);
        row.bit_shift_carry[limb] = F::from_u32(carry);
        range_checker.add_count(carry, shift);
        range_checker.add_count(aux, BITMANIP_LIMB_BITS - shift);
    }

    let shifted = if uw {
        (limbs_to_u64(&rs1) as u32 as u64) << shift
    } else {
        limbs_to_u64(&rs1).wrapping_shl(shift as u32)
    };
    let rs2 = limbs_to_u64(&rs2);
    let mut add_carry = 0u8;
    row.add_carry[0] = F::ZERO;
    for i in 0..BITMANIP_NUM_BITS {
        if i % BITMANIP_LIMB_BITS == 0 {
            row.add_carry[i / BITMANIP_LIMB_BITS] = F::from_u8(add_carry);
        }
        let total = ((shifted >> i) & 1) as u8 + ((rs2 >> i) & 1) as u8 + add_carry;
        add_carry = total >> 1;
    }
    row.add_carry[BITMANIP_NUM_LIMBS] = F::from_u8(add_carry);
}

fn fill_slli_uw_aux<F: PrimeField32>(
    row: &mut BitManipSlliUwCoreCols<F>,
    rs1: [u16; BITMANIP_NUM_LIMBS],
    imm: usize,
    range_checker: &SharedVariableRangeCheckerChip,
) {
    let bit_shift = imm % BITMANIP_LIMB_BITS;
    let limb_shift = imm / BITMANIP_LIMB_BITS;
    row.bit_shift_marker = [F::ZERO; BITMANIP_LIMB_BITS];
    row.bit_shift_marker[bit_shift] = F::ONE;
    row.limb_shift_marker = [F::ZERO; BITMANIP_NUM_LIMBS];
    row.limb_shift_marker[limb_shift] = F::ONE;
    row.bit_shift_carry = [F::ZERO; BITMANIP_NUM_LIMBS];
    row.bit_shift_aux = [F::ZERO; BITMANIP_NUM_LIMBS];

    let carry_mask = (1u32 << bit_shift) - 1;
    let aux_mask = (1u32 << (BITMANIP_LIMB_BITS - bit_shift)) - 1;
    for (limb, &rs1_limb) in rs1.iter().enumerate() {
        let source = if limb >= 2 { 0 } else { rs1_limb as u32 };
        let aux = source & aux_mask;
        let carry = (source >> (BITMANIP_LIMB_BITS - bit_shift)) & carry_mask;
        row.bit_shift_aux[limb] = F::from_u32(aux);
        row.bit_shift_carry[limb] = F::from_u32(carry);
        range_checker.add_count(carry, bit_shift);
        range_checker.add_count(aux, BITMANIP_LIMB_BITS - bit_shift);
    }
}

fn constrain_clz_marker<AB: InteractionBuilder>(
    builder: &mut AB,
    bits: &[AB::Var; BITMANIP_NUM_BITS],
    active_marker: AB::Expr,
    count: usize,
    width: usize,
) {
    if count == width {
        for bit in bits.iter().take(width) {
            builder.assert_zero(active_marker.clone() * *bit);
        }
    } else if count < width {
        for bit in bits.iter().take(width).skip(width - count) {
            builder.assert_zero(active_marker.clone() * *bit);
        }
        builder.assert_zero(active_marker * (bits[width - count - 1] - AB::Expr::ONE));
    }
}

fn constrain_ctz_marker<AB: InteractionBuilder>(
    builder: &mut AB,
    bits: &[AB::Var; BITMANIP_NUM_BITS],
    active_marker: AB::Expr,
    count: usize,
    width: usize,
) {
    if count == width {
        for bit in bits.iter().take(width) {
            builder.assert_zero(active_marker.clone() * *bit);
        }
    } else if count < width {
        for bit in bits.iter().take(count) {
            builder.assert_zero(active_marker.clone() * *bit);
        }
        builder.assert_zero(active_marker * (bits[count] - AB::Expr::ONE));
    }
}
