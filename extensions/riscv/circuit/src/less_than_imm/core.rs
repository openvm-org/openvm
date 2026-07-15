use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    utils::not,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBytesBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{LessThanImmOpcode, LessThanOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::less_than::run_less_than;

/// Core columns for comparisons with a signed 12-bit immediate.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct LessThanImmCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [T; NUM_LIMBS],
    /// Bits `[10:0]` of the 12-bit signed immediate.
    pub imm_low11: T,
    /// Sign bit (bit 11) of the immediate.
    pub imm_sign: T,
    pub cmp_result: T,

    pub opcode_slt_flag: T,
    pub opcode_sltu_flag: T,

    pub b_msb_f: T,
    pub c_msb_f: T,

    pub diff_marker: [T; NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(LessThanImmCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct LessThanImmCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub range_bus: VariableRangeCheckerBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        LessThanImmCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for LessThanImmCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<ImmInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LessThanImmCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_slt_flag, cols.opcode_sltu_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);
        builder.assert_bool(cols.imm_sign);

        // Range check the low 11 bits of the immediate so the (imm_low11, imm_sign)
        // decomposition of the 24-bit operand is unique.
        self.range_bus
            .range_check(cols.imm_low11, 11)
            .eval(builder, is_valid.clone());

        // Sign-extended u16 limbs of the immediate, as expressions.
        let sign_u16 = cols.imm_sign * AB::Expr::from_u32(u16::MAX as u32);
        let c: [AB::Expr; NUM_LIMBS] = array::from_fn(|i| {
            if i == 0 {
                cols.imm_low11 + cols.imm_sign * AB::Expr::from_u32(0xF800)
            } else {
                sign_u16.clone()
            }
        });

        let b = &cols.b;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        let b_diff = b[NUM_LIMBS - 1] - cols.b_msb_f;
        builder.assert_zero(b_diff.clone() * (AB::Expr::from_u32(1 << LIMB_BITS) - b_diff));
        builder.assert_eq(
            cols.c_msb_f,
            cols.imm_sign
                * (AB::Expr::from_u32((1 << LIMB_BITS) - 1)
                    - cols.opcode_slt_flag * AB::Expr::from_u32(1 << LIMB_BITS)),
        );

        for i in (0..NUM_LIMBS).rev() {
            let diff = (if i == NUM_LIMBS - 1 {
                cols.c_msb_f - cols.b_msb_f
            } else {
                c[i].clone() - b[i]
            }) * (AB::Expr::from_u8(2) * cols.cmp_result - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }

        builder.assert_bool(prefix_sum.clone());
        builder
            .when(not::<AB::Expr>(prefix_sum.clone()))
            .assert_zero(cols.cmp_result);

        let sign_shift = AB::Expr::from_u32(1 << (LIMB_BITS - 1)) * cols.opcode_slt_flag;
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift.clone(), LIMB_BITS)
            .eval(builder, is_valid.clone());

        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, LIMB_BITS)
            .eval(builder, prefix_sum);

        let expected_opcode = flags
            .iter()
            .zip(LessThanImmOpcode::iter())
            .fold(AB::Expr::ZERO, |acc, (flag, opcode)| {
                acc + (*flag).into() * AB::Expr::from_u8(opcode as u8)
            })
            + AB::Expr::from_usize(self.offset);
        let mut a: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        a[0] = cols.cmp_result.into();

        // 24-bit encoding matching i12_to_u24 in the transpiler.
        let imm = cols.imm_low11 + cols.imm_sign * AB::Expr::from_u32(0xFFF800);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into)].into(),
            writes: [a].into(),
            instruction: ImmInstruction {
                is_valid,
                opcode: expected_opcode,
                immediate: imm,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LessThanImmCoreRecord<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub b: [u16; NUM_LIMBS],
    pub imm_low11: u16,
    pub imm_sign: u16,
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LessThanImmExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone, derive_new::new)]
pub struct LessThanImmFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

pub(crate) fn imm_to_u16_limbs<const NUM_LIMBS: usize>(
    imm_low11: u16,
    imm_sign: u16,
) -> [u16; NUM_LIMBS] {
    let mut c = [imm_sign * 0xFFFF; NUM_LIMBS];
    c[0] = imm_low11 + imm_sign * 0xF800;
    c
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for LessThanImmExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; NUM_LIMBS]; 1]>,
            WriteData: From<[[u16; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut LessThanImmCoreRecord<NUM_LIMBS, LIMB_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", LessThanImmOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, c, .. } = instruction;

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let c_u32 = c.as_canonical_u32();
        core_record.imm_low11 = (c_u32 & 0x7FF) as u16;
        core_record.imm_sign = ((c_u32 >> 11) & 1) as u16;
        core_record.local_opcode = opcode.local_opcode_idx(self.offset) as u8;

        let c_limbs = imm_to_u16_limbs::<NUM_LIMBS>(core_record.imm_low11, core_record.imm_sign);
        let (cmp_result, _, _, _) = run_less_than::<NUM_LIMBS, LIMB_BITS>(
            core_record.local_opcode == LessThanImmOpcode::SLTI as u8,
            &core_record.b,
            &c_limbs,
        );

        let mut output = [0u16; NUM_LIMBS];
        output[0] = cmp_result as u16;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for LessThanImmFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &LessThanImmCoreRecord<NUM_LIMBS, LIMB_BITS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };

        let core_row: &mut LessThanImmCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        let is_slt = record.local_opcode == LessThanImmOpcode::SLTI as u8;
        let b = record.b;
        let imm_low11 = record.imm_low11;
        let imm_sign = record.imm_sign;
        let c = imm_to_u16_limbs::<NUM_LIMBS>(imm_low11, imm_sign);
        let (cmp_result, diff_idx, b_sign, c_sign) =
            run_less_than::<NUM_LIMBS, LIMB_BITS>(is_slt, &b, &c);

        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_u16(b[NUM_LIMBS - 1].wrapping_neg()),
                b[NUM_LIMBS - 1] as u32 - (1u32 << (LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_u16(b[NUM_LIMBS - 1]),
                b[NUM_LIMBS - 1] as u32 + ((is_slt as u32) << (LIMB_BITS - 1)),
            )
        };
        let c_msb_f = if c_sign && is_slt {
            -F::ONE
        } else {
            F::from_u16(c[NUM_LIMBS - 1])
        };

        let diff_val = if diff_idx == NUM_LIMBS {
            F::ZERO
        } else if diff_idx == (NUM_LIMBS - 1) {
            if cmp_result {
                c_msb_f - b_msb_f
            } else {
                b_msb_f - c_msb_f
            }
        } else if cmp_result {
            F::from_u16((c[diff_idx] as u32 - b[diff_idx] as u32) as u16)
        } else {
            F::from_u16((b[diff_idx] as u32 - c[diff_idx] as u32) as u16)
        };

        self.range_checker_chip.add_count(imm_low11 as u32, 11);
        self.range_checker_chip.add_count(b_msb_range, LIMB_BITS);

        core_row.diff_val = diff_val;
        core_row.diff_marker = [F::ZERO; NUM_LIMBS];
        if diff_idx != NUM_LIMBS {
            self.range_checker_chip
                .add_count(diff_val.as_canonical_u32() - 1, LIMB_BITS);
            core_row.diff_marker[diff_idx] = F::ONE;
        }

        core_row.c_msb_f = c_msb_f;
        core_row.b_msb_f = b_msb_f;
        core_row.opcode_sltu_flag = F::from_bool(!is_slt);
        core_row.opcode_slt_flag = F::from_bool(is_slt);
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.imm_sign = F::from_u16(imm_sign);
        core_row.imm_low11 = F::from_u16(imm_low11);
        core_row.b = b.map(F::from_u16);
    }
}

// Keep the opcode mapping in sync with LessThanOpcode used by `run_less_than`.
const _: () = assert!(
    LessThanImmOpcode::SLTI as usize == LessThanOpcode::SLT as usize
        && LessThanImmOpcode::SLTIU as usize == LessThanOpcode::SLTU as usize
);
