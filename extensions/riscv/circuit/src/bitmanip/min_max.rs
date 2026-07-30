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
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use super::{
    core::{
        is_min_max_opcode, limbs_to_u64, run_bitmanip_reg, u64_to_limbs, BITMANIP_OFFSET, MAX,
        MAXU, MIN, MINU,
    },
    BITMANIP_LIMB_BITS, BITMANIP_NUM_LIMBS,
};
use crate::less_than::run_less_than;

/// MIN/MINU/MAX/MAXU: the `less_than` comparison gadget plus a `pick_b`
/// selector; the written value is an expression over the operand limbs, so no
/// output columns are needed.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct BitManipMinMaxCoreCols<T> {
    pub b: [T; BITMANIP_NUM_LIMBS],
    pub c: [T; BITMANIP_NUM_LIMBS],
    /// b < c (signed for MIN/MAX, unsigned for MINU/MAXU).
    pub cmp_result: T,
    /// 1 exactly when the written value is `b`.
    pub pick_b: T,
    pub opcode_min_flag: T,
    pub opcode_minu_flag: T,
    pub opcode_max_flag: T,
    pub opcode_maxu_flag: T,
    /// Most significant limb of b and c as a field element; range checked to
    /// [-2^(LIMB_BITS-1), 2^(LIMB_BITS-1)) if signed, [0, 2^LIMB_BITS) if not.
    pub b_msb_f: T,
    pub c_msb_f: T,
    /// 1 at the most significant index where b and c differ, otherwise 0.
    pub diff_marker: [T; BITMANIP_NUM_LIMBS],
    pub diff_val: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BitManipMinMaxCoreCols<u8>)]
pub struct BitManipMinMaxCoreAir {
    pub range_bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for BitManipMinMaxCoreAir {
    fn width(&self) -> usize {
        BitManipMinMaxCoreCols::<F>::width()
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for BitManipMinMaxCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for BitManipMinMaxCoreAir
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
        let cols: &BitManipMinMaxCoreCols<_> = local_core.borrow();
        let flags = [
            (cols.opcode_min_flag, MIN),
            (cols.opcode_minu_flag, MINU),
            (cols.opcode_max_flag, MAX),
            (cols.opcode_maxu_flag, MAXU),
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &(flag, _)| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.cmp_result);

        // The comparison machinery below mirrors `LessThanCoreAir`: find the
        // most significant differing limb via a one-hot marker with prefix sum
        // and derive `cmp_result = (b < c)`, with the top limbs replaced by
        // sign-adjusted field values when the comparison is signed.
        let b = &cols.b;
        let c = &cols.c;
        let marker = &cols.diff_marker;
        let mut prefix_sum = AB::Expr::ZERO;

        let b_diff = b[BITMANIP_NUM_LIMBS - 1] - cols.b_msb_f;
        let c_diff = c[BITMANIP_NUM_LIMBS - 1] - cols.c_msb_f;
        builder
            .assert_zero(b_diff.clone() * (AB::Expr::from_u32(1 << BITMANIP_LIMB_BITS) - b_diff));
        builder
            .assert_zero(c_diff.clone() * (AB::Expr::from_u32(1 << BITMANIP_LIMB_BITS) - c_diff));

        for i in (0..BITMANIP_NUM_LIMBS).rev() {
            let diff = (if i == BITMANIP_NUM_LIMBS - 1 {
                cols.c_msb_f - cols.b_msb_f
            } else {
                c[i] - b[i]
            }) * (AB::Expr::from_u8(2) * cols.cmp_result - AB::Expr::ONE);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero((AB::Expr::ONE - prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.diff_val, diff);
        }

        builder.assert_bool(prefix_sum.clone());
        builder.assert_zero((AB::Expr::ONE - prefix_sum.clone()) * cols.cmp_result);

        let signed_flag = cols.opcode_min_flag + cols.opcode_max_flag;
        let sign_shift = AB::Expr::from_u32(1 << (BITMANIP_LIMB_BITS - 1)) * signed_flag;
        self.range_bus
            .range_check(cols.b_msb_f + sign_shift.clone(), BITMANIP_LIMB_BITS)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(cols.c_msb_f + sign_shift, BITMANIP_LIMB_BITS)
            .eval(builder, is_valid.clone());

        // Ensures diff_val is non-zero at the marked limb.
        self.range_bus
            .range_check(cols.diff_val - AB::Expr::ONE, BITMANIP_LIMB_BITS)
            .eval(builder, prefix_sum);

        // pick_b selects the written operand: the smaller one for MIN/MINU,
        // the larger one for MAX/MAXU. Zero on padding rows.
        let min_flags = cols.opcode_min_flag + cols.opcode_minu_flag;
        let max_flags = cols.opcode_max_flag + cols.opcode_maxu_flag;
        builder.assert_eq(
            cols.pick_b,
            min_flags * cols.cmp_result + max_flags * (AB::Expr::ONE - cols.cmp_result),
        );

        let writes: [AB::Expr; BITMANIP_NUM_LIMBS] =
            array::from_fn(|i| cols.pick_b * cols.b[i] + (AB::Expr::ONE - cols.pick_b) * cols.c[i]);

        let expected_local_opcode = flags
            .iter()
            .fold(AB::Expr::ZERO, |acc, &(flag, local_opcode)| {
                acc + flag * AB::Expr::from_usize(local_opcode)
            });
        let expected_opcode = expected_local_opcode + AB::Expr::from_usize(BITMANIP_OFFSET);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [writes].into(),
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
pub struct BitManipMinMaxCoreRecord {
    pub b: [u16; BITMANIP_NUM_LIMBS],
    pub c: [u16; BITMANIP_NUM_LIMBS],
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BitManipMinMaxExecutor<A> {
    adapter: A,
}

#[derive(Clone, derive_new::new)]
pub struct BitManipMinMaxFiller<A> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

impl<F, A, RA> PreflightExecutor<F, RA> for BitManipMinMaxExecutor<A>
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
        (A::RecordMut<'buf>, &'buf mut BitManipMinMaxCoreRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("Rv64BMinMax({})", opcode - BITMANIP_OFFSET)
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let local_opcode = instruction.opcode.local_opcode_idx(BITMANIP_OFFSET);
        debug_assert!(is_min_max_opcode(local_opcode));

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

impl<F, A> TraceFiller<F> for BitManipMinMaxFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let record: &BitManipMinMaxCoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let b = record.b;
        let c = record.c;
        let local_opcode = record.local_opcode as usize;

        let core_row: &mut BitManipMinMaxCoreCols<F> = core_row.borrow_mut();

        let is_signed = matches!(local_opcode, MIN | MAX);
        let is_min = matches!(local_opcode, MIN | MINU);
        let (cmp_result, diff_idx, b_sign, c_sign) =
            run_less_than::<BITMANIP_NUM_LIMBS, BITMANIP_LIMB_BITS>(is_signed, &b, &c);

        // Range check (msb_f + 2^(LIMB_BITS-1)) if signed, msb_f if not.
        let (b_msb_f, b_msb_range) = if b_sign {
            (
                -F::from_u16(b[BITMANIP_NUM_LIMBS - 1].wrapping_neg()),
                b[BITMANIP_NUM_LIMBS - 1] as u32 - (1u32 << (BITMANIP_LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_u16(b[BITMANIP_NUM_LIMBS - 1]),
                b[BITMANIP_NUM_LIMBS - 1] as u32 + ((is_signed as u32) << (BITMANIP_LIMB_BITS - 1)),
            )
        };
        let (c_msb_f, c_msb_range) = if c_sign {
            (
                -F::from_u16(c[BITMANIP_NUM_LIMBS - 1].wrapping_neg()),
                c[BITMANIP_NUM_LIMBS - 1] as u32 - (1u32 << (BITMANIP_LIMB_BITS - 1)),
            )
        } else {
            (
                F::from_u16(c[BITMANIP_NUM_LIMBS - 1]),
                c[BITMANIP_NUM_LIMBS - 1] as u32 + ((is_signed as u32) << (BITMANIP_LIMB_BITS - 1)),
            )
        };

        core_row.diff_val = if diff_idx == BITMANIP_NUM_LIMBS {
            F::ZERO
        } else if diff_idx == BITMANIP_NUM_LIMBS - 1 {
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

        self.range_checker_chip
            .add_count(b_msb_range, BITMANIP_LIMB_BITS);
        self.range_checker_chip
            .add_count(c_msb_range, BITMANIP_LIMB_BITS);

        core_row.diff_marker = [F::ZERO; BITMANIP_NUM_LIMBS];
        if diff_idx != BITMANIP_NUM_LIMBS {
            self.range_checker_chip
                .add_count(core_row.diff_val.as_canonical_u32() - 1, BITMANIP_LIMB_BITS);
            core_row.diff_marker[diff_idx] = F::ONE;
        }

        core_row.c_msb_f = c_msb_f;
        core_row.b_msb_f = b_msb_f;
        core_row.opcode_maxu_flag = F::from_bool(local_opcode == MAXU);
        core_row.opcode_max_flag = F::from_bool(local_opcode == MAX);
        core_row.opcode_minu_flag = F::from_bool(local_opcode == MINU);
        core_row.opcode_min_flag = F::from_bool(local_opcode == MIN);
        core_row.pick_b = F::from_bool(if is_min { cmp_result } else { !cmp_result });
        core_row.cmp_result = F::from_bool(cmp_result);
        core_row.c = c.map(F::from_u16);
        core_row.b = b.map(F::from_u16);
    }
}
