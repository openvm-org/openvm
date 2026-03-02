use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    AlignedBytesBorrow,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::BaseAluWOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::RV64_WORD_NUM_LIMBS;

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct BaseAluCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_addw_flag: T,
    pub opcode_subw_flag: T,

    // Sign bit of the low 32-bit result. Upper bytes are constrained from this single bit.
    pub word_sign: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct BaseAluCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BaseAluCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BaseAluCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_addw_flag, cols.opcode_subw_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // ADDW/SUBW are 32-bit operations. We only constrain the low word limbs with carries.
        let mut carry_add: [AB::Expr; RV64_WORD_NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let mut carry_sub: [AB::Expr; RV64_WORD_NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..RV64_WORD_NUM_LIMBS {
            carry_add[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_addw_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (a[i] + c[i] - b[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_subw_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // Range-check low word bytes.
        for &a_limb in &a[..RV64_WORD_NUM_LIMBS] {
            self.bus
                .send_xor(a_limb, a_limb, AB::Expr::ZERO)
                .eval(builder, is_valid.clone());
        }

        // Constrain upper 32 bits to sign extension of low 32-bit result.
        builder.assert_bool(cols.word_sign);
        let sign_mask = AB::Expr::from_canonical_u32(1 << (LIMB_BITS - 1));
        self.bus
            .send_xor(
                a[RV64_WORD_NUM_LIMBS - 1],
                sign_mask.clone(),
                a[RV64_WORD_NUM_LIMBS - 1] + sign_mask.clone()
                    - AB::Expr::from_canonical_u32(2) * cols.word_sign * sign_mask,
            )
            .eval(builder, is_valid.clone());
        let sign_extend_limb = AB::Expr::from_canonical_u32((1 << LIMB_BITS) - 1) * cols.word_sign;
        for &a_limb in &a[RV64_WORD_NUM_LIMBS..] {
            builder
                .when(is_valid.clone())
                .assert_eq(a_limb, sign_extend_limb.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(BaseAluWOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
        );

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
        self.offset
    }
}

#[repr(C, align(4))]
#[derive(AlignedBytesBorrow, Debug)]
pub struct BaseAluCoreRecord<const NUM_LIMBS: usize> {
    pub b: [u8; NUM_LIMBS],
    pub c: [u8; NUM_LIMBS],
    // Use u8 instead of usize for better packing
    pub local_opcode: u8,
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BaseAluExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct BaseAluFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
    pub offset: usize,
}

impl<F, A, RA, const NUM_LIMBS: usize, const LIMB_BITS: usize> PreflightExecutor<F, RA>
    for BaseAluExecutor<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u8; NUM_LIMBS]; 2]>,
            WriteData: From<[[u8; NUM_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut BaseAluCoreRecord<NUM_LIMBS>),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluWOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        let local_opcode = BaseAluWOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &core_record.b, &core_record.c);

        core_record.local_opcode = local_opcode as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> TraceFiller<F>
    for BaseAluFiller<A, NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // BaseAluCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        // SAFETY: core_row contains a valid BaseAluCoreRecord written by the executor
        // during trace generation
        let record: &BaseAluCoreRecord<NUM_LIMBS> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        // SAFETY: the following is highly unsafe. We are going to cast `core_row` to a record
        // buffer, and then do an _overlapping_ write to the `core_row` as a row of field elements.
        // This requires:
        // - Cols and Record structs should be repr(C) and we write in reverse order (to ensure
        //   non-overlapping)
        // - Do not overwrite any reference in `record` before it has already been used or moved
        // - alignment of `F` must be >= alignment of Record (AlignedBytesBorrow will panic
        //   otherwise)

        let local_opcode = BaseAluWOpcode::from_usize(record.local_opcode as usize);
        let a = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &record.b, &record.c);
        core_row.opcode_addw_flag = F::from_bool(local_opcode == BaseAluWOpcode::ADDW);
        core_row.opcode_subw_flag = F::from_bool(local_opcode == BaseAluWOpcode::SUBW);
        core_row.word_sign =
            F::from_canonical_u8(a[RV64_WORD_NUM_LIMBS - 1] >> (LIMB_BITS as u8 - 1));

        for &a_limb in &a[..RV64_WORD_NUM_LIMBS] {
            self.bitwise_lookup_chip
                .request_xor(a_limb as u32, a_limb as u32);
        }
        self.bitwise_lookup_chip.request_xor(
            a[RV64_WORD_NUM_LIMBS - 1] as u32,
            (1u32 << (LIMB_BITS - 1)) as u32,
        );

        core_row.c = record.c.map(F::from_canonical_u8);
        core_row.b = record.b.map(F::from_canonical_u8);
        core_row.a = a.map(F::from_canonical_u8);
    }
}

#[inline(always)]
pub(super) fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluWOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 8, "specialize for bytes");
    debug_assert!(NUM_LIMBS >= RV64_WORD_NUM_LIMBS);
    match opcode {
        BaseAluWOpcode::ADDW => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluWOpcode::SUBW => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
    }
}

#[inline(always)]
fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; RV64_WORD_NUM_LIMBS];
    let mut carry = [0u8; RV64_WORD_NUM_LIMBS];
    for i in 0..RV64_WORD_NUM_LIMBS {
        let mut overflow =
            (x[i] as u16) + (y[i] as u16) + if i > 0 { carry[i - 1] as u16 } else { 0 };
        carry[i] = (overflow >> LIMB_BITS) as u8;
        overflow &= (1u16 << LIMB_BITS) - 1;
        z[i] = overflow as u8;
    }
    sign_extend_word::<NUM_LIMBS, LIMB_BITS>(&z)
}

#[inline(always)]
fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; RV64_WORD_NUM_LIMBS];
    let mut carry = [0u8; RV64_WORD_NUM_LIMBS];
    for i in 0..RV64_WORD_NUM_LIMBS {
        let rhs = y[i] as u16 + if i > 0 { carry[i - 1] as u16 } else { 0 };
        if x[i] as u16 >= rhs {
            z[i] = x[i] - rhs as u8;
            carry[i] = 0;
        } else {
            z[i] = (x[i] as u16 + (1u16 << LIMB_BITS) - rhs) as u8;
            carry[i] = 1;
        }
    }
    sign_extend_word::<NUM_LIMBS, LIMB_BITS>(&z)
}

#[inline(always)]
fn sign_extend_word<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    low_word: &[u8; RV64_WORD_NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(NUM_LIMBS >= RV64_WORD_NUM_LIMBS);
    let sign_extend_limb = ((1u16 << LIMB_BITS) - 1) as u8
        * (low_word[RV64_WORD_NUM_LIMBS - 1] >> (LIMB_BITS as u8 - 1));
    let mut out = [sign_extend_limb; NUM_LIMBS];
    out[..RV64_WORD_NUM_LIMBS].copy_from_slice(low_word);
    out
}
