use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct AddSubCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
}

/// If `RANGE_CHECK_TOP_LIMB` is false, the adapter must constrain the top output limb to
/// `[0, 2^LIMB_BITS)`.
#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(AddSubCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct AddSubCoreAir<
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    pub range_bus: VariableRangeCheckerBus,
    pub offset: usize,
}

impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAir<F> for AddSubCoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
    fn width(&self) -> usize {
        AddSubCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<
        F: Field,
        const NUM_LIMBS: usize,
        const LIMB_BITS: usize,
        const RANGE_CHECK_TOP_LIMB: bool,
    > BaseAirWithPublicValues<F> for AddSubCoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize, const RANGE_CHECK_TOP_LIMB: bool>
    VmCoreAir<AB, I> for AddSubCoreAir<NUM_LIMBS, LIMB_BITS, RANGE_CHECK_TOP_LIMB>
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
        let cols: &AddSubCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [cols.opcode_add_flag, cols.opcode_sub_flag];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // For ADD, define carry[i] = (b[i] + c[i] + carry[i - 1] - a[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= a[i] < 2^LIMB_BITS, it can be proven that
        // a[i] = (b[i] + c[i]) % 2^LIMB_BITS as necessary. The same holds for SUB when
        // carry[i] is (a[i] + c[i] - b[i] + carry[i - 1]) / 2^LIMB_BITS.
        let mut carry_add: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let mut carry_sub: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            // We explicitly separate the constraints for ADD and SUB in order to keep degree
            // cubic. Because we constrain that the carry (which is arbitrary) is bool, if
            // carry has degree larger than 1 the max-degree constrain could be at least 4.
            carry_add[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_add_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (a[i] + c[i] - b[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_sub_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // Range check a to [0, 2^LIMB_BITS): the carry constraints above only prove
        // a[i] = (b[i] op c[i]) mod 2^LIMB_BITS given this bound, and `a` is written to
        // memory, which requires canonical u16 cells.
        let range_limb_count = NUM_LIMBS - usize::from(!RANGE_CHECK_TOP_LIMB);
        for &a_limb in &a[..range_limb_count] {
            self.range_bus
                .range_check(a_limb, LIMB_BITS)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            cols.opcode_add_flag * AB::Expr::from_u8(BaseAluOpcode::ADD as u8)
                + cols.opcode_sub_flag * AB::Expr::from_u8(BaseAluOpcode::SUB as u8),
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

#[derive(Clone, Copy, derive_new::new)]
pub struct AddSubExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(derive_new::new)]
pub struct AddSubFiller<
    A,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const RANGE_CHECK_TOP_LIMB: bool,
> {
    adapter: A,
    pub range_checker_chip: SharedVariableRangeCheckerChip,
}

#[inline(always)]
pub(crate) fn run_add_sub<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> [u16; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 16, "specialize for u16 limbs");
    match opcode {
        BaseAluOpcode::ADD => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::SUB => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        _ => unreachable!("AddSubExecutor received non-ADD/SUB opcode"),
    }
}

#[inline(always)]
fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> [u16; NUM_LIMBS] {
    let mut z = [0u16; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut overflow = (x[i] as u32) + (y[i] as u32) + if i > 0 { carry[i - 1] } else { 0 };
        carry[i] = overflow >> LIMB_BITS;
        overflow &= (1u32 << LIMB_BITS) - 1;
        z[i] = overflow as u16;
    }
    z
}

#[inline(always)]
fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u16; NUM_LIMBS],
    y: &[u16; NUM_LIMBS],
) -> [u16; NUM_LIMBS] {
    let mut z = [0u16; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] as u32 + if i > 0 { carry[i - 1] } else { 0 };
        if x[i] as u32 >= rhs {
            z[i] = (x[i] as u32 - rhs) as u16;
            carry[i] = 0;
        } else {
            z[i] = (x[i] as u32 + (1u32 << LIMB_BITS) - rhs) as u16;
            carry[i] = 1;
        }
    }
    z
}
