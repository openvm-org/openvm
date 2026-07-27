use std::{array, borrow::Borrow};

use openvm_circuit::arch::*;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_riscv_transpiler::MulOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiplicationCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new, ColumnsAir)]
#[columns_via(MultiplicationCoreCols<u8, NUM_LIMBS, LIMB_BITS>)]
pub struct MultiplicationCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        MultiplicationCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for MultiplicationCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &MultiplicationCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // Define carry[i] = (sum_{k=0}^{i} b[k] * c[i - k] + carry[i - 1] - a[i]) / 2^LIMB_BITS.
        // If 0 <= a[i], carry[i] < 2^LIMB_BITS, it can be proven that a[i] = sum_{k=0}^{i} (b[k] *
        // c[i - k]) % 2^LIMB_BITS as necessary.
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_u32(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            let expected_limb = if i == 0 {
                AB::Expr::ZERO
            } else {
                carry[i - 1].clone()
            } + (0..=i).fold(AB::Expr::ZERO, |acc, k| acc + (b[k] * c[i - k]));
            carry[i] = AB::Expr::from(carry_divide) * (expected_limb - a[i]);
        }

        for (a, carry) in a.iter().zip(carry.iter()) {
            self.bus
                .send(vec![(*a).into(), carry.clone()])
                .eval(builder, cols.is_valid);
        }

        // Memory bus checks only packed u16 values; these byte limbs need separate bounds.
        for i in 0..NUM_LIMBS {
            self.bitwise_lookup_bus
                .send_range(b[i], c[i])
                .eval(builder, cols.is_valid);
        }

        let expected_opcode = VmCoreAir::<AB, I>::opcode_to_global_expr(self, MulOpcode::MUL);

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
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
pub struct MultiplicationExecutor<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
}

#[derive(Clone)]
pub struct MultiplicationFiller<A, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    adapter: A,
    pub offset: usize,
    pub range_tuple_chip: SharedRangeTupleCheckerChip<2>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<A, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    MultiplicationFiller<A, NUM_LIMBS, LIMB_BITS>
{
    pub fn new(
        adapter: A,
        range_tuple_chip: SharedRangeTupleCheckerChip<2>,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        // The RangeTupleChecker is used to range check (a[i], carry[i]) pairs where 0 <= i
        // < NUM_LIMBS. a[i] must have LIMB_BITS bits and carry[i] is the sum of i + 1 bytes
        // (with LIMB_BITS bits).
        debug_assert!(
            range_tuple_chip.sizes()[0] == 1 << LIMB_BITS,
            "First element of RangeTupleChecker must have size {}",
            1 << LIMB_BITS
        );
        debug_assert!(
            range_tuple_chip.sizes()[1] >= (1 << LIMB_BITS) * NUM_LIMBS as u32,
            "Second element of RangeTupleChecker must have size of at least {}",
            (1 << LIMB_BITS) * NUM_LIMBS as u32
        );

        Self {
            adapter,
            offset,
            range_tuple_chip,
            bitwise_lookup_chip,
        }
    }
}

pub(crate) fn fill_core_row<F: PrimeField32, const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    range_tuple_chip: &SharedRangeTupleCheckerChip<2>,
    bitwise_lookup_chip: &SharedBitwiseOperationLookupChip<LIMB_BITS>,
    core_row: &mut MultiplicationCoreCols<F, NUM_LIMBS, LIMB_BITS>,
    b: [u8; NUM_LIMBS],
    c: [u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let (a, carry) = run_mul::<NUM_LIMBS, LIMB_BITS>(&b, &c);
    fill_core_row_with_result(
        range_tuple_chip,
        bitwise_lookup_chip,
        core_row,
        b,
        c,
        a,
        carry,
    );
    a
}

pub(crate) fn fill_core_row_with_result<
    F: PrimeField32,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
>(
    range_tuple_chip: &SharedRangeTupleCheckerChip<2>,
    bitwise_lookup_chip: &SharedBitwiseOperationLookupChip<LIMB_BITS>,
    core_row: &mut MultiplicationCoreCols<F, NUM_LIMBS, LIMB_BITS>,
    b: [u8; NUM_LIMBS],
    c: [u8; NUM_LIMBS],
    a: [u8; NUM_LIMBS],
    carry: [u32; NUM_LIMBS],
) {
    for (a, carry) in a.iter().zip(carry.iter()) {
        range_tuple_chip.add_count(&[*a as u32, *carry]);
    }
    for (b_val, c_val) in b.iter().zip(c.iter()) {
        bitwise_lookup_chip.request_range(*b_val as u32, *c_val as u32);
    }

    core_row.is_valid = F::ONE;
    core_row.c = c.map(F::from_u8);
    core_row.b = b.map(F::from_u8);
    core_row.a = a.map(F::from_u8);
}

// returns mul, carry
#[inline(always)]
pub(crate) fn run_mul<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> ([u8; NUM_LIMBS], [u32; NUM_LIMBS]) {
    let mut result = [0u8; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut res = 0u32;
        if i > 0 {
            res = carry[i - 1];
        }
        for j in 0..=i {
            res += (x[j] as u32) * (y[i - j] as u32);
        }
        carry[i] = res >> LIMB_BITS;
        res %= 1u32 << LIMB_BITS;
        result[i] = res as u8;
    }
    (result, carry)
}
