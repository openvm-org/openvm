use std::{
    array,
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    bigint::utils::big_uint_to_num_limbs,
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    utils::not,
    xor::{bus::XorBus, lookup::XorLookupChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use itertools::fold;
use num_bigint_dig::BigUint;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use strum::IntoEnumIterator;

use crate::{
    arch::{
        instructions::{DivRemOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DivRemCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    // b = c * q + r for some 0 <= |r| < |c| and sign(r) = sign(b).
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub q: [T; NUM_LIMBS],
    pub r: [T; NUM_LIMBS],

    // Flags to indicate special cases.
    pub zero_divisor: T,
    pub signed_overflow: T,

    // Sign of b and c respectively, while q_sign = b_sign ^ c_sign if q is non-zero
    // and is 0 otherwise. sign_xor = b_sign ^ c_sign always.
    pub b_sign: T,
    pub c_sign: T,
    pub q_sign: T,
    pub sign_xor: T,

    // Auxiliary columns to constrain that 0 <= |r| < |c|. When sign_xor == 1 we have
    // r_prime = -r, and when sign_xor == 0 we have r_prime = r. Each r_inv[i] is the
    // field inverse of r_prime - 2^LIMB_BITS, ensures each r_prime[i] is in range.
    pub r_prime: [T; NUM_LIMBS],
    pub r_inv: [T; NUM_LIMBS],
    pub lt_marker: [T; NUM_LIMBS],
    pub lt_diff: T,

    // Opcode flags
    pub opcode_div_flag: T,
    pub opcode_divu_flag: T,
    pub opcode_rem_flag: T,
    pub opcode_remu_flag: T,
}

#[derive(Copy, Clone, Debug)]
pub struct DivRemCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub xor_bus: XorBus,
    pub range_tuple_bus: RangeTupleCheckerBus<2>,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        DivRemCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
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
        let cols: &DivRemCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_div_flag,
            cols.opcode_divu_flag,
            cols.opcode_rem_flag,
            cols.opcode_remu_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let b = &cols.b;
        let c = &cols.c;
        let q = &cols.q;
        let r = &cols.r;

        // Constrain that b = (c * q + r') % 2^{NUM_LIMBS * LIMB_BITS} and range check
        // each element in q.
        let c_ext = cols.c_sign * AB::F::from_canonical_u32((1 << LIMB_BITS) - 1);
        let carry_divide = AB::F::from_canonical_u32(1 << LIMB_BITS).inverse();
        let mut carry: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());

        for i in 0..NUM_LIMBS {
            let expected_limb = if i == 0 {
                AB::Expr::zero()
            } else {
                carry[i - 1].clone()
            } + (0..=i).fold(r[i].into(), |ac, k| ac + (c[k] * q[i - k]));
            carry[i] = AB::Expr::from(carry_divide) * (expected_limb - b[i]);
        }

        for (q, carry) in q.iter().zip(carry.iter()) {
            self.range_tuple_bus
                .send(vec![(*q).into(), carry.clone()])
                .eval(builder, is_valid.clone());
        }

        // Constrain that the upper limbs of b = c * q + r' are all equal to b_ext and
        // range check each element in r.
        let q_ext = cols.q_sign * AB::F::from_canonical_u32((1 << LIMB_BITS) - 1);
        let mut carry_ext: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());

        for j in 0..NUM_LIMBS {
            let expected_limb = if j == 0 {
                carry[NUM_LIMBS - 1].clone()
            } else {
                carry_ext[j - 1].clone()
            } + ((j + 1)..NUM_LIMBS).fold(AB::Expr::zero(), |acc, k| {
                acc + (c[k] * q[NUM_LIMBS + j - k])
            }) + (0..(j + 1)).fold(AB::Expr::zero(), |acc, k| {
                acc + (c[k] * q_ext.clone()) + (q[k] * c_ext.clone())
            });
            // Technically there are ways to constrain that c * q is in range without
            // using a range checker, but because we already have to range check each
            // limb of r it requires no additional columns to also range check each
            // carry_ext.
            //
            // Note also that really we should have added the sign extension of r to
            // expected_limb and set carry_ext = (expected_limb - r_sign) / 2^LIMB_BITS,
            // but because we expect the sign of r to be the sign of b both changes
            // cancel each other out.
            carry_ext[j] = AB::Expr::from(carry_divide) * expected_limb;
        }

        for (r, carry) in r.iter().zip(carry_ext.iter()) {
            self.range_tuple_bus
                .send(vec![(*r).into(), carry.clone()])
                .eval(builder, is_valid.clone());
        }

        // Handle special cases. We can have either a zero divisor or signed overflow,
        // or neither.
        let signed = cols.opcode_div_flag + cols.opcode_rem_flag;
        let special_case = cols.zero_divisor + cols.signed_overflow;
        builder.assert_bool(special_case.clone());
        builder
            .when(special_case.clone())
            .assert_eq(cols.q_sign, signed.clone());

        builder.assert_bool(cols.zero_divisor);
        let mut when_zero_divisor = builder.when(cols.zero_divisor);
        for i in 0..NUM_LIMBS {
            when_zero_divisor.assert_zero(c[i]);
            when_zero_divisor.assert_eq(q[i], AB::F::from_canonical_u32((1 << LIMB_BITS) - 1));
            when_zero_divisor.assert_eq(r[i], b[i]);
        }

        builder.assert_bool(cols.signed_overflow);
        let mut when_signed_overflow = builder.when(cols.signed_overflow);
        when_signed_overflow.assert_one(signed.clone());
        for i in 0..NUM_LIMBS {
            // Note we do not need to check b = 0b100... here, as it is the only in-range
            // integer such that b = q and b = - q (as constrained below)
            when_signed_overflow.assert_eq(c[i], AB::F::from_canonical_u32((1 << LIMB_BITS) - 1));
            when_signed_overflow.assert_eq(q[i], b[i]);
            when_signed_overflow.assert_zero(r[i]);
        }

        // Constrain the correctness of b_sign and c_sign. Note that we do not need to
        // check that the sign of r is b_sign since we cannot have r' <_u c if this is
        // not the case.
        //
        // To constrain the correctness of q_sign we make sure if q is non-zero then
        // q_sign = b_sign ^ c_sign, and if q_sign = 1 then q is non-zero. Note q_sum
        // is guaranteed to be non-zero if q is non-zero since we've range checked each
        // limb of q to be within [0, 2^LIMB_BITS) already.
        let mask = AB::F::from_canonical_u32(1 << (LIMB_BITS - 1));

        builder.assert_bool(cols.b_sign);
        builder.assert_bool(cols.c_sign);
        builder
            .when(not::<AB::Expr>(signed.clone()))
            .assert_zero(cols.b_sign);
        builder
            .when(not::<AB::Expr>(signed.clone()))
            .assert_zero(cols.c_sign);
        builder.assert_eq(
            cols.b_sign + cols.c_sign - AB::Expr::from_canonical_u32(2) * cols.b_sign * cols.c_sign,
            cols.sign_xor,
        );

        let nonzero_q = q.iter().fold(AB::Expr::zero(), |acc, q| acc + *q);
        builder.assert_bool(cols.q_sign);
        builder
            .when(nonzero_q * not::<AB::Expr>(special_case.clone()))
            .assert_eq(cols.q_sign, cols.sign_xor);
        builder
            .when((cols.q_sign - cols.sign_xor) * not::<AB::Expr>(special_case.clone()))
            .assert_zero(cols.q_sign);

        self.xor_bus
            .send(
                b[NUM_LIMBS - 1],
                mask,
                b[NUM_LIMBS - 1] + mask - cols.b_sign * mask * AB::Expr::from_canonical_u32(2),
            )
            .eval(builder, signed.clone());
        self.xor_bus
            .send(
                c[NUM_LIMBS - 1],
                mask,
                c[NUM_LIMBS - 1] + mask - cols.c_sign * mask * AB::Expr::from_canonical_u32(2),
            )
            .eval(builder, signed.clone());

        // Constrain that 0 <= |r| < |c| by checking that r' <_u c (unsigned LT). By
        // definition, the sign of r must be b_sign. If c is negative then we want
        // to constrain c <_u r'. If c is positive, then we want to constrain r' <_u c.
        //
        // Because we already constrain that r and q are correct for special cases,
        // we skip the range check when special_case = 1.
        let r_p = &cols.r_prime;
        let mut carry_lt: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::zero());

        for i in 0..NUM_LIMBS {
            // When the signs of r (i.e. b) and c are the same, r' = r.
            builder.when(not(cols.sign_xor)).assert_eq(r[i], r_p[i]);

            // When the signs of r and c are different, r' = -r. To constrain this, we
            // first ensure each r[i] + r'[i] + carry[i - 1] is in {0, 2^LIMB_BITS}, and
            // that when the sum is 0 then r'[i] = 0 as well. Passing both constraints
            // implies that 0 <= r'[i] <= 2^LIMB_BITS, and in order to ensure r'[i] !=
            // 2^LIMB_BITS we check that r'[i] - 2^LIMB_BITS has an inverse in F.
            let last_carry = if i > 0 {
                carry_lt[i - 1].clone()
            } else {
                AB::Expr::zero()
            };
            carry_lt[i] = AB::Expr::from(carry_divide) * (last_carry.clone() + r[i] + r_p[i]);
            builder.when(cols.sign_xor).assert_zero(
                (carry_lt[i].clone() - last_carry) * (carry_lt[i].clone() - AB::Expr::one()),
            );
            builder
                .when(cols.sign_xor)
                .assert_one((r_p[i] - AB::F::from_canonical_u32(1 << LIMB_BITS)) * cols.r_inv[i]);
            builder
                .when(cols.sign_xor * not::<AB::Expr>(carry_lt[i].clone()))
                .assert_zero(r_p[i]);
        }

        let marker = &cols.lt_marker;
        let mut prefix_sum = special_case.clone();

        for i in (0..NUM_LIMBS).rev() {
            let diff = r_p[i] * (AB::Expr::from_canonical_u8(2) * cols.c_sign - AB::Expr::one())
                + c[i] * (AB::Expr::one() - AB::Expr::from_canonical_u8(2) * cols.c_sign);
            prefix_sum += marker[i].into();
            builder.assert_bool(marker[i]);
            builder.assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker[i]).assert_eq(cols.lt_diff, diff);
        }

        builder.when(is_valid.clone()).assert_one(prefix_sum);
        self.xor_bus
            .send(
                cols.lt_diff - AB::Expr::one(),
                cols.lt_diff - AB::Expr::one(),
                AB::F::zero(),
            )
            .eval(builder, is_valid.clone() - special_case);

        // Generate expected opcode and output a to pass to the adapter.
        let expected_opcode = flags.iter().zip(DivRemOpcode::iter()).fold(
            AB::Expr::zero(),
            |acc, (flag, local_opcode)| {
                acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
            },
        ) + AB::Expr::from_canonical_usize(self.offset);

        let is_div = cols.opcode_div_flag + cols.opcode_divu_flag;
        let a =
            array::from_fn(|i| (is_div.clone() * q[i]) + (not::<AB::Expr>(is_div.clone()) * r[i]));

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct DivRemCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: DivRemCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> DivRemCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        xor_lookup_chip: Arc<XorLookupChip<LIMB_BITS>>,
        range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
        offset: usize,
    ) -> Self {
        // The RangeTupleChecker is used to range check (a[i], carry[i]) pairs where 0 <= i
        // < 2 * NUM_LIMBS. a[i] must have LIMB_BITS bits and carry[i] is the sum of i + 1
        // bytes (with LIMB_BITS bits). XorLookup is used to sign check bytes.
        debug_assert!(
            range_tuple_chip.sizes()[0] == 1 << LIMB_BITS,
            "First element of RangeTupleChecker must have size {}",
            1 << LIMB_BITS
        );
        debug_assert!(
            range_tuple_chip.sizes()[1] >= (1 << LIMB_BITS) * 2 * NUM_LIMBS as u32,
            "Second element of RangeTupleChecker must have size of at least {}",
            (1 << LIMB_BITS) * 2 * NUM_LIMBS as u32
        );

        Self {
            air: DivRemCoreAir {
                xor_bus: xor_lookup_chip.bus(),
                range_tuple_bus: *range_tuple_chip.bus(),
                offset,
            },
            xor_lookup_chip,
            range_tuple_chip,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DivRemCoreRecord<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub opcode: DivRemOpcode,
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],
    pub q: [T; NUM_LIMBS],
    pub r: [T; NUM_LIMBS],
    pub zero_divisor: T,
    pub signed_overflow: T,
    pub b_sign: T,
    pub c_sign: T,
    pub q_sign: T,
    pub sign_xor: T,
    pub r_prime: [T; NUM_LIMBS],
    pub r_inv: [T; NUM_LIMBS],
    pub lt_diff_val: T,
    pub lt_diff_idx: usize,
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, I> for DivRemCoreChip<NUM_LIMBS, LIMB_BITS>
where
    I::Reads: Into<[[F; NUM_LIMBS]; 2]>,
    I::Writes: From<[[F; NUM_LIMBS]; 1]>,
{
    type Record = DivRemCoreRecord<F, NUM_LIMBS, LIMB_BITS>;
    type Air = DivRemCoreAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let divrem_opcode = DivRemOpcode::from_usize(opcode - self.air.offset);

        let is_div = divrem_opcode == DivRemOpcode::DIV || divrem_opcode == DivRemOpcode::DIVU;
        let is_signed = divrem_opcode == DivRemOpcode::DIV || divrem_opcode == DivRemOpcode::REM;

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let b = data[0].map(|x| x.as_canonical_u32());
        let c = data[1].map(|y| y.as_canonical_u32());
        let (q, r, b_sign, c_sign, case) = run_divrem::<NUM_LIMBS, LIMB_BITS>(is_signed, &b, &c);

        let carries = run_mul_carries::<NUM_LIMBS, LIMB_BITS>(is_signed, &c, &q, &r);
        for i in 0..NUM_LIMBS {
            self.range_tuple_chip.add_count(&[q[i], carries[i]]);
            self.range_tuple_chip
                .add_count(&[r[i], carries[i + NUM_LIMBS]]);
        }

        let mask = 1 << (LIMB_BITS - 1);
        let q_sign = (q[NUM_LIMBS - 1] & mask) != 0 && is_signed;
        let sign_xor = b_sign ^ c_sign;
        let r_prime = if sign_xor {
            negate::<NUM_LIMBS, LIMB_BITS>(&r)
        } else {
            r
        };

        if is_signed {
            self.xor_lookup_chip.request(b[NUM_LIMBS - 1], mask);
            self.xor_lookup_chip.request(c[NUM_LIMBS - 1], mask);
        }

        let (lt_diff_idx, lt_diff_val) = if case == 0 {
            let idx = run_sltu_diff_idx(&c, &r_prime, c_sign);
            let val = if c_sign {
                r_prime[idx] - c[idx]
            } else {
                c[idx] - r_prime[idx]
            };
            self.xor_lookup_chip.request(val - 1, val - 1);
            (idx, val)
        } else {
            (NUM_LIMBS, 0)
        };

        let r_prime_f = r_prime.map(F::from_canonical_u32);
        let output = AdapterRuntimeContext::without_pc([
            (if is_div { &q } else { &r }).map(F::from_canonical_u32)
        ]);
        let record = DivRemCoreRecord {
            opcode: divrem_opcode,
            b: data[0],
            c: data[1],
            q: q.map(F::from_canonical_u32),
            r: r.map(F::from_canonical_u32),
            zero_divisor: F::from_bool(case == 1),
            signed_overflow: F::from_bool(case == 2),
            b_sign: F::from_bool(b_sign),
            c_sign: F::from_bool(c_sign),
            q_sign: F::from_bool(q_sign),
            sign_xor: F::from_bool(sign_xor),
            r_prime: r_prime_f,
            r_inv: r_prime_f.map(|r| (r - F::from_canonical_u32(256)).inverse()),
            lt_diff_val: F::from_canonical_u32(lt_diff_val),
            lt_diff_idx,
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", DivRemOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let row_slice: &mut DivRemCoreCols<_, NUM_LIMBS, LIMB_BITS> = row_slice.borrow_mut();
        row_slice.b = record.b;
        row_slice.c = record.c;
        row_slice.q = record.q;
        row_slice.r = record.r;
        row_slice.zero_divisor = record.zero_divisor;
        row_slice.signed_overflow = record.signed_overflow;
        row_slice.b_sign = record.b_sign;
        row_slice.c_sign = record.c_sign;
        row_slice.q_sign = record.q_sign;
        row_slice.sign_xor = record.sign_xor;
        row_slice.r_prime = record.r_prime;
        row_slice.r_inv = record.r_inv;
        row_slice.lt_marker = array::from_fn(|i| F::from_bool(i == record.lt_diff_idx));
        row_slice.lt_diff = record.lt_diff_val;
        row_slice.opcode_div_flag = F::from_bool(record.opcode == DivRemOpcode::DIV);
        row_slice.opcode_divu_flag = F::from_bool(record.opcode == DivRemOpcode::DIVU);
        row_slice.opcode_rem_flag = F::from_bool(record.opcode == DivRemOpcode::REM);
        row_slice.opcode_remu_flag = F::from_bool(record.opcode == DivRemOpcode::REMU);
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// Returns (quotient, remainder, x_sign, y_sign, case) where case = 0 for normal, 1 for zero
// divisor, and 2 for signed overflow
pub(super) fn run_divrem<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    signed: bool,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], [u32; NUM_LIMBS], bool, bool, u8) {
    let x_sign = if signed {
        x[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1
    } else {
        false
    };
    let y_sign = if signed {
        y[NUM_LIMBS - 1] >> (LIMB_BITS - 1) == 1
    } else {
        false
    };

    let zero_divisor = fold(y, true, |b, y_val| b && (*y_val == 0));
    let overflow = fold(
        &x[..(NUM_LIMBS - 1)],
        x[NUM_LIMBS - 1] == 1 << (LIMB_BITS - 1),
        |b, x_val| b && (*x_val == 0),
    ) && fold(y, true, |b, y_val| b && (*y_val == (1 << LIMB_BITS) - 1))
        && x_sign
        && y_sign;

    if zero_divisor {
        return ([(1 << LIMB_BITS) - 1; NUM_LIMBS], *x, x_sign, y_sign, 1);
    } else if overflow {
        return (*x, [0; NUM_LIMBS], x_sign, y_sign, 2);
    }

    let x_abs = if x_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(x)
    } else {
        *x
    };
    let y_abs = if y_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(y)
    } else {
        *y
    };

    let x_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&x_abs);
    let y_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&y_abs);
    let q_big = x_big.clone() / y_big.clone();
    let r_big = x_big.clone() % y_big.clone();

    let q = if x_sign ^ y_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_big))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_big)
    };

    // In C |q * y| <= |x|, which means if x is negative then r <= 0 and vice versa.
    let r = if x_sign {
        negate::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_big))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_big)
    };

    (q, r, x_sign, y_sign, 0)
}

pub(super) fn run_sltu_diff_idx<const NUM_LIMBS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
    cmp: bool,
) -> usize {
    for i in (0..NUM_LIMBS).rev() {
        if x[i] != y[i] {
            assert!((x[i] < y[i]) == cmp);
            return i;
        }
    }
    assert!(!cmp);
    NUM_LIMBS
}

// returns carries of d * q + r
pub(super) fn run_mul_carries<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    signed: bool,
    d: &[u32; NUM_LIMBS],
    q: &[u32; NUM_LIMBS],
    r: &[u32; NUM_LIMBS],
) -> Vec<u32> {
    let mut carry = vec![0u32; 2 * NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let mut val = r[i] + if i > 0 { carry[i - 1] } else { 0 };
        for j in 0..=i {
            val += d[j] * q[i - j];
        }
        carry[i] = val >> LIMB_BITS;
    }

    let d_ext =
        (d[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * if signed { (1 << LIMB_BITS) - 1 } else { 0 };
    let q_ext =
        (q[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * if signed { (1 << LIMB_BITS) - 1 } else { 0 };
    let mut d_prefix = 0;
    let mut q_prefix = 0;

    for i in 0..NUM_LIMBS {
        d_prefix += d[i];
        q_prefix += q[i];
        let mut val = carry[NUM_LIMBS + i - 1] + d_prefix * q_ext + q_prefix * d_ext;
        for j in (i + 1)..NUM_LIMBS {
            val += d[j] * q[NUM_LIMBS + i - j];
        }
        carry[NUM_LIMBS + i] = val >> LIMB_BITS;
    }
    carry
}

fn limbs_to_biguint<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
) -> BigUint {
    let base = BigUint::new(vec![1 << LIMB_BITS]);
    let mut res = BigUint::new(vec![0]);
    for val in x.iter().rev() {
        res *= base.clone();
        res += BigUint::new(vec![*val]);
    }
    res
}

fn biguint_to_limbs<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &BigUint,
) -> [u32; NUM_LIMBS] {
    let res_vec = big_uint_to_num_limbs(x, LIMB_BITS, NUM_LIMBS);
    array::from_fn(|i| res_vec[i] as u32)
}

fn negate<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut carry = 1;
    array::from_fn(|i| {
        let val = (1 << LIMB_BITS) + carry - 1 - x[i];
        carry = val >> LIMB_BITS;
        val % (1 << LIMB_BITS)
    })
}
