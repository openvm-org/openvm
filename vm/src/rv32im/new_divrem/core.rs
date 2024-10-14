use std::{array, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    bigint::utils::big_uint_to_num_limbs,
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use itertools::fold;
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{DivRemOpcode, UsizeOpcode},
        AdapterAirContext, AdapterRuntimeContext, Reads, Result, VmAdapterChip, VmAdapterInterface,
        VmCoreAir, VmCoreChip, Writes,
    },
    system::program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DivRemCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    // TODO: add auxiliary columns
    pub opcode_div_flag: T,
    pub opcode_divu_flag: T,
    pub opcode_rem_flag: T,
    pub opcode_remu_flag: T,
}

#[derive(Copy, Clone, Debug)]
pub struct DivRemCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: RangeTupleCheckerBus<2>,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        DivRemCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB: InteractionBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {
        todo!();
    }
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for DivRemCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    fn eval(
        &self,
        _builder: &mut AB,
        _local: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        todo!()
    }
}

#[derive(Debug)]
pub struct DivRemCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: DivRemCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    offset: usize,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> DivRemCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(range_tuple_chip: Arc<RangeTupleCheckerChip<2>>, offset: usize) -> Self {
        Self {
            air: DivRemCoreAir {
                bus: *range_tuple_chip.bus(),
            },
            range_tuple_chip,
            offset,
        }
    }
}

impl<F: PrimeField32, A: VmAdapterChip<F>, const NUM_LIMBS: usize, const LIMB_BITS: usize>
    VmCoreChip<F, A> for DivRemCoreChip<NUM_LIMBS, LIMB_BITS>
where
    Reads<F, A::Interface<F>>: Into<[[F; NUM_LIMBS]; 2]>,
    Writes<F, A::Interface<F>>: From<[F; NUM_LIMBS]>,
{
    // TODO: update for trace generation
    type Record = u32;
    type Air = DivRemCoreAir<NUM_LIMBS, LIMB_BITS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: F,
        reads: <A::Interface<F> as VmAdapterInterface<F>>::Reads,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)> {
        let Instruction { opcode, .. } = instruction;
        let local_opcode_index = DivRemOpcode::from_usize(opcode - self.offset);

        let data: [[F; NUM_LIMBS]; 2] = reads.into();
        let x = data[0].map(|x| x.as_canonical_u32());
        let y = data[1].map(|y| y.as_canonical_u32());
        let (q, r, _x_sign, _y_sign) = solve_divrem::<NUM_LIMBS, LIMB_BITS>(
            local_opcode_index == DivRemOpcode::DIV || local_opcode_index == DivRemOpcode::REM,
            &x,
            &y,
        );

        let z = if local_opcode_index == DivRemOpcode::DIV
            || local_opcode_index == DivRemOpcode::DIVU
        {
            &q
        } else {
            &r
        };

        // Core doesn't modify PC directly, so we let Adapter handle the increment
        let output: AdapterRuntimeContext<F, A::Interface<F>> = AdapterRuntimeContext {
            to_pc: None,
            writes: z.map(F::from_canonical_u32).into(),
        };

        // TODO: send RangeTupleChecker requests
        // TODO: create Record and return

        Ok((output, 0))
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        todo!()
    }

    fn generate_trace_row(&self, _row_slice: &mut [F], _record: Self::Record) {
        todo!()
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

// Returns (quotient, remainder, x_sign, y_sign)
pub(super) fn solve_divrem<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    signed: bool,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> ([u32; NUM_LIMBS], [u32; NUM_LIMBS], u32, u32) {
    let x_sign = if signed {
        x[NUM_LIMBS - 1] >> (LIMB_BITS - 1)
    } else {
        0
    };
    let y_sign = if signed {
        y[NUM_LIMBS - 1] >> (LIMB_BITS - 1)
    } else {
        0
    };

    assert!(x_sign == 0 || x_sign == 1);
    assert!(y_sign == 0 || y_sign == 1);

    let zero_divisor = fold(y, true, |b, y_val| b && (*y_val == 0));
    let overflow = fold(&y[1..], y[0] == 1 << (LIMB_BITS - 1), |b, y_val| {
        b && (*y_val == 0)
    }) && fold(x, true, |b, x_val| b && (*x_val == (1 << LIMB_BITS) - 1))
        && x_sign == 1
        && y_sign == 1;

    if zero_divisor {
        return ([(1 << LIMB_BITS) - 1; NUM_LIMBS], *x, x_sign, y_sign);
    } else if overflow {
        return (*x, [0; NUM_LIMBS], x_sign, y_sign);
    }

    let x_abs = if x_sign == 1 {
        get_2s_complement::<NUM_LIMBS, LIMB_BITS>(x)
    } else {
        *x
    };
    let y_abs = if y_sign == 1 {
        get_2s_complement::<NUM_LIMBS, LIMB_BITS>(y)
    } else {
        *y
    };

    let x_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&x_abs);
    let y_big = limbs_to_biguint::<NUM_LIMBS, LIMB_BITS>(&y_abs);
    let q_big = x_big.clone() / y_big.clone();
    let r_big = x_big.clone() % y_big.clone();

    let q = if x_sign + y_sign == 1 {
        get_2s_complement::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_big))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&q_big)
    };

    // In C |q * y| <= |x|, which means if x is negative then r <= 0 and vice versa.
    let r = if x_sign == 1 {
        get_2s_complement::<NUM_LIMBS, LIMB_BITS>(&biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_big))
    } else {
        biguint_to_limbs::<NUM_LIMBS, LIMB_BITS>(&r_big)
    };

    (q, r, x_sign, y_sign)
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

fn get_2s_complement<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut carry = 1u32;
    let mut result = [0; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        result[i] = (1 << LIMB_BITS) + carry - 1 - x[i];
        carry = result[i] >> LIMB_BITS;
        result[i] %= 1 << LIMB_BITS;
    }
    result
}
