use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::{
        check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
        check_carry_to_zero::get_carry_max_abs_and_bits,
        utils::*,
        OverflowInt,
    },
    sub_chip::{AirConfig, LocalTraceInstructions},
    var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::{BigInt, BigUint, Sign};
use num_traits::Zero;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeField64};
use p3_matrix::Matrix;

use super::{FieldVariable, SymbolicExpr, LIMB_BITS};

#[derive(Clone)]
pub struct ExprBuilder {
    // The prime field.
    pub prime: BigUint,

    pub num_input: usize,

    // Each constraint introduces one new variable, so this is also the number of variables.
    pub num_constraint: usize,

    // The number of limbs of the quotient for each constraint.
    pub q_limbs: Vec<usize>,
    // The number of limbs of the carries for each constraint.
    pub carry_limbs: Vec<usize>,

    // The constraints that should be evaluated to zero mod p (so doesn't include - pq part).
    pub constraints: Vec<SymbolicExpr>,

    // The equations to compute the newly introduced variables.
    pub computes: Vec<SymbolicExpr>,
}

impl ExprBuilder {
    pub fn new(prime: BigUint) -> Self {
        Self {
            prime,
            num_input: 0,
            num_constraint: 0,
            q_limbs: vec![],
            carry_limbs: vec![],
            constraints: vec![],
            computes: vec![],
        }
    }

    pub fn new_input(builder: Rc<RefCell<ExprBuilder>>) -> FieldVariable {
        let num_input = {
            let mut borrowed = builder.borrow_mut();
            borrowed.num_input += 1;
            borrowed.num_input
        };
        FieldVariable {
            expr: SymbolicExpr::Input(num_input - 1),
            builder: builder.clone(),
        }
    }
}

pub struct FieldExprChip {
    pub builder: ExprBuilder,

    // Number of limbs of a field element.
    pub num_limbs: usize,

    pub check_carry_mod_to_zero: CheckCarryModToZeroSubAir,
}

impl<F: Field> BaseAir<F> for FieldExprChip {
    fn width(&self) -> usize {
        self.num_limbs * (self.builder.num_input + self.builder.num_constraint)
            + self.builder.q_limbs.iter().sum::<usize>()
            + self.builder.carry_limbs.iter().sum::<usize>()
            + 1 // is_valid
    }
}

type Vecs<T> = Vec<Vec<T>>;

impl<AB: InteractionBuilder> Air<AB> for FieldExprChip {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let (is_valid, inputs, vars, q_limbs, carry_limbs) = self.load_vars(&local);
        let inputs = load_overflow::<AB>(inputs);
        let vars = load_overflow::<AB>(vars);

        for i in 0..self.builder.num_constraint {
            let expr = self.builder.constraints[i].evaluate(&inputs, &vars);
            self.check_carry_mod_to_zero.constrain_carry_mod_to_zero(
                builder,
                expr,
                CheckCarryModToZeroCols {
                    carries: carry_limbs[i].clone(),
                    quotient: q_limbs[i].clone(),
                },
                is_valid,
            )
        }

        // TODO: above range check q and c, still need to range check vars.
    }
}

impl AirConfig for FieldExprChip {
    // No column struct.
    type Cols<T> = Vec<T>;
}

impl<F: PrimeField64> LocalTraceInstructions<F> for FieldExprChip {
    type LocalInput = (Vec<BigUint>, Arc<VariableRangeCheckerChip>);

    fn generate_trace_row(&self, local_input: Self::LocalInput) -> Self::Cols<F> {
        let (inputs, range_checker) = local_input;
        assert_eq!(inputs.len(), self.builder.num_input);

        let mut vars = vec![BigUint::zero(); self.builder.num_constraint];

        // BigInt type is required for computing the quotient.
        let input_bigint = inputs
            .iter()
            .map(|x| BigInt::from_biguint(Sign::Plus, x.clone()))
            .collect::<Vec<BigInt>>();
        let prime_bigint = BigInt::from_biguint(Sign::Plus, self.builder.prime.clone());
        let mut vars_bigint = vec![BigInt::zero(); self.builder.num_constraint];

        // OverflowInt type is required for computing the carries.
        let input_overflow = input_bigint
            .iter()
            .map(|x| to_overflow_int(x, self.num_limbs))
            .collect::<Vec<_>>();
        let zero = OverflowInt::<isize>::from_vec(vec![0], LIMB_BITS);
        let mut vars_overflow = vec![zero; self.builder.num_constraint];
        let prime_overflow = to_overflow_int(&prime_bigint, self.num_limbs);

        let mut all_q = vec![];
        let mut all_carry = vec![];
        for i in 0..self.builder.num_constraint {
            let r = self.builder.computes[i].compute(&inputs, &vars, &self.builder.prime);
            vars[i] = r.clone();
            vars_bigint[i] = BigInt::from_biguint(Sign::Plus, r);
            vars_overflow[i] = to_overflow_int(&vars_bigint[i], self.num_limbs);
            // expr = q * p
            let expr_bigint = self.builder.constraints[i].evaluate(&input_bigint, &vars_bigint);
            let q = expr_bigint / &prime_bigint;
            let q_limbs = big_int_to_num_limbs(&q, LIMB_BITS, self.builder.q_limbs[i]);
            for &q in q_limbs.iter() {
                range_checker.add_count((q + (1 << LIMB_BITS)) as u32, LIMB_BITS + 1);
            }
            let q_overflow = OverflowInt {
                limbs: q_limbs.clone(),
                max_overflow_bits: LIMB_BITS + 1, // q can be negative, so this is the constraint we have when range check.
                limb_max_abs: (1 << LIMB_BITS),
            };
            // compute carries of (expr - q * p)
            let expr = self.builder.constraints[i].evaluate(&input_overflow, &vars_overflow);
            let expr = expr - q_overflow * prime_overflow.clone();
            let carries = expr.calculate_carries(LIMB_BITS);
            let max_overflow_bits = expr.max_overflow_bits;
            let (carry_min_abs, carry_bits) =
                get_carry_max_abs_and_bits(max_overflow_bits, LIMB_BITS);
            for &carry in carries.iter() {
                range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
            }
            all_q.push(vec_isize_to_f::<F>(q_limbs));
            all_carry.push(vec_isize_to_f::<F>(carries));
        }
        // TODO: range check vars libs

        let input_limbs = input_overflow
            .iter()
            .map(|x| vec_isize_to_f::<F>(x.limbs.clone()))
            .collect::<Vec<_>>();
        let vars_limbs = vars_overflow
            .iter()
            .map(|x| vec_isize_to_f::<F>(x.limbs.clone()))
            .collect::<Vec<_>>();

        [
            vec![F::one()],
            input_limbs.concat(),
            vars_limbs.concat(),
            all_q.concat(),
            all_carry.concat(),
        ]
        .concat()
    }
}

impl FieldExprChip {
    fn load_vars<T: Clone>(&self, arr: &[T]) -> (T, Vecs<T>, Vecs<T>, Vecs<T>, Vecs<T>) {
        let is_valid = arr[0].clone();
        let mut idx = 1;
        let mut inputs = vec![];
        for _ in 0..self.builder.num_input {
            inputs.push(arr[idx..idx + self.num_limbs].to_vec());
            idx += self.num_limbs;
        }
        let mut vars = vec![];
        for _ in 0..self.builder.num_constraint {
            vars.push(arr[idx..idx + self.num_limbs].to_vec());
            idx += self.num_limbs;
        }
        let mut q_limbs = vec![];
        for q in self.builder.q_limbs.iter() {
            q_limbs.push(arr[idx..idx + q].to_vec());
            idx += q;
        }
        let mut carry_limbs = vec![];
        for c in self.builder.carry_limbs.iter() {
            carry_limbs.push(arr[idx..idx + c].to_vec());
            idx += c;
        }
        (is_valid, inputs, vars, q_limbs, carry_limbs)
    }
}

fn load_overflow<AB: AirBuilder>(arr: Vecs<AB::Var>) -> Vec<OverflowInt<AB::Expr>> {
    let mut result = vec![];
    for x in arr.into_iter() {
        result.push(OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            x, LIMB_BITS,
        ));
    }
    result
}

fn to_overflow_int(x: &BigInt, num_limbs: usize) -> OverflowInt<isize> {
    let x_limbs = big_int_to_num_limbs(x, LIMB_BITS, num_limbs);
    OverflowInt::from_vec(x_limbs, LIMB_BITS)
}
