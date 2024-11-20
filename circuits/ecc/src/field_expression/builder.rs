use std::{cell::RefCell, ops::Deref, rc::Rc};

use ax_circuit_primitives::{
    bigint::{
        check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
        check_carry_to_zero::get_carry_max_abs_and_bits,
        utils::*,
        OverflowInt,
    },
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};
use ax_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use num_bigint_dig::{BigInt, BigUint, Sign};
use num_traits::Zero;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};

use super::{FieldVariable, SymbolicExpr};

#[derive(Clone)]
pub struct ExprBuilderConfig {
    pub modulus: BigUint,
    pub num_limbs: usize,
    pub limb_bits: usize,
}

impl ExprBuilderConfig {
    pub fn check_valid(&self) {
        assert!(self.modulus.bits() <= self.num_limbs * self.limb_bits);
    }
}

#[derive(Clone)]
pub struct ExprBuilder {
    // The prime field.
    pub prime: BigUint,
    // Same value, but we need BigInt for computing the quotient.
    pub prime_bigint: BigInt,

    pub num_input: usize,
    pub num_flags: usize,

    // This should be equal to number of constraints, but declare it to be explicit.
    pub num_variables: usize,

    /// The number of bits in a canonical representation of a limb.
    pub limb_bits: usize,
    /// Number of limbs in canonical representation of the bigint field element.
    pub num_limbs: usize,
    // The max bits to range check.
    pub range_checker_bits: usize,

    // The number of limbs of the quotient for each constraint.
    pub q_limbs: Vec<usize>,
    // The number of limbs of the carries for each constraint.
    pub carry_limbs: Vec<usize>,

    // The constraints that should be evaluated to zero mod p (doesn't include - p * q part).
    pub constraints: Vec<SymbolicExpr>,

    // The equations to compute the newly introduced variables. For trace gen only.
    pub computes: Vec<SymbolicExpr>,

    pub output_indices: Vec<usize>,
}

impl ExprBuilder {
    pub fn new(config: ExprBuilderConfig, range_checker_bits: usize) -> Self {
        let prime_bigint = BigInt::from_biguint(Sign::Plus, config.modulus.clone());
        Self {
            prime: config.modulus,
            prime_bigint,
            num_input: 0,
            num_flags: 0,
            limb_bits: config.limb_bits,
            num_limbs: config.num_limbs,
            range_checker_bits,
            num_variables: 0,
            q_limbs: vec![],
            carry_limbs: vec![],
            constraints: vec![],
            computes: vec![],
            output_indices: vec![],
        }
    }

    pub fn new_input(builder: Rc<RefCell<ExprBuilder>>) -> FieldVariable {
        let mut borrowed = builder.borrow_mut();
        let num_limbs = borrowed.num_limbs;
        let limb_bits = borrowed.limb_bits;
        borrowed.num_input += 1;
        let (num_input, range_checker_bits) = (borrowed.num_input, borrowed.range_checker_bits);
        drop(borrowed);
        FieldVariable {
            expr: SymbolicExpr::Input(num_input - 1),
            builder: builder.clone(),
            limb_max_abs: (1 << limb_bits) - 1,
            max_overflow_bits: limb_bits,
            expr_limbs: num_limbs,
            range_checker_bits,
        }
    }

    pub fn new_flag(&mut self) -> usize {
        self.num_flags += 1;
        self.num_flags - 1
    }

    // Below functions are used when adding variables and constraints manually, need to be careful.
    // Number of variables, constraints and computes should be consistent,
    // so there should be same number of calls to the new_var, add_constraint and add_compute.
    pub fn new_var(&mut self) -> (usize, SymbolicExpr) {
        self.num_variables += 1;
        // Allocate space for the new variable, to make sure they are corresponding to the same variable index.
        self.constraints.push(SymbolicExpr::Input(0));
        self.computes.push(SymbolicExpr::Input(0));
        self.q_limbs.push(0);
        self.carry_limbs.push(0);
        (
            self.num_variables - 1,
            SymbolicExpr::Var(self.num_variables - 1),
        )
    }

    pub fn set_constraint(&mut self, index: usize, constraint: SymbolicExpr) {
        let (q_limbs, carry_limbs) =
            constraint.constraint_limbs(&self.prime, self.limb_bits, self.num_limbs);
        self.constraints[index] = constraint;
        self.q_limbs[index] = q_limbs;
        self.carry_limbs[index] = carry_limbs;
    }

    pub fn set_compute(&mut self, index: usize, compute: SymbolicExpr) {
        self.computes[index] = compute;
    }
}

#[derive(Clone)]
pub struct FieldExpr {
    pub builder: ExprBuilder,

    pub check_carry_mod_to_zero: CheckCarryModToZeroSubAir,

    pub range_bus: VariableRangeCheckerBus,
}

impl FieldExpr {
    pub fn new(builder: ExprBuilder, range_bus: VariableRangeCheckerBus) -> Self {
        let subair = CheckCarryModToZeroSubAir::new(
            builder.prime.clone(),
            builder.limb_bits,
            range_bus.index,
            range_bus.range_max_bits,
        );
        FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus,
        }
    }
}

impl Deref for FieldExpr {
    type Target = ExprBuilder;

    fn deref(&self) -> &ExprBuilder {
        &self.builder
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExpr {}
impl<F: Field> PartitionedBaseAir<F> for FieldExpr {}
impl<F: Field> BaseAir<F> for FieldExpr {
    fn width(&self) -> usize {
        self.num_limbs * (self.builder.num_input + self.builder.num_variables)
            + self.builder.q_limbs.iter().sum::<usize>()
            + self.builder.carry_limbs.iter().sum::<usize>()
            + self.builder.num_flags
            + 1 // is_valid
            + 1 // is_setup
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for FieldExpr {
    /// The sub-row slice owned by the expression builder.
    type AirContext<'a>
        = &'a [AB::Var]
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, local: &'a [AB::Var])
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let FieldExprCols {
            is_valid,
            is_setup,
            inputs,
            vars,
            q_limbs,
            carry_limbs,
            flags,
        } = self.load_vars(local);

        {
            let modulus_limbs = big_uint_to_limbs(&self.builder.prime, self.builder.limb_bits);
            for i in 0..inputs[0].len().max(modulus_limbs.len()) {
                let lhs = if i < inputs[0].len() {
                    inputs[0][i].into()
                } else {
                    AB::Expr::ZERO
                };
                let rhs = if i < modulus_limbs.len() {
                    AB::Expr::from_canonical_usize(modulus_limbs[i])
                } else {
                    AB::Expr::ZERO
                };
                builder.when(is_setup).assert_eq(lhs, rhs);
            }
        }

        let inputs = load_overflow::<AB>(inputs, self.limb_bits);
        let vars = load_overflow::<AB>(vars, self.limb_bits);

        // TODO: turn back on once we also support this in ecc and everywhere
        // builder.when_first_row().assert_one(is_setup);

        for flag in flags.iter() {
            builder.assert_bool(*flag);
        }
        builder.assert_bool(is_setup);
        builder.assert_eq(
            is_valid.into(),
            flags.iter().fold(is_setup.into(), |acc, &x| acc + x),
        );
        for i in 0..self.constraints.len() {
            let expr = self.constraints[i].evaluate_overflow_expr::<AB>(&inputs, &vars, &flags);
            self.check_carry_mod_to_zero.eval(
                builder,
                (
                    expr,
                    CheckCarryModToZeroCols {
                        carries: carry_limbs[i].clone(),
                        quotient: q_limbs[i].clone(),
                    },
                    is_valid.into(),
                ),
            );
        }

        for var in vars.iter() {
            for limb in var.limbs.iter() {
                range_check(
                    builder,
                    self.range_bus.index,
                    self.range_bus.range_max_bits,
                    self.limb_bits,
                    limb.clone(),
                    is_valid,
                );
            }
        }
    }
}

type Vecs<T> = Vec<Vec<T>>;

pub struct FieldExprCols<T> {
    pub is_valid: T,
    pub is_setup: T,
    pub inputs: Vecs<T>,
    pub vars: Vecs<T>,
    pub q_limbs: Vecs<T>,
    pub carry_limbs: Vecs<T>,
    pub flags: Vec<T>,
}

impl<F: PrimeField64> TraceSubRowGenerator<F> for FieldExpr {
    type TraceContext<'a> = (&'a VariableRangeCheckerChip, Vec<BigUint>, Vec<bool>);
    type ColsMut<'a> = &'a mut [F];

    fn generate_subrow<'a>(
        &'a self,
        (range_checker, inputs, flags): (&'a VariableRangeCheckerChip, Vec<BigUint>, Vec<bool>),
        sub_row: &'a mut [F],
    ) {
        assert_eq!(inputs.len(), self.num_input);
        // Remove this if this is no longer the case in the future.
        assert_eq!(self.num_variables, self.constraints.len());
        let limb_bits = self.limb_bits;

        let mut vars = vec![BigUint::zero(); self.num_variables];

        // BigInt type is required for computing the quotient.
        let input_bigint = inputs
            .iter()
            .map(|x| BigInt::from_biguint(Sign::Plus, x.clone()))
            .collect::<Vec<BigInt>>();
        let mut vars_bigint = vec![BigInt::zero(); self.num_variables];

        // OverflowInt type is required for computing the carries.
        let input_overflow = input_bigint
            .iter()
            .map(|x| to_overflow_int(x, self.num_limbs, self.limb_bits))
            .collect::<Vec<_>>();
        let zero = OverflowInt::<isize>::from_vec(vec![0], limb_bits);
        let mut vars_overflow = vec![zero; self.num_variables];
        let prime_overflow = to_overflow_int(&self.prime_bigint, self.num_limbs, self.limb_bits);

        let mut all_q = vec![];
        let mut all_carry = vec![];
        for i in 0..self.constraints.len() {
            let r = self.computes[i].compute(&inputs, &vars, &flags, &self.prime);
            vars[i] = r.clone();
            vars_bigint[i] = BigInt::from_biguint(Sign::Plus, r);
            vars_overflow[i] = to_overflow_int(&vars_bigint[i], self.num_limbs, self.limb_bits);
        }
        // We need to have all variables computed first because, e.g. constraints[2] might need variables[3].
        for i in 0..self.constraints.len() {
            // expr = q * p
            let expr_bigint =
                self.constraints[i].evaluate_bigint(&input_bigint, &vars_bigint, &flags);
            let q = &expr_bigint / &self.prime_bigint;
            // If this is not true then the evaluated constraint is not divisible by p.
            debug_assert_eq!(expr_bigint, &q * &self.prime_bigint);
            let q_limbs = big_int_to_num_limbs(&q, limb_bits, self.q_limbs[i]);
            assert_eq!(q_limbs.len(), self.q_limbs[i]); // If this fails, the q_limbs estimate is wrong.
            for &q in q_limbs.iter() {
                range_checker.add_count((q + (1 << limb_bits)) as u32, limb_bits + 1);
            }
            let q_overflow = OverflowInt {
                limbs: q_limbs.clone(),
                max_overflow_bits: limb_bits + 1, // q can be negative, so this is the constraint we have when range check.
                limb_max_abs: (1 << limb_bits),
            };
            // compute carries of (expr - q * p)
            let expr = self.constraints[i].evaluate_overflow_isize(
                &input_overflow,
                &vars_overflow,
                &flags,
            );
            let expr = expr - q_overflow * prime_overflow.clone();
            let carries = expr.calculate_carries(limb_bits);
            assert_eq!(carries.len(), self.carry_limbs[i]); // If this fails, the carry limbs estimate is wrong.
            let max_overflow_bits = expr.max_overflow_bits;
            let (carry_min_abs, carry_bits) =
                get_carry_max_abs_and_bits(max_overflow_bits, limb_bits);
            for &carry in carries.iter() {
                range_checker.add_count((carry + carry_min_abs as isize) as u32, carry_bits);
            }
            all_q.push(vec_isize_to_f::<F>(q_limbs));
            all_carry.push(vec_isize_to_f::<F>(carries));
        }
        for var in vars_overflow.iter() {
            for limb in var.limbs.iter() {
                range_checker.add_count(*limb as u32, limb_bits);
            }
        }

        let input_limbs = input_overflow
            .iter()
            .map(|x| vec_isize_to_f::<F>(x.limbs.clone()))
            .collect::<Vec<_>>();
        let vars_limbs = vars_overflow
            .iter()
            .map(|x| vec_isize_to_f::<F>(x.limbs.clone()))
            .collect::<Vec<_>>();

        // TODO: avoid all these copies and directly allocate
        sub_row.copy_from_slice(
            &[
                vec![F::ONE, F::ZERO],
                input_limbs.concat(),
                vars_limbs.concat(),
                all_q.concat(),
                all_carry.concat(),
                flags.iter().map(|x| F::from_bool(*x)).collect::<Vec<_>>(),
            ]
            .concat(),
        );
    }
}

impl FieldExpr {
    pub fn canonical_num_limbs(&self) -> usize {
        self.builder.num_limbs
    }

    pub fn canonical_limb_bits(&self) -> usize {
        self.builder.limb_bits
    }

    pub fn execute(&self, inputs: Vec<BigUint>, flags: Vec<bool>) -> Vec<BigUint> {
        let mut vars = vec![BigUint::zero(); self.num_variables];
        for i in 0..self.constraints.len() {
            let r = self.computes[i].compute(&inputs, &vars, &flags, &self.prime);
            vars[i] = r.clone();
        }
        vars
    }

    pub fn execute_with_output(&self, inputs: Vec<BigUint>, flags: Vec<bool>) -> Vec<BigUint> {
        let vars = self.execute(inputs, flags);
        self.builder
            .output_indices
            .iter()
            .map(|i| vars[*i].clone())
            .collect()
    }

    pub fn load_vars<T: Clone>(&self, arr: &[T]) -> FieldExprCols<T> {
        let is_valid = arr[0].clone();
        let is_setup = arr[1].clone();
        let mut idx = 2;
        let mut inputs = vec![];
        for _ in 0..self.num_input {
            inputs.push(arr[idx..idx + self.num_limbs].to_vec());
            idx += self.num_limbs;
        }
        let mut vars = vec![];
        for _ in 0..self.num_variables {
            vars.push(arr[idx..idx + self.num_limbs].to_vec());
            idx += self.num_limbs;
        }
        let mut q_limbs = vec![];
        for q in self.q_limbs.iter() {
            q_limbs.push(arr[idx..idx + q].to_vec());
            idx += q;
        }
        let mut carry_limbs = vec![];
        for c in self.carry_limbs.iter() {
            carry_limbs.push(arr[idx..idx + c].to_vec());
            idx += c;
        }
        let flags = arr[idx..idx + self.num_flags].to_vec();
        FieldExprCols {
            is_valid,
            is_setup,
            inputs,
            vars,
            q_limbs,
            carry_limbs,
            flags,
        }
    }
}

fn load_overflow<AB: AirBuilder>(
    arr: Vecs<AB::Var>,
    limb_bits: usize,
) -> Vec<OverflowInt<AB::Expr>> {
    let mut result = vec![];
    for x in arr.into_iter() {
        result.push(OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(
            x, limb_bits,
        ));
    }
    result
}

fn to_overflow_int(x: &BigInt, num_limbs: usize, limb_bits: usize) -> OverflowInt<isize> {
    let x_limbs = big_int_to_num_limbs(x, limb_bits, num_limbs);
    OverflowInt::from_vec(x_limbs, limb_bits)
}
