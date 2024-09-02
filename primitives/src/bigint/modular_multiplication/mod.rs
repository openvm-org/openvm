use std::{iter::repeat, sync::Arc};

use afs_stark_backend::interaction::InteractionBuilder;
use num_bigint_dig::BigUint;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField64};
use p3_matrix::Matrix;

use crate::{
    bigint::{
        check_carry_mod_to_zero::{CheckCarryModToZeroCols, CheckCarryModToZeroSubAir},
        utils::big_uint_to_limbs,
        OverflowInt,
    },
    range_gate::RangeCheckerGateChip,
    sub_chip::{AirConfig, LocalTraceInstructions},
};

// x * y = q * p + r => x * y = r (mod p)
#[derive(Clone)]
pub struct ModularMultiplicationCols<T> {
    pub x: Vec<T>,
    pub y: Vec<T>,
    pub q: Vec<T>,
    pub r: Vec<T>,
    pub carries: Vec<T>,
}

impl<T: Clone> ModularMultiplicationCols<T> {
    pub fn from_slice(slc: &[T], num_limbs: usize) -> Self {
        // The modulus p has num_limbs limbs.
        // So all the numbers (x, y, q, r) we operate on have num_limbs limbs.
        // The carries are for the expression xy - pq -r so it should be 2 * num_limbs - 1.
        let x = slc[0..num_limbs].to_vec();
        let y = slc[num_limbs..2 * num_limbs].to_vec();
        let q = slc[2 * num_limbs..3 * num_limbs].to_vec();
        let r = slc[3 * num_limbs..4 * num_limbs].to_vec();
        let carries = slc[4 * num_limbs..6 * num_limbs - 1].to_vec();

        Self {
            x,
            y,
            q,
            r,
            carries,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];

        flattened.extend_from_slice(&self.x);
        flattened.extend_from_slice(&self.y);
        flattened.extend_from_slice(&self.q);
        flattened.extend_from_slice(&self.r);
        flattened.extend_from_slice(&self.carries);

        flattened
    }
}

pub struct ModularMultiplicationAir {
    pub check_carry_sub_air: CheckCarryModToZeroSubAir,
    // The modulus p
    pub modulus: BigUint,
    // The number of limbs of the big numbers we operate on. Should be the number of limbs of modulus.
    pub num_limbs: usize,
    pub limb_bits: usize,
    pub range_decomp: usize,
}

impl ModularMultiplicationAir {
    pub fn new(
        modulus: BigUint,
        limb_bits: usize,
        max_overflow_bits: usize,
        num_limbs: usize,
        range_bus: usize,
        range_decomp: usize,
    ) -> Self {
        let check_carry_sub_air = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            limb_bits,
            range_bus,
            range_decomp,
            max_overflow_bits,
        );

        Self {
            check_carry_sub_air,
            modulus,
            num_limbs,
            limb_bits,
            range_decomp,
        }
    }

    fn get_carry_min_value_abs(&self) -> usize {
        self.check_carry_sub_air
            .check_carry_to_zero
            .carry_min_value_abs
    }

    fn get_carry_bits(&self) -> usize {
        self.check_carry_sub_air.check_carry_to_zero.carry_bits
    }
}

impl AirConfig for ModularMultiplicationAir {
    type Cols<T> = ModularMultiplicationCols<T>;
}

impl<F: Field> BaseAir<F> for ModularMultiplicationAir {
    fn width(&self) -> usize {
        6 * self.num_limbs - 1
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularMultiplicationAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularMultiplicationCols::<AB::Var>::from_slice(&local, self.num_limbs);

        let ModularMultiplicationCols {
            x,
            y,
            q,
            r,
            carries,
        } = local;

        let x_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(x, self.limb_bits);
        let y_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(y, self.limb_bits);
        let r_overflow = OverflowInt::<AB::Expr>::from_var_vec::<AB, AB::Var>(r, self.limb_bits);
        let expr = x_overflow * y_overflow - r_overflow;

        self.check_carry_sub_air.constrain_carry_mod_to_zero(
            builder,
            expr,
            CheckCarryModToZeroCols {
                carries,
                quotient: q,
            },
        );
    }
}

impl<F: PrimeField64> LocalTraceInstructions<F> for ModularMultiplicationAir {
    type LocalInput = (BigUint, BigUint, Arc<RangeCheckerGateChip>);

    fn generate_trace_row(&self, input: Self::LocalInput) -> Self::Cols<F> {
        let (x, y, range_checker) = input;
        //  todo: verify this is integer division
        let quotient = (x.clone() * y.clone()) / self.modulus.clone();
        let r = x.clone() * y.clone() - self.modulus.clone() * quotient.clone();
        // Quotient and result can be smaller, but padding it to the same size.
        let quotient_f: Vec<F> = big_uint_to_limbs(quotient.clone(), self.limb_bits)
            .iter()
            .chain(repeat(&0))
            .take(self.num_limbs)
            .map(|&x| F::from_canonical_usize(x))
            .collect();
        let r_f: Vec<F> = big_uint_to_limbs(r.clone(), self.limb_bits)
            .iter()
            .chain(repeat(&0))
            .take(self.num_limbs)
            .map(|&x| F::from_canonical_usize(x))
            .collect();
        let range_check = |bits: usize, value: usize| {
            let value = value as u32;
            if bits == self.range_decomp {
                range_checker.add_count(value);
            } else {
                range_checker.add_count(value);
                range_checker.add_count(value + (1 << self.range_decomp) - (1 << bits));
            }
        };
        let x_overflow =
            OverflowInt::<isize>::from_big_uint(x, self.limb_bits, Some(self.num_limbs));
        let y_overflow =
            OverflowInt::<isize>::from_big_uint(y, self.limb_bits, Some(self.num_limbs));
        let r_overflow =
            OverflowInt::<isize>::from_big_uint(r, self.limb_bits, Some(self.num_limbs));
        let p_overflow = OverflowInt::<isize>::from_big_uint(
            self.modulus.clone(),
            self.limb_bits,
            Some(self.num_limbs),
        );
        let q_overflow =
            OverflowInt::<isize>::from_big_uint(quotient, self.limb_bits, Some(self.num_limbs));
        for &q in q_overflow.limbs.iter() {
            range_check(self.limb_bits, q as usize);
        }
        let expr = x_overflow.clone() * y_overflow.clone() - r_overflow - p_overflow * q_overflow;
        let carries = expr.calculate_carries(self.limb_bits);
        let mut carries_f = vec![F::zero(); carries.len()];
        let carry_min_abs = self.get_carry_min_value_abs() as isize;
        for (i, &carry) in carries.iter().enumerate() {
            range_check(self.get_carry_bits(), (carry + carry_min_abs) as usize);
            carries_f[i] = F::from_canonical_usize(carry.unsigned_abs())
                * if carry >= 0 { F::one() } else { F::neg_one() };
        }

        ModularMultiplicationCols {
            x: x_overflow
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            y: y_overflow
                .limbs
                .iter()
                .map(|x| F::from_canonical_usize(*x as usize))
                .collect(),
            q: quotient_f,
            r: r_f,
            carries: carries_f,
        }
    }
}

#[cfg(test)]
mod test {
    use ax_sdk::{config::baby_bear_blake3::run_simple_test_no_pis, utils::create_seeded_rng};
    use num_traits::{FromPrimitive, One, Zero};
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_ceil_usize;
    use rand::RngCore;

    use super::{super::utils::secp256k1_prime, *};
    // 256 bit prime, 10 limb bits -> 26 limbs.
    const LIMB_BITS: usize = 10;
    const NUM_LIMB: usize = 26;

    fn evaluate_bigint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
        let mut res = BigUint::zero();
        let base = BigUint::from_u64(1 << limb_bits).unwrap();
        for limb in limbs.iter().rev() {
            res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
        }
        res
    }

    fn get_air_and_range_checker(
        prime: BigUint,
        limb_bits: usize,
        num_limbs: usize,
    ) -> (ModularMultiplicationAir, Arc<RangeCheckerGateChip>) {
        // The equation: x*y - p*q - r, with num_limbs N = 26
        // Abs of each limb of the equation can be as much as 2^10 * 2^10 * N * 2 + 2^10
        let limb_max_abs = (1 << (2 * limb_bits)) * num_limbs * 2 + (1 << limb_bits);
        // overflow bits: log(max_abs) => 26
        let max_overflow_bits = log2_ceil_usize(limb_max_abs);

        let range_bus = 1;
        let range_decomp = 17;
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus, 1 << range_decomp));
        let air = ModularMultiplicationAir::new(
            prime,
            limb_bits,
            max_overflow_bits,
            num_limbs,
            range_bus,
            range_decomp,
        );
        (air, range_checker)
    }

    fn generate_xy() -> (BigUint, BigUint) {
        let mut rng = create_seeded_rng();
        let len = 8; // in bytes -> 256 bits.
        let x = (0..len).map(|_| rng.next_u32()).collect();
        let x = BigUint::new(x);
        let y = (0..len).map(|_| rng.next_u32()).collect();
        let y = BigUint::new(y);
        (x, y)
    }

    #[test]
    fn test_x_mul_y() {
        let prime = secp256k1_prime();
        let (x, y) = generate_xy();

        let (air, range_checker) = get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB);
        let expected_r = x.clone() * y.clone() % prime.clone();
        let expected_q = x.clone() * y.clone() / prime;
        let cols = air.generate_trace_row((x, y, range_checker.clone()));
        let ModularMultiplicationCols {
            x: _x,
            y: _y,
            q,
            r,
            carries: _carries,
        } = cols.clone();
        let generated_r = evaluate_bigint(&r, LIMB_BITS);
        let generated_q = evaluate_bigint(&q, LIMB_BITS);
        assert_eq!(generated_r, expected_r);
        assert_eq!(generated_q, expected_q);

        let row = cols.flatten();
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
        let range_trace = range_checker.generate_trace();

        run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
            .expect("Verification failed");
    }

    #[test]
    fn test_x_mul_zero() {
        let prime = secp256k1_prime();
        let (x, _) = generate_xy();
        let y = BigUint::zero();

        let (air, range_checker) = get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB);
        let cols = air.generate_trace_row((x, y, range_checker.clone()));

        let row = cols.flatten();
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
        let range_trace = range_checker.generate_trace();

        run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
            .expect("Verification failed");
    }

    #[test]
    fn test_x_mul_one() {
        let prime = secp256k1_prime();
        let (x, _) = generate_xy();
        let y = BigUint::one();

        let (air, range_checker) = get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB);
        let cols = air.generate_trace_row((x, y, range_checker.clone()));

        let row = cols.flatten();
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
        let range_trace = range_checker.generate_trace();

        run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
            .expect("Verification failed");
    }

    #[test]
    #[should_panic]
    fn test_x_mul_y_wrong_trace() {
        let prime = secp256k1_prime();
        let (x, y) = generate_xy();

        let (air, range_checker) = get_air_and_range_checker(prime.clone(), LIMB_BITS, NUM_LIMB);
        let cols = air.generate_trace_row((x, y, range_checker.clone()));

        let row = cols.flatten();
        let mut trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&air));
        trace.row_mut(0)[0] += BabyBear::one();
        let range_trace = range_checker.generate_trace();

        run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
            .expect("Verification failed");
    }
}
