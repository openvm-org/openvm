use std::{cmp::min, collections::HashSet};

use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::Matrix;
use prime_factorization::Factorization;

use crate::{
    modular_multiplication::{
        air::constrain_limbs, cast_primes::columns::ModularMultiplicationPrimesCols, FullLimbs,
        LimbDimensions,
    },
    sub_chip::AirConfig,
};

pub struct SmallModulusSystem<F: Field> {
    pub small_modulus: usize,
    pub small_modulus_inverse: F,
    pub io_coefficients: Vec<Vec<usize>>,
    pub q_coefficients: Vec<usize>,
}

pub struct ModularMultiplicationPrimesAir<F: Field> {
    pub modulus: BigUint,
    pub total_bits: usize,

    pub decomp: usize,
    pub range_bus: usize,

    pub limb_dimensions: LimbDimensions,

    pub small_modulus_bits: usize,
    pub quotient_bits: usize,
    pub small_moduli_systems: Vec<SmallModulusSystem<F>>,
}

/// Has IO columns (a, b, r)
/// It is guaranteed that if (a, b, r) is verifiable then a * b == r (mod `modulus`)
/// However, any of a, b, r may be >= `modulus`
/// Furthermore, (a, b, r) is guaranteed to be verifiable if a, b, r < `modulus` and a * b == r (mod `modulus`)
/// If a * b == r (mod `modulus`) but one of a, b, r is >= `modulus`, then (a, b, r) may not be verifiable
impl<F: PrimeField64> ModularMultiplicationPrimesAir<F> {
    // `F` should have size at least 2^`bits_per_elem`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modulus: BigUint,
        total_bits: usize,
        // global parameters: range checker
        decomp: usize,
        range_bus: usize,
        // global parameter: how many bits of an elem are used
        repr_bits: usize,
        // local parameters
        // max_limb_bits and small_modulus_bits should be maximized subject to some constraints
        max_limb_bits: usize,
        small_modulus_bits: usize,
        small_modulus_limit: usize,
        quotient_bits: usize,
    ) -> Self {
        let field_size = F::neg_one().as_canonical_u64() as usize;
        assert!((1 << repr_bits) <= field_size);
        assert!(max_limb_bits <= decomp);
        assert!(small_modulus_bits <= decomp);
        assert!(quotient_bits <= decomp);
        // `total_bits` should be sufficient to represent numbers 0..`modulus`
        assert!(total_bits >= (modulus.clone() - BigUint::one()).bits() as usize);
        assert!(small_modulus_limit <= (1 << small_modulus_bits));

        let mut io_limb_sizes = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let mut limbs_here = vec![];
            let mut rem_bits_here = min(rem_bits, repr_bits);
            rem_bits -= rem_bits_here;
            while rem_bits_here > 0 {
                let limb = min(rem_bits_here, max_limb_bits);
                rem_bits_here -= limb;
                limbs_here.push(limb);
            }
            io_limb_sizes.push(limbs_here);
        }

        let mut q_limb_sizes = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let limb = min(rem_bits, max_limb_bits);
            rem_bits -= limb;
            q_limb_sizes.push(limb);
        }

        let mut max_sum_io_limbs: usize = 0;
        for limbs in io_limb_sizes.iter() {
            for &limb in limbs.iter() {
                max_sum_io_limbs += (1 << limb) - 1;
            }
        }
        let mut max_sum_q_limbs: usize = 0;
        for &limb in q_limb_sizes.iter() {
            max_sum_q_limbs += (1 << limb) - 1;
        }
        let max_sum_pq_r = max_sum_io_limbs + max_sum_q_limbs;

        // ensures that expression for a, b is at most 2^small_modulus_bits * 2^quotient_bits
        assert!(max_sum_io_limbs <= (1 << quotient_bits));
        // ensures that the range of ab - (pq + r) is at most 2^small_modulus_bits * 2^quotient_bits
        assert!((1 << small_modulus_bits) + max_sum_pq_r <= (1 << quotient_bits));
        // ensures no overflow of (small_modulus * quotient) + residue
        assert!(
            (small_modulus_limit * ((1 << quotient_bits) - 1)) + (1 << small_modulus_bits)
                <= field_size
        );

        let small_moduli = Self::choose_small_moduli_prime_powers(
            BigUint::one() << (2 * total_bits),
            small_modulus_limit,
        );

        let small_moduli_systems = small_moduli
            .iter()
            .map(|&small_modulus| {
                let mut curr = 1;
                let elem_coefficients = io_limb_sizes
                    .iter()
                    .map(|limbs| {
                        limbs
                            .iter()
                            .map(|limb| {
                                let result = curr;
                                curr <<= limb;
                                curr %= small_modulus;
                                result
                            })
                            .collect()
                    })
                    .collect();
                let mut curr = (modulus.clone() % small_modulus).to_u64().unwrap() as usize;
                let pure_coefficients = q_limb_sizes
                    .iter()
                    .map(|limb| {
                        let result = curr;
                        curr <<= limb;
                        curr %= small_modulus;
                        result
                    })
                    .collect();
                SmallModulusSystem {
                    small_modulus,
                    small_modulus_inverse: F::from_canonical_usize(small_modulus).inverse(),
                    io_coefficients: elem_coefficients,
                    q_coefficients: pure_coefficients,
                }
            })
            .collect();

        Self {
            modulus,
            total_bits,
            decomp,
            range_bus,
            limb_dimensions: LimbDimensions::new(io_limb_sizes, q_limb_sizes),
            small_modulus_bits,
            quotient_bits,
            small_moduli_systems,
        }
    }

    // greedy algorithm, not necessarily optimal
    // consider (modulus, small_modulus_limit) = (2520, 10)
    // greedy will choose [10, 9, 7] then fail because nothing left
    // optimal is [9, 7, 8, 5]
    // algorithm that only considers prime powers may be useful alternative
    fn choose_small_moduli_greedy(need: BigUint, small_modulus_limit: usize) -> Vec<usize> {
        let mut small_moduli = vec![];
        let mut small_mod_prod = BigUint::one();
        let mut candidate = small_modulus_limit;
        while small_mod_prod < need {
            if candidate == 1 {
                panic!("Not able to find sufficiently large set of small moduli");
            }
            if small_moduli.iter().all(|&x| gcd(x, candidate) == 1) {
                small_moduli.push(candidate);
                small_mod_prod *= candidate;
            }
            candidate -= 1;
        }
        small_moduli
    }

    fn choose_small_moduli_prime_powers(need: BigUint, small_modulus_limit: usize) -> Vec<usize> {
        let mut small_moduli = vec![];
        let mut small_mod_prod = BigUint::one();
        let mut candidate = small_modulus_limit;
        let mut seen_primes = HashSet::new();
        while small_mod_prod < need {
            if candidate == 1 {
                panic!("Not able to find sufficiently large set of small moduli");
            }
            let prime_factors = Factorization::run(candidate as u64);
            if prime_factors.prime_factor_repr().len() == 1
                && seen_primes.insert(prime_factors.factors[0])
            {
                small_moduli.push(candidate);
                small_mod_prod *= candidate;
            }
            candidate -= 1;
        }
        small_moduli
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl<F: Field> AirConfig for ModularMultiplicationPrimesAir<F> {
    type Cols<T> = ModularMultiplicationPrimesCols<T>;
}

impl<F: Field> BaseAir<F> for ModularMultiplicationPrimesAir<F> {
    fn width(&self) -> usize {
        ModularMultiplicationPrimesCols::<F>::get_width(self)
    }
}

impl<F: Field> ModularMultiplicationPrimesAir<F> {
    fn range_check<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        bits: usize,
        into_expr: impl Into<AB::Expr>,
    ) {
        assert!(bits <= self.decomp);
        let expr = into_expr.into();
        if bits == self.decomp {
            builder.push_send(self.range_bus, [expr], AB::F::one());
        } else {
            builder.push_send(self.range_bus, [expr.clone()], AB::F::one());
            builder.push_send(
                self.range_bus,
                [expr + AB::F::from_canonical_usize((1 << self.decomp) - (1 << bits))],
                AB::F::one(),
            );
        }
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularMultiplicationPrimesAir<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularMultiplicationPrimesCols::<AB::Var>::from_slice(&local, self);

        let ModularMultiplicationPrimesCols {
            general,
            system_cols,
        } = local;

        let FullLimbs {
            a_limbs,
            b_limbs,
            r_limbs,
            q_limbs,
        } = constrain_limbs(
            builder,
            self.range_bus,
            self.decomp,
            &self.limb_dimensions,
            general,
        );

        for (system, system_cols_here) in self.small_moduli_systems.iter().zip_eq(system_cols) {
            let [a_reduced, b_reduced, r_reduced] = [&a_limbs, &b_limbs, &r_limbs].map(|limbs| {
                let mut reduced = AB::Expr::zero();
                for (coefficients, limbs_here) in system.io_coefficients.iter().zip_eq(limbs) {
                    for (&coefficient, limb) in coefficients.iter().zip_eq(limbs_here) {
                        reduced += AB::Expr::from_canonical_usize(coefficient) * limb.clone();
                    }
                }
                reduced
            });

            let [a_residue, b_residue] = [
                (a_reduced, system_cols_here.a_quotient),
                (b_reduced, system_cols_here.b_quotient),
            ]
            .map(|(reduced, quotient)| {
                self.range_check(builder, self.quotient_bits, quotient);
                let residue =
                    reduced - (AB::Expr::from_canonical_usize(system.small_modulus) * quotient);
                self.range_check(builder, self.small_modulus_bits, residue.clone());
                residue
            });

            let mut pq_reduced = AB::Expr::zero();
            for (&coefficient, limb) in system.q_coefficients.iter().zip_eq(&q_limbs) {
                pq_reduced += AB::Expr::from_canonical_usize(coefficient) * limb.clone();
            }

            let reduced = (a_residue * b_residue) - (pq_reduced + r_reduced);
            let quotient = reduced * system.small_modulus_inverse;
            self.range_check(
                builder,
                self.quotient_bits,
                quotient
                    + AB::F::from_canonical_usize(
                        (1 << self.quotient_bits) - (1 << self.small_modulus_bits),
                    ),
            );
        }
    }
}
