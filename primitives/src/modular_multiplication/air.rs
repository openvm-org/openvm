use std::cmp::min;

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::modular_multiplication::columns::ModularMultiplicationCols;
use crate::sub_chip::AirConfig;

pub struct SmallModulusSystem {
    pub small_modulus: usize,
    pub io_coefficients: Vec<Vec<usize>>,
    pub q_coefficients: Vec<usize>,
}

pub struct ModularMultiplicationAir {
    pub modulus: BigUint,

    pub decomp: usize,
    pub range_bus: usize,

    pub io_limb_sizes: Vec<Vec<usize>>,
    pub num_io_limbs: usize,
    pub q_limb_sizes: Vec<usize>,

    pub small_modulus_bits: usize,
    pub quotient_bits: usize,
    pub small_moduli_systems: Vec<SmallModulusSystem>,
}

impl ModularMultiplicationAir {
    // resulting Air can only be used with fields of size at least 2^`bits_per_elem`
    pub fn new(
        modulus: BigUint,
        bits_per_elem: usize,
        // global parameters: range checker
        decomp: usize,
        range_bus: usize,
        // global parameter: how many bits of an elem are used
        repr_bits: usize,
        // local parameters
        // max_limb_bits and small_modulus_bits should be maximized subject to some constraints
        max_limb_bits: usize,
        small_modulus_bits: usize,
        small_modulus_lower_bound: usize,
        quotient_bits: usize,
    ) -> Self {
        assert!(2 * decomp <= bits_per_elem);
        assert!(repr_bits <= bits_per_elem);
        assert!(max_limb_bits <= decomp);
        assert!(small_modulus_bits <= decomp);
        assert!(quotient_bits <= decomp);
        // `total_bits` is # of bits necessary to represent numbers 0..`modulus`
        let total_bits = (modulus.clone() - BigUint::one()).bits() as usize;

        let mut io_limb_sizes = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let mut limbs_here = vec![];
            let mut rem_bits_here = min(rem_bits, repr_bits);
            rem_bits -= rem_bits_here;
            while rem_bits_here > 0 {
                let limb = min(rem_bits_here, decomp);
                rem_bits_here -= limb;
                limbs_here.push(limb);
            }
            io_limb_sizes.push(limbs_here);
        }

        let mut q_limb_sizes = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let limb = min(rem_bits, decomp);
            rem_bits -= limb;
            q_limb_sizes.push(limb);
        }

        let mut max_sum_elem_limbs: usize = 0;
        for limbs in io_limb_sizes.iter() {
            for &limb in limbs.iter() {
                max_sum_elem_limbs += (1 << limb) - 1;
            }
        }
        let mut max_sum_pure_limbs: usize = 0;
        for &limb in q_limb_sizes.iter() {
            max_sum_pure_limbs += (1 << limb) - 1;
        }
        let max_sum_pq_r = max_sum_elem_limbs + max_sum_pure_limbs;

        // ensures that expression for a, b is at most 2^small_modulus_bits * 2^quotient_bits
        assert!(max_sum_elem_limbs <= (1 << quotient_bits));
        // ensures that the range of ab - (pq + r) is at most 2^small_modulus_bits * 2^quotient_bits
        assert!((1 << small_modulus_bits) + max_sum_pq_r <= (1 << quotient_bits));
        // ensures no overflow of (small_modulus * quotient) + residue
        assert!(small_modulus_bits + quotient_bits <= bits_per_elem);

        let small_moduli =
            Self::choose_small_moduli(BigUint::one() << (2 * total_bits), 1 << small_modulus_bits);

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
                    io_coefficients: elem_coefficients,
                    q_coefficients: pure_coefficients,
                }
            })
            .collect();

        let num_io_limbs = io_limb_sizes.iter().map(|limbs| limbs.len()).sum();

        Self {
            modulus,
            decomp,
            range_bus,
            io_limb_sizes,
            num_io_limbs,
            q_limb_sizes,
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
    fn choose_small_moduli(need: BigUint, small_modulus_limit: usize) -> Vec<usize> {
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
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl AirConfig for ModularMultiplicationAir {
    type Cols<T> = ModularMultiplicationCols<T>;
}

impl<F: Field> BaseAir<F> for ModularMultiplicationAir {
    fn width(&self) -> usize {
        ModularMultiplicationCols::<F>::get_width(&self)
    }
}

impl ModularMultiplicationAir {
    fn range_check<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        bits: usize,
        var: impl Into<AB::Expr>,
    ) {
        assert!(bits <= self.decomp);
        if bits == self.decomp {
            builder.push_send(self.range_bus, [var], AB::F::one());
        } else {
            builder.push_send(self.range_bus, [var], AB::F::one());
            builder.push_send(
                self.range_bus,
                [var * AB::F::from_canonical_usize(1 << (self.decomp - bits))],
                AB::F::one(),
            );
        }
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularMultiplicationAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularMultiplicationCols::<AB::Var>::from_slice(&local, &self);

        for (elems, limbs) in [
            (local.io.a_elems, &local.aux.a_limbs),
            (local.io.b_elems, &local.aux.b_limbs),
            (local.io.r_elems, &local.aux.r_limbs),
        ] {
            for (limb_sizes, (&elem, limbs_here)) in
                self.io_limb_sizes.iter().zip_eq(elems.iter().zip_eq(limbs))
            {
                let mut elem_check = AB::Expr::zero();
                let mut shift = 0;
                for (&limb_size, &limb) in limb_sizes.iter().zip_eq(limbs_here) {
                    self.range_check(builder, limb_size, limb);
                    elem_check += AB::Expr::from_canonical_usize(1 << shift) * limb;
                    shift += limb_size;
                }
                builder.assert_eq(elem, elem_check);
            }
        }

        for (&limb_size, &limb) in self.q_limb_sizes.iter().zip_eq(&local.aux.q_limbs) {
            self.range_check(builder, limb_size, limb);
        }

        for (system, system_cols) in self
            .small_moduli_systems
            .iter()
            .zip_eq(local.aux.system_cols)
        {
            let [a_reduced, b_reduced, r_reduced] =
                [&local.aux.a_limbs, &local.aux.b_limbs, &local.aux.r_limbs].map(|limbs| {
                    let mut reduced = AB::Expr::zero();
                    for (coefficients, limbs_here) in system.io_coefficients.iter().zip_eq(limbs) {
                        for (&coefficient, &limb) in coefficients.iter().zip_eq(limbs_here) {
                            reduced += AB::Expr::from_canonical_usize(coefficient) * limb;
                        }
                    }
                    reduced
                });

            for (reduced, residue, quotient) in [
                (a_reduced, system_cols.a_residue, system_cols.a_quotient),
                (b_reduced, system_cols.b_residue, system_cols.b_quotient),
            ] {
                self.range_check(builder, self.small_modulus_bits, residue);
                self.range_check(builder, self.quotient_bits, quotient);
                builder.assert_eq(
                    reduced,
                    (AB::Expr::from_canonical_usize(system.small_modulus) * quotient) + residue,
                );
            }

            let mut pq_reduced = AB::Expr::zero();
            for (&coefficient, &limb) in system.q_coefficients.iter().zip_eq(&local.aux.q_limbs) {
                pq_reduced += AB::Expr::from_canonical_usize(coefficient) * limb;
            }

            let reduced =
                (system_cols.a_residue * system_cols.b_residue) - (pq_reduced + r_reduced);
            self.range_check(
                builder,
                self.quotient_bits,
                system_cols.total_quotient
                    + AB::F::from_canonical_usize(
                        (1 << self.quotient_bits) - (1 << (2 * self.small_modulus_bits)),
                    ),
            );
            builder.assert_eq(
                reduced,
                AB::Expr::from_canonical_usize(system.small_modulus) * system_cols.total_quotient,
            );
        }
    }
}
