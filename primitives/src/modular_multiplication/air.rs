use std::cmp::{max, min};

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

struct ModularMultiplicationAir {
    pub decomp: usize,
    pub elem_limbs: Vec<Vec<usize>>,
    pub pure_limbs: Vec<usize>,

    pub max_sum_bits: usize,
    pub small_moduli: Vec<usize>,
    pub elem_coefficients: Vec<Vec<Vec<usize>>>,
    pub pure_coefficients: Vec<Vec<usize>>,
    pub modulus_residues: Vec<usize>,
}

impl ModularMultiplicationAir {
    // resulting Air can only be used with fields of size at least `field_size`
    pub fn new(modulus: BigUint, bits_per_elem: usize, decomp: usize) -> Self {
        // `total_bits` = ceil(lg(modulus))
        let mut total_bits = modulus.bits() as usize;
        if modulus.bits() == modulus.trailing_zeros().unwrap() + 1 {
            total_bits -= 1;
        }

        let mut elem_limbs = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let mut limbs_here = vec![];
            let mut rem_bits_here = min(rem_bits, bits_per_elem);
            rem_bits -= rem_bits_here;
            while rem_bits_here > 0 {
                let limb = min(rem_bits_here, decomp);
                rem_bits_here -= limb;
                limbs_here.push(limb);
            }
            elem_limbs.push(limbs_here);
        }

        let mut pure_limbs = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let limb = min(rem_bits, decomp);
            rem_bits -= limb;
            pure_limbs.push(limb);
        }

        let mut max_sum_1: usize = 0;
        for limbs in elem_limbs.iter() {
            for &limb in limbs.iter() {
                max_sum_1 += (1 << limb) - 1;
            }
        }
        let mut max_sum_2: usize = 0;
        for &limb in pure_limbs.iter() {
            max_sum_2 += (1 << limb) - 1;
        }
        // some small room for optimization:
        // small_modulus_limit should be maximized, subject to
        // max_sum * (small_modulus_limit - 1) < small_modulus_limit * 2^max_sum_bits <= 2^bits_per_elem
        let max_sum = max(max_sum_1, max_sum_2);
        let max_sum_bits = max_sum.ilog2() as usize;
        let small_modulus_limit = 1 << (bits_per_elem - max_sum_bits);
        let small_moduli = Self::choose_small_moduli(&modulus, small_modulus_limit);

        let elem_coefficients = small_moduli
            .iter()
            .map(|small_modulus| {
                let mut curr = 1;
                elem_limbs
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
                    .collect()
            })
            .collect();

        let pure_coefficients = small_moduli
            .iter()
            .map(|small_modulus| {
                let mut curr = 1;
                pure_limbs
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

        let modulus_residues = small_moduli
            .iter()
            .map(|small_modulus| (modulus.clone() % small_modulus).to_u64().unwrap() as usize)
            .collect();

        Self {
            decomp,
            elem_limbs,
            pure_limbs,
            max_sum_bits,
            small_moduli,
            elem_coefficients,
            pure_coefficients,
            modulus_residues,
        }
    }

    // greedy algorithm, not necessarily optimal
    // consider (modulus, small_modulus_limit) = (2520, 10)
    // greedy will choose [10, 9, 7] then fail because nothing left
    // optimal is [9, 7, 8, 5]
    fn choose_small_moduli(modulus: &BigUint, small_modulus_limit: usize) -> Vec<usize> {
        let mut small_moduli = vec![];
        let mut small_mod_prod = BigUint::one();
        let mut candidate = small_modulus_limit;
        while small_mod_prod < *modulus {
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
