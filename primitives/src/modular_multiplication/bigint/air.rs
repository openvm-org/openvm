use std::{cmp::min, str::FromStr};

use afs_stark_backend::interaction::InteractionBuilder;
use itertools::Itertools;
use num_bigint_dig::BigUint;
use num_traits::One;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::{
    modular_multiplication::{
        air::{constrain_limbs, range_check},
        bigint::columns::ModularArithmeticBigIntCols,
        trace::{big_uint_to_bits, take_limb},
        FullLimbs, LimbDimensions,
    },
    sub_chip::AirConfig,
};

#[derive(Clone, Debug)]
pub struct ModularArithmeticBigIntAir {
    pub modulus: BigUint,
    pub total_bits: usize,
    pub decomp: usize,
    pub range_bus: usize,

    pub limb_dimensions: LimbDimensions,
    pub repr_bits: usize,
    pub max_limb_bits: usize,
    pub carry_bits: usize,
    pub carry_min_value_abs: usize,
    pub num_carries: usize,
    pub modulus_limbs: Vec<usize>,
}

impl ModularArithmeticBigIntAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        modulus: BigUint,
        total_bits: usize,
        decomp: usize,
        range_bus: usize,
        bits_per_elem: usize,
        repr_bits: usize,
        max_limb_bits: usize,
        carry_bits: usize,
        carry_min_value_abs: usize,
    ) -> Self {
        assert_eq!(repr_bits % max_limb_bits, 0);
        // `total_bits` should be sufficient to represent numbers 0..`modulus`
        assert!(total_bits >= (modulus.clone() - BigUint::one()).bits());
        assert!(max_limb_bits <= decomp);
        assert!(carry_bits <= decomp);

        let mut limb_sizes = vec![];
        let mut rem_bits = total_bits;
        while rem_bits > 0 {
            let limb_size = min(rem_bits, max_limb_bits);
            rem_bits -= limb_size;
            limb_sizes.push(limb_size);
        }
        let num_carries = (2 * limb_sizes.len()) - 2;

        let limb_max_value = (1 << max_limb_bits) - 1;

        let sum_min_value_abs = (limb_sizes.len() * limb_max_value * limb_max_value)
            + limb_max_value
            + carry_min_value_abs;
        assert!(sum_min_value_abs <= (carry_min_value_abs << max_limb_bits));

        let carry_max_value = (1 << carry_bits) - carry_min_value_abs;
        let sum_max_value = (limb_sizes.len() * limb_max_value * limb_max_value) + carry_max_value;
        assert!(sum_max_value <= (carry_max_value << max_limb_bits));
        assert!(((carry_min_value_abs + carry_max_value) << max_limb_bits) <= (1 << bits_per_elem));

        let limbs_per_elem = repr_bits / max_limb_bits;
        let limb_dimensions = LimbDimensions::new_same_sizes(limb_sizes, limbs_per_elem);

        let total_limbs = (total_bits + max_limb_bits - 1) / max_limb_bits;

        let mut modulus_bits = big_uint_to_bits(modulus.clone());

        let modulus_limbs = (0..total_limbs)
            .map(|_| take_limb(&mut modulus_bits, max_limb_bits))
            .collect();
        Self {
            modulus,
            total_bits,
            decomp,
            range_bus,
            limb_dimensions,
            repr_bits,
            max_limb_bits,
            carry_bits,
            carry_min_value_abs,
            num_carries,
            modulus_limbs,
        }
    }

    pub fn secp256k1_coord_prime() -> BigUint {
        let mut result = BigUint::one() << 256;
        for power in [32, 9, 8, 7, 6, 4, 0] {
            result -= BigUint::one() << power;
        }
        result
    }

    pub fn secp256k1_scalar_prime() -> BigUint {
        BigUint::from_str(
            "115792089237316195423570985008687907852837564279074904382605163141518161494337",
        )
        .unwrap()
    }

    pub fn default_for_secp256k1_coord(limb_bits: usize) -> Self {
        Self::new(
            Self::secp256k1_coord_prime(),
            256,
            16,
            0,
            30,
            30,
            limb_bits,
            16,
            1 << 15,
        )
    }

    pub fn default_for_secp256k1_scalar(limb_bits: usize) -> Self {
        Self::new(
            Self::secp256k1_scalar_prime(),
            256,
            16,
            0,
            30,
            30,
            limb_bits,
            16,
            1 << 15,
        )
    }
}

impl AirConfig for ModularArithmeticBigIntAir {
    type Cols<T> = ModularArithmeticBigIntCols<T>;
}

impl<F: Field> BaseAir<F> for ModularArithmeticBigIntAir {
    fn width(&self) -> usize {
        ModularArithmeticBigIntCols::<F>::get_width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularArithmeticBigIntAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ModularArithmeticBigIntCols::<AB::Var>::from_slice(&local, self);
        self.eval(builder, local);
    }
}

impl ModularArithmeticBigIntAir {
    pub fn eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: ModularArithmeticBigIntCols<AB::Var>,
    ) {
        let ModularArithmeticBigIntCols { general, carries } = cols;

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

        let [a_limbs, b_limbs, r_limbs] =
            [a_limbs, b_limbs, r_limbs].map(|limbs| limbs.into_iter().flatten().collect_vec());
        let p_limbs = self
            .modulus_limbs
            .iter()
            .map(|&x| AB::Expr::from_canonical_usize(x))
            .collect_vec();

        let mut carry_checks = vec![AB::Expr::zero(); self.num_carries + 1];
        for (i, a_limb) in a_limbs.iter().enumerate() {
            for (j, b_limb) in b_limbs.iter().enumerate() {
                carry_checks[i + j] += a_limb.clone() * b_limb.clone();
            }
        }
        for (i, p_limb) in p_limbs.iter().enumerate() {
            for (j, q_limb) in q_limbs.iter().enumerate() {
                carry_checks[i + j] -= p_limb.clone() * q_limb.clone();
            }
        }
        for (i, r_limb) in r_limbs.iter().enumerate() {
            carry_checks[i] -= r_limb.clone();
        }
        for (i, &carry) in carries.iter().enumerate() {
            carry_checks[i + 1] += carry.into();
        }
        for (&carry, carry_check) in carries.iter().zip_eq(carry_checks.iter().dropping_back(1)) {
            builder.assert_eq(
                carry * AB::F::from_canonical_usize(1 << self.max_limb_bits),
                carry_check.clone(),
            );
            range_check(
                builder,
                self.range_bus,
                self.decomp,
                self.carry_bits,
                carry + AB::F::from_canonical_usize(self.carry_min_value_abs),
            );
        }
        builder.assert_zero(carry_checks.last().unwrap().clone());
    }
}
