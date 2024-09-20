use std::borrow::Cow;

use afs_compiler::{
    asm::AsmBuilder,
    ir::{Array, Builder, Config, Var},
    util::execute_program_with_config,
};
use itertools::Itertools;
use num_bigint_dig::{algorithms::mod_inverse, BigUint};
use num_traits::{abs, signum, FromPrimitive};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use stark_vm::{
    modular_addsub::{big_uint_to_num_limbs, secp256k1_coord_prime},
    vm::config::VmConfig,
};

const NUM_LIMBS: usize = 32;
const LIMB_SIZE: usize = 8;

struct Fraction {
    num: isize,
    denom: isize,
}

impl Fraction {
    fn new(num: isize, denom: isize) -> Self {
        Self { num, denom }
    }

    fn to_biguint(&self) -> BigUint {
        let sign = signum(self.num) * signum(self.denom);
        let num = BigUint::from_isize(abs(self.num)).unwrap();
        let denom = BigUint::from_isize(abs(self.denom)).unwrap();
        let mut value = num
            * mod_inverse(
                Cow::Borrowed(&denom),
                Cow::Borrowed(&secp256k1_coord_prime()),
            )
            .unwrap()
            .to_biguint()
            .unwrap();
        if sign == -1 {
            value = secp256k1_coord_prime() - value;
        }
        value
    }
}

impl From<isize> for Fraction {
    fn from(value: isize) -> Self {
        Self::new(value, 1)
    }
}

struct Point {
    x: Fraction,
    y: Fraction,
}

impl Point {
    fn new(x: impl Into<Fraction>, y: impl Into<Fraction>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
    fn load_const<C: Config>(&self, builder: &mut Builder<C>) -> Array<C, Var<C::N>> {
        let array = builder.dyn_array(2 * NUM_LIMBS);
        let x = self.x.to_biguint();
        let y = self.y.to_biguint();

        let [x, y] = [x, y].map(|x| {
            big_uint_to_num_limbs(&x, LIMB_SIZE, NUM_LIMBS)
                .into_iter()
                .map(C::N::from_canonical_usize)
                .collect_vec()
        });
        for (i, &elem) in x.iter().chain(y.iter()).enumerate() {
            builder.set(&array, i, elem);
        }
        array
    }
}

fn test_secp256k1_add(point_1: Point, point_2: Point, point_3: Point) {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::bigint_builder();

    let p1 = point_1.load_const(&mut builder);
    let p2 = point_2.load_const(&mut builder);
    let expected = point_3.load_const(&mut builder);

    let res = builder.secp256k1_add(p1, p2);

    builder.assert_var_array_eq(&res, &expected);

    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program_with_config(
        VmConfig {
            secp256k1_enabled: true,
            u256_arithmetic_enabled: true,
            modular_addsub_enabled: true,
            modular_multdiv_enabled: true,
            ..Default::default()
        },
        program,
        vec![],
    );
}

// tests for x^3 = y^2 + 7

#[test]
fn test_compiler_ec_double() {
    test_secp256k1_add(Point::new(2, 1), Point::new(2, 1), Point::new(32, -181));
}

#[test]
fn test_compiler_ec_ne_add() {
    test_secp256k1_add(Point::new(2, 1), Point::new(32, 181), Point::new(2, -1));
}

#[test]
fn test_compiler_ec_add_to_zero() {
    test_secp256k1_add(Point::new(2, 1), Point::new(2, -1), Point::new(0, 0));
}

#[test]
fn test_compiler_ec_add_zero_left() {
    test_secp256k1_add(Point::new(0, 0), Point::new(2, 1), Point::new(2, 1))
}

#[test]
fn test_compiler_ec_add_zero_right() {
    test_secp256k1_add(Point::new(2, 1), Point::new(0, 0), Point::new(2, 1))
}

#[test]
fn test_compiler_ec_double_zero() {
    test_secp256k1_add(Point::new(0, 0), Point::new(0, 0), Point::new(0, 0))
}
