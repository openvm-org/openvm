use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldExtensionAlgebra};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use zkhash::ark_ff::UniformRand;

use crate::babybearext4::BabyBearExt4;

pub type OldBabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

#[test]
pub fn arith_tests() {
    let mut rng = create_seeded_rng();
    let a = BabyBearExt4::rand(&mut rng);
    let b = BabyBearExt4::rand(&mut rng);
    let a1 = OldBabyBearExt4::from_base_slice(&a.value);
    let b1 = OldBabyBearExt4::from_base_slice(&b.value);
    println!("{:?}", a);
    println!("{:?}", a1);
    assert_same(a1 + b1, a + b);
    assert_same(a1 * b1, a * b);
    assert_same(a1 - b1, a - b);
    assert_same(a1 / b1, a / b);
}

pub fn assert_same(a: OldBabyBearExt4, b: BabyBearExt4) {
    use openvm_stark_backend::p3_field::FieldExtensionAlgebra;
    for i in 0..4 {
        assert_eq!(
            FieldExtensionAlgebra::<BabyBear>::as_base_slice(&a)[i],
            b.value[i]
        )
    }
}
