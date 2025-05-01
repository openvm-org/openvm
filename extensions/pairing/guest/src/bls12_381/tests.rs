use crate::bls12_381::{BLS12_381_PSEUDO_BINARY_ENCODING, BLS12_381_SEED_ABS};

#[test]
fn test_bls12381_pseudo_binary_encoding() {
    let mut x: i128 = 0;
    let mut power_of_2 = 1;
    for b in BLS12_381_PSEUDO_BINARY_ENCODING.iter() {
        x += (*b as i128) * power_of_2;
        power_of_2 *= 2;
    }
    assert_eq!(x.unsigned_abs(), BLS12_381_SEED_ABS as u128);
}
