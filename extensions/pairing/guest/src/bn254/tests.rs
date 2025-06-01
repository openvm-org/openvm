use crate::bn254::{BN254_PSEUDO_BINARY_ENCODING, BN254_SEED};

#[test]
fn test_bn254_pseudo_binary_encoding() {
    let mut x: i128 = 0;
    let mut power_of_2 = 1;
    for b in BN254_PSEUDO_BINARY_ENCODING.iter() {
        x += (*b as i128) * power_of_2;
        power_of_2 *= 2;
    }
    assert_eq!(x.unsigned_abs(), 6 * (BN254_SEED as u128) + 2);
}
