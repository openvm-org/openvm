pub mod bls12_381;
pub mod bn254;
pub mod k256;
pub mod p256;

use num_bigint::BigUint;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    K256 = 0,
    P256 = 1,
    BN254 = 2,
    BLS12_381 = 3,
    Generic = 4,
}

pub fn get_curve_type_from_modulus(modulus: &BigUint) -> CurveType {
    if modulus == &k256::modulus() {
        return CurveType::K256;
    }

    if modulus == &p256::modulus() {
        return CurveType::P256;
    }

    if modulus == &bn254::modulus() {
        return CurveType::BN254;
    }

    if modulus == &bls12_381::modulus() {
        return CurveType::BLS12_381;
    }

    CurveType::Generic
}

pub fn get_curve_type(modulus: &BigUint, a_coeff: &BigUint) -> CurveType {
    if modulus == &k256::modulus() && a_coeff == &k256::a() {
        return CurveType::K256;
    }

    if modulus == &p256::modulus() && a_coeff == &p256::a() {
        return CurveType::P256;
    }

    if modulus == &bn254::modulus() && a_coeff == &bn254::a() {
        return CurveType::BN254;
    }

    if modulus == &bls12_381::modulus() && a_coeff == &bls12_381::a() {
        return CurveType::BLS12_381;
    }

    CurveType::Generic
}
