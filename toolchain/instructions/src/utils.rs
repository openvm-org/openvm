use num_bigint_dig::BigUint;
use num_traits::Num;
use p3_field::Field;

pub fn parse_biguint_auto(s: &str) -> Option<BigUint> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        BigUint::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with("0b") || s.starts_with("0B") {
        BigUint::from_str_radix(&s[2..], 2).ok()
    } else {
        BigUint::from_str_radix(s, 10).ok()
    }
}

pub fn isize_to_field<F: Field>(value: isize) -> F {
    if value < 0 {
        return F::NEG_ONE * F::from_canonical_usize(value.unsigned_abs());
    }
    F::from_canonical_usize(value as usize)
}
