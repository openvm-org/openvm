use num_bigint::BigUint;
use num_traits::Num;

fn parse_biguint_auto(s: &str) -> Option<BigUint> {
    let s = s.trim();
    if s.starts_with("0x") || s.starts_with("0X") {
        BigUint::from_str_radix(&s[2..], 16).ok()
    } else if s.starts_with("0b") || s.starts_with("0B") {
        BigUint::from_str_radix(&s[2..], 2).ok()
    } else {
        BigUint::from_str_radix(s, 10).ok()
    }
}
