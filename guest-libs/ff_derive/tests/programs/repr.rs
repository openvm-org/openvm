use ff::PrimeField;
use num_bigint::BigUint;

pub fn check<F: PrimeField>(big_endian: bool, storage_bytes: usize) {
    let modulus = BigUint::parse_bytes(F::MODULUS.trim_start_matches("0x").as_bytes(), 16).unwrap();
    let width = F::Repr::default().as_ref().len();
    let mut cases = alloc::vec![
        BigUint::from(0u8),
        BigUint::from(1u8),
        &modulus - 1u8,
        modulus.clone(),
        &modulus + 1u8,
        (BigUint::from(1u8) << (width * 8)) - 1u8,
    ];
    if width > storage_bytes {
        cases.push(BigUint::from(1u8) << (storage_bytes * 8));
        cases.push((BigUint::from(1u8) << (storage_bytes * 8)) + 1u8);
        cases.push((BigUint::from(1u8) << ((width - 1) * 8)) + 1u8);
    }
    for value in cases {
        let mut repr = F::Repr::default();
        let bytes = value.to_bytes_le();
        if big_endian {
            for (dst, src) in repr.as_mut().iter_mut().rev().zip(bytes) {
                *dst = src;
            }
        } else {
            repr.as_mut()[..bytes.len()].copy_from_slice(&bytes);
        }
        let expected = value < modulus;
        let parsed = F::from_repr(repr);
        let vartime = F::from_repr_vartime(repr);
        assert_eq!(bool::from(parsed.is_some()), expected);
        assert_eq!(vartime.is_some(), expected);
        if expected {
            assert_eq!(parsed.unwrap().to_repr().as_ref(), repr.as_ref());
            assert_eq!(vartime.unwrap().to_repr().as_ref(), repr.as_ref());
        } else {
            assert_eq!(parsed.unwrap_or(F::ONE), F::ONE);
        }
    }
}
