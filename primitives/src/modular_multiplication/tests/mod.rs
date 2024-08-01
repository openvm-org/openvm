use std::sync::Arc;

use num_bigint::BigUint;
use num_traits::One;
use rand::RngCore;

use afs_test_utils::config::baby_bear_poseidon2::run_simple_test_no_pis;
use afs_test_utils::utils::create_seeded_rng;

use crate::modular_multiplication::air::ModularMultiplicationAir;
use crate::modular_multiplication::columns::ModularMultiplicationCols;
use crate::range_gate::RangeCheckerGateChip;

fn secp256k1_prime() -> BigUint {
    let mut result = BigUint::one() << 256;
    for power in [32, 9, 8, 7, 6, 4, 1] {
        result -= BigUint::one() << power;
    }
    result
}

fn default_air() -> ModularMultiplicationAir {
    ModularMultiplicationAir::new(secp256k1_prime(), 256, 31, 15, 0, 27, 9, 9, 15)
}

#[test]
fn test_flatten_fromslice_roundtrip() {
    let air = default_air();

    let num_cols = ModularMultiplicationCols::<usize>::get_width(&air);
    let all_cols = (0..num_cols).collect::<Vec<usize>>();

    let cols_numbered = ModularMultiplicationCols::<usize>::from_slice(&all_cols, &air);
    let flattened = cols_numbered.flatten();

    for (i, col) in flattened.iter().enumerate() {
        assert_eq!(*col, all_cols[i]);
    }

    assert_eq!(num_cols, flattened.len());
}

#[test]
fn test_modular_multiplication() {
    //std::env::set_var("RUST_BACKTRACE", "1");
    let air = default_air();
    let num_digits = 8;
    let range_checker = Arc::new(RangeCheckerGateChip::new(air.range_bus, 1 << air.decomp));

    let mut rng = create_seeded_rng();
    let a_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let a = BigUint::new(a_digits);
    let b_digits = (0..num_digits).map(|_| rng.next_u32()).collect();
    let b = BigUint::new(b_digits);
    // if these are not true then trace is not guaranteed to be verifiable
    assert!(a < secp256k1_prime());
    assert!(b < secp256k1_prime());

    let trace = air.generate_trace(vec![(a, b)], range_checker.clone());
    let range_trace = range_checker.generate_trace();
    run_simple_test_no_pis(vec![&air, &range_checker.air], vec![trace, range_trace])
        .expect("Verification failed");
}
