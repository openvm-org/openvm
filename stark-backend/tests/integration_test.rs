#![feature(trait_upcasting)]
#![allow(incomplete_features)]

/// Test utils
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::run_simple_test},
    utils,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

mod cached_lookup;
mod fib_air;
mod fib_selector_air;
mod fib_triples_air;
pub mod interaction;
mod partitioned_sum_air;

#[test]
fn test_single_fib_stark() {
    use fib_air::air::FibonacciAir;
    use fib_air::trace::generate_trace_rows;

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_fib_triples_stark() {
    use fib_triples_air::air::FibonacciAir;
    use fib_triples_air::trace::generate_trace_rows;

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let pis = [a, b, get_fib_number(n + 1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciAir;

    let trace = generate_trace_rows::<Val>(a, b, n);

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_single_fib_selector_stark() {
    use fib_selector_air::air::FibonacciSelectorAir;
    use fib_selector_air::trace::generate_trace_rows;

    let log_trace_degree = 3;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n = 1usize << log_trace_degree;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let pis = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air = FibonacciSelectorAir::new(sels, false);

    let trace = generate_trace_rows::<Val>(a, b, air.sels());

    run_simple_test(vec![&air], vec![trace], vec![pis]).expect("Verification failed");
}

#[test]
fn test_double_fib_starks() {
    use fib_air::air::FibonacciAir;
    use fib_selector_air::air::FibonacciSelectorAir;

    let log_n1 = 3;
    let log_n2 = 5;

    // Public inputs:
    let a = 0u32;
    let b = 1u32;
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;

    type Val = BabyBear;
    let sels: Vec<bool> = (0..n2).map(|i| i % 2 == 0).collect(); // Evens
    let pis1 = [a, b, get_fib_number(n1)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();
    let pis2 = [a, b, get_conditional_fib_number(&sels)]
        .map(BabyBear::from_canonical_u32)
        .to_vec();

    let air1 = FibonacciAir;
    let air2 = FibonacciSelectorAir::new(sels, false);

    let trace1 = fib_air::trace::generate_trace_rows::<Val>(a, b, n1);
    let trace2 = fib_selector_air::trace::generate_trace_rows::<Val>(a, b, air2.sels());

    run_simple_test(vec![&air1, &air2], vec![trace1, trace2], vec![pis1, pis2])
        .expect("Verification failed");
}

fn get_fib_number(n: usize) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn get_conditional_fib_number(sels: &[bool]) -> u32 {
    let mut a = 0;
    let mut b = 1;
    for &s in sels[0..sels.len() - 1].iter() {
        if s {
            let c = a + b;
            a = b;
            b = c;
        }
    }
    b
}
