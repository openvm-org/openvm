//-----------------------------------------------------------------------
// Program: STREAM in Rust for rv32im
// Adapted from the original STREAM benchmark by John D. McCalpin
//-----------------------------------------------------------------------
// Copyright adaptation: Based on stream.c by John D. McCalpin
//-----------------------------------------------------------------------

use openvm as _;
use std::mem::size_of;

// Constants for the benchmark
const STREAM_ARRAY_SIZE: usize = 10_000;
const NTIMES: usize = 10;
const OFFSET: usize = 0;

// Scale factor
const SCALAR: f32 = 3.0;

// Main benchmark functions
fn stream_copy(a: &[f32], c: &mut [f32]) {
    for j in 0..STREAM_ARRAY_SIZE {
        c[j] = a[j];
    }
}

fn stream_scale(c: &[f32], b: &mut [f32], scalar: f32) {
    for j in 0..STREAM_ARRAY_SIZE {
        b[j] = scalar * c[j];
    }
}

fn stream_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    for j in 0..STREAM_ARRAY_SIZE {
        c[j] = a[j] + b[j];
    }
}

fn stream_triad(b: &[f32], c: &[f32], a: &mut [f32], scalar: f32) {
    for j in 0..STREAM_ARRAY_SIZE {
        a[j] = b[j] + scalar * c[j];
    }
}

fn check_stream_results(a: &[f32], b: &[f32], c: &[f32]) -> bool {
    // Expected values after running the benchmark
    let mut aj = 1.0f32;
    let mut bj = 2.0f32;
    let mut cj = 0.0f32;

    // a[] is modified during timing check
    aj = 2.0f32 * aj;

    // Execute timing loop
    let scalar = SCALAR;
    for _ in 0..NTIMES {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    // Accumulate deltas between observed and expected results
    let mut a_sum_err = 0.0f32;
    let mut b_sum_err = 0.0f32;
    let mut c_sum_err = 0.0f32;

    for j in 0..STREAM_ARRAY_SIZE {
        a_sum_err += (a[j] - aj).abs();
        b_sum_err += (b[j] - bj).abs();
        c_sum_err += (c[j] - cj).abs();
    }

    let a_avg_err = a_sum_err / STREAM_ARRAY_SIZE as f32;
    let b_avg_err = b_sum_err / STREAM_ARRAY_SIZE as f32;
    let c_avg_err = c_sum_err / STREAM_ARRAY_SIZE as f32;

    // For f32, we need a small epsilon
    let epsilon = if size_of::<f32>() == 4 {
        1.0e-6
    } else {
        1.0e-13
    };

    let mut err = 0;
    if a_avg_err / aj.abs() > epsilon {
        err += 1;
        println!(
            "Failed Validation on array a[], AvgRelAbsErr > epsilon ({})",
            epsilon
        );
        println!(
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}",
            aj,
            a_avg_err,
            a_avg_err / aj.abs()
        );

        let mut ierr = 0;
        for j in 0..STREAM_ARRAY_SIZE {
            if (a[j] / aj - 1.0).abs() > epsilon {
                ierr += 1;
                // Limited verbose output for debugging
                if ierr < 10 {
                    println!(
                        "         array a: index: {}, expected: {}, observed: {}, relative error: {}",
                        j, aj, a[j], (aj - a[j]).abs() / a_avg_err
                    );
                }
            }
        }
        println!("     For array a[], {} errors were found.", ierr);
    }

    if b_avg_err / bj.abs() > epsilon {
        err += 1;
        println!(
            "Failed Validation on array b[], AvgRelAbsErr > epsilon ({})",
            epsilon
        );
        println!(
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}",
            bj,
            b_avg_err,
            b_avg_err / bj.abs()
        );

        let mut ierr = 0;
        for j in 0..STREAM_ARRAY_SIZE {
            if (b[j] / bj - 1.0).abs() > epsilon {
                ierr += 1;
                if ierr < 10 {
                    println!(
                        "         array b: index: {}, expected: {}, observed: {}, relative error: {}",
                        j, bj, b[j], (bj - b[j]).abs() / b_avg_err
                    );
                }
            }
        }
        println!("     For array b[], {} errors were found.", ierr);
    }

    if c_avg_err / cj.abs() > epsilon {
        err += 1;
        println!(
            "Failed Validation on array c[], AvgRelAbsErr > epsilon ({})",
            epsilon
        );
        println!(
            "     Expected Value: {}, AvgAbsErr: {}, AvgRelAbsErr: {}",
            cj,
            c_avg_err,
            c_avg_err / cj.abs()
        );

        let mut ierr = 0;
        for j in 0..STREAM_ARRAY_SIZE {
            if (c[j] / cj - 1.0).abs() > epsilon {
                ierr += 1;
                if ierr < 10 {
                    println!(
                        "         array c: index: {}, expected: {}, observed: {}, relative error: {}",
                        j, cj, c[j], (cj - c[j]).abs() / c_avg_err
                    );
                }
            }
        }
        println!("     For array c[], {} errors were found.", ierr);
    }

    if err == 0 {
        println!(
            "Solution Validates: avg error less than {} on all three arrays",
            epsilon
        );
        true
    } else {
        false
    }
}

fn main() {
    println!("-------------------------------------------------------------");
    println!("STREAM benchmark - Rust Floating Point Version");
    println!("-------------------------------------------------------------");

    let bytes_per_word = size_of::<f32>();

    println!(
        "This system uses {} bytes per array element.",
        bytes_per_word
    );

    println!("-------------------------------------------------------------");
    println!(
        "Array size = {} (elements), Offset = {} (elements)",
        STREAM_ARRAY_SIZE, OFFSET
    );
    println!(
        "Memory per array = {:.1} MiB (= {:.1} GiB).",
        bytes_per_word as f64 * (STREAM_ARRAY_SIZE as f64) / 1024.0 / 1024.0,
        bytes_per_word as f64 * (STREAM_ARRAY_SIZE as f64) / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "Total memory required = {:.1} MiB (= {:.1} GiB).",
        (3.0 * bytes_per_word as f64) * (STREAM_ARRAY_SIZE as f64) / 1024.0 / 1024.0,
        (3.0 * bytes_per_word as f64) * (STREAM_ARRAY_SIZE as f64) / 1024.0 / 1024.0 / 1024.0
    );
    println!("Each kernel will be executed {} times.", NTIMES);
    println!(" The *best* time for each kernel (excluding the first iteration)");
    println!(" will be used to compute the reported bandwidth.");

    // Allocate arrays
    let mut a = vec![1.0; STREAM_ARRAY_SIZE + OFFSET];
    let mut b = vec![2.0; STREAM_ARRAY_SIZE + OFFSET];
    let mut c = vec![0.0; STREAM_ARRAY_SIZE + OFFSET];

    // Initialize arrays
    for j in 0..STREAM_ARRAY_SIZE {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    for j in 0..STREAM_ARRAY_SIZE {
        a[j] = 2.0 * a[j];
    }

    println!("-------------------------------------------------------------");
    println!("No timing information available for rv32im target");
    println!("Running operations without timing");
    println!("-------------------------------------------------------------");

    // --- MAIN LOOP --- repeat test cases NTIMES times
    let scalar = SCALAR;

    for _ in 0..NTIMES {
        // Copy
        stream_copy(&a, &mut c);

        // Scale
        stream_scale(&c, &mut b, scalar);

        // Add
        stream_add(&a, &b, &mut c);

        // Triad
        stream_triad(&b, &c, &mut a, scalar);
    }

    println!("-------------------------------------------------------------");
    println!("No timing results available for rv32im target");
    println!("-------------------------------------------------------------");

    // --- Check Results ---
    check_stream_results(&a, &b, &c);

    println!("-------------------------------------------------------------");
}
