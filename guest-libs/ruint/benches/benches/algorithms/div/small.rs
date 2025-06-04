use crate::prelude::*;
use ruint::algorithms::div::*;

pub fn group(criterion: &mut Criterion) {
    bench_div_2x1_ref(criterion);
    bench_div_2x1_mg10(criterion);
    bench_div_3x2_ref(criterion);
    bench_div_3x2_mg10(criterion);
}

fn bench_div_2x1_ref(criterion: &mut Criterion) {
    let mut rng = rng();
    criterion.bench_function("algo/div/2x1/ref", move |bencher| {
        bencher.iter_batched(
            || {
                let q: u64 = rng.random();
                let r: u64 = rng.random();
                let d = rng.random::<u64>() | (1 << 63);
                let r = r % d;
                let n = u128::from(q) * u128::from(d) + u128::from(r);
                (n, d)
            },
            |(u, d)| black_box(div_2x1_ref(u, d)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_div_2x1_mg10(criterion: &mut Criterion) {
    let mut rng = rng();
    criterion.bench_function("algo/div/2x1/mg10", move |bencher| {
        bencher.iter_batched(
            || {
                let q: u64 = rng.random();
                let r: u64 = rng.random();
                let d = rng.random::<u64>() | (1 << 63);
                let r = r % d;
                let n = u128::from(q) * u128::from(d) + u128::from(r);
                let v = reciprocal(d);
                (n, d, v)
            },
            |(u, d, v)| black_box(div_2x1_mg10(u, d, v)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_div_3x2_ref(criterion: &mut Criterion) {
    let mut rng = rng();
    criterion.bench_function("algo/div/3x2/ref", move |bencher| {
        bencher.iter_batched(
            || {
                let q: u64 = rng.random();
                let r: u128 = rng.random();
                let d = rng.random::<u128>() | (1 << 127);
                let r = r % d;
                let (n21, n0) = {
                    let d1 = (d >> 64) as u64;
                    let d0 = d as u64;
                    let r1 = (r >> 64) as u64;
                    let r0 = r as u64;
                    // n = q * d + r
                    let n10 = u128::from(q) * u128::from(d0) + u128::from(r0);
                    let n0 = n10 as u64;
                    let n21 = (n10 >> 64) + u128::from(q) * u128::from(d1) + u128::from(r1);
                    (n21, n0)
                };
                (n21, n0, d)
            },
            |(n21, n0, d)| black_box(div_3x2_ref(n21, n0, d)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_div_3x2_mg10(criterion: &mut Criterion) {
    let mut rng = rng();
    criterion.bench_function("algo/div/3x2/mg10", move |bencher| {
        bencher.iter_batched(
            || {
                let q: u64 = rng.random();
                let r: u128 = rng.random();
                let d = rng.random::<u128>() | (1 << 127);
                let r = r % d;
                let (n21, n0) = {
                    let d1 = (d >> 64) as u64;
                    let d0 = d as u64;
                    let r1 = (r >> 64) as u64;
                    let r0 = r as u64;
                    // n = q * d + r
                    let n10 = u128::from(q) * u128::from(d0) + u128::from(r0);
                    let n0 = n10 as u64;
                    let n21 = (n10 >> 64) + u128::from(q) * u128::from(d1) + u128::from(r1);
                    (n21, n0)
                };
                let v = reciprocal_2(d);
                (n21, n0, d, v)
            },
            |(n21, n0, d, v)| black_box(div_3x2_mg10(n21, n0, d, v)),
            BatchSize::SmallInput,
        );
    });
}
