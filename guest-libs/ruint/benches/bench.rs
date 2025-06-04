mod benches;

#[allow(unused_imports)]
mod prelude {
    pub use criterion::{black_box, BatchSize, Criterion};
    pub use proptest::{
        arbitrary::Arbitrary,
        strategy::{Strategy, ValueTree},
        test_runner::TestRunner,
    };
    pub use rand_09::{self as rand, prelude::*, rng};
    pub use ruint::{const_for, nlimbs, uint, Bits, Uint, UintTryFrom, UintTryTo};
}

fn main() {
    let mut c = criterion::Criterion::default().configure_from_args();
    benches::group(&mut c);
    c.final_summary();
}
