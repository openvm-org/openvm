use core::hint::black_box;
use openvm as _;
use openvm_bigint_guest::U256;

// This will overflow but that is fine
const N: u32 = 65_000;

pub fn main() {
    let mut acc = U256::from_u32(1);
    let mut i = U256::from_u32(N);
    while i > black_box(U256::ZERO) {
        acc *= i.clone();
        i -= U256::from_u32(1);
    }
    black_box(acc);
}
