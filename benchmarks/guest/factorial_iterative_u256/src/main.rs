use core::hint::black_box;
use openvm as _;
use openvm_ruint::aliases::U256;

// This will overflow but that is fine
const N: u32 = 65_000;

pub fn main() {
    let mut acc = U256::from(1u32);
    let mut i = U256::from(N);
    while i > black_box(U256::ZERO) {
        acc *= i.clone();
        i -= U256::from(1u32);
    }
    black_box(acc);
}
