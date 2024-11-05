#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::intrinsics::IntModN;

axvm::entry!(main);

pub fn main() {
    let mut pow = IntModN::modulus();
    pow[0] -= 2;

    let mut a = IntModN::from_u32(1234);
    let mut res = IntModN::from_u32(1);
    let inv = &res / &a;

    for i in 0..32 {
        for j in 0..8 {
            if pow[i] & (1 << j) != 0 {
                res = res * &a;
            }
            a *= a.clone();
        }
    }

    // https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
    if res != inv {
        axvm::process::panic();
    }

    let two = IntModN::from_u32(2);
    let minus_two = IntModN::from_bytes(pow);

    if (res - minus_two) != (inv + two) {
        axvm::process::panic();
    }
}
