use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{sign::Signed, One, Zero};

/// Jacobi returns the Jacobi symbol (x/y), either +1, -1, or 0.
/// The y argument must be an odd integer.
pub fn jacobi(x: &BigInt, y: &BigInt) -> isize {
    if !y.is_odd() {
        panic!(
            "invalid arguments, y must be an odd integer,but got {:?}",
            y
        );
    }

    let mut a = x.clone();
    let mut b = y.clone();
    let mut j = 1;

    if b.is_negative() {
        if a.is_negative() {
            j = -1;
        }
        b = -b;
    }

    loop {
        if b.is_one() {
            return j;
        }
        if a.is_zero() {
            return 0;
        }

        a = a.mod_floor(&b);
        if a.is_zero() {
            return 0;
        }

        // a > 0

        // handle factors of 2 in a
        let s = a.trailing_zeros().unwrap();
        if s & 1 != 0 {
            //let bmod8 = b.get_limb(0) & 7;
            let bmod8 = mod_2_to_the_k(&b, 3);
            if bmod8 == BigInt::from(3) || bmod8 == BigInt::from(5) {
                j = -j;
            }
        }

        let c = &a >> s; // a = 2^s*c

        // swap numerator and denominator
        if mod_2_to_the_k(&b, 2) == BigInt::from(3) && mod_2_to_the_k(&c, 2) == BigInt::from(3) {
            j = -j
        }

        a = b;
        b = c;
    }
}

fn mod_2_to_the_k(x: &BigInt, k: u32) -> BigInt {
    x & BigInt::from(2u32.pow(k) - 1)
}
#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;

    use super::*;

    #[test]
    fn test_jacobi() {
        let cases = [
            [0, 1, 1],
            [0, -1, 1],
            [1, 1, 1],
            [1, -1, 1],
            [0, 5, 0],
            [1, 5, 1],
            [2, 5, -1],
            [-2, 5, -1],
            [2, -5, -1],
            [-2, -5, 1],
            [3, 5, -1],
            [5, 5, 0],
            [-5, 5, 0],
            [6, 5, 1],
            [6, -5, 1],
            [-6, 5, 1],
            [-6, -5, -1],
        ];

        for case in cases.iter() {
            let x = BigInt::from_i64(case[0]).unwrap();
            let y = BigInt::from_i64(case[1]).unwrap();

            assert_eq!(case[2] as isize, jacobi(&x, &y), "jacobi({}, {})", x, y);
        }
    }
}
