#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::reveal_u32;

extern "C" {
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

const N: usize = 10000;

openvm::entry!(main);

fn main() {
    // test equal arrays
    {
        let mut a: [u32; N] = [1; N];
        let mut b: [u32; N] = [1; N];

        let res = unsafe {
            memcmp(
                a.as_mut_ptr() as *const u8,
                b.as_mut_ptr() as *const u8,
                N * size_of::<u32>(),
            )
        };

        assert_eq!(res.signum(), 0);
    }

    // test unequal array
    {
        let a: [u32; N] = [1; N];
        let b: [u32; N] = [2; N];

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                N * size_of::<u32>(),
            )
        };

        assert_eq!(res.signum(), -1);
    }

    {
        let mut a: [u32; N] = [1; N];
        let mut b: [u32; N] = [1; N];

        a[N - 1] = 100;

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                N * size_of::<u32>(),
            )
        };

        assert_eq!(res.signum(), 1);
    }

    // test varied n parameter
    {
        let mut a: [u32; N] = [1; N];
        let mut b: [u32; N] = [1; N];

        a[N - 1] = 100;

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                (N - 1) * size_of::<u32>(),
            )
        };

        assert_eq!(res.signum(), 0);
    }

    // test incorrect size parameter
    {
        let mut a: [u32; N] = [1; N];
        let mut b: [u32; N] = [1; N];

        a[N - 1] = 100;

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                N * size_of::<u8>(),
            )
        };

        assert_eq!(res.signum(), 0);
    }

    {
        let mut a: [u32; N] = [1; N];
        let mut b: [u32; N] = [7; N];

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                0 * size_of::<u32>(),
            )
        };

        assert_eq!(res.signum(), 0);
    }

    // test other data types (u128, i64, strings)
    {
        let mut a: [u128; N] = [1; N];
        let mut b: [u128; N] = [7; N];

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                N * size_of::<u128>(),
            )
        };

        assert_eq!(res.signum(), -1);
    }

    {
        let mut a: [i64; N] = [7; N];
        let mut b: [i64; N] = [1; N];

        let res = unsafe {
            memcmp(
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                N * size_of::<i64>(),
            )
        };

        assert_eq!(res.signum(), 1);
    }

    {
        let a = "abc";
        let b = "abd";

        let res = unsafe { memcmp(a.as_ptr() as *const u8, b.as_ptr() as *const u8, a.len()) };

        assert_eq!(res.signum(), -1);
    }

    // test memory on the heap
    {
        let mut a: Vec<u8> = vec![1; N];
        let mut b: Vec<u8> = vec![1; N];
        a[N - 1] = 2;

        let res = unsafe { memcmp(a.as_ptr(), b.as_ptr(), N) };

        assert_eq!(res.signum(), 1);
    }

    // test that memcmp compares as unsigned bytes
    {
        let a: [i8; 2] = [-1, 0];
        let b: [i8; 2] = [1, 0];

        let res = unsafe { memcmp(a.as_ptr() as *const u8, b.as_ptr() as *const u8, 2) };

        assert_eq!(res.signum(), 1);
    }

    // test unaligned access
    {
        let buffer_a: [u8; 17] = [0; 17];
        let buffer_b: [u8; 17] = [0; 17];

        // Compare from unaligned offsets
        let res = unsafe { memcmp(buffer_a.as_ptr().offset(1), buffer_b.as_ptr().offset(1), 15) };

        assert_eq!(res.signum(), 0);
    }

    // test overlapping memory (same source)
    {
        let buffer: [u8; 10] = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6];

        let res = unsafe { memcmp(buffer.as_ptr(), buffer.as_ptr().offset(5), 5) };

        // [1, 2, 3, 4, 5] < [1, 2, 3, 4, 6]
        assert_eq!(res.signum(), -1);
    }

    {
        for pos in [0, 1, N / 2, N - 1] {
            let mut a: [u8; N] = [5; N];
            let mut b: [u8; N] = [5; N];
            b[pos] = 6;

            let res = unsafe { memcmp(a.as_ptr(), b.as_ptr(), pos + 1) };

            assert_eq!(res.signum(), -1);
        }
    }
}
