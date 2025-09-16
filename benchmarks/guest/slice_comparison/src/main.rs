use openvm::io::reveal_u32;

/*
this memcmp will be linked with the memcmp.s assembly implementation by the global_asm! macro defined in crates/toolchain/openvm/src/lib.rs
*/
extern "C" {
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

const N: usize = 32_768;

fn main() {
    let mut a: [u128; N] = [u128::MAX - 1; N];
    let mut b: [u128; N] = [u128::MAX - 1; N];

    let mut total_res: i32 = 0;

    // equal case
    {
        let res = unsafe {
            memcmp(
                a.as_mut_ptr() as *const u8,
                b.as_mut_ptr() as *const u8,
                N * size_of::<u128>(),
            )
        };
        total_res += res.signum();
    }

    let indices = [N - 1, N - 10, N - 100, N - 10000];

    // not equal case
    for i in 0..indices.len() {
        let idx = indices[i];
        a[idx] = u128::MAX;
        let res = unsafe {
            memcmp(
                a.as_mut_ptr() as *const u8,
                b.as_mut_ptr() as *const u8,
                N * size_of::<u128>(),
            )
        };
        a[idx] = u128::MAX - 1;
        total_res += res.signum();
    }

    reveal_u32(total_res as u32, 0);
}
