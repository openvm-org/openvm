use openvm::io::reveal_u32;

extern "C" {
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32;
}

const N: usize = 100_000;

fn main() {
    let mut a: [u8; N] = [20; N];
    let mut b: [u8; N] = [20; N];

    let indices = [
        66228, 93081, 58212, 28452, 60127, 51663, 42909, 93461, 44209, 29007,
    ];

    let mut total_res = 0;

    for i in 0..indices.len() {
        let idx = indices[i];
        a[idx] = 21;
        let res = unsafe { memcmp(a.as_ptr(), b.as_ptr(), N) };
        a[idx] = 20;

        total_res += res;
    }

    reveal_u32(total_res as u32, 0);
}
