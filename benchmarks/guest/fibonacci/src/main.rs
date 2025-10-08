use core::ptr;
#[cfg(test)]
use rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(test)]
use test_case::test_case;

openvm::entry!(main);

#[no_mangle]
pub fn append<T>(dst: &mut [T], src: &mut [T], shift: usize) {
    let src_len = src.len();
    let _dst_len = dst.len();

    unsafe {
        // The call to add is always safe because `Vec` will never
        // allocate more than `isize::MAX` bytes.
        let dst_ptr = dst.as_mut_ptr().wrapping_add(shift);
        let src_ptr = src.as_ptr();
        println!("dst_ptr: {}", dst_ptr as usize); // these have the same pointer destination (basically), in between runs
        println!("src_ptr: {}", src_ptr as usize);
        println!("src_len: {}", src_len);

        ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_len);
    }
}
#[cfg_attr(test, test_case(0, 100, 42))] // shift, length
#[cfg_attr(test, test_case(1, 100, 42))]
#[cfg_attr(test, test_case(2, 100, 42))]
#[cfg_attr(test, test_case(3, 100, 42))]
fn test1(shift: usize, length: usize, seed: u64) {
    let n: usize = length;

    let mut a: Vec<u8> = vec![0; 2 * n];
    let mut b: Vec<u8> = vec![2; n];

    let mut rng = StdRng::seed_from_u64(seed); // fixed seed
    for i in 0..n {
        b[i] = rng.gen::<u8>();
    }
    println!("b: {:?}", b);
    append(&mut a[..], &mut b[..], shift);

    println!("a: {:?}", a);
    println!("b: {:?}", b);
    let mut idx = 0;
    for i in 0..(2 * n) {
        if i < shift || i >= shift + b.len() {
            assert_eq!(a[i], 0);
        } else {
            assert_eq!(a[i], b[idx]);
            idx += 1;
        }
    }
}
pub fn main() {
    const n: usize = 32;

    let mut a: [u8; 2 * n] = [0; 2 * n];
    let mut b: [u8; n] = [2; n];

    let shift: usize = 1;
    for i in 0..n {
        b[i] = (7 * i + 13) as u8;
    }
    println!("b: {:?}", b);
    append(&mut a, &mut b, shift);

    println!("a: {:?}", a);
    println!("b: {:?}", b);
    let mut idx = 0;
    for i in 0..2 * n {
        if i < shift || i >= shift + b.len() {
            assert_eq!(a[i], 0);
        } else {
            assert_eq!(a[i], b[idx]);
            idx += 1;
        }
    }
}
