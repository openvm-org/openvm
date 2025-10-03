use core::ptr;

openvm::entry!(main);

#[no_mangle]
pub fn append<T>(dst: &mut [T], src: &mut [T], shift: usize) {
    let src_len = src.len();
    let dst_len = dst.len();

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

pub fn main() {
    const n: usize = 32;

    let mut a: [u8; 2 * n] = [0; 2 * n];
    let mut b: [u8; n] = [2; n];

    let shift: usize = 1;
    for i in 0..n {
        b[i] = i as u8 + 1 as u8;
    }
    println!("b: {:?}", b);
    append(&mut a, &mut b, shift);
    let mut idx = 0;
    for i in 0..2 * n {
        if i < shift || i >= shift + b.len() {
            assert_eq!(a[i], 0);
        } else {
            assert_eq!(a[i], b[idx]);
            idx += 1;
        }
    }

    println!("a: {:?}", a);
    println!("b: {:?}", b);
}
