use core::ptr;

openvm::entry!(main);

/// Moves all the elements of `src` into `dst`, leaving `src` empty.
#[no_mangle]
pub fn append<T>(dst: &mut [T], src: &mut [T], shift: usize) {
    let src_len = src.len();
    let dst_len = dst.len();

    unsafe {
        // The call to add is always safe because `Vec` will never
        // allocate more than `isize::MAX` bytes.
        let dst_ptr = dst.as_mut_ptr().wrapping_add(shift);
        let src_ptr = src.as_ptr();
        println!("dst_ptr: {:?}", dst_ptr);
        println!("src_ptr: {:?}", src_ptr);
        println!("src_len: {:?}", src_len);

        // The two regions cannot overlap because mutable references do
        // not alias, and two different vectors cannot own the same
        // memory.
        ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_len);
    }
}

pub fn main() {
    let mut a: [u8; 1000] = [1; 1000];
    let mut b: [u8; 500] = [2; 500];

    let shift: usize = 0;
    append(&mut a, &mut b, shift);

    for i in 0..1000 {
        if i < shift || i >= shift + b.len() {
            assert_eq!(a[i], 1);
        } else {
            assert_eq!(a[i], 2);
        }
    }

    println!("a: {:?}", a);
    println!("b: {:?}", b);
}