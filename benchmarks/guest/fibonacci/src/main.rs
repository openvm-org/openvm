use core::ptr;

openvm::entry!(main);

// Explicitly use the C memcpy function to ensure we're using custom memcpy
extern "C" {
    // in rust, u8 is 1 byte of memory
    fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

/// Test function that explicitly calls memcpy to verify custom implementation
pub fn test_custom_memcpy(dst: &mut [u8], src: &[u8], shift: usize) {
    let src_len = src.len();
    let dst_len = dst.len();

    // Bounds checking
    if shift + src_len > dst_len {
        return; // Just return on bounds error
    }

    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(shift);
        let src_ptr = src.as_ptr();

        // This will definitely use our custom memcpy implementation
        memcpy(dst_ptr, src_ptr, src_len);
    }
}

pub fn main() {
    let mut a: [u8; 1000] = [1; 1000];
    for i in 0..1000 {
        a[i] = 1 as u8;
    }
    let mut b: [u8; 500] = [2; 500];
    for i in 0..500 {
        b[i] = 2 as u8;
    }

    let shift: usize = 3;

    // Test the custom memcpy
    test_custom_memcpy(&mut a, &b, shift);

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
/*
ok what lolll
memcpy works (??)
*/
