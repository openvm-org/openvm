#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use alloc::vec;

use openvm::io::read;
extern crate openvm;

use openvm::platform::memory::sys_alloc_aligned;
extern crate openvm_rv32im_guest;

extern crate alloc;

openvm::entry!(main);

fn main() {
    let n: u32 = read();

    let c = n * 2;

    let v1 = vec![c as u32];
    let old_v1_0 = v1[0];

    let heap_ptr = unsafe { sys_alloc_aligned(4, 4) };
    let missing = u32::MAX - (heap_ptr as u32);
    let _alloc_overflow = unsafe { sys_alloc_aligned(missing as usize, 4) };
    let _alloc_overlap = unsafe { sys_alloc_aligned((v1.as_ptr() as usize) - 4, 4) };

    let v2 = vec![(c + 1) as u32];

    assert_eq!(v1, v2);
    assert!(old_v1_0 != v1[0]);
}
