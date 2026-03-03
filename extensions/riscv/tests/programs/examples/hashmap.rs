#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use std::collections::HashMap;

openvm::entry!(main);

fn main() {
    let mut map = HashMap::new();
    map.insert(1, 2);
    map.insert(2, 8);
    assert!(map.get(&1) == Some(&2));
    assert!(map.get(&2) == Some(&8));
    assert!(!map.contains_key(&3));
    println!("{:?}", map.get(&1).unwrap());
}
