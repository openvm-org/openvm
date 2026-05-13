// Exercises libstd compiled for riscv64im-unknown-openvm-elf: HashMap and BTreeMap
// from std::collections, plus a Vec<String> allocation, all going through the
// PAL (sys_alloc_aligned, sys_rand if HashMap seeds, etc.).
#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

#[cfg(feature = "std")]
pub fn main() {
    use std::collections::{BTreeMap, HashMap};

    let mut hm: HashMap<&str, u32> = HashMap::new();
    hm.insert("hello", 1);
    hm.insert("world", 2);
    assert_eq!(hm.get("hello"), Some(&1));
    assert_eq!(hm.get("world"), Some(&2));
    assert!(!hm.contains_key("missing"));

    let mut bt: BTreeMap<u32, u32> = BTreeMap::new();
    for i in 0..16u32 {
        bt.insert(i, i.wrapping_mul(3));
    }
    let sum: u32 = bt.values().copied().sum();
    assert_eq!(sum, (0..16u32).map(|i| i.wrapping_mul(3)).sum());

    let owned: Vec<String> = (0..4).map(|i| format!("item-{i}")).collect();
    assert_eq!(owned[3], "item-3");
}

#[cfg(not(feature = "std"))]
pub fn main() {
    // No-op when built without std: the test_rv64im_std test case always passes
    // --features std, so this branch only exists to keep the file compilable
    // under default `cargo check`/`clippy`.
}
