#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
use axvm::io::read;

axvm::entry!(main);

#[derive(serde::Deserialize)]
struct Foo {
    bar: u32,
    baz: alloc::vec::Vec<u32>,
}

pub fn main() {
    let foo: Foo = read();
    if foo.baz.len() != 4 {
        axvm::process::panic();
    }
    if foo.bar != 42 {
        axvm::process::panic();
    }
}
