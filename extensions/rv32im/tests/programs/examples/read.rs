#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
use openvm::io::read;

#[derive(serde::Deserialize)]
struct Foo {
    bar: u32,
    baz: alloc::vec::Vec<u32>,
}

#[allow(clippy::disallowed_names)]
#[openvm::entry]
pub fn main() {
    let foo: Foo = read();
    if foo.baz.len() != 4 {
        openvm::process::panic();
    }
    if foo.bar != 42 {
        openvm::process::panic();
    }
}
