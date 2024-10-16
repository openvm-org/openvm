#![feature(proc_macro_hygiene)]
use axvm_macros::axvm;

#[test]
fn main() {
    #[axvm]
    let z = addmod::<777>(x, y);
}
