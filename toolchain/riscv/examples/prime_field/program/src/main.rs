#![no_main]
#![no_std]

use axvm_macros::axvm;

axvm::entry!(main);

pub fn main() {
    let x = [1u8; 32];
    let y = [2u8; 32];
    // Currently must alloc z first; not ideal
    let mut z = [0u8; 32];

    // the generic is a placeholder; real primes are bigints - what's the best way to specify? probably the proc macro should provide a list of constants from the axiom.toml
    // z will be auto converted to the pointer &mut z
    axvm!(z = addmod::<777>(&x, &y););
}
