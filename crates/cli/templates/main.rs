#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

use openvm::process::exit;

openvm::entry!(main);

pub fn main() {
    exit();
}
