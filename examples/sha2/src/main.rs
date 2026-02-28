// [!region imports]
#![cfg_attr(all(target_os = "zkvm", not(feature = "std")), no_main)]
#![cfg_attr(all(target_os = "zkvm", not(feature = "std")), no_std)]

extern crate alloc;

use alloc::{format, string::String};

use openvm_sha2::{Digest, Sha256, Sha384, Sha512};

openvm::entry!(main);
// [!endregion imports]

// [!region main]
fn println(s: String) {
    #[cfg(target_os = "zkvm")]
    openvm::io::println(s);
    #[cfg(not(target_os = "zkvm"))]
    println!("{}", s);
}
pub fn main() {
    let mut sha256 = Sha256::new();
    sha256.update(b"Hello, world!");
    sha256.update(b"some other input");
    let output = sha256.finalize();
    println(format!("SHA-256: {:?}", output));

    let mut sha512 = Sha512::new();
    sha512.update(b"Hello, world!");
    sha512.update(b"some other input");
    let output = sha512.finalize();
    println(format!("SHA-512: {:?}", output));

    let mut sha384 = Sha384::new();
    sha384.update(b"Hello, world!");
    sha384.update(b"some other input");
    let output = sha384.finalize();
    println(format!("SHA-384: {:?}", output));
}

// [!endregion main]
