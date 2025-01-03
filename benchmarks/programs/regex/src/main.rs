#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::mem::transmute;

use regex::Regex;

openvm::entry!(main);

const PATTERN: &str = r"(?m)(\r\n|^)From:([^\r\n]+<)?(?P<email>[^<>]+)>?";

pub fn main() {
    let data = openvm::io::read_vec();
    let data = core::str::from_utf8(&data).expect("Invalid UTF-8");

    // Compile the regex
    let re = Regex::new(PATTERN).expect("Invalid regex");

    let caps = re.captures(data).expect("No match found.");
    let email = caps.name("email").expect("No email found.");
    let email_hash = openvm_keccak256_guest::keccak256(email.as_str().as_bytes());

    let email_hash = unsafe { transmute::<[u8; 32], [u32; 8]>(email_hash) };

    email_hash
        .into_iter()
        .enumerate()
        .for_each(|(i, x)| openvm::io::reveal(x, i));
}
