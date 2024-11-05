#![no_main]
#![no_std]

use core::{hint::black_box, mem::transmute};

use regex::Regex;

axvm::entry!(main);

pub fn main() {
    let pattern = r"(?m)(\r\n|^)From:([^\r\n]+<)?(?P<email>[^<>]+)>?";
    let data = axvm::io::read_vec();
    let data = core::str::from_utf8(&data).expect("Invalid UTF-8");

    // Compile the regex
    let re = Regex::new(pattern).expect("Invalid regex");

    let caps = re.captures(data).expect("No match found.");
    let email = caps.name("email").expect("No email found.");
    let email_hash = axvm::intrinsics::keccak256(email.as_str().as_bytes());
    let email_hash = unsafe { transmute::<[u8; 32], [u32; 8]>(email_hash) };

    email_hash.into_iter().enumerate().for_each(|(i, x)| {
        // axvm::io::reveal(x, i);
        black_box(x);
        black_box(i);
    });
}
