#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::{hint::black_box, mem::transmute};

use regex::Regex;

axvm::entry!(main);

// use std::fs::File;
// use std::io::{self, Write};

// fn write_string_as_bytes_to_file(filename: &str, content: &str) -> io::Result<()> {
//     let mut file = File::create(filename)?;  // Create or overwrite the file
//     file.write_all(content.as_bytes())?;     // Write bytes directly
//     Ok(())
// }

// fn main() -> io::Result<()> {
//     let content = DATA;
//     write_string_as_bytes_to_file("output.bin", content)?; // Writes binary bytes to file
//     Ok(())
// }

const pattern: &str = r"(?m)(\r\n|^)From:([^\r\n]+<)?(?P<email>[^<>]+)>?";

pub fn main() {
    let data = axvm::io::read_vec();
    let data = core::str::from_utf8(&data).expect("Invalid UTF-8");

    // Compile the regex
    let re = Regex::new(pattern).expect("Invalid regex");

    let caps = re.captures(data).expect("No match found.");
    // black_box(caps);
    let email = caps.name("email").expect("No email found.");
    let email_hash = axvm::intrinsics::keccak256(email.as_str().as_bytes());

    let email_hash = unsafe { transmute::<[u8; 32], [u32; 8]>(email_hash) };

    email_hash
        .into_iter()
        .enumerate()
        .for_each(|(i, x)| axvm::io::reveal(x, i));
}
