// #![no_std]
// #![no_main]

use regex::Regex;

const PATTERN: &str = r"(?m)(\r\n|^)From:([^\r\n]+<)?(?P<email>[^<>]+)>?";

// openvm::entry!(main);

pub fn main() {
    let data = core::hint::black_box(include_str!("../regex_email.txt"));

    // Compile the regex
    let re = Regex::new(PATTERN).expect("Invalid regex");

    let caps = re.captures(data).expect("No match found.");
    let email = caps.name("email").expect("No email found.");
    // let email_hash = openvm_keccak256_guest::keccak256(email.as_str().as_bytes());

    // openvm::io::reveal_bytes32(email_hash);
}
