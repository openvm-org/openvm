#![no_main]
#![no_std]

use core::hint::black_box;

use regex::Regex;

axvm::entry!(main);

pub fn main() {
    let input = axvm::io::read_vec();
    let input = core::str::from_utf8(&input).unwrap();

    let regex = Regex::new(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$").unwrap();
    let result = regex.is_match(input);
    let _ = black_box(result);
}
