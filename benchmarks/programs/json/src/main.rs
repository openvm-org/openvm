#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::result::Result;
use axvm_json_program::UserProfile;
use serde_json_core::de::from_str;

axvm::entry!(main);

fn main() {
    let json_string = axvm::io::read_vec();
    let json_string = core::str::from_utf8(&json_string).expect("Failed to convert to string");

    let deserialized: Result<(UserProfile, usize), _> = from_str(json_string);
    deserialized.expect("Failed to deserialize JSON");
}
