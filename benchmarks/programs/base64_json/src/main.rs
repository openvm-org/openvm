#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::result::Result;

use axvm_json_program::UserProfile;
use base64::engine::{general_purpose, Engine};
use serde_json_core::de::from_str;

axvm::entry!(main);

fn main() {
    let data_b64 = axvm::io::read_vec();
    let data_b64 = core::str::from_utf8(&data_b64).expect("Invalid UTF-8");

    let decoded = general_purpose::STANDARD
        .decode(data_b64)
        .expect("Failed to decode base64");
    let json_string = core::str::from_utf8(&decoded).expect("Failed to decode base64");
    let json_string = json_string.replace("\r\n", "\n");

    let deserialized: Result<(UserProfile, usize), _> = from_str(&json_string);
    deserialized.expect("Failed to deserialize JSON");
}
