pub fn bytes_to_fixed_bytes_be<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut result = [0u8; N];
    result[..bytes.len()].copy_from_slice(bytes);
    result
}

/// Converts a string to a fixed-size byte array, big-endian, left-padded with zeros.
/// If the string starts with "0x", it is removed before conversion.
/// If the string does not start with "0x", it is parsed as a number or string
pub fn string_to_fixed_bytes_be<const N: usize>(s: String) -> [u8; N] {
    if s.starts_with("0x") {
        let bytes = s.strip_prefix("0x").unwrap().as_bytes();
        bytes_to_fixed_bytes_be(bytes)
    } else {
        let num = s.parse::<u64>();
        match num {
            Ok(num) => {
                let bytes = num.to_be_bytes();
                bytes_to_fixed_bytes_be(&bytes)
            }
            Err(_) => {
                let bytes = s.as_bytes();
                bytes_to_fixed_bytes_be(bytes)
            }
        }
    }
}
