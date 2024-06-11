/// Converts byte array to an N-size byte array, big-endian, left-padded with zeros.
pub fn bytes_to_fixed_bytes_be_vec<const N: usize>(bytes: &[u8]) -> Vec<u8> {
    let mut result = [0u8; N];
    let start = N.saturating_sub(bytes.len());
    result[start..].copy_from_slice(&bytes[..N.min(bytes.len())]);
    result.to_vec()
}

/// Converts a string to an N-size byte array, big-endian, left-padded with zeros.
/// If the string starts with "0x", it is removed before conversion.
/// If the string does not start with "0x", it is parsed as a number or string
pub fn string_to_fixed_bytes_be_vec<const N: usize>(s: String) -> Vec<u8> {
    if s.starts_with("0x") {
        let hex_str = s.strip_prefix("0x").unwrap();
        let bytes_vec = hex::decode(hex_str).unwrap();
        let bytes = bytes_vec.as_slice();
        bytes_to_fixed_bytes_be_vec::<N>(bytes)
    } else {
        let num = s.parse::<u64>();
        match num {
            Ok(num) => {
                let bytes = num.to_be_bytes();
                bytes_to_fixed_bytes_be_vec::<N>(&bytes)
            }
            Err(_) => {
                let bytes = s.as_bytes();
                bytes_to_fixed_bytes_be_vec::<N>(bytes)
            }
        }
    }
}
