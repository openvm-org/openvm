/// Converts byte vector to a byte vector of a target size, big-endian, left-padded with zeros.
pub fn bytes_to_fixed_bytes_be_vec(bytes: &[u8], target_bytes: usize) -> Vec<u8> {
    let bytes_len = bytes.len();
    if bytes_len > target_bytes {
        panic!(
            "Data size exceeds the target size ({} > {})",
            bytes_len, target_bytes
        );
    }
    let zeros_len = target_bytes - bytes_len;
    let mut fixed_bytes = vec![0; zeros_len];
    fixed_bytes.extend_from_slice(bytes);
    if fixed_bytes.len() != target_bytes {
        panic!(
            "Invalid data size: {} for this codec, expected: {}",
            fixed_bytes.len(),
            target_bytes
        );
    }
    fixed_bytes
}

/// Converts a string to a byte vector of a target size, big-endian, left-padded with zeros.
/// If the string starts with "0x", it is removed before conversion.
/// If the string does not start with "0x", it is parsed as a number or string
pub fn string_to_fixed_bytes_be_vec(s: String, target_bytes: usize) -> Vec<u8> {
    if s.starts_with("0x") {
        let hex_str = s.strip_prefix("0x").unwrap();
        let hex_str = if hex_str.len() % 2 != 0 {
            let formatted_hex_str = format!("0{}", hex_str);
            formatted_hex_str
        } else {
            hex_str.to_string()
        };
        let bytes_vec = hex::decode(hex_str).unwrap();
        let bytes = bytes_vec.as_slice();
        bytes_to_fixed_bytes_be_vec(bytes, target_bytes)
    } else {
        let num = s.parse::<u64>();
        match num {
            Ok(num) => {
                let bytes = num.to_be_bytes();
                bytes_to_fixed_bytes_be_vec(&bytes, target_bytes)
            }
            Err(_) => {
                let bytes = s.as_bytes();
                bytes_to_fixed_bytes_be_vec(bytes, target_bytes)
            }
        }
    }
}
