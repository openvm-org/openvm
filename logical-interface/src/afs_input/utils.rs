use alloy_primitives::U256;

pub fn parse_string_to_bytes(s: String) -> Vec<u8> {
    let s_str = s.as_str();
    if s_str.starts_with("0x") {
        hex::decode(s_str.strip_prefix("0x").unwrap()).unwrap()
    } else if let Ok(s_u256) = U256::from_str_radix(s_str, 10) {
        s_u256.to_be_bytes_vec()
    } else {
        s_str.as_bytes().to_vec()
    }
}
