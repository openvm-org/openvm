use num::{BigUint, Num};

pub fn str_to_u64_arr(s: &str, radix: usize) -> Vec<u64> {
    let x = BigUint::from_str_radix(s, radix as u32).unwrap();
    let x_bytes = x.to_bytes_be();
    let mut x_bytes = x_bytes
        .chunks(8)
        .map(|chunk| u64::from_be_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    x_bytes.reverse();
    x_bytes
}

pub fn str_to_hex_str(s: &str, radix: usize) -> Vec<String> {
    let a = str_to_u64_arr(s, radix);
    a.iter()
        .map(|num_le| {
            let hex_str = format!("0x{:x}", num_le);
            hex_str
        })
        .collect::<Vec<_>>()
}
