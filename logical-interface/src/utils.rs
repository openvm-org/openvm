use alloy_primitives::{U256, U512};

pub trait MemorySize {
    const MEMORY_SIZE: usize;
}

macro_rules! impl_memory_size_for_uint {
    ($($t:ty),*) => {
        $(
            impl MemorySize for $t {
                const MEMORY_SIZE: usize = std::mem::size_of::<$t>();
            }
        )*
    }
}

impl_memory_size_for_uint!(u8, u16, u32, u64, u128, U256, U512);

macro_rules! impl_memory_size_for_array {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> MemorySize for [$t; N] {
                const MEMORY_SIZE: usize = N * std::mem::size_of::<$t>();
            }
        )*
    }
}

impl_memory_size_for_array!(u8, u16, u32, u64, u128, U256, U512);

/// Converts byte vector to a byte vector of a target size, big-endian, left-padded with zeros.
pub fn bytes_to_be_vec(bytes: &[u8], size: usize) -> Vec<u8> {
    let truncated_bytes = if bytes.len() > size {
        &bytes[bytes.len() - size..]
    } else {
        bytes
    };
    let zeros_len = size - truncated_bytes.len();
    let mut fixed_bytes = vec![0; zeros_len];
    fixed_bytes.extend_from_slice(truncated_bytes);
    fixed_bytes
}

pub fn uint_to_be_vec(value: usize, size: usize) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    bytes_to_be_vec(&bytes, size)
}

/// Converts a string to a byte vector of a target size, big-endian, left-padded with zeros.
/// If the string starts with "0x", it is removed before conversion.
/// If the string does not start with "0x", it is parsed as a number or string
pub fn string_to_be_vec(s: String, size: usize) -> Vec<u8> {
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
        bytes_to_be_vec(bytes, size)
    } else {
        let num = s.parse::<u64>();
        match num {
            Ok(num) => {
                let bytes = num.to_be_bytes();
                bytes_to_be_vec(&bytes, size)
            }
            Err(_) => {
                let bytes = s.as_bytes();
                bytes_to_be_vec(bytes, size)
            }
        }
    }
}

/// Converts a byte vector to a vector of Page elements, where each Page element is a u32
/// that represents a 31-bit field element and contains 2 big-endian bytes from the byte vector.
/// 2 MSBs of each Page element are set to 0 and 2 LSBs are set to two
/// bytes from the byte vector.
pub fn fixed_bytes_to_field_vec(value: Vec<u8>) -> Vec<u32> {
    if value.len() == 1 {
        return vec![value[0] as u32];
    } else if value.len() % 2 != 0 {
        panic!("Invalid field size: {}", value.len());
    }
    value
        .chunks(2)
        .map(|chunk| {
            let mut bytes = [0u8; 2];
            bytes.copy_from_slice(chunk);
            u16::from_be_bytes(bytes) as u32
        })
        .collect()
}

pub fn parse_literal_to_bytes(s: &str) -> Vec<u8> {
    let s = stringify!(s);
    if s.starts_with('"') {
        // handle str
        let s = s.strip_prefix('"').unwrap();
        let s = s.strip_suffix('"').unwrap();
        if s.starts_with("0x") {
            let hex_str = s.strip_prefix("0x").unwrap();
            hex::decode(hex_str).unwrap()
        } else {
            bytes_to_be_vec(s.as_bytes(), s.len())
        }
    } else {
        // handle number
        let num = s.parse::<u128>().unwrap();
        let num = num.to_be_bytes().to_vec();
        // remove leading zeros
        num.iter().fold(Vec::<u8>::new(), |mut acc, &x| {
            if x != 0 || !acc.is_empty() {
                acc.push(x);
            }
            acc
        })
    }
}

#[macro_export]
macro_rules! u8_vec {
    ( $s:literal, $len:expr ) => {
        {
            let bytes_vec = u8_vec!($s);
            let zeros_len = $len - bytes_vec.len();
            let mut fixed_bytes = vec![0; zeros_len];
            fixed_bytes.extend_from_slice(&bytes_vec);
            fixed_bytes
        }
    };
    ( $s:literal ) => {
        {
            let s = stringify!($s);
            if s.starts_with('"') {
                // handle str
                let s = s.strip_prefix('"').unwrap();
                let s = s.strip_suffix('"').unwrap();
                if s.starts_with("0x") {
                    let hex_str = s.strip_prefix("0x").unwrap();
                    hex::decode(hex_str).unwrap()
                } else {
                    bytes_to_be_vec(s.as_bytes(), s.len())
                }
            } else {
                // handle number
                let num = s.parse::<u128>().unwrap();
                let num = num.to_be_bytes().to_vec();
                num.iter().fold(Vec::<u8>::new(), |mut acc, &x| {
                    if x != 0 || !acc.is_empty() {
                        acc.push(x);
                    }
                    acc
                })
            }
        }
    };
    ( [$($x:expr),*], $len: expr ) => {
        {
            let bytes_vec = u8_vec!([$($x),*]);
            let zeros_len = $len - bytes_vec.len();
            let mut fixed_bytes = vec![0; zeros_len];
            fixed_bytes.extend_from_slice(&bytes_vec);
            fixed_bytes
        }
    };
    ( [$($x:expr),*] ) => {
        vec![$($x as u8),*]
    };
}

#[macro_export]
macro_rules! field_vec {
    ( $s:literal, $len:expr ) => {
        {
            let v = field_vec!($s);
            let zeros_len = $len - v.len();
            let mut fixed_bytes = vec![0; zeros_len];
            fixed_bytes.extend_from_slice(&v);
            fixed_bytes
        }
    };
    ( $s:literal ) => {
        {
            let v = u8_vec!($s);
            v.chunks(2)
                .map(|chunk| {
                    let mut bytes = [0u8; 2];
                    if chunk.len() == 1 {
                        bytes[1] = chunk[0];
                    } else {
                        bytes.copy_from_slice(chunk);
                    }
                    u16::from_be_bytes(bytes) as u32
                })
                .collect::<Vec<u32>>()
        }
    };
    ( [$($x:expr),*], $len: expr ) => {
        {
            let bytes_vec = field_vec!([$($x),*]);
            let zeros_len = $len - bytes_vec.len();
            let mut fixed_bytes = vec![0; zeros_len];
            fixed_bytes.extend_from_slice(&bytes_vec);
            fixed_bytes
        }
    };
    ([$($x:expr),*]) => {
        vec![$($x as u16 as u32),*]
    };
}

#[test]
pub fn test_macro_u8_vec() {
    assert_eq!(u8_vec!("0x010101", 6), vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(u8_vec!("0x010101"), vec![1, 1, 1]);
    assert_eq!(u8_vec!(1), vec![1]);
    assert_eq!(u8_vec!([1, 2, 3], 5), vec![0, 0, 1, 2, 3]);
    assert_eq!(u8_vec!("010101", 6), vec![48, 49, 48, 49, 48, 49]);
    assert_eq!(u8_vec!([10, 10, 10]), vec![10, 10, 10]);
    assert_eq!(u8_vec!(1, 2), vec![0, 1]);

    let len = 4;
    assert_eq!(u8_vec!("0x010101", len), vec![0, 1, 1, 1]);
    let len = 10;
    assert_eq!(u8_vec!("0x010101", len), vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
}

#[test]
pub fn test_macro_field_vec() {
    assert_eq!(field_vec!(5), vec![5]);
    assert_eq!(field_vec!("0x0101"), vec![257]);
    assert_eq!(field_vec!("0x0101", 6), vec![0, 0, 0, 0, 0, 257]);
    assert_eq!(field_vec!([10, 10, 10]), vec![10, 10, 10]);
    assert_eq!(
        field_vec!("0x003700370037", 8),
        vec![0, 0, 0, 0, 0, 55, 55, 55]
    );
}
