// use alloy_primitives::{Uint, U256, U512};

pub trait IntoBeBytes: Sized {
    fn into_be_bytes(&self) -> Vec<u8>;
}

// impl IntoBeBytes for u8 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         vec![*self]
//     }
// }

impl IntoBeBytes for u16 {
    fn into_be_bytes(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

// impl IntoBeBytes for u32 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         self.to_be_bytes().to_vec()
//     }
// }

// impl IntoBeBytes for u64 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         self.to_be_bytes().to_vec()
//     }
// }

// impl IntoBeBytes for u128 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         self.to_be_bytes().to_vec()
//     }
// }

// impl IntoBeBytes for U256 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         self.to_be_bytes::<32>().to_vec()
//     }
// }

// impl IntoBeBytes for U512 {
//     fn into_be_bytes(&self) -> Vec<u8> {
//         self.to_be_bytes::<64>().to_vec()
//     }
// }

// pub trait FromBeBytes: Sized {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self>;
// }

// impl FromBeBytes for u8 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         Some(bytes[0])
//     }
// }

// impl FromBeBytes for u16 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 2 {
//             return None;
//         }
//         Some(u16::from_be_bytes(bytes.try_into().ok()?))
//     }
// }

// impl FromBeBytes for u32 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 4 {
//             return None;
//         }
//         Some(u32::from_be_bytes(bytes.try_into().ok()?))
//     }
// }

// impl FromBeBytes for u64 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 8 {
//             return None;
//         }
//         Some(u64::from_be_bytes(bytes.try_into().ok()?))
//     }
// }

// impl FromBeBytes for u128 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 16 {
//             return None;
//         }
//         Some(u128::from_be_bytes(bytes.try_into().ok()?))
//     }
// }

// impl FromBeBytes for U256 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 32 {
//             return None;
//         }
//         Some(U256::from_be_bytes::<32>(bytes.try_into().ok()?))
//     }
// }

// impl FromBeBytes for U512 {
//     fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
//         if bytes.len() != 64 {
//             return None;
//         }
//         Some(U512::from_be_bytes::<64>(bytes.try_into().ok()?))
//     }
// }
