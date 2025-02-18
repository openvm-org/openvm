use eyre::{bail, Result};

use super::ChipSerde;

// Max length of inner vec is 2^64 - 1
// First 8 bytes store the total length of the vec
// For each inner vec, we store the length as u64, then the vec
impl ChipSerde for Vec<Vec<u8>> {
    fn size(&self) -> usize {
        todo!()
    }

    fn dump(&self) -> Vec<u8> {
        let mut total_len = 8;
        for inner_vec in self {
            total_len += (inner_vec.len() as u64) + 8;
        }
        let mut result = Vec::with_capacity(total_len.try_into().unwrap());
        result.extend_from_slice(&total_len.to_le_bytes());
        for inner_vec in self {
            result.extend_from_slice(&(inner_vec.len() as u64).to_le_bytes());
            result.extend_from_slice(inner_vec);
        }
        result
    }

    fn load(data: &[u8]) -> Result<Box<Self>> {
        let num_vecs = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let mut result = Vec::with_capacity(num_vecs.try_into().unwrap());
        let mut offset = 8;

        for _ in 0..num_vecs {
            if offset + 8 > data.len() {
                bail!("failed to load vec<vec<u8>>");
            }
            let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            if offset + len > data.len() {
                bail!("failed to load vec<vec<u8>>");
            }
            // TODO: single allocation
            let inner_vec = data[offset..offset + len].to_vec();
            offset += len;
            result.push(inner_vec);
        }

        Ok(Box::new(result))
    }
}
