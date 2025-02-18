use super::SerdeError;

// TODO: allocate once and split
// First 8 bytes store the length of the vec
// For each inner vec, we store the length as u64, then the vec
pub fn size(v: &Vec<Vec<u8>>) -> usize {
    let mut total_len = 8;
    for inner_vec in v {
        total_len += inner_vec.len() + 8;
    }
    total_len
}

pub fn dump(v: &Vec<Vec<u8>>, data: &mut [u8]) -> usize {
    let mut current_pos = 0;
    data[current_pos..current_pos + 8].copy_from_slice(&(v.len() as u64).to_le_bytes());
    current_pos += 8;
    for inner_vec in v {
        data[current_pos..current_pos + 8].copy_from_slice(&(inner_vec.len() as u64).to_le_bytes());
        current_pos += 8;
        data[current_pos..current_pos + inner_vec.len()].copy_from_slice(inner_vec);
        current_pos += inner_vec.len();
    }
    current_pos
}

pub fn load(data: &[u8]) -> Result<Vec<Vec<u8>>, SerdeError> {
    let num_vecs = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let mut result = Vec::with_capacity(num_vecs as usize);
    let mut offset = 8;

    for _ in 0..num_vecs {
        if offset + 8 > data.len() {
            return Err(SerdeError::InvalidData(
                "Insufficient data for Read variant".to_string(),
            ));
        }
        let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + len > data.len() {
            return Err(SerdeError::InvalidData(
                "Insufficient data for Read variant".to_string(),
            ));
        }
        let inner_vec = data[offset..offset + len].to_vec();
        offset += len;
        result.push(inner_vec);
    }

    Ok(result)
}
