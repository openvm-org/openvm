use bytes::Bytes;

use super::SerdeError;

// First 8 bytes store the length of the vec
// Next 8 bytes, store the sum of the lengths of all inner vecs
// For each inner vec, we store the length as u64, then the vec
pub fn size(v: &Vec<Bytes>) -> usize {
    let mut total_len = 16;
    for inner_vec in v {
        total_len += inner_vec.len() + 8;
    }
    total_len
}

pub fn dump(v: &Vec<Bytes>, data: &mut [u8]) -> usize {
    let mut current_pos = 0;
    data[current_pos..current_pos + 8].copy_from_slice(&(v.len() as u64).to_le_bytes());
    current_pos += 8;
    let sum_len = v.iter().map(|v| v.len()).sum::<usize>();
    data[current_pos..current_pos + 8].copy_from_slice(&(sum_len as u64).to_le_bytes());
    current_pos += 8;
    for inner_vec in v {
        data[current_pos..current_pos + 8].copy_from_slice(&(inner_vec.len() as u64).to_le_bytes());
        current_pos += 8;
        data[current_pos..current_pos + inner_vec.len()].copy_from_slice(inner_vec);
        current_pos += inner_vec.len();
    }
    current_pos
}

pub fn load(data: &[u8]) -> Result<Vec<Bytes>, SerdeError> {
    if data.len() < 16 {
        return Err(SerdeError::InvalidData(
            "Insufficient data for VecVec".to_string(),
        ));
    }
    let num_vecs = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let sum_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let mut result: Vec<Bytes> = Vec::with_capacity(num_vecs);
    let mut internal_vecs = Bytes::from(vec![0; sum_len]);

    if data.len() < 16 + sum_len + 8 * num_vecs {
        return Err(SerdeError::InvalidData(
            "Insufficient data for VecVec".to_string(),
        ));
    }

    let mut offset = 16;
    let mut internal_vec_count = 0;

    println!("start unsafe");
    unsafe {
        for i in 0..num_vecs {
            let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;
            // let mut inner_vec =
            //     Vec::from_raw_parts(internal_vecs.as_mut_ptr().add(internal_vec_count), len, len);
            // inner_vec.copy_from_slice(&data[offset..offset + len]);
            let inner_vec = internal_vecs.split_to(len);

            offset += len;
            internal_vec_count += len;
            result.push(inner_vec);
        }
        assert_eq!(internal_vec_count, sum_len);
        std::mem::forget(internal_vecs);
    }
    println!("end unsafe");

    Ok(result)
}
