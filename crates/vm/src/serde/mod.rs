use eyre::Result;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::VmChipComplexState,
    system::memory::{online::MemoryLogEntry, MemoryImage, PagedVec},
};

pub mod vecvec;

#[cfg(test)]
mod tests;

pub trait ChipSerde {
    fn size(&self) -> usize;
    fn dump(&self) -> Vec<u8>;
    fn load(data: &[u8]) -> Result<Box<Self>>;
}

impl<F: PrimeField32> ChipSerde for MemoryLogEntry<F> {
    fn size(&self) -> usize {
        match self {
            Self::Read { .. } => 1 + 4 + 4 + 8,
            Self::Write { data, .. } => 1 + 8 + data.len() * 4,
            Self::IncrementTimestampBy(_) => 1 + 4,
        }
    }

    fn dump(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.size());
        match self {
            Self::Read {
                address_space,
                pointer,
                len,
            } => {
                result.push(0); // enum variant 0 for Read
                result.extend_from_slice(&address_space.to_le_bytes());
                result.extend_from_slice(&pointer.to_le_bytes());
                result.extend_from_slice(&(*len as u64).to_le_bytes());
            }
            Self::Write {
                address_space,
                pointer,
                data,
            } => {
                result.push(1); // enum variant 1 for Write
                result.extend_from_slice(&address_space.to_le_bytes());
                result.extend_from_slice(&pointer.to_le_bytes());
                for value in data {
                    result.extend_from_slice(&value.as_canonical_u32().to_le_bytes());
                }
            }
            Self::IncrementTimestampBy(increment) => {
                result.push(2); // enum variant 2 for IncrementTimestampBy
                result.extend_from_slice(&increment.to_le_bytes());
            }
        }
        result
    }

    fn load(data: &[u8]) -> Result<Box<Self>> {
        if data.is_empty() {
            return Err(eyre::eyre!("Empty data"));
        }

        let variant = data[0];
        match variant {
            0 => {
                // Read
                if data.len() != 1 + 4 + 4 + 8 {
                    return Err(eyre::eyre!("Insufficient data for Read variant"));
                }
                let address_space = u32::from_le_bytes(data[1..5].try_into()?);
                let pointer = u32::from_le_bytes(data[5..9].try_into()?);
                let len = u64::from_le_bytes(data[9..17].try_into()?) as usize;
                Ok(Box::new(Self::Read {
                    address_space,
                    pointer,
                    len,
                }))
            }
            1 => {
                // Write
                if data.len() <= 1 + 4 + 4 {
                    return Err(eyre::eyre!("Insufficient data for Write variant"));
                }
                let address_space = u32::from_le_bytes(data[1..5].try_into()?);
                let pointer = u32::from_le_bytes(data[5..9].try_into()?);
                let remaining_bytes = &data[9..];
                if remaining_bytes.len() % 4 != 0 {
                    return Err(eyre::eyre!("Invalid data length for Write variant"));
                }
                let mut data_vec = Vec::with_capacity(remaining_bytes.len() / 4);
                for chunk in remaining_bytes.chunks_exact(4) {
                    let value = u32::from_le_bytes(chunk.try_into()?);
                    data_vec.push(F::from_canonical_u32(value));
                }
                Ok(Box::new(Self::Write {
                    address_space,
                    pointer,
                    data: data_vec,
                }))
            }
            2 => {
                // IncrementTimestampBy
                if data.len() != 1 + 4 {
                    return Err(eyre::eyre!(
                        "Invalid data length for IncrementTimestampBy variant"
                    ));
                }
                let increment = u32::from_le_bytes(data[1..5].try_into()?);
                Ok(Box::new(Self::IncrementTimestampBy(increment)))
            }
            _ => Err(eyre::eyre!("Invalid variant tag")),
        }
    }
}

// single field pages: Vec<Option<Vec<T>>>
// first 8 bytes: number of pages
// each page:
// - first byte: 1/0 indicating if the page is nonempty (initialized)
// - remaining bytes: page data if not empty (1)
impl<F: PrimeField32, const PAGE_SIZE: usize> ChipSerde for PagedVec<F, PAGE_SIZE> {
    fn size(&self) -> usize {
        let mut size = 8 + self.pages.len();
        for _ in self.pages.iter().flatten() {
            size += PAGE_SIZE * 4;
        }
        size
    }

    fn dump(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.size());
        result.extend_from_slice(&(self.pages.len() as u64).to_le_bytes());
        for page in self.pages.iter() {
            if let Some(data) = page {
                result.push(1);
                result.extend(data.iter().flat_map(|x| x.as_canonical_u32().to_le_bytes()));
            } else {
                result.push(0);
            }
        }
        result
    }

    fn load(data: &[u8]) -> Result<Box<Self>> {
        if data.len() < 8 {
            return Err(eyre::eyre!("Insufficient data for PagedVec header"));
        }

        // Read number of pages
        let num_pages = u64::from_le_bytes(data[0..8].try_into()?);
        let mut current_pos = 8;

        // Create vector to store pages
        let mut pages = Vec::with_capacity(num_pages as usize);

        // Read each page
        for _ in 0..num_pages {
            if current_pos >= data.len() {
                return Err(eyre::eyre!("Unexpected end of data while reading pages"));
            }

            // Read page flag (0 = empty, 1 = has data)
            let has_data = data[current_pos] != 0;
            current_pos += 1;

            if has_data {
                // Check if we have enough data for a full page
                if current_pos + PAGE_SIZE * 4 > data.len() {
                    return Err(eyre::eyre!("Insufficient data for page contents"));
                }

                // TODO: can we allocate all internal vec at once?
                let mut page_data = Vec::with_capacity(PAGE_SIZE);
                for _ in 0..PAGE_SIZE {
                    let value = u32::from_le_bytes(data[current_pos..current_pos + 4].try_into()?);
                    page_data.push(F::from_canonical_u32(value));
                    current_pos += 4;
                }
                pages.push(Some(page_data));
            } else {
                pages.push(None);
            }
        }

        Ok(Box::new(Self { pages }))
    }
}

// first 8 bytes: number of page_vec
// And then each page_vec
// finally as_offset.
impl<F: PrimeField32> ChipSerde for MemoryImage<F> {
    fn size(&self) -> usize {
        self.paged_vecs.iter().map(|p| p.size()).sum::<usize>() + 8 + 4
    }

    fn dump(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.size());
        result.extend_from_slice(&(self.paged_vecs.len() as u64).to_le_bytes());
        for paged_vec in self.paged_vecs.iter() {
            result.extend(paged_vec.dump());
        }
        result.extend_from_slice(&self.as_offset.to_le_bytes());
        result
    }

    fn load(data: &[u8]) -> Result<Box<Self>> {
        if data.len() < 8 {
            return Err(eyre::eyre!("Insufficient data for MemoryImage header"));
        }
        let num_paged_vecs = u64::from_le_bytes(data[0..8].try_into()?);
        let mut current_pos = 8;
        let mut paged_vecs = Vec::with_capacity(num_paged_vecs as usize);
        for _ in 0..num_paged_vecs {
            let paged_vec = PagedVec::load(&data[current_pos..])?;
            current_pos += paged_vec.size();
            paged_vecs.push(*paged_vec);
        }
        let as_offset = u32::from_le_bytes(data[current_pos..current_pos + 4].try_into()?);
        Ok(Box::new(Self {
            paged_vecs,
            as_offset,
        }))
    }
}

// base: range checker, initial memory, memory logs, connector, program
// inventory: executors, periphery
impl<F> ChipSerde for VmChipComplexState<F> {
    fn size(&self) -> usize {
        todo!()
    }

    fn dump(&self) -> Vec<u8> {
        todo!()
    }

    fn load(_data: &[u8]) -> Result<Box<Self>> {
        todo!()
    }
}
