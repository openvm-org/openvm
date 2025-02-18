use std::array::TryFromSliceError;

use openvm_stark_backend::p3_field::PrimeField32;
use thiserror::Error;

use crate::{
    arch::{SystemBaseState, VmChipComplexState, VmInventoryState},
    system::memory::{online::MemoryLogEntry, MemoryImage, PagedVec, PAGE_SIZE},
};

pub mod vecvec;

#[cfg(test)]
mod tests;

// Define your custom error type
#[derive(Error, Debug)]
pub enum SerdeError {
    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("failed to convert slice: {0}")]
    SliceConversion(#[from] TryFromSliceError),
}

// pub trait ChipSerde {
//     fn size(&self) -> usize;
//     fn dump(&self) -> Vec<u8>;
//     fn load(data: &[u8]) -> Result<Box<Self, String>>;
// }

impl<F: PrimeField32> MemoryLogEntry<F> {
    // 1 byte for enum variant
    // Read: 4 + 4 + 8
    // Write: 4 + 4 + 8 bytes for the len + len * 4
    // IncrementTimestampBy: 4
    pub fn size(&self) -> usize {
        match self {
            Self::Read { .. } => 1 + 4 + 4 + 8,
            Self::Write { data, .. } => 1 + 4 + 4 + 8 + data.len() * 4,
            Self::IncrementTimestampBy(_) => 1 + 4,
        }
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        match self {
            Self::Read {
                address_space,
                pointer,
                len,
            } => {
                data[current_pos] = 0; // enum variant 0 for Read
                current_pos += 1;
                data[current_pos..current_pos + 4].copy_from_slice(&address_space.to_le_bytes());
                current_pos += 4;
                data[current_pos..current_pos + 4].copy_from_slice(&pointer.to_le_bytes());
                current_pos += 4;
                data[current_pos..current_pos + 8].copy_from_slice(&(*len as u64).to_le_bytes());
                current_pos += 8;
            }
            Self::Write {
                address_space,
                pointer,
                data: write_data,
            } => {
                data[current_pos] = 1; // enum variant 1 for Write
                current_pos += 1;
                data[current_pos..current_pos + 4].copy_from_slice(&address_space.to_le_bytes());
                current_pos += 4;
                data[current_pos..current_pos + 4].copy_from_slice(&pointer.to_le_bytes());
                current_pos += 4;
                data[current_pos..current_pos + 8]
                    .copy_from_slice(&(write_data.len() as u64).to_le_bytes());
                current_pos += 8;
                for value in write_data {
                    data[current_pos..current_pos + 4]
                        .copy_from_slice(&value.as_canonical_u32().to_le_bytes());
                    current_pos += 4;
                }
            }
            Self::IncrementTimestampBy(increment) => {
                data[current_pos] = 2; // enum variant 2 for IncrementTimestampBy
                current_pos += 1;
                data[current_pos..current_pos + 4].copy_from_slice(&increment.to_le_bytes());
                current_pos += 4;
            }
        }
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        if data.is_empty() {
            return Err(SerdeError::InvalidData("Empty data".to_string()));
        }

        let variant = data[0];
        match variant {
            0 => {
                // Read
                if data.len() < 1 + 4 + 4 + 8 {
                    return Err(SerdeError::InvalidData(
                        "Insufficient data for Read variant".to_string(),
                    ));
                }
                let address_space = u32::from_le_bytes(data[1..5].try_into()?);
                let pointer = u32::from_le_bytes(data[5..9].try_into()?);
                let len = u64::from_le_bytes(data[9..17].try_into()?) as usize;
                Ok(Self::Read {
                    address_space,
                    pointer,
                    len,
                })
            }
            1 => {
                // Write
                if data.len() < 1 + 4 + 4 + 8 {
                    return Err(SerdeError::InvalidData(
                        "Insufficient data for Write variant".to_string(),
                    ));
                }
                let address_space = u32::from_le_bytes(data[1..5].try_into()?);
                let pointer = u32::from_le_bytes(data[5..9].try_into()?);
                let remaining_bytes = &data[9..];
                let len = u64::from_le_bytes(remaining_bytes[0..8].try_into()?) as usize;
                let mut data_vec = Vec::with_capacity(len);
                if remaining_bytes.len() < 8 + len * 4 {
                    return Err(SerdeError::InvalidData(
                        "Invalid data length for Write variant".to_string(),
                    ));
                }
                for chunk in remaining_bytes[8..8 + len * 4].chunks_exact(4) {
                    let value = u32::from_le_bytes(chunk.try_into()?);
                    data_vec.push(F::from_canonical_u32(value));
                }
                Ok(Self::Write {
                    address_space,
                    pointer,
                    data: data_vec,
                })
            }
            2 => {
                // IncrementTimestampBy
                if data.len() < 1 + 4 {
                    return Err(SerdeError::InvalidData(
                        "Invalid data length for IncrementTimestampBy variant".to_string(),
                    ));
                }
                let increment = u32::from_le_bytes(data[1..5].try_into()?);
                Ok(Self::IncrementTimestampBy(increment))
            }
            _ => Err(SerdeError::InvalidData("Invalid variant tag".to_string())),
        }
    }
}

// single field pages: Vec<Option<Vec<T>>>
// first 8 bytes: number of pages
// each page:
// - first byte: 1/0 indicating if the page is nonempty (initialized)
// - remaining bytes: page data if not empty (1)
impl<F: PrimeField32> PagedVec<F, PAGE_SIZE> {
    pub fn size(&self) -> usize {
        let mut size = 8 + self.pages.len();
        for _ in self.pages.iter().flatten() {
            size += PAGE_SIZE * 4;
        }
        size
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.pages.len() as u64).to_le_bytes());
        current_pos += 8;
        for page in self.pages.iter() {
            if let Some(page_data) = page {
                assert_eq!(page_data.len(), PAGE_SIZE);
                data[current_pos] = 1;
                current_pos += 1;
                for (i, value) in page_data.iter().enumerate() {
                    data[current_pos + i * 4..current_pos + (i + 1) * 4]
                        .copy_from_slice(&value.as_canonical_u32().to_le_bytes());
                }
                current_pos += PAGE_SIZE * 4;
            } else {
                data[current_pos] = 0;
                current_pos += 1;
            }
        }
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        if data.len() < 8 {
            return Err(SerdeError::InvalidData(
                "Insufficient data for PagedVec header".to_string(),
            ));
        }

        // Read number of pages
        let num_pages = u64::from_le_bytes(data[0..8].try_into()?);
        let mut current_pos = 8;

        // Create vector to store pages
        let mut pages = Vec::with_capacity(num_pages as usize);

        // Read each page
        for _ in 0..num_pages {
            if current_pos >= data.len() {
                return Err(SerdeError::InvalidData(
                    "Unexpected end of data while reading pages".to_string(),
                ));
            }

            // Read page flag (0 = empty, 1 = has data)
            let has_data = data[current_pos] != 0;
            current_pos += 1;

            if has_data {
                // Check if we have enough data for a full page
                if current_pos + PAGE_SIZE * 4 > data.len() {
                    return Err(SerdeError::InvalidData(
                        "Insufficient data for page contents".to_string(),
                    ));
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

        Ok(Self { pages })
    }
}

// first 8 bytes: number of page_vec
// And then each page_vec
// finally as_offset.
impl<F: PrimeField32> MemoryImage<F> {
    pub fn size(&self) -> usize {
        self.paged_vecs.iter().map(|p| p.size()).sum::<usize>() + 8 + 4
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.paged_vecs.len() as u64).to_le_bytes());
        current_pos += 8;
        for paged_vec in self.paged_vecs.iter() {
            current_pos += paged_vec.dump(&mut data[current_pos..]);
        }
        data[current_pos..current_pos + 4].copy_from_slice(&self.as_offset.to_le_bytes());
        current_pos += 4;
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        if data.len() < 8 {
            return Err(SerdeError::InvalidData(
                "Insufficient data for MemoryImage header".to_string(),
            ));
        }
        let num_paged_vecs = u64::from_le_bytes(data[0..8].try_into()?);
        let mut current_pos = 8;
        let mut paged_vecs = Vec::with_capacity(num_paged_vecs as usize);
        for _ in 0..num_paged_vecs {
            let paged_vec = PagedVec::load(&data[current_pos..])?;
            current_pos += paged_vec.size();
            paged_vecs.push(paged_vec);
        }
        let as_offset = u32::from_le_bytes(data[current_pos..current_pos + 4].try_into()?);
        Ok(Self {
            paged_vecs,
            as_offset,
        })
    }
}

fn load_length_prefixed_vec(data: &[u8], current_pos: &mut usize) -> Result<Vec<u8>, SerdeError> {
    if *current_pos + 8 > data.len() {
        return Err(SerdeError::InvalidData(
            "Insufficient data for length prefix".to_string(),
        ));
    }
    let len = u64::from_le_bytes(data[*current_pos..*current_pos + 8].try_into()?) as usize;
    *current_pos += 8;

    if *current_pos + len > data.len() {
        return Err(SerdeError::InvalidData("Invalid data length".to_string()));
    }
    let vec_data = data[*current_pos..*current_pos + len].to_vec();
    *current_pos += len;

    Ok(vec_data)
}

// range checker, initial memory, memory logs, connector, program
impl<F: PrimeField32> SystemBaseState<F> {
    pub fn size(&self) -> usize {
        8 + self.range_checker_chip.len() // range checker size + data
        + 1 + self.initial_memory.as_ref().map_or(0, |m| m.size()) // flag + data
        + 8 + self.memory_logs.iter().map(|l| l.size()).sum::<usize>() // memory logs len + data
        + 8 + self.connector_chip.len() // connector size + data
        + 8 + self.program_chip.len() // program size + data
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        // range_checker_chip
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.range_checker_chip.len() as u64).to_le_bytes());
        current_pos += 8;
        data[current_pos..current_pos + self.range_checker_chip.len()]
            .copy_from_slice(&self.range_checker_chip);
        current_pos += self.range_checker_chip.len();

        // initial_memory
        if let Some(initial_memory) = &self.initial_memory {
            data[current_pos] = 1;
            current_pos += 1;
            current_pos += initial_memory.dump(&mut data[current_pos..]);
        } else {
            data[current_pos] = 0;
            current_pos += 1;
        }

        // memory_logs
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.memory_logs.len() as u64).to_le_bytes());
        current_pos += 8;
        for log in self.memory_logs.iter() {
            current_pos += log.dump(&mut data[current_pos..]);
        }

        // connector_chip
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.connector_chip.len() as u64).to_le_bytes());
        current_pos += 8;
        data[current_pos..current_pos + self.connector_chip.len()]
            .copy_from_slice(&self.connector_chip);
        current_pos += self.connector_chip.len();

        // program_chip
        data[current_pos..current_pos + 8]
            .copy_from_slice(&(self.program_chip.len() as u64).to_le_bytes());
        current_pos += 8;
        data[current_pos..current_pos + self.program_chip.len()]
            .copy_from_slice(&self.program_chip);
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        let mut current_pos = 0;

        // Load range_checker_chip
        let range_checker_chip = load_length_prefixed_vec(data, &mut current_pos)?;

        // Load initial_memory
        if current_pos >= data.len() {
            return Err(SerdeError::InvalidData(
                "Unexpected end of data".to_string(),
            ));
        }
        let has_initial_memory = data[current_pos] != 0;
        current_pos += 1;
        let initial_memory = if has_initial_memory {
            Some(MemoryImage::load(&data[current_pos..])?)
        } else {
            None
        };
        if let Some(mem) = &initial_memory {
            current_pos += mem.size();
        }

        // Load memory_logs
        if current_pos + 8 > data.len() {
            return Err(SerdeError::InvalidData(
                "Invalid memory_logs length".to_string(),
            ));
        }
        let memory_logs_len =
            u64::from_le_bytes(data[current_pos..current_pos + 8].try_into()?) as usize;
        current_pos += 8;
        let mut memory_logs = Vec::with_capacity(memory_logs_len);
        for _ in 0..memory_logs_len {
            let log = MemoryLogEntry::load(&data[current_pos..])?;
            current_pos += log.size();
            memory_logs.push(log);
        }

        // Load connector_chip
        let connector_chip = load_length_prefixed_vec(data, &mut current_pos)?;

        // Load program_chip
        let program_chip = load_length_prefixed_vec(data, &mut current_pos)?;

        Ok(Self {
            range_checker_chip,
            initial_memory,
            memory_logs,
            connector_chip,
            program_chip,
        })
    }
}

impl VmInventoryState {
    pub fn size(&self) -> usize {
        vecvec::size(&self.executors) + vecvec::size(&self.periphery)
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        current_pos += vecvec::dump(&self.executors, &mut data[current_pos..]);
        current_pos += vecvec::dump(&self.periphery, &mut data[current_pos..]);
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        let mut current_pos = 0;
        let executors = vecvec::load(&data[current_pos..])?;
        current_pos += vecvec::size(&executors);
        let periphery = vecvec::load(&data[current_pos..])?;
        Ok(Self {
            executors,
            periphery,
        })
    }
}

impl<F: PrimeField32> VmChipComplexState<F> {
    pub fn size(&self) -> usize {
        self.base.size() + self.inventory.size()
    }

    pub fn dump(&self, data: &mut [u8]) -> usize {
        let mut current_pos = 0;
        current_pos += self.base.dump(&mut data[current_pos..]);
        current_pos += self.inventory.dump(&mut data[current_pos..]);
        current_pos
    }

    pub fn load(data: &[u8]) -> Result<Self, SerdeError> {
        let mut current_pos = 0;
        let base = SystemBaseState::load(&data[current_pos..])?;
        current_pos += base.size();
        let inventory = VmInventoryState::load(&data[current_pos..])?;
        Ok(Self { base, inventory })
    }
}
