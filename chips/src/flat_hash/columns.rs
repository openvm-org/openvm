use crate::flat_hash::FlatHashAir;

pub const NUM_COLS: usize = 3;

pub struct FlatHashCols<T> {
    pub io: FlatHashIOCols<T>,
    pub aux: FlatHashInternalCols<T>,
}

pub struct FlatHashIOCols<T> {
    pub is_alloc: T,
    pub page: Vec<T>,
}

/// Hash state indices match to the nth round index vector
/// Hash chunk indices match to the input for the nth round
/// Hash output indices match to the output for the nth round (i.e. the next round's input)
/// All done on the same row
pub struct FlatHashColIndices {
    pub is_alloc_index: usize,
    pub hash_state_indices: Vec<Vec<usize>>,
    pub hash_chunk_indices: Vec<Vec<usize>>,
    pub hash_output_indices: Vec<Vec<usize>>,
}

/// Hashes is just a sequential list of all intermediate hash states
/// Starting from all zeros initial state up to and including final state
#[derive(Clone)]
pub struct FlatHashInternalCols<T> {
    pub hashes: Vec<T>,
}

impl<T: Clone> FlatHashCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut combined = vec![self.io.is_alloc.clone()];
        combined.extend(self.io.page.clone());
        combined.extend(self.aux.hashes.clone());
        combined
    }

    pub fn hash_col_indices(
        page_width: usize,
        hash_width: usize,
        hash_rate: usize,
    ) -> FlatHashColIndices {
        let num_hashes = page_width / hash_rate;
        let hash_state_indices = (0..num_hashes)
            .map(|i| {
                let start = page_width + i * hash_width + 1;
                let end = start + hash_width;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        let hash_chunk_indices = (0..num_hashes)
            .map(|i| {
                let start = i * hash_rate + 1;
                let end = start + hash_rate;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        let hash_output_indices = (0..num_hashes)
            .map(|i| {
                let start = page_width + (i + 1) * hash_width + 1;
                let end = start + hash_width;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        FlatHashColIndices {
            is_alloc_index: 0,
            hash_state_indices,
            hash_chunk_indices,
            hash_output_indices,
        }
    }

    pub fn from_slice(slice: &[T], chip: &FlatHashAir) -> Self {
        let (page, hashes) = slice.split_at(chip.page_width + 1);

        Self {
            io: FlatHashIOCols {
                is_alloc: slice[0].clone(),
                page: page[1..].to_vec(),
            },
            aux: FlatHashInternalCols {
                hashes: hashes.to_vec(),
            },
        }
    }
}
