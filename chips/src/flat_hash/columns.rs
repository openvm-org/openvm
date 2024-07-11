use crate::flat_hash::FlatHashAir;

pub const NUM_COLS: usize = 3;

pub struct FlatHashCols<T> {
    pub io: FlatHashIOCols<T>,
    pub aux: FlatHashInternalCols<T>,
}

pub struct FlatHashIOCols<T> {
    pub page: Vec<T>,
    pub digest: Vec<T>,
}

/// Hash state indices match to the nth round index vector
/// Hash chunk indices match to the input for the nth round
/// Hash output indices match to the output for the nth round (i.e. the next round's input)
/// All done on the same row
pub struct FlatHashColIndices {
    pub hash_state_indices: Vec<Vec<usize>>,
    pub hash_chunk_indices: Vec<Vec<usize>>,
    pub hash_output_indices: Vec<Vec<usize>>,
}

#[derive(Clone)]
pub struct FlatHashInternalCols<T> {
    pub hashes: Vec<T>,
}

impl<T: Clone> FlatHashCols<T> {
    pub fn flatten(&self) -> Vec<T> {
        let mut combined = self.io.page.clone();
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
                let start = page_width + i * hash_width;
                let end = start + hash_width;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        let hash_chunk_indices = (0..num_hashes)
            .map(|i| {
                let start = i * hash_rate;
                let end = start + hash_rate;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        let hash_output_indices = (0..num_hashes)
            .map(|i| {
                let start = page_width + (i + 1) * hash_width;
                let end = start + hash_width;
                (start..end).collect::<Vec<usize>>()
            })
            .collect();

        FlatHashColIndices {
            hash_state_indices,
            hash_chunk_indices,
            hash_output_indices,
        }
    }

    pub fn from_slice(slice: &[T], chip: &FlatHashAir) -> Self {
        let (page, hashes) = slice.split_at(chip.page_width);
        let num_hashes = chip.page_width / chip.hash_rate;
        let digest_start = (num_hashes - 1) * chip.hash_width;
        let digest = hashes[digest_start..digest_start + chip.digest_width].to_vec();

        Self {
            io: FlatHashIOCols {
                page: page.to_vec(),
                digest,
            },
            aux: FlatHashInternalCols {
                hashes: hashes.to_vec(),
            },
        }
    }
}
