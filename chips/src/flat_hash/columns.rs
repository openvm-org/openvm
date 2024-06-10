use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FlatHashCols<T> {
    pub page_width: usize,
    pub page_height: usize,
    pub hash_width: usize,
    pub hash_rate: usize,
    pub digest_width: usize,
    pub io: FlatHashIOCols<T>,
    pub aux: FlatHashInternalCols<T>,
}

#[derive(Clone)]
pub struct FlatHashIOCols<T> {
    pub page: Vec<T>,
    pub digest: Vec<T>,
}

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
    pub fn new(
        page_width: usize,
        page_height: usize,
        hash_width: usize,
        hash_rate: usize,
        page: Vec<T>,
        hashes: Vec<T>,
        digest_width: usize,
    ) -> FlatHashCols<T> {
        let num_hashes = page_width / hash_rate;
        let digest = hashes
            [((num_hashes - 1) * hash_width)..((num_hashes - 1) * hash_width + digest_width)]
            .to_vec();
        FlatHashCols {
            page_width,
            page_height,
            hash_width,
            hash_rate,
            io: FlatHashIOCols { page, digest },
            aux: FlatHashInternalCols { hashes },
            digest_width,
        }
    }

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

    pub fn from_slice(
        slice: &[T],
        page_width: usize,
        page_height: usize,
        hash_width: usize,
        hash_rate: usize,
        digest_width: usize,
    ) -> Self {
        let (page, hashes) = slice.split_at(page_width);
        Self::new(
            page_width,
            page_height,
            hash_width,
            hash_rate,
            page.to_vec(),
            hashes.to_vec(),
            digest_width,
        )
    }

    pub fn get_width(&self) -> usize {
        self.page_width + (self.page_width / self.hash_rate + 1) * self.hash_width
    }
}
