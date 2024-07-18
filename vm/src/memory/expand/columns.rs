use std::array::from_fn;

pub struct ExpandCols<const CHUNK: usize, T> {
    pub direction: T,
    pub address_space: T,
    pub parent_height: T,
    pub parent_label: T,
    pub parent_hash: [T; CHUNK],
    pub child_hashes: [[T; CHUNK]; 2],
    pub are_final: [T; 2],
}

impl<const CHUNK: usize, T: Clone> ExpandCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut slc_index = 0;
        let mut take = || {
            slc_index += 1;
            slc[slc_index - 1].clone()
        };

        let direction = take();
        let address_space = take();
        let height = take();
        let parent_label = take();
        let parent_hash = from_fn(|_| take());
        let child_hashes = from_fn(|_| from_fn(|_| take()));
        let are_final = from_fn(|_| take());

        Self {
            direction,
            address_space,
            parent_height: height,
            parent_label,
            parent_hash,
            child_hashes,
            are_final,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.direction.clone(),
            self.address_space.clone(),
            self.parent_height.clone(),
            self.parent_label.clone(),
        ];
        result.extend(self.parent_hash.clone());
        result.extend(self.child_hashes.concat());
        result.extend(self.are_final.clone());
        result
    }

    pub fn get_width() -> usize {
        4 + (3 * CHUNK) + 2
    }
}
