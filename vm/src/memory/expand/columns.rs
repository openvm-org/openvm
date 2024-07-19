pub struct ExpandCols<const CHUNK: usize, T> {
    pub direction: T,
    pub address_space: T,
    pub parent_height: T,
    pub parent_label: T,
    pub parent_hash: [T; CHUNK],
    pub left_child_hash: [T; CHUNK],
    pub right_child_hash: [T; CHUNK],
    pub left_is_final: T,
    pub right_is_final: T,
}

impl<const CHUNK: usize, T: Clone> ExpandCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut iter = slc.iter();
        let mut take = || iter.next().unwrap().clone();

        let direction = take();
        let address_space = take();
        let height = take();
        let parent_label = take();
        let parent_hash = std::array::from_fn(|_| take());
        let left_child_hash = std::array::from_fn(|_| take());
        let right_child_hash = std::array::from_fn(|_| take());
        let left_is_final = take();
        let right_is_final = take();

        Self {
            direction,
            address_space,
            parent_height: height,
            parent_label,
            parent_hash,
            left_child_hash,
            right_child_hash,
            left_is_final,
            right_is_final,
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
        result.extend(self.left_child_hash.clone());
        result.extend(self.right_child_hash.clone());
        result.push(self.left_is_final.clone());
        result.push(self.right_is_final.clone());
        result
    }

    pub fn get_width() -> usize {
        4 + (3 * CHUNK) + 2
    }
}
