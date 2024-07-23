pub struct MemoryInterfaceCols<const CHUNK: usize, T> {
    // direction = 1 corresponds to initial memory state
    // direction = -1 corresponds to final memory state
    // direction = 0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub direction: T,
    pub address_space: T,
    pub leaf_label: T,
    pub values: [T; CHUNK],
    // auxes represents: multiplicity when direction = 1, is_final when direction = -1
    pub auxes: [T; CHUNK],
    pub temp_multiplicity: [T; CHUNK],
    pub temp_is_final: [T; CHUNK],
}

impl<const CHUNK: usize, T: Clone> MemoryInterfaceCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut iter = slc.iter().cloned();
        let mut take = || iter.next().unwrap();

        let direction = take();
        let address_space = take();
        let leaf_label = take();
        let values = std::array::from_fn(|_| take());
        let auxes = std::array::from_fn(|_| take());
        let temp_multiplicity = std::array::from_fn(|_| take());
        let temp_is_final = std::array::from_fn(|_| take());

        Self {
            direction,
            address_space,
            leaf_label,
            values,
            auxes,
            temp_multiplicity,
            temp_is_final,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.direction.clone(),
            self.address_space.clone(),
            self.leaf_label.clone(),
        ];
        result.extend(self.values.clone());
        result.extend(self.auxes.clone());
        result.extend(self.temp_multiplicity.clone());
        result.extend(self.temp_is_final.clone());
        result
    }

    pub fn get_width() -> usize {
        3 + (4 * CHUNK)
    }
}
