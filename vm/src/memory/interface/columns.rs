use std::array::from_fn;

pub struct MemoryInterfaceCols<const CHUNK: usize, T> {
    pub direction: T,
    pub address_space: T,
    pub leaf_label: T,
    pub values: [T; CHUNK],
    pub auxes: [T; CHUNK],
    pub temp_multiplicity: [T; CHUNK],
    pub temp_is_final: [T; CHUNK],
}

impl<const CHUNK: usize, T: Clone> MemoryInterfaceCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut slc_index = 0;
        let mut take = || {
            slc_index += 1;
            slc[slc_index - 1].clone()
        };

        let direction = take();
        let address_space = take();
        let leaf_label = take();
        let values = from_fn(|_| take());
        let auxes = from_fn(|_| take());
        let temp_multiplicity = from_fn(|_| take());
        let temp_is_final = from_fn(|_| take());

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
        3 + (2 * CHUNK)
    }
}
