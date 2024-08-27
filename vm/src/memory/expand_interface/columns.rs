#[derive(Debug)]
pub struct MemoryExpandInterfaceCols<const NUM_WORDS: usize, const WORD_SIZE: usize, T> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,
    pub address_space: T,
    pub leaf_label: T,
    pub values: [[T; WORD_SIZE]; NUM_WORDS],
    // timestamp used for each word in this row
    pub clks: [T; NUM_WORDS],
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, T: Clone>
    MemoryExpandInterfaceCols<NUM_WORDS, WORD_SIZE, T>
{
    pub fn from_slice(slc: &[T]) -> Self {
        let mut iter = slc.iter().cloned();
        let mut take = || iter.next().unwrap();

        let expand_direction = take();
        let address_space = take();
        let leaf_label = take();
        let values = std::array::from_fn(|_| std::array::from_fn(|_| take()));
        let clks = std::array::from_fn(|_| take());

        Self {
            expand_direction,
            address_space,
            leaf_label,
            values,
            clks,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.expand_direction.clone(),
            self.address_space.clone(),
            self.leaf_label.clone(),
        ];
        result.extend(self.values.clone().into_iter().flat_map(|x| x.into_iter()));
        result.extend(self.clks.clone());
        result
    }

    pub fn width() -> usize {
        3 + NUM_WORDS * WORD_SIZE + NUM_WORDS
    }
}
