pub struct ExpandCols<const CHUNK: usize, T> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,

    // height_section = 0 indicates that as_label is being expanded
    // height_within = 1 indicates that address_label is being expanded
    // height can be computed as (height_section * address_bits) + height_within
    pub children_height_section: T,
    pub children_height_within: T,
    // aux column used to constrain that (height_section, height_within) != (0, address_bits)
    // because that should instead be (height_section, height_within) = (1, 0)
    pub height_inverse: T,

    pub parent_as_label: T,
    pub parent_address_label: T,

    pub parent_hash: [T; CHUNK],
    pub left_child_hash: [T; CHUNK],
    pub right_child_hash: [T; CHUNK],

    // indicate whether `expand_direction` is different from origin
    // when `expand_direction` != -1, must be 0
    pub left_direction_different: T,
    pub right_direction_different: T,
}

impl<const CHUNK: usize, T: Clone> ExpandCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut iter = slc.iter();
        let mut take = || iter.next().unwrap().clone();

        let expand_direction = take();
        let parent_height_section = take();
        let parent_height_within = take();
        let height_inverse = take();
        let parent_as_label = take();
        let parent_label = take();
        let parent_hash = std::array::from_fn(|_| take());
        let left_child_hash = std::array::from_fn(|_| take());
        let right_child_hash = std::array::from_fn(|_| take());
        let left_direction_different = take();
        let right_direction_different = take();

        Self {
            expand_direction,
            parent_as_label,
            children_height_section: parent_height_section,
            children_height_within: parent_height_within,
            height_inverse,
            parent_address_label: parent_label,
            parent_hash,
            left_child_hash,
            right_child_hash,
            left_direction_different,
            right_direction_different,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![
            self.expand_direction.clone(),
            self.children_height_section.clone(),
            self.children_height_within.clone(),
            self.height_inverse.clone(),
            self.parent_as_label.clone(),
            self.parent_address_label.clone(),
        ];
        result.extend(self.parent_hash.clone());
        result.extend(self.left_child_hash.clone());
        result.extend(self.right_child_hash.clone());
        result.push(self.left_direction_different.clone());
        result.push(self.right_direction_different.clone());
        result
    }

    pub fn get_width() -> usize {
        6 + (3 * CHUNK) + 2
    }
}
