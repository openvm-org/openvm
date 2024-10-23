use afs_derive::AlignedBorrow;

#[derive(Debug)]
pub struct MemoryMerkleCols<const CHUNK: usize, T> {
    // `expand_direction` =  1 corresponds to initial memory state
    // `expand_direction` = -1 corresponds to final memory state
    // `expand_direction` =  0 corresponds to irrelevant row (all interactions multiplicity 0)
    pub expand_direction: T,

    // height_section = 1 indicates that as_label is being expanded
    // height_section = 0 indicates that address_label is being expanded
    pub height_section: T,
    pub parent_height: T,
    pub is_root: T,

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

impl<const CHUNK: usize, T: Clone> MemoryMerkleCols<CHUNK, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let mut iter = slc.iter();
        let mut take = || iter.next().unwrap().clone();

        let expand_direction = take();
        let height_section = take();
        let parent_height = take();
        let is_root = take();
        let parent_as_label = take();
        let parent_address_label = take();

        let parent_hash = std::array::from_fn(|_| take());
        let left_child_hash = std::array::from_fn(|_| take());
        let right_child_hash = std::array::from_fn(|_| take());

        let left_direction_different = take();
        let right_direction_different = take();

        Self {
            expand_direction,
            parent_as_label,
            height_section,
            parent_height,
            is_root,
            parent_address_label,
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
            self.height_section.clone(),
            self.parent_height.clone(),
            self.is_root.clone(),
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

#[derive(Debug, AlignedBorrow)]
#[repr(C)]
pub struct MemoryMerklePvs<T, const CHUNK: usize> {
    pub initial_root: [T; CHUNK],
    pub final_root: [T; CHUNK],
}
