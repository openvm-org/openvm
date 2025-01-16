use rustc_hash::FxHashMap;

pub trait VanEmdeBoas: Clone {
    /// Constructs a new empty van Emde Boas tree.
    fn new() -> Self;

    /// Inserts a key into the tree.
    fn insert(&mut self, key: u32);

    /// Removes a key from the tree. The key must be present in the tree.
    fn erase(&mut self, key: u32);

    /// Checks if a key is present in the tree.
    fn contains(&self, key: u32) -> bool;

    /// Checks if the tree is empty.
    fn empty(&self) -> bool;

    /// Returns the smallest key in the tree.
    fn min(&self) -> Option<u32>;

    /// Returns the largest key in the tree.
    fn max(&self) -> Option<u32>;

    /// Returns the largest key in the tree less than or equal to the given key.
    fn max_not_exceeding(&self, key: u32) -> Option<u32>;
}

#[derive(Clone)]
pub struct VebLeaf {
    mask: u64,
}

impl VanEmdeBoas for VebLeaf {
    fn new() -> Self {
        Self { mask: 0 }
    }

    fn insert(&mut self, key: u32) {
        self.mask |= 1 << key;
    }

    fn erase(&mut self, key: u32) {
        self.mask &= !(1 << key);
    }

    fn contains(&self, key: u32) -> bool {
        (self.mask & (1 << key)) > 0
    }

    fn empty(&self) -> bool {
        self.mask == 0
    }

    fn min(&self) -> Option<u32> {
        if self.mask == 0 {
            None
        } else {
            Some(self.mask.trailing_zeros())
        }
    }

    fn max(&self) -> Option<u32> {
        if self.mask == 0 {
            None
        } else {
            Some(self.mask.ilog2())
        }
    }

    fn max_not_exceeding(&self, key: u32) -> Option<u32> {
        if key == 63 {
            return self.max();
        }
        let mask = self.mask & ((1 << (key + 1)) - 1);
        if mask == 0 {
            None
        } else {
            Some(mask.ilog2())
        }
    }
}

#[derive(Clone)]
pub struct VebNode<const LOW_BITS: usize, const HIGH_CNT: usize, SmallKeys, HighKeys>
where
    SmallKeys: VanEmdeBoas,
    HighKeys: VanEmdeBoas,
{
    existing_children: HighKeys,
    subtrees: Vec<Option<SmallKeys>>,
    min: Option<u32>,
    max: Option<u32>,
}

impl<const LOW_BITS: usize, const HIGH_CNT: usize, SmallKeys, HighKeys>
    VebNode<LOW_BITS, HIGH_CNT, SmallKeys, HighKeys>
where
    SmallKeys: VanEmdeBoas,
    HighKeys: VanEmdeBoas,
{
    fn high(key: u32) -> u32 {
        key >> LOW_BITS
    }

    fn low(key: u32) -> u32 {
        key & ((1 << LOW_BITS) - 1)
    }
}

impl<const LOW_BITS: usize, const HIGH_CNT: usize, SmallKeys, HighKeys> VanEmdeBoas
    for VebNode<LOW_BITS, HIGH_CNT, SmallKeys, HighKeys>
where
    SmallKeys: VanEmdeBoas,
    HighKeys: VanEmdeBoas,
{
    fn new() -> Self {
        Self {
            existing_children: HighKeys::new(),
            subtrees: vec![None; HIGH_CNT],
            min: None,
            max: None,
        }
    }

    fn insert(&mut self, key: u32) {
        self.existing_children.insert(Self::high(key));
        self.subtrees[Self::high(key) as usize]
            .get_or_insert(SmallKeys::new())
            .insert(Self::low(key));
        if let Some(k) = self.min {
            if key < k {
                self.min = Some(key);
            }
        } else {
            self.min = Some(key);
        }
        if let Some(k) = self.max {
            if key > k {
                self.max = Some(key);
            }
        } else {
            self.max = Some(key);
        }
    }

    fn erase(&mut self, key: u32) {
        self.subtrees[Self::high(key) as usize]
            .get_or_insert(SmallKeys::new())
            .erase(Self::low(key));
        if self.subtrees[Self::high(key) as usize]
            .as_ref()
            .unwrap()
            .empty()
        {
            self.existing_children.erase(Self::high(key));
            self.subtrees[Self::high(key) as usize] = None;
        }
        if key == self.min.unwrap() {
            if key == self.max.unwrap() {
                self.min = None;
                self.max = None;
            } else {
                let high = <HighKeys as VanEmdeBoas>::min(&self.existing_children).unwrap();
                let low = self.subtrees[high as usize]
                    .as_ref()
                    .unwrap()
                    .min()
                    .unwrap();
                self.min = Some(high << LOW_BITS | low);
            }
        } else if key == self.max.unwrap() {
            let high = <HighKeys as VanEmdeBoas>::max(&self.existing_children).unwrap();
            let low = self.subtrees[high as usize]
                .as_ref()
                .unwrap()
                .max()
                .unwrap();
            self.max = Some(high << LOW_BITS | low);
        }
    }

    fn contains(&self, key: u32) -> bool {
        self.subtrees[Self::high(key) as usize]
            .as_ref()
            .unwrap()
            .contains(Self::low(key))
    }

    fn empty(&self) -> bool {
        self.min.is_none()
    }

    fn min(&self) -> Option<u32> {
        self.min
    }

    fn max(&self) -> Option<u32> {
        self.max
    }

    fn max_not_exceeding(&self, key: u32) -> Option<u32> {
        if let Some(subtree) = self.subtrees[Self::high(key) as usize].as_ref() {
            if let Some(low) = subtree.max_not_exceeding(Self::low(key)) {
                return Some(Self::high(key) << LOW_BITS | low);
            }
        }
        if Self::high(key) > 0 {
            if let Some(high) = self
                .existing_children
                .max_not_exceeding(Self::high(key) - 1)
            {
                let low = self.subtrees[high as usize]
                    .as_ref()
                    .unwrap()
                    .max()
                    .unwrap();
                Some(high << LOW_BITS | low)
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub type _VebTree1 = VebNode<6, 64, VebLeaf, VebLeaf>; // 12 bits
pub type _VebTree2 = VebNode<12, 64, _VebTree1, VebLeaf>; // 18 bits
pub type _VebTree3 = VebNode<12, 262144, _VebTree1, _VebTree2>; // 30 bits

pub type VebTree = _VebTree3;

#[derive(Clone)]
pub struct VebMap<T> {
    tree: VebTree,
    values: FxHashMap<u32, T>,
}

impl<T> VebMap<T> {
    pub fn new() -> Self {
        Self {
            tree: VebTree::new(),
            values: FxHashMap::default(),
        }
    }

    pub fn insert(&mut self, key: u32, value: T) -> Option<T> {
        self.tree.insert(key);
        self.values.insert(key, value)
    }

    pub fn erase(&mut self, key: u32) {
        self.tree.erase(key);
        self.values.remove(&key);
    }

    pub fn get(&self, key: u32) -> Option<&T> {
        self.values.get(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.values.iter().map(|(key, value)| (*key, value))
    }

    pub fn max_not_exceeding(&self, key: u32) -> Option<u32> {
        self.tree.max_not_exceeding(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_veb_tree() {
        let mut tree = VebTree::new();
        for i in 0..100 {
            tree.insert(i * i);
        }
        for i in 0..100 {
            assert!(tree.contains(i * i));
        }

        assert_eq!(tree.min(), Some(0));
        assert_eq!(tree.max(), Some(99 * 99));
        assert_eq!(tree.max_not_exceeding(99 * 99 - 1), Some(98 * 98));
        assert_eq!(tree.max_not_exceeding(99 * 99), Some(99 * 99));
        assert_eq!(tree.max_not_exceeding(99 * 99 + 1), Some(99 * 99));
        assert_eq!(tree.max_not_exceeding(u32::MAX), Some(99 * 99));
        tree.erase(0);
        assert_eq!(tree.min(), Some(1));
        assert_eq!(tree.max(), Some(99 * 99));
        assert_eq!(tree.max_not_exceeding(0), None);
        for i in 1..100 {
            tree.erase(i * i);
        }
        for i in 0..100 {
            assert!(!tree.contains(i * i));
        }
    }
}
