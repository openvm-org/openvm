#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ScopePath(pub Vec<(usize, String)>);

impl ScopePath {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    pub fn then(&self, index: usize, name: String) -> Self {
        Self(
            self.0
                .clone()
                .into_iter()
                .chain(vec![(index, name)])
                .collect(),
        )
    }

    pub fn concat(&self, offset: usize, other: &Self) -> Self {
        let mut result = self.0.clone();
        result.push(other.0[0].clone());
        result.last_mut().unwrap().0 += offset;
        result.extend(other.0.clone().into_iter().skip(1));
        Self(result)
    }

    pub fn prepend(&mut self, other: &Self, offset: usize) {
        *self = other.concat(offset, self);
    }

    pub fn disjoint(&self, other: &Self) -> bool {
        // NOT zip_eq
        for ((index1, constructor1), (index2, constructor2)) in self.0.iter().zip(other.0.iter()) {
            if index1 != index2 {
                return false;
            }
            if constructor1 != constructor2 {
                return true;
            }
        }
        false
    }

    pub fn prefixes<'a>(&'a self) -> impl DoubleEndedIterator<Item = Self> + 'a {
        (0..=self.0.len()).map(move |i| ScopePath(self.0[..i].to_vec()))
    }

    pub fn is_prefix(&self, other: &Self, index: usize) -> bool {
        other.0.len() < self.0.len()
            && self.0[..other.0.len()] == other.0
            && self.0[other.0.len()].0 == index
    }
}
