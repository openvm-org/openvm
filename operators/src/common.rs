#[derive(PartialEq, Clone, Debug)]
pub struct Commitment<const LEN: usize> {
    commit: [u32; LEN],
}

impl<const LEN: usize> Default for Commitment<LEN> {
    fn default() -> Self {
        Self { commit: [0; LEN] }
    }
}

impl<const LEN: usize> Commitment<LEN> {
    pub fn from_slice(slice: &[u32]) -> Self {
        Self {
            commit: slice.try_into().unwrap(),
        }
    }

    pub fn flatten(&self) -> Vec<u32> {
        self.commit.to_vec()
    }
}
