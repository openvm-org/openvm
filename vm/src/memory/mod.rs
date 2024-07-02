pub mod offline_checker;
#[cfg(test)]
pub mod tests;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryAccess<const WORD_SIZE: usize, F> {
    pub timestamp: usize,
    pub op_type: OpType,
    pub address_space: F,
    pub address: F,
    pub data: [F; WORD_SIZE],
}
