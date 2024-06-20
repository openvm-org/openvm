pub mod offline_checker;
#[cfg(test)]
pub mod tests;

#[derive(PartialEq, Clone, Debug)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

#[derive(Clone, Debug)]
pub struct Operation {
    pub clk: usize,
    pub addr_space: u32,
    pub pointer: u32,
    pub data: Vec<u32>,
    pub op_type: OpType,
}
