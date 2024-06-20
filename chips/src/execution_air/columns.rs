#[derive(Debug)]
pub struct ExecutionCols<T> {
    pub mult: T,
    pub clk: T,
    pub idx: Vec<T>,
    pub data: Vec<T>,
    pub op_type: T,
}

impl<T: Clone> ExecutionCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, data_len: usize) -> ExecutionCols<T> {
        ExecutionCols {
            mult: cols[0].clone(),
            clk: cols[1].clone(),
            idx: cols[2..2 + idx_len].to_vec(),
            data: cols[2 + idx_len..2 + idx_len + data_len].to_vec(),
            op_type: cols[2 + idx_len + data_len].clone(),
        }
    }
}
