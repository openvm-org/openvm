use core::mem::size_of;

#[derive(Default, Copy, Clone)]
pub struct RangeTupleCols<T> {
    pub mult: T,
}

#[derive(Default, Clone)]
pub struct RangeTuplePreprocessedCols<T> {
    pub tuple: Vec<T>,
}

pub const NUM_RANGE_TUPLE_COLS: usize = size_of::<RangeTupleCols<u8>>();
