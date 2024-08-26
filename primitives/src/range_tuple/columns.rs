#[derive(Default, Copy, Clone)]
pub struct RangeTupleCols<T> {
    pub mult: T,
}

#[derive(Default, Clone)]
pub struct RangeTuplePreprocessedCols<T> {
    pub counters: Vec<T>,
}
