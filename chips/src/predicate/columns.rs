use super::Comp;

pub struct PredicateIOCols<T> {
    pub x: T,
    pub y: T,
    pub cmp: Comp,
}
