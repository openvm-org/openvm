pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

pub enum Comp {
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}

pub struct PredicateAir {}

pub struct PredicateChip {
    pub air: PredicateAir,
}
