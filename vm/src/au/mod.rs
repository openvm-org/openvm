use super::cpu::trace::ArithmeticOperation;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, Copy)]
pub struct AUAir {}

pub struct AUChip<T> {
    pub air: AUAir,
    pub arithmetic_operations: Vec<ArithmeticOperation<T>>,
}

impl AUAir {
    pub const BASE_OP: u8 = 5;

    pub fn new() -> Self {
        Self {}
    }
}
