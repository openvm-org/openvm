use crate::cpu::trace::Instruction;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct ProgramAir<T> {
    program: Vec<Instruction<T>>,
}
