mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod execution;
pub mod memory;
pub mod program;
mod utils;

use std::marker::PhantomData;

pub use cpu::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use execution::ExecutionTester;
pub use memory::MemoryTester;
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{interaction::BusIndex, p3_air::BaseAir};
use p3_field::Field;
pub use utils::*;

use crate::arch::{Arena, ExecutionState, MatrixRecordArena, PreflightExecutor, Streams};

pub const EXECUTION_BUS: BusIndex = 0;
pub const MEMORY_BUS: BusIndex = 1;
pub const POSEIDON2_DIRECT_BUS: BusIndex = 6;
pub const READ_INSTRUCTION_BUS: BusIndex = 8;
pub const BITWISE_OP_LOOKUP_BUS: BusIndex = 9;
pub const BYTE_XOR_BUS: BusIndex = 10;
pub const RANGE_TUPLE_CHECKER_BUS: BusIndex = 11;
pub const MEMORY_MERKLE_BUS: BusIndex = 12;

pub const RANGE_CHECKER_BUS: BusIndex = 4;

pub type ArenaId = usize;

pub struct TestChipHarness<F, E, A, C, RA = MatrixRecordArena<F>> {
    pub executor: E,
    pub air: A,
    pub chip: C,
    pub arena: RA,
    phantom: PhantomData<F>,
}

impl<F, E, A, C, RA> TestChipHarness<F, E, A, C, RA>
where
    F: Field,
    A: BaseAir<F>,
    RA: Arena,
{
    pub fn with_capacity(executor: E, air: A, chip: C, height: usize) -> Self {
        let width = air.width();
        let height = next_power_of_two_or_zero(height);
        let arena = RA::with_capacity(height, width);
        Self {
            executor,
            air,
            chip,
            arena,
            phantom: PhantomData,
        }
    }
}

pub trait TestBuilder<F> {
    fn execute<E: PreflightExecutor<F, RA>, RA: Arena>(
        &mut self,
        executor: &mut E,
        arena: &mut RA,
        instruction: &Instruction<F>,
    );

    fn execute_with_pc<E: PreflightExecutor<F, RA>, RA: Arena>(
        &mut self,
        executor: &mut E,
        arena: &mut RA,
        instruction: &Instruction<F>,
        initial_pc: u32,
    );

    fn write_cell(&mut self, address_space: usize, pointer: usize, value: F);
    fn read_cell(&mut self, address_space: usize, pointer: usize) -> F;

    fn write<const N: usize>(&mut self, address_space: usize, pointer: usize, value: [F; N]);
    fn read<const N: usize>(&mut self, address_space: usize, pointer: usize) -> [F; N];

    fn write_usize<const N: usize>(
        &mut self,
        address_space: usize,
        pointer: usize,
        value: [usize; N],
    );

    fn address_bits(&self) -> usize;

    fn last_to_pc(&self) -> F;
    fn last_from_pc(&self) -> F;

    fn execution_final_state(&self) -> ExecutionState<F>;
    fn streams_mut(&mut self) -> &mut Streams<F>;

    fn get_default_register(&mut self, increment: usize) -> usize;
    fn get_default_pointer(&mut self, increment: usize) -> usize;

    fn write_heap_pointer_default(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
    ) -> (usize, usize);

    fn write_heap_default<const NUM_LIMBS: usize>(
        &mut self,
        reg_increment: usize,
        pointer_increment: usize,
        writes: Vec<[F; NUM_LIMBS]>,
    ) -> (usize, usize);
}
