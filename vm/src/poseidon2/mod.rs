use crate::vm::VirtualMachine;
use afs_chips::sub_chip::LocalTraceInstructions;
use columns::{Poseidon2ChipCols, Poseidon2ChipIoCols};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeField32};
use poseidon2::poseidon2::Poseidon2Air;
use std::usize;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Poseidon2 chip.
///
/// Carries the requested rows and the underlying air.

pub struct Poseidon2Chip<const WIDTH: usize, F: Clone> {
    pub air: Poseidon2Air<WIDTH, F>,
    pub rows: Vec<Poseidon2ChipCols<WIDTH, F>>,
}

pub struct Poseidon2Query<F> {
    pub clk: usize,
    pub a: F,
    pub b: F,
    pub c: F,
    pub d: F,
    pub e: F,
    pub cmp: F,
}

impl<F: Field> Poseidon2Query<F> {
    pub fn to_io_cols(&self) -> Poseidon2ChipIoCols<F> {
        Poseidon2ChipIoCols::<F> {
            is_alloc: F::from_canonical_u32(1),
            clk: F::from_canonical_usize(self.clk),
            a: self.a,
            b: self.b,
            c: self.c,
            d: self.d,
            e: self.e,
            cmp: self.cmp,
        }
    }
}

impl Poseidon2Chip<16, BabyBear> {
    pub fn new() -> Self {
        let air = Poseidon2Air::<16, BabyBear>::new_p3_baby_bear_16();
        Self { air, rows: vec![] }
    }
}

impl<const WIDTH: usize, F: PrimeField32> Poseidon2Chip<WIDTH, F> {
    pub fn poseidon2_perm(vm: &mut VirtualMachine<WIDTH, F>, op: Poseidon2Query<F>) {
        let data_1: [F; 8] = core::array::from_fn(|i| {
            vm.memory_chip
                .read_elem(20 * op.clk + i, op.d, op.a + F::from_canonical_usize(i))
        });
        let data_2: [F; 8] = core::array::from_fn(|i| {
            vm.memory_chip
                .read_elem(20 * op.clk + 8 + i, op.d, op.b + F::from_canonical_usize(i))
        });
        let input_state: [F; 16] = [data_1, data_2].concat().try_into().unwrap();
        let aux = vm.poseidon2_chip.air.generate_trace_row(input_state);
        vm.poseidon2_chip.rows.push(Poseidon2ChipCols {
            io: op.to_io_cols(),
            aux,
        });
    }
}
