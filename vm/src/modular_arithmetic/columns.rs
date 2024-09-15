use std::iter;

use super::{ModularArithmeticAirVariant, ModularArithmeticVmAir, NUM_LIMBS};
use crate::{
    arch::columns::ExecutionState,
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapDataIoCols,
    },
};

pub struct ModularArithmeticCols<T: Clone> {
    pub io: ModularArithmeticIoCols<T>,
    pub aux: ModularArithmeticAuxCols<T>,
}

impl<T: Clone> ModularArithmeticCols<T> {
    pub fn width(air: &ModularArithmeticVmAir<ModularArithmeticAirVariant>) -> usize {
        ModularArithmeticIoCols::<T>::width() + ModularArithmeticAuxCols::<T>::width(air)
    }

    pub fn from_iterator(
        mut iter: impl Iterator<Item = T>,
        air: &ModularArithmeticVmAir<ModularArithmeticAirVariant>,
    ) -> Self {
        Self {
            io: ModularArithmeticIoCols::from_iterator(iter.by_ref()),
            aux: ModularArithmeticAuxCols::from_iterator(iter.by_ref(), air),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }
}

pub struct ModularArithmeticIoCols<T: Clone> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub y: MemoryHeapDataIoCols<T, NUM_LIMBS>,
    pub z: MemoryHeapDataIoCols<T, NUM_LIMBS>,
}

impl<T: Clone> ModularArithmeticIoCols<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            from_state: ExecutionState::from_iter(iter.by_ref()),
            x: MemoryHeapDataIoCols::from_iterator(iter.by_ref()),
            y: MemoryHeapDataIoCols::from_iterator(iter.by_ref()),
            z: MemoryHeapDataIoCols::from_iterator(iter.by_ref()),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .cloned()
            .collect()
    }

    pub fn width() -> usize {
        // from_state = 2, memory_data = 2 + len,
        // 2 + 3 * (len + 2) + 3 * 3
        NUM_LIMBS * 3 + 17
    }
}

pub struct ModularArithmeticAuxCols<T: Clone> {
    // 0 for padding rows.
    pub is_valid: T,
    pub read_x_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryHeapReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryHeapWriteAuxCols<T, NUM_LIMBS>,

    pub carries: Vec<T>,
    pub q: Vec<T>,
    pub opcode: T,
}

impl<T: Clone> ModularArithmeticAuxCols<T> {
    pub fn width(air: &ModularArithmeticVmAir<ModularArithmeticAirVariant>) -> usize {
        MemoryHeapReadAuxCols::<T, NUM_LIMBS>::width() * 2
            + MemoryHeapWriteAuxCols::<T, NUM_LIMBS>::width()
            + air.carry_limbs
            + air.q_limbs
            + 2
    }

    pub fn from_iterator(
        mut iter: impl Iterator<Item = T>,
        air: &ModularArithmeticVmAir<ModularArithmeticAirVariant>,
    ) -> Self {
        let is_valid = iter.next().unwrap();
        let read_x_aux_cols = MemoryHeapReadAuxCols::<T, NUM_LIMBS>::from_iterator(&mut iter);
        let read_y_aux_cols = MemoryHeapReadAuxCols::<T, NUM_LIMBS>::from_iterator(&mut iter);
        let write_z_aux_cols = MemoryHeapWriteAuxCols::<T, NUM_LIMBS>::from_iterator(&mut iter);

        let carries = iter.by_ref().take(air.carry_limbs).collect::<Vec<_>>();
        let q = iter.by_ref().take(air.q_limbs).collect::<Vec<_>>();
        let opcode = iter.next().unwrap();
        Self {
            is_valid,
            read_x_aux_cols,
            read_y_aux_cols,
            write_z_aux_cols,
            carries,
            q,
            opcode,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let valid = iter::once(&self.is_valid).cloned().collect::<Vec<_>>();
        let mem = [
            self.read_x_aux_cols.clone().flatten(),
            self.read_y_aux_cols.clone().flatten(),
            self.write_z_aux_cols.clone().flatten(),
        ]
        .concat();

        [
            valid,
            mem,
            self.carries.clone(),
            self.q.clone(),
            vec![self.opcode.clone()],
        ]
        .concat()
    }
}

#[derive(Clone)]
pub struct MemoryData<T: Clone> {
    pub data: Vec<T>,
    pub address_space: T,
    pub address: T,
}

impl<T: Clone> MemoryData<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>, data_len: usize) -> Self {
        Self {
            data: iter.by_ref().take(data_len).collect(),
            address_space: iter.next().unwrap(),
            address: iter.next().unwrap(),
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.data
            .iter()
            .chain(iter::once(&self.address_space))
            .chain(iter::once(&self.address))
    }
}

impl<T: Clone + Default> Default for MemoryData<T> {
    fn default() -> Self {
        Self {
            data: vec![Default::default(); NUM_LIMBS],
            address_space: Default::default(),
            address: Default::default(),
        }
    }
}
