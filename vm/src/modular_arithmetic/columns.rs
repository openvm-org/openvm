use std::iter;

use super::{ModularArithmeticAir, NUM_LIMBS};
use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

pub struct ModularArithmeticCols<T: Clone> {
    pub io: ModularArithmeticIoCols<T>,
    pub aux: ModularArithmeticAuxCols<T>,
}

impl<T: Clone> ModularArithmeticCols<T> {
    pub fn width(air: &ModularArithmeticAir) -> usize {
        ModularArithmeticIoCols::<T>::width() + ModularArithmeticAuxCols::<T>::width(air)
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>, air: &ModularArithmeticAir) -> Self {
        Self {
            io: ModularArithmeticIoCols::from_iterator(iter.by_ref()),
            aux: ModularArithmeticAuxCols::from_iterator(iter.by_ref(), air),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }
}

#[derive(Default)]
pub struct ModularArithmeticIoCols<T: Clone> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<T>,
    pub y: MemoryData<T>,
    pub z: MemoryData<T>,
    pub x_address: MemoryData<T>,
    pub y_address: MemoryData<T>,
    pub z_address: MemoryData<T>,
}

impl<T: Clone> ModularArithmeticIoCols<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            from_state: ExecutionState::from_iter(iter.by_ref()),
            x: MemoryData::from_iterator(iter.by_ref()),
            y: MemoryData::from_iterator(iter.by_ref()),
            z: MemoryData::from_iterator(iter.by_ref()),
            x_address: MemoryData::from_iterator(iter.by_ref()),
            y_address: MemoryData::from_iterator(iter.by_ref()),
            z_address: MemoryData::from_iterator(iter.by_ref()),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .chain(self.x_address.flatten())
            .chain(self.y_address.flatten())
            .chain(self.z_address.flatten())
            .cloned()
            .collect()
    }

    pub fn width() -> usize {
        NUM_LIMBS * 3 + 17
    }
}

pub struct ModularArithmeticAuxCols<T: Clone> {
    // 0 for padding rows.
    pub is_valid: T,
    pub read_x_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub read_y_aux_cols: MemoryReadAuxCols<T, NUM_LIMBS>,
    pub write_z_aux_cols: MemoryWriteAuxCols<T, NUM_LIMBS>,
    pub x_address_aux_cols: MemoryReadAuxCols<T, 1>,
    pub y_address_aux_cols: MemoryReadAuxCols<T, 1>,
    pub z_address_aux_cols: MemoryReadAuxCols<T, 1>,

    pub carries: Vec<T>,
    pub q: Vec<T>,
}

impl<T: Clone> ModularArithmeticAuxCols<T> {
    pub fn width(air: &ModularArithmeticAir) -> usize {
        // FIXME: the length of carries and q depend on operation
        MemoryReadAuxCols::<T, NUM_LIMBS>::width()
            + MemoryReadAuxCols::<T, NUM_LIMBS>::width()
            + MemoryWriteAuxCols::<T, NUM_LIMBS>::width()
            + MemoryReadAuxCols::<T, 1>::width()
            + MemoryReadAuxCols::<T, 1>::width()
            + MemoryReadAuxCols::<T, 1>::width()
            + air.carry_limbs
            + air.q_limbs
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>, air: &ModularArithmeticAir) -> Self {
        let is_valid = iter.next().unwrap();
        let width = MemoryReadAuxCols::<T, NUM_LIMBS>::width();
        let read_x_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let read_x_aux_cols = MemoryReadAuxCols::<T, NUM_LIMBS>::from_slice(&read_x_slice);

        let read_y_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let read_y_aux_cols = MemoryReadAuxCols::<T, NUM_LIMBS>::from_slice(&read_y_slice);

        let write_z_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let write_z_aux_cols = MemoryWriteAuxCols::<T, NUM_LIMBS>::from_slice(&write_z_slice);

        let width2 = MemoryReadAuxCols::<T, 1>::width();
        let x_address_slice = iter.by_ref().take(width2).collect::<Vec<_>>();
        let x_address_aux_cols = MemoryReadAuxCols::<T, 1>::from_slice(&x_address_slice);

        let y_address_slice = iter.by_ref().take(width2).collect::<Vec<_>>();
        let y_address_aux_cols = MemoryReadAuxCols::<T, 1>::from_slice(&y_address_slice);

        let z_address_slice = iter.by_ref().take(width2).collect::<Vec<_>>();
        let z_address_aux_cols = MemoryReadAuxCols::<T, 1>::from_slice(&z_address_slice);

        let carries = iter.by_ref().take(air.carry_limbs).collect::<Vec<_>>();
        let q = iter.by_ref().take(air.q_limbs).collect::<Vec<_>>();

        Self {
            is_valid,
            read_x_aux_cols,
            read_y_aux_cols,
            write_z_aux_cols,
            x_address_aux_cols,
            y_address_aux_cols,
            z_address_aux_cols,
            carries,
            q,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let valid = iter::once(&self.is_valid).cloned().collect::<Vec<_>>();
        let mem = [
            self.read_x_aux_cols.clone().flatten(),
            self.read_y_aux_cols.clone().flatten(),
            self.write_z_aux_cols.clone().flatten(),
            self.x_address_aux_cols.clone().flatten(),
            self.y_address_aux_cols.clone().flatten(),
            self.z_address_aux_cols.clone().flatten(),
        ]
        .concat();

        [valid, mem, self.carries.clone(), self.q.clone()].concat()
    }
}

pub struct MemoryData<T: Clone> {
    pub data: Vec<T>,
    pub address_space: T,
    pub address: T,
}

impl<T: Clone> MemoryData<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            data: iter.by_ref().take(NUM_LIMBS).collect(),
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
