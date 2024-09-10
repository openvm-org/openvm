use std::iter;

use super::{ModularArithmeticAir, NUM_LIMBS};
use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
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
}

impl<T: Clone> ModularArithmeticIoCols<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            from_state: ExecutionState::from_iter(iter.by_ref()),
            x: MemoryData::from_iterator(iter.by_ref()),
            y: MemoryData::from_iterator(iter.by_ref()),
            z: MemoryData::from_iterator(iter.by_ref()),
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
        NUM_LIMBS * 3 + 8
    }
}

pub struct ModularArithmeticAuxCols<T: Clone> {
    pub read_x_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub read_y_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<NUM_LIMBS, T>,
}

impl<T: Clone> ModularArithmeticAuxCols<T> {
    pub fn width(air: &ModularArithmeticAir) -> usize {
        MemoryReadAuxCols::<NUM_LIMBS, T>::width(&air.mem_oc)
            + MemoryReadAuxCols::<NUM_LIMBS, T>::width(&air.mem_oc)
            + MemoryWriteAuxCols::<NUM_LIMBS, T>::width(&air.mem_oc)
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>, air: &ModularArithmeticAir) -> Self {
        let mem_oc = &air.mem_oc;
        let width = MemoryReadAuxCols::<NUM_LIMBS, T>::width(mem_oc);
        let read_x_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let read_x_aux_cols = MemoryReadAuxCols::<NUM_LIMBS, T>::from_slice(&read_x_slice, mem_oc);

        let read_y_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let read_y_aux_cols = MemoryReadAuxCols::<NUM_LIMBS, T>::from_slice(&read_y_slice, mem_oc);

        let write_z_slice = iter.by_ref().take(width).collect::<Vec<_>>();
        let write_z_aux_cols =
            MemoryWriteAuxCols::<NUM_LIMBS, T>::from_slice(&write_z_slice, mem_oc);

        Self {
            read_x_aux_cols,
            read_y_aux_cols,
            write_z_aux_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            self.read_x_aux_cols.clone().flatten(),
            self.read_y_aux_cols.clone().flatten(),
            self.write_z_aux_cols.clone().flatten(),
        ]
        .concat()
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
