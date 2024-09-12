use std::{array, iter, mem::size_of};

use p3_field::PrimeField32;

use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

pub struct UintMultiplicationCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> {
    pub io: UintMultiplicationIoCols<NUM_LIMBS, LIMB_BITS, T>,
    pub aux: UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, T>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone>
    UintMultiplicationCols<NUM_LIMBS, LIMB_BITS, T>
{
    pub fn width() -> usize {
        UintMultiplicationAuxCols::<NUM_LIMBS, LIMB_BITS, T>::width()
            + UintMultiplicationIoCols::<NUM_LIMBS, LIMB_BITS, T>::width()
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let io = UintMultiplicationIoCols::from_iterator(iter.by_ref());
        let aux = UintMultiplicationAuxCols::from_iterator(iter.by_ref());
        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, F: PrimeField32> Default
    for UintMultiplicationCols<NUM_LIMBS, LIMB_BITS, F>
{
    fn default() -> Self {
        Self {
            io: Default::default(),
            aux: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct UintMultiplicationIoCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub y: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub z: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub ptr_as: T,
    pub address_as: T,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone>
    UintMultiplicationIoCols<NUM_LIMBS, LIMB_BITS, T>
{
    pub fn width() -> usize {
        size_of::<UintMultiplicationIoCols<NUM_LIMBS, LIMB_BITS, u8>>()
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let from_state = ExecutionState::from_iter(iter.by_ref());
        let x = MemoryData::from_iterator(iter.by_ref());
        let y = MemoryData::from_iterator(iter.by_ref());
        let z = MemoryData::from_iterator(iter.by_ref());
        let ptr_as = iter.next().unwrap();
        let address_as = iter.next().unwrap();
        Self {
            from_state,
            x,
            y,
            z,
            ptr_as,
            address_as,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .chain(iter::once(&self.ptr_as))
            .chain(iter::once(&self.address_as))
            .cloned()
            .collect()
    }
}

pub struct UintMultiplicationAuxCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> {
    pub is_valid: T,
    pub carry: [T; NUM_LIMBS],
    pub read_ptr_aux_cols: [MemoryReadAuxCols<1, T>; 3],
    pub read_x_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub read_y_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<NUM_LIMBS, T>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone>
    UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, T>
{
    pub fn width() -> usize {
        size_of::<UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, u8>>()
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let is_valid = iter.next().unwrap();
        let carry = array::from_fn(|_| iter.next().unwrap());

        let read_width = MemoryReadAuxCols::<NUM_LIMBS, T>::width();
        let write_width = MemoryWriteAuxCols::<NUM_LIMBS, T>::width();

        let ptr_read_width = MemoryReadAuxCols::<1, T>::width();
        let read_ptr_aux_cols = array::from_fn(|_| {
            MemoryReadAuxCols::<1, T>::from_slice(
                &iter.by_ref().take(ptr_read_width).collect::<Vec<_>>(),
            )
        });

        let read_x_slice = iter.by_ref().take(read_width).collect::<Vec<_>>();
        let read_y_slice = iter.by_ref().take(read_width).collect::<Vec<_>>();
        let write_z_slice = iter.by_ref().take(write_width).collect::<Vec<_>>();
        let read_x_aux_cols = MemoryReadAuxCols::<NUM_LIMBS, T>::from_slice(&read_x_slice);
        let read_y_aux_cols = MemoryReadAuxCols::<NUM_LIMBS, T>::from_slice(&read_y_slice);
        let write_z_aux_cols = MemoryWriteAuxCols::<NUM_LIMBS, T>::from_slice(&write_z_slice);

        Self {
            is_valid,
            carry,
            read_ptr_aux_cols,
            read_x_aux_cols,
            read_y_aux_cols,
            write_z_aux_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            vec![self.is_valid.clone()],
            self.carry.to_vec(),
            self.read_ptr_aux_cols
                .iter()
                .flat_map(|c| c.clone().flatten())
                .collect::<Vec<_>>(),
            self.read_x_aux_cols.clone().flatten(),
            self.read_y_aux_cols.clone().flatten(),
            self.write_z_aux_cols.clone().flatten(),
        ]
        .concat()
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, F: PrimeField32> Default
    for UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, F>
{
    fn default() -> Self {
        Self {
            is_valid: Default::default(),
            carry: array::from_fn(|_| Default::default()),
            read_ptr_aux_cols: array::from_fn(|_| MemoryReadAuxCols::disabled()),
            read_x_aux_cols: MemoryReadAuxCols::disabled(),
            read_y_aux_cols: MemoryReadAuxCols::disabled(),
            write_z_aux_cols: MemoryWriteAuxCols::disabled(),
        }
    }
}

pub struct MemoryData<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> {
    pub data: [T; NUM_LIMBS],
    pub address: T,
    pub ptr_to_address: T,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Default> Default
    for MemoryData<NUM_LIMBS, LIMB_BITS, T>
{
    fn default() -> Self {
        Self {
            data: array::from_fn(|_| Default::default()),
            address: Default::default(),
            ptr_to_address: Default::default(),
        }
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> MemoryData<NUM_LIMBS, LIMB_BITS, T> {
    pub fn width() -> usize {
        size_of::<MemoryData<NUM_LIMBS, LIMB_BITS, u8>>()
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            data: array::from_fn(|_| iter.next().unwrap()),
            address: iter.next().unwrap(),
            ptr_to_address: iter.next().unwrap(),
        }
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.data
            .iter()
            .chain(iter::once(&self.address))
            .chain(iter::once(&self.ptr_to_address))
    }
}
