use std::iter;

use p3_field::PrimeField32;

use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

pub struct UintMultiplicationCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone> {
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
pub struct UintMultiplicationIoCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub y: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub z: MemoryData<NUM_LIMBS, LIMB_BITS, T>,
    pub d: T,
    pub e: T,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone>
    UintMultiplicationIoCols<NUM_LIMBS, LIMB_BITS, T>
{
    pub fn width() -> usize {
        MemoryData::<NUM_LIMBS, LIMB_BITS, T>::width() * 3 + ExecutionState::<T>::get_width() + 2
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let from_state = ExecutionState::from_iter(iter.by_ref());
        let x = MemoryData::from_iterator(iter.by_ref());
        let y = MemoryData::from_iterator(iter.by_ref());
        let z = MemoryData::from_iterator(iter.by_ref());
        let d = iter.next().unwrap();
        let e = iter.next().unwrap();
        Self {
            from_state,
            x,
            y,
            z,
            d,
            e,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .chain(iter::once(&self.d))
            .chain(iter::once(&self.e))
            .cloned()
            .collect()
    }
}

pub struct UintMultiplicationAuxCols<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone> {
    pub is_valid: T,
    pub carry: Vec<T>,
    pub read_ptr_aux_cols: [MemoryReadAuxCols<1, T>; 3],
    pub read_x_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub read_y_aux_cols: MemoryReadAuxCols<NUM_LIMBS, T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<NUM_LIMBS, T>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone>
    UintMultiplicationAuxCols<NUM_LIMBS, LIMB_BITS, T>
{
    pub fn width() -> usize {
        3 * MemoryReadAuxCols::<1, T>::width()
            + MemoryReadAuxCols::<NUM_LIMBS, T>::width()
            + MemoryReadAuxCols::<NUM_LIMBS, T>::width()
            + MemoryWriteAuxCols::<NUM_LIMBS, T>::width()
            + NUM_LIMBS
            + 1
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let is_valid = iter.next().unwrap();
        let carry = iter.by_ref().take(NUM_LIMBS).collect();

        let read_width = MemoryReadAuxCols::<NUM_LIMBS, T>::width();
        let write_width = MemoryWriteAuxCols::<NUM_LIMBS, T>::width();

        let ptr_read_width = MemoryReadAuxCols::<1, T>::width();
        let read_ptr_aux_cols = [(); 3].map(|_| {
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
            self.carry.clone(),
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
            carry: vec![Default::default(); NUM_LIMBS],
            read_ptr_aux_cols: [(); 3].map(|_| MemoryReadAuxCols::disabled()),
            read_x_aux_cols: MemoryReadAuxCols::disabled(),
            read_y_aux_cols: MemoryReadAuxCols::disabled(),
            write_z_aux_cols: MemoryWriteAuxCols::disabled(),
        }
    }
}

pub struct MemoryData<const NUM_LIMBS: usize, const LIMB_BITS: usize, T> {
    pub data: Vec<T>,
    pub address: T,
    pub ptr_to_address: T,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone + Default> Default
    for MemoryData<NUM_LIMBS, LIMB_BITS, T>
{
    fn default() -> Self {
        Self {
            data: vec![Default::default(); NUM_LIMBS],
            address: Default::default(),
            ptr_to_address: Default::default(),
        }
    }
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize, T: Clone> MemoryData<NUM_LIMBS, LIMB_BITS, T> {
    pub fn width() -> usize {
        NUM_LIMBS + 2
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        Self {
            data: iter.by_ref().take(NUM_LIMBS).collect(),
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
