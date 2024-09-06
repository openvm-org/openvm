use std::iter;

use super::{num_limbs, LongArithmeticAir};
use crate::{
    arch::columns::ExecutionState,
    memory::offline_checker::columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

pub struct LongArithmeticCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone> {
    pub io: LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>,
    pub aux: LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>,
}

#[derive(Default)]
pub struct LongArithmeticIoCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone> {
    pub from_state: ExecutionState<T>,
    pub x: MemoryData<ARG_SIZE, LIMB_SIZE, T>,
    pub y: MemoryData<ARG_SIZE, LIMB_SIZE, T>,
    pub z: MemoryData<ARG_SIZE, LIMB_SIZE, T>,
    pub cmp_result: T,
}

pub struct MemoryData<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub data: Vec<T>,
    pub address_space: T,
    pub address: T,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone + Default> Default
    for MemoryData<ARG_SIZE, LIMB_SIZE, T>
{
    fn default() -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        Self {
            data: vec![Default::default(); num_limbs],
            address_space: Default::default(),
            address: Default::default(),
        }
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone> MemoryData<ARG_SIZE, LIMB_SIZE, T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        Self {
            data: iter.by_ref().take(num_limbs).collect(),
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

pub struct LongArithmeticAuxCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub is_valid: T,
    pub opcode_add_flag: T, // 1 if z_limbs should contain the result of addition
    pub opcode_sub_flag: T, // 1 if z_limbs should contain the result of subtraction (means that opcode is SUB or LT)
    pub opcode_lt_flag: T,  // 1 if opcode is LT
    pub opcode_eq_flag: T,  // 1 if opcode is EQ
    // buffer is the carry of the addition/subtraction,
    // or may serve as a single-nonzero-inverse helper vector for EQ256.
    // Refer to air.rs for more details.
    pub buffer: Vec<T>,

    // TODO: 16 -> normal thing
    pub read_x_aux_cols: MemoryReadAuxCols<16, T>,
    pub read_y_aux_cols: MemoryReadAuxCols<16, T>,
    pub write_z_aux_cols: MemoryWriteAuxCols<16, T>,
    pub write_cmp_aux_cols: MemoryWriteAuxCols<1, T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn from_iterator(
        mut iter: impl Iterator<Item = T>,
        air: &LongArithmeticAir<ARG_SIZE, LIMB_SIZE>,
    ) -> Self {
        let io = LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::from_iterator(iter.by_ref());
        let aux =
            LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::from_iterator(iter.by_ref(), air);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }

    // TODO get rid of get_width somehow?
    pub fn get_width(air: &LongArithmeticAir<ARG_SIZE, LIMB_SIZE>) -> usize {
        LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::get_width()
            + LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::get_width(air)
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        3 * num_limbs::<ARG_SIZE, LIMB_SIZE>() + 9
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let from_state = ExecutionState::from_iter(iter.by_ref());
        let x = MemoryData::from_iterator(iter.by_ref());
        let y = MemoryData::from_iterator(iter.by_ref());
        let z = MemoryData::from_iterator(iter.by_ref());
        let cmp_result = iter.next().unwrap();

        Self {
            from_state,
            x,
            y,
            z,
            cmp_result,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.x.flatten())
            .chain(self.y.flatten())
            .chain(self.z.flatten())
            .chain(iter::once(&self.cmp_result))
            .cloned()
            .collect()
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn get_width(air: &LongArithmeticAir<ARG_SIZE, LIMB_SIZE>) -> usize {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        MemoryReadAuxCols::<16, T>::width(&air.mem_oc)
            + MemoryReadAuxCols::<16, T>::width(&air.mem_oc)
            + MemoryWriteAuxCols::<16, T>::width(&air.mem_oc)
            + MemoryWriteAuxCols::<1, T>::width(&air.mem_oc)
            + (5 + num_limbs)
    }

    pub fn from_iterator(
        mut iter: impl Iterator<Item = T>,
        air: &LongArithmeticAir<ARG_SIZE, LIMB_SIZE>,
    ) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let is_valid = iter.next().unwrap();
        let opcode_add_flag = iter.next().unwrap();
        let opcode_sub_flag = iter.next().unwrap();
        let opcode_lt_flag = iter.next().unwrap();
        let opcode_eq_flag = iter.next().unwrap();
        let buffer = iter.by_ref().take(num_limbs).collect();

        let mem_oc = &air.mem_oc;
        let read_x_slice = {
            let width = MemoryReadAuxCols::<16, T>::width(mem_oc);
            iter.by_ref().take(width).collect::<Vec<_>>()
        };
        let read_y_slice = {
            let width = MemoryReadAuxCols::<16, T>::width(mem_oc);
            iter.by_ref().take(width).collect::<Vec<_>>()
        };
        let write_z_slice = {
            let width = MemoryWriteAuxCols::<16, T>::width(mem_oc);
            iter.by_ref().take(width).collect::<Vec<_>>()
        };
        let write_cmp_slice = {
            let width = MemoryWriteAuxCols::<1, T>::width(mem_oc);
            iter.by_ref().take(width).collect::<Vec<_>>()
        };

        let read_x_aux_cols = MemoryReadAuxCols::<16, T>::from_slice(&read_x_slice[..], mem_oc);
        let read_y_aux_cols = MemoryReadAuxCols::<16, T>::from_slice(&read_y_slice[..], mem_oc);
        let write_z_aux_cols = MemoryWriteAuxCols::<16, T>::from_slice(&write_z_slice[..], mem_oc);
        let write_cmp_aux_cols =
            MemoryWriteAuxCols::<1, T>::from_slice(&write_cmp_slice[..], mem_oc);

        Self {
            is_valid,
            opcode_add_flag,
            opcode_sub_flag,
            opcode_lt_flag,
            opcode_eq_flag,
            buffer,

            read_x_aux_cols,
            read_y_aux_cols,
            write_z_aux_cols,
            write_cmp_aux_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let our_cols = iter::once(&self.is_valid)
            .chain(iter::once(&self.opcode_add_flag))
            .chain(iter::once(&self.opcode_sub_flag))
            .chain(iter::once(&self.opcode_lt_flag))
            .chain(iter::once(&self.opcode_eq_flag))
            .chain(self.buffer.iter())
            .cloned()
            .collect::<Vec<_>>();
        let memory_aux_cols = [
            self.read_x_aux_cols.clone().flatten(),
            self.read_y_aux_cols.clone().flatten(),
            self.write_z_aux_cols.clone().flatten(),
            self.write_cmp_aux_cols.clone().flatten(),
        ]
        .concat();
        [our_cols, memory_aux_cols].concat()
    }
}
