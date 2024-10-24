use std::collections::BTreeMap;

use itertools::Itertools;
use p3_field::{Field, PrimeField32};
use strum::{EnumCount, IntoEnumIterator};

use crate::arch::instructions::CoreOpcode;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreIoCols<T> {
    pub timestamp: T,
    pub pc: T,
    pub opcode: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
}

impl<T: Clone> CoreIoCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            timestamp: slc[0].clone(),
            pc: slc[1].clone(),
            opcode: slc[2].clone(),
            a: slc[3].clone(),
            b: slc[4].clone(),
            c: slc[5].clone(),
            d: slc[6].clone(),
            e: slc[7].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.timestamp.clone(),
            self.pc.clone(),
            self.opcode.clone(),
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
            self.e.clone(),
        ]
    }

    pub fn get_width() -> usize {
        8
    }
}

impl<T: Field> CoreIoCols<T> {
    pub fn blank_row() -> Self {
        Self {
            timestamp: T::default(),
            pc: T::default(),
            opcode: T::default(),
            a: T::default(),
            b: T::default(),
            c: T::default(),
            d: T::default(),
            e: T::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreAuxCols<T> {
    pub is_valid: T,
    pub operation_flags: BTreeMap<CoreOpcode, T>,
}

impl<T: Clone> CoreAuxCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let start = 0;
        let end = CoreOpcode::COUNT;
        let operation_flags_vec = slc[start..end].to_vec();
        let mut operation_flags = BTreeMap::new();
        for (opcode, operation_flag) in CoreOpcode::iter().zip_eq(operation_flags_vec) {
            operation_flags.insert(opcode, operation_flag);
        }
        Self {
            is_valid: slc[end].clone(),
            operation_flags,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        for opcode in CoreOpcode::iter() {
            flattened.push(self.operation_flags.get(&opcode).unwrap().clone());
        }
        flattened.push(self.is_valid.clone());
        flattened
    }

    pub fn get_width() -> usize {
        CoreOpcode::COUNT + 1
    }
}

impl<F: PrimeField32> CoreAuxCols<F> {
    pub fn blank_row() -> Self {
        let mut operation_flags = BTreeMap::new();
        for opcode in CoreOpcode::iter() {
            operation_flags.insert(opcode, F::zero());
        }

        Self {
            is_valid: F::zero(),
            operation_flags,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoreCols<T> {
    pub io: CoreIoCols<T>,
    pub aux: CoreAuxCols<T>,
}

impl<T: Clone> CoreCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        let io = CoreIoCols::<T>::from_slice(&slc[..CoreIoCols::<T>::get_width()]);
        let aux = CoreAuxCols::<T>::from_slice(&slc[CoreIoCols::<T>::get_width()..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width() -> usize {
        CoreIoCols::<T>::get_width() + CoreAuxCols::<T>::get_width()
    }
}

impl<F: PrimeField32> CoreCols<F> {
    pub fn blank_row() -> Self {
        Self {
            io: CoreIoCols::<F>::blank_row(),
            aux: CoreAuxCols::<F>::blank_row(),
        }
    }
}
