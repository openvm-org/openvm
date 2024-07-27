use std::{array::from_fn, collections::BTreeMap};

use afs_primitives::is_equal_vec::columns::IsEqualVecAuxCols;
use itertools::Itertools;

use super::{CpuOptions, OpCode, CPU_MAX_ACCESSES_PER_CYCLE};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuIoCols<T> {
    pub timestamp: T,
    pub pc: T,

    pub opcode: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub d: T,
    pub e: T,
}

impl<T: Clone> CpuIoCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            timestamp: slc[0].clone(),
            pc: slc[1].clone(),
            opcode: slc[2].clone(),
            op_a: slc[3].clone(),
            op_b: slc[4].clone(),
            op_c: slc[5].clone(),
            d: slc[6].clone(),
            e: slc[7].clone(),
        }
    }

    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.push(self.timestamp);
        buf.push(self.pc.clone());
        buf.push(self.opcode.clone());
        buf.push(self.op_a.clone());
        buf.push(self.op_b.clone());
        buf.push(self.op_c.clone());
        buf.push(self.d.clone());
        buf.push(self.e.clone());
    }

    pub fn get_width() -> usize {
        8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryAccessCols<const WORD_SIZE: usize, T> {
    pub enabled: T,

    pub address_space: T,
    pub is_immediate: T,
    pub is_zero_aux: T,

    pub address: T,

    pub data: [T; WORD_SIZE],
}

impl<const WORD_SIZE: usize, T: Clone> MemoryAccessCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            enabled: slc[0].clone(),
            address_space: slc[1].clone(),
            is_immediate: slc[2].clone(),
            is_zero_aux: slc[3].clone(),
            address: slc[4].clone(),
            data: from_fn(|i| slc[5 + i].clone()),
        }
    }
    pub fn to_buf(self, buf: &mut Vec<T>) {
        buf.push(self.enabled);
        buf.push(self.address_space);
        buf.push(self.is_immediate);
        buf.push(self.is_zero_aux);
        buf.push(self.address);
        buf.extend(self.data);
    }

    pub fn get_width() -> usize {
        5 + WORD_SIZE
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuAuxCols<const WORD_SIZE: usize, T> {
    pub operation_flags: BTreeMap<OpCode, T>,
    pub accesses: [MemoryAccessCols<WORD_SIZE, T>; CPU_MAX_ACCESSES_PER_CYCLE],
    pub read0_equals_read1: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

impl<const WORD_SIZE: usize, T: Clone> CpuAuxCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], options: CpuOptions) -> Self {
        let mut start = 0;
        let mut end = options.num_enabled_instructions();
        let operation_flags_vec = slc[start..end].to_vec();
        let mut operation_flags = BTreeMap::new();
        for (opcode, operation_flag) in options
            .enabled_instructions()
            .iter()
            .zip_eq(operation_flags_vec)
        {
            operation_flags.insert(*opcode, operation_flag);
        }

        let accesses = from_fn(|_| {
            start = end;
            end += MemoryAccessCols::<WORD_SIZE, T>::get_width();
            MemoryAccessCols::from_slice(&slc[start..end])
        });

        let beq_check = slc[end].clone();
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(&slc[end + 1..], WORD_SIZE);

        Self {
            operation_flags,
            accesses,
            read0_equals_read1: beq_check,
            is_equal_vec_aux,
        }
    }

    pub fn to_buf(self, buf: &mut Vec<T>, options: CpuOptions) {
        for opcode in options.enabled_instructions() {
            buf.push(self.operation_flags.get(&opcode).unwrap().clone());
        }
        for access in self.accesses.into_iter() {
            access.to_buf(buf);
        }
        buf.push(self.read0_equals_read1.clone());
        self.is_equal_vec_aux.to_buf(buf);
    }

    pub fn get_width(options: CpuOptions) -> usize {
        options.num_enabled_instructions()
            + (CPU_MAX_ACCESSES_PER_CYCLE * MemoryAccessCols::<WORD_SIZE, T>::get_width())
            + 1
            + IsEqualVecAuxCols::<T>::width(WORD_SIZE)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuCols<const WORD_SIZE: usize, T> {
    pub io: CpuIoCols<T>,
    pub aux: CpuAuxCols<WORD_SIZE, T>,
}

impl<const WORD_SIZE: usize, T: Clone> CpuCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], options: CpuOptions) -> Self {
        let io = CpuIoCols::<T>::from_slice(&slc[..CpuIoCols::<T>::get_width()]);
        let aux =
            CpuAuxCols::<WORD_SIZE, T>::from_slice(&slc[CpuIoCols::<T>::get_width()..], options);

        Self { io, aux }
    }

    pub fn to_buf(self, buf: &mut Vec<T>, options: CpuOptions) {
        self.io.to_buf(buf);
        self.aux.to_buf(buf, options);
    }

    pub fn flatten(self, options: CpuOptions) -> Vec<T> {
        let mut buf = Vec::with_capacity(Self::get_width(options));
        self.to_buf(&mut buf, options);
        buf
    }

    pub fn get_width(options: CpuOptions) -> usize {
        CpuIoCols::<T>::get_width() + CpuAuxCols::<WORD_SIZE, T>::get_width(options)
    }
}
