use std::{array, collections::BTreeMap};

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecAuxCols, IsEqualVecAir},
    sub_chip::LocalTraceInstructions,
};
use itertools::Itertools;
use p3_field::{Field, PrimeField32};

use super::{
    CpuAir, CpuChip, Opcode, CPU_MAX_READS_PER_CYCLE, CPU_MAX_WRITES_PER_CYCLE, WORD_SIZE,
};
use crate::{
    arch::instructions::CORE_INSTRUCTIONS,
    memory::{
        offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        MemoryReadRecord, MemoryWriteRecord,
    },
};

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
    pub op_f: T,
    pub op_g: T,
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
            op_f: slc[8].clone(),
            op_g: slc[9].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.timestamp.clone(),
            self.pc.clone(),
            self.opcode.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.op_f.clone(),
            self.op_g.clone(),
        ]
    }

    pub fn get_width() -> usize {
        10
    }
}

impl<T: Field> CpuIoCols<T> {
    pub fn nop_row(pc: T, timestamp: T) -> Self {
        Self {
            timestamp,
            pc,
            opcode: T::from_canonical_usize(Opcode::NOP as usize),
            op_a: T::default(),
            op_b: T::default(),
            op_c: T::default(),
            d: T::default(),
            e: T::default(),
            op_f: T::default(),
            op_g: T::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuMemoryAccessCols<T> {
    pub address_space: T,
    pub pointer: T,
    pub timestamp: T,
    pub enabled: T,
    pub value: T,
}

impl<F: Field> CpuMemoryAccessCols<F> {
    pub fn disabled() -> Self {
        CpuMemoryAccessCols {
            address_space: F::one(),
            pointer: F::zero(),
            timestamp: F::zero(),
            enabled: F::zero(),
            value: F::zero(),
        }
    }

    pub fn from_read_record(read: MemoryReadRecord<1, F>) -> Self {
        CpuMemoryAccessCols {
            address_space: read.address_space,
            pointer: read.pointer,
            timestamp: read.timestamp,
            enabled: F::one(),
            value: read.value(),
        }
    }

    pub fn from_write_record(write: MemoryWriteRecord<1, F>) -> Self {
        CpuMemoryAccessCols {
            address_space: write.address_space,
            pointer: write.pointer,
            timestamp: write.timestamp,
            enabled: F::one(),
            value: write.value(),
        }
    }
}

impl<T: Clone> CpuMemoryAccessCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            address_space: slc[0].clone(),
            pointer: slc[1].clone(),
            timestamp: slc[2].clone(),
            enabled: slc[3].clone(),
            value: slc[4].clone(),
        }
    }
}

impl<T> CpuMemoryAccessCols<T> {
    pub fn flatten(self) -> Vec<T> {
        vec![
            self.address_space,
            self.pointer,
            self.timestamp,
            self.enabled,
            self.value,
        ]
    }

    pub fn width() -> usize {
        5
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuAuxCols<T> {
    pub operation_flags: BTreeMap<Opcode, T>,
    pub public_value_flags: Vec<T>,
    pub reads: [CpuMemoryAccessCols<T>; CPU_MAX_READS_PER_CYCLE],
    pub writes: [CpuMemoryAccessCols<T>; CPU_MAX_WRITES_PER_CYCLE],
    pub read0_equals_read1: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
    pub reads_aux_cols: [MemoryReadOrImmediateAuxCols<T>; CPU_MAX_READS_PER_CYCLE],
    pub writes_aux_cols: [MemoryWriteAuxCols<1, T>; CPU_MAX_WRITES_PER_CYCLE],
}

impl<T: Clone> CpuAuxCols<T> {
    pub fn from_slice(slc: &[T], cpu_air: &CpuAir) -> Self {
        let mut start = 0;
        let mut end = CORE_INSTRUCTIONS.len();
        let operation_flags_vec = slc[start..end].to_vec();
        let mut operation_flags = BTreeMap::new();
        for (opcode, operation_flag) in CORE_INSTRUCTIONS.iter().zip_eq(operation_flags_vec) {
            operation_flags.insert(*opcode, operation_flag);
        }

        start = end;
        end += cpu_air.options.num_public_values;
        let public_value_flags = slc[start..end].to_vec();

        let reads = array::from_fn(|_| {
            start = end;
            end += CpuMemoryAccessCols::<T>::width();
            CpuMemoryAccessCols::<T>::from_slice(&slc[start..end])
        });
        let writes = array::from_fn(|_| {
            start = end;
            end += CpuMemoryAccessCols::<T>::width();
            CpuMemoryAccessCols::<T>::from_slice(&slc[start..end])
        });

        start = end;
        end += 1;
        let beq_check = slc[start].clone();

        start = end;
        end += IsEqualVecAuxCols::<T>::width(WORD_SIZE);
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(&slc[start..end], WORD_SIZE);

        let reads_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryReadOrImmediateAuxCols::<T>::width();
            MemoryReadOrImmediateAuxCols::from_slice(&slc[start..end])
        });
        let writes_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryWriteAuxCols::<WORD_SIZE, T>::width();
            MemoryWriteAuxCols::from_slice(&slc[start..end])
        });

        Self {
            operation_flags,
            public_value_flags,
            reads,
            writes,
            read0_equals_read1: beq_check,
            is_equal_vec_aux,
            reads_aux_cols,
            writes_aux_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        for opcode in CORE_INSTRUCTIONS {
            flattened.push(self.operation_flags.get(&opcode).unwrap().clone());
        }
        flattened.extend(self.public_value_flags.clone());
        flattened.extend(
            self.reads
                .iter()
                .cloned()
                .flat_map(CpuMemoryAccessCols::<T>::flatten),
        );
        flattened.extend(
            self.writes
                .iter()
                .cloned()
                .flat_map(CpuMemoryAccessCols::<T>::flatten),
        );
        flattened.push(self.read0_equals_read1.clone());
        flattened.extend(self.is_equal_vec_aux.flatten());
        flattened.extend(
            self.reads_aux_cols
                .iter()
                .cloned()
                .flat_map(MemoryReadOrImmediateAuxCols::flatten),
        );
        flattened.extend(
            self.writes_aux_cols
                .iter()
                .cloned()
                .flat_map(MemoryWriteAuxCols::flatten),
        );
        flattened
    }

    pub fn get_width(cpu_air: &CpuAir) -> usize {
        CORE_INSTRUCTIONS.len()
            + cpu_air.options.num_public_values
            + CPU_MAX_READS_PER_CYCLE
                * (CpuMemoryAccessCols::<T>::width() + MemoryReadOrImmediateAuxCols::<T>::width())
            + CPU_MAX_WRITES_PER_CYCLE
                * (CpuMemoryAccessCols::<T>::width() + MemoryWriteAuxCols::<WORD_SIZE, T>::width())
            + 1
            + IsEqualVecAuxCols::<T>::width(WORD_SIZE)
    }
}

impl<F: PrimeField32> CpuAuxCols<F> {
    pub fn nop_row(chip: &CpuChip<F>) -> Self {
        let mut operation_flags = BTreeMap::new();
        for opcode in CORE_INSTRUCTIONS {
            operation_flags.insert(opcode, F::from_bool(opcode == Opcode::NOP));
        }

        let is_equal_vec_cols = LocalTraceInstructions::generate_trace_row(
            &IsEqualVecAir::new(WORD_SIZE),
            (vec![F::zero()], vec![F::zero()]),
        );
        Self {
            operation_flags,
            public_value_flags: vec![F::zero(); chip.air.options.num_public_values],
            reads: array::from_fn(|_| CpuMemoryAccessCols::disabled()),
            writes: array::from_fn(|_| CpuMemoryAccessCols::disabled()),
            read0_equals_read1: F::one(),
            is_equal_vec_aux: is_equal_vec_cols.aux,
            reads_aux_cols: array::from_fn(|_| MemoryReadOrImmediateAuxCols::disabled()),
            writes_aux_cols: array::from_fn(|_| MemoryWriteAuxCols::disabled()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuCols<T> {
    pub io: CpuIoCols<T>,
    pub aux: CpuAuxCols<T>,
}

impl<T: Clone> CpuCols<T> {
    pub fn from_slice(slc: &[T], cpu_air: &CpuAir) -> Self {
        let io = CpuIoCols::<T>::from_slice(&slc[..CpuIoCols::<T>::get_width()]);
        let aux = CpuAuxCols::<T>::from_slice(&slc[CpuIoCols::<T>::get_width()..], cpu_air);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(cpu_air: &CpuAir) -> usize {
        CpuIoCols::<T>::get_width() + CpuAuxCols::<T>::get_width(cpu_air)
    }
}

impl<F: PrimeField32> CpuCols<F> {
    /// This function mutates internal state of some chips. It should be called once for every
    /// NOP row---results should not be cloned.
    /// TODO[zach]: Make this less surprising, probably by not doing less-than checks on dummy rows.
    pub fn nop_row(chip: &CpuChip<F>, pc: F, timestamp: F) -> Self {
        Self {
            io: CpuIoCols::<F>::nop_row(pc, timestamp),
            aux: CpuAuxCols::<F>::nop_row(chip),
        }
    }
}
