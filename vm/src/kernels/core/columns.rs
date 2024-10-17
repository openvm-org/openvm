use std::{array, collections::BTreeMap};

use afs_primitives::{
    is_equal::{columns::IsEqualAuxCols, IsEqualAir},
    sub_chip::LocalTraceInstructions,
};
use itertools::Itertools;
use p3_field::{Field, PrimeField32};
use strum::{EnumCount, IntoEnumIterator};

use super::{CoreAir, CoreChip, CORE_MAX_READS_PER_CYCLE, CORE_MAX_WRITES_PER_CYCLE};
use crate::{
    arch::instructions::CoreOpcode,
    system::memory::{
        offline_checker::{MemoryReadOrImmediateAuxCols, MemoryWriteAuxCols},
        MemoryReadRecord, MemoryWriteRecord,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreIoCols<T> {
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

impl<T: Clone> CoreIoCols<T> {
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

impl<T: Field> CoreIoCols<T> {
    pub fn nop_row(pc: T) -> Self {
        Self {
            timestamp: T::default(),
            pc,
            opcode: T::from_canonical_usize(CoreOpcode::NOP as usize),
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
pub struct CoreMemoryAccessCols<T> {
    pub address_space: T,
    pub pointer: T,
    pub value: T,
}

impl<F: Field> CoreMemoryAccessCols<F> {
    pub fn disabled() -> Self {
        CoreMemoryAccessCols {
            address_space: F::one(),
            pointer: F::zero(),
            value: F::zero(),
        }
    }

    pub fn from_read_record(read: MemoryReadRecord<F, 1>) -> Self {
        CoreMemoryAccessCols {
            address_space: read.address_space,
            pointer: read.pointer,
            value: read.value(),
        }
    }

    pub fn from_write_record(write: MemoryWriteRecord<F, 1>) -> Self {
        CoreMemoryAccessCols {
            address_space: write.address_space,
            pointer: write.pointer,
            value: write.value(),
        }
    }
}

impl<T: Clone> CoreMemoryAccessCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            address_space: slc[0].clone(),
            pointer: slc[1].clone(),
            value: slc[2].clone(),
        }
    }
}

impl<T> CoreMemoryAccessCols<T> {
    pub fn flatten(self) -> Vec<T> {
        vec![self.address_space, self.pointer, self.value]
    }

    pub fn width() -> usize {
        3
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreAuxCols<T> {
    pub operation_flags: BTreeMap<CoreOpcode, T>,
    pub public_value_flags: Vec<T>,
    pub reads: [CoreMemoryAccessCols<T>; CORE_MAX_READS_PER_CYCLE],
    pub writes: [CoreMemoryAccessCols<T>; CORE_MAX_WRITES_PER_CYCLE],
    pub read0_equals_read1: T,
    pub is_equal_aux: IsEqualAuxCols<T>,
    pub reads_aux_cols: [MemoryReadOrImmediateAuxCols<T>; CORE_MAX_READS_PER_CYCLE],
    pub writes_aux_cols: [MemoryWriteAuxCols<T, 1>; CORE_MAX_WRITES_PER_CYCLE],

    pub next_pc: T,
}

impl<T: Clone> CoreAuxCols<T> {
    pub fn from_slice(slc: &[T], core_air: &CoreAir) -> Self {
        let mut start = 0;
        let mut end = CoreOpcode::COUNT;
        let operation_flags_vec = slc[start..end].to_vec();
        let mut operation_flags = BTreeMap::new();
        for (opcode, operation_flag) in CoreOpcode::iter().zip_eq(operation_flags_vec) {
            operation_flags.insert(opcode, operation_flag);
        }

        start = end;
        end += core_air.options.num_public_values;
        let public_value_flags = slc[start..end].to_vec();

        let reads = array::from_fn(|_| {
            start = end;
            end += CoreMemoryAccessCols::<T>::width();
            CoreMemoryAccessCols::<T>::from_slice(&slc[start..end])
        });
        let writes = array::from_fn(|_| {
            start = end;
            end += CoreMemoryAccessCols::<T>::width();
            CoreMemoryAccessCols::<T>::from_slice(&slc[start..end])
        });

        start = end;
        end += 1;
        let beq_check = slc[start].clone();

        start = end;
        end += IsEqualAuxCols::<T>::width();
        let is_equal_aux = IsEqualAuxCols::from_slice(&slc[start..end]);

        let reads_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryReadOrImmediateAuxCols::<T>::width();
            MemoryReadOrImmediateAuxCols::from_slice(&slc[start..end])
        });
        let writes_aux_cols = array::from_fn(|_| {
            start = end;
            end += MemoryWriteAuxCols::<T, 1>::width();
            MemoryWriteAuxCols::from_slice(&slc[start..end])
        });
        let next_pc = slc[end].clone();

        Self {
            operation_flags,
            public_value_flags,
            reads,
            writes,
            read0_equals_read1: beq_check,
            is_equal_aux,
            reads_aux_cols,
            writes_aux_cols,
            next_pc,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        for opcode in CoreOpcode::iter() {
            flattened.push(self.operation_flags.get(&opcode).unwrap().clone());
        }
        flattened.extend(self.public_value_flags.clone());
        flattened.extend(
            self.reads
                .iter()
                .cloned()
                .flat_map(CoreMemoryAccessCols::<T>::flatten),
        );
        flattened.extend(
            self.writes
                .iter()
                .cloned()
                .flat_map(CoreMemoryAccessCols::<T>::flatten),
        );
        flattened.push(self.read0_equals_read1.clone());
        flattened.extend(self.is_equal_aux.flatten());
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
        flattened.push(self.next_pc.clone());
        flattened
    }

    pub fn get_width(core_air: &CoreAir) -> usize {
        CoreOpcode::COUNT
            + core_air.options.num_public_values
            + CORE_MAX_READS_PER_CYCLE
                * (CoreMemoryAccessCols::<T>::width() + MemoryReadOrImmediateAuxCols::<T>::width())
            + CORE_MAX_WRITES_PER_CYCLE
                * (CoreMemoryAccessCols::<T>::width() + MemoryWriteAuxCols::<T, 1>::width())
            + 1
            + IsEqualAuxCols::<T>::width()
            + 1
    }
}

impl<F: PrimeField32> CoreAuxCols<F> {
    pub fn nop_row(chip: &CoreChip<F>) -> Self {
        let mut operation_flags = BTreeMap::new();
        for opcode in CoreOpcode::iter() {
            operation_flags.insert(opcode, F::from_bool(opcode == CoreOpcode::NOP));
        }

        let is_equal_cols =
            LocalTraceInstructions::generate_trace_row(&IsEqualAir, (F::zero(), F::zero()));
        Self {
            operation_flags,
            public_value_flags: vec![F::zero(); chip.air.options.num_public_values],
            reads: array::from_fn(|_| CoreMemoryAccessCols::disabled()),
            writes: array::from_fn(|_| CoreMemoryAccessCols::disabled()),
            read0_equals_read1: F::one(),
            is_equal_aux: is_equal_cols.aux,
            reads_aux_cols: array::from_fn(|_| MemoryReadOrImmediateAuxCols::disabled()),
            writes_aux_cols: array::from_fn(|_| MemoryWriteAuxCols::disabled()),
            next_pc: F::from_canonical_u32(chip.state.pc),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreCols<T> {
    pub io: CoreIoCols<T>,
    pub aux: CoreAuxCols<T>,
}

impl<T: Clone> CoreCols<T> {
    pub fn from_slice(slc: &[T], core_air: &CoreAir) -> Self {
        let io = CoreIoCols::<T>::from_slice(&slc[..CoreIoCols::<T>::get_width()]);
        let aux = CoreAuxCols::<T>::from_slice(&slc[CoreIoCols::<T>::get_width()..], core_air);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten());
        flattened
    }

    pub fn get_width(core_air: &CoreAir) -> usize {
        CoreIoCols::<T>::get_width() + CoreAuxCols::<T>::get_width(core_air)
    }
}

impl<F: PrimeField32> CoreCols<F> {
    /// This function mutates internal state of some chips. It should be called once for every
    /// NOP row---results should not be cloned.
    /// TODO[zach]: Make this less surprising, probably by not doing less-than checks on dummy rows.
    pub fn nop_row(chip: &CoreChip<F>, pc: F) -> Self {
        Self {
            io: CoreIoCols::<F>::nop_row(pc),
            aux: CoreAuxCols::<F>::nop_row(chip),
        }
    }
}
