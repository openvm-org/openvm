use std::{array::from_fn, collections::BTreeMap};

use afs_primitives::{
    is_equal_vec::{columns::IsEqualVecAuxCols, IsEqualVecAir},
    sub_chip::LocalTraceInstructions,
};
use itertools::Itertools;
use p3_field::{Field, PrimeField32};

use super::{CpuAir, CpuOptions, OpCode, CPU_MAX_ACCESSES_PER_CYCLE};
use crate::{
    memory::{
        manager::operation::MemoryOperation, offline_checker::columns::MemoryOfflineCheckerAuxCols,
    },
    vm::ExecutionSegment,
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
            opcode: T::from_canonical_usize(OpCode::NOP as usize),
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
pub struct CpuAuxCols<const WORD_SIZE: usize, T> {
    pub operation_flags: BTreeMap<OpCode, T>,
    pub public_value_flags: Vec<T>,
    pub mem_ops: [MemoryOperation<WORD_SIZE, T>; CPU_MAX_ACCESSES_PER_CYCLE],
    pub read0_equals_read1: T,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
    pub mem_oc_aux_cols: [MemoryOfflineCheckerAuxCols<WORD_SIZE, T>; CPU_MAX_ACCESSES_PER_CYCLE],
}

impl<const WORD_SIZE: usize, T: Clone> CpuAuxCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], cpu_air: &CpuAir<WORD_SIZE>) -> Self {
        let mut start = 0;
        let mut end = cpu_air.options.num_enabled_instructions();
        let operation_flags_vec = slc[start..end].to_vec();
        let mut operation_flags = BTreeMap::new();
        for (opcode, operation_flag) in cpu_air
            .options
            .enabled_instructions()
            .iter()
            .zip_eq(operation_flags_vec)
        {
            operation_flags.insert(*opcode, operation_flag);
        }

        start = end;
        end += cpu_air.options.num_public_values;
        let public_value_flags = slc[start..end].to_vec();

        let mem_ops = from_fn(|_| {
            start = end;
            end += MemoryOperation::<WORD_SIZE, T>::width();
            MemoryOperation::<WORD_SIZE, T>::from_slice(&slc[start..end])
        });

        start = end;
        end += 1;
        let beq_check = slc[start].clone();

        start = end;
        end += IsEqualVecAuxCols::<T>::width(WORD_SIZE);
        let is_equal_vec_aux = IsEqualVecAuxCols::from_slice(&slc[start..end], WORD_SIZE);

        let mem_oc_aux_cols = from_fn(|_| {
            start = end;
            end +=
                MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(&cpu_air.memory_offline_checker);
            MemoryOfflineCheckerAuxCols::from_slice(&slc[start..end])
        });

        Self {
            operation_flags,
            public_value_flags,
            mem_ops,
            read0_equals_read1: beq_check,
            is_equal_vec_aux,
            mem_oc_aux_cols,
        }
    }

    pub fn flatten(&self, options: CpuOptions) -> Vec<T> {
        let mut flattened = vec![];
        for opcode in options.enabled_instructions() {
            flattened.push(self.operation_flags.get(&opcode).unwrap().clone());
        }
        flattened.extend(self.public_value_flags.clone());
        flattened.extend(
            self.mem_ops
                .iter()
                .cloned()
                .flat_map(MemoryOperation::<WORD_SIZE, T>::flatten),
        );
        flattened.push(self.read0_equals_read1.clone());
        flattened.extend(self.is_equal_vec_aux.flatten());
        flattened.extend(
            self.mem_oc_aux_cols
                .iter()
                .cloned()
                .flat_map(MemoryOfflineCheckerAuxCols::flatten),
        );
        flattened
    }

    pub fn get_width(cpu_air: &CpuAir<WORD_SIZE>) -> usize {
        cpu_air.options.num_enabled_instructions()
            + cpu_air.options.num_public_values
            + CPU_MAX_ACCESSES_PER_CYCLE
                * (MemoryOperation::<WORD_SIZE, T>::width()
                    + MemoryOfflineCheckerAuxCols::<WORD_SIZE, T>::width(
                        &cpu_air.memory_offline_checker,
                    ))
            + 1
            + IsEqualVecAuxCols::<T>::width(WORD_SIZE)
    }
}

impl<const WORD_SIZE: usize, T: PrimeField32> CpuAuxCols<WORD_SIZE, T> {
    pub fn nop_row<const NUM_WORDS: usize>(vm: &ExecutionSegment<NUM_WORDS, WORD_SIZE, T>) -> Self {
        let mut operation_flags = BTreeMap::new();
        for opcode in vm.options().enabled_instructions() {
            operation_flags.insert(opcode, T::from_bool(opcode == OpCode::NOP));
        }
        // TODO[osama]: consider using MemoryTraceBuilder here
        let oc_cols: [_; CPU_MAX_ACCESSES_PER_CYCLE] = from_fn(|_| {
            vm.cpu_chip
                .air
                .memory_offline_checker
                .disabled_memory_checker_cols(vm.range_checker.clone())
        });
        let is_equal_vec_cols = LocalTraceInstructions::generate_trace_row(
            &IsEqualVecAir::new(WORD_SIZE),
            (
                oc_cols[0].io.cell.data.to_vec(),
                oc_cols[1].io.cell.data.to_vec(),
            ),
        );
        Self {
            operation_flags,
            public_value_flags: vec![T::zero(); vm.options().num_public_values],
            mem_ops: from_fn(|i| oc_cols[i].io.clone()),
            read0_equals_read1: T::one(),
            is_equal_vec_aux: is_equal_vec_cols.aux,
            mem_oc_aux_cols: from_fn(|i| oc_cols[i].aux.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuCols<const WORD_SIZE: usize, T> {
    pub io: CpuIoCols<T>,
    pub aux: CpuAuxCols<WORD_SIZE, T>,
}

impl<const WORD_SIZE: usize, T: Clone> CpuCols<WORD_SIZE, T> {
    pub fn from_slice(slc: &[T], cpu_air: &CpuAir<WORD_SIZE>) -> Self {
        let io = CpuIoCols::<T>::from_slice(&slc[..CpuIoCols::<T>::get_width()]);
        let aux =
            CpuAuxCols::<WORD_SIZE, T>::from_slice(&slc[CpuIoCols::<T>::get_width()..], cpu_air);

        Self { io, aux }
    }

    pub fn flatten(&self, options: CpuOptions) -> Vec<T> {
        let mut flattened = self.io.flatten();
        flattened.extend(self.aux.flatten(options));
        flattened
    }

    pub fn get_width(cpu_air: &CpuAir<WORD_SIZE>) -> usize {
        CpuIoCols::<T>::get_width() + CpuAuxCols::<WORD_SIZE, T>::get_width(cpu_air)
    }
}

impl<const WORD_SIZE: usize, T: PrimeField32> CpuCols<WORD_SIZE, T> {
    pub fn nop_row<const NUM_WORDS: usize>(
        vm: &ExecutionSegment<NUM_WORDS, WORD_SIZE, T>,
        pc: T,
        timestamp: T,
    ) -> Self {
        Self {
            io: CpuIoCols::<T>::nop_row(pc, timestamp),
            aux: CpuAuxCols::<WORD_SIZE, T>::nop_row(vm),
        }
    }
}
