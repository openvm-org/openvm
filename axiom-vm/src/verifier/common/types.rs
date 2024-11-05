use std::{array, borrow::BorrowMut};

use ax_stark_sdk::ax_stark_backend::p3_field::PrimeField32;
use axvm_circuit::{
    circuit_derive::AlignedBorrow,
    system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs},
};
use axvm_native_compiler::prelude::*;

#[derive(Debug, AlignedBorrow)]
#[repr(C)]
pub struct VmVerifierPvs<T> {
    /// The commitment of the app program.
    pub app_commit: [T; DIGEST_SIZE],
    /// The merged execution state of all the segments this circuit aggregates.
    pub connector: VmConnectorPvs<T>,
    /// The memory state before/after all the segments this circuit aggregates.
    pub memory: MemoryMerklePvs<T, DIGEST_SIZE>,
    /// The merkle root of all public values. This is only meaningful when the last segment is
    /// aggregated by this circuit.
    pub public_values_commit: [T; DIGEST_SIZE],
}

impl<F: PrimeField32> VmVerifierPvs<Felt<F>> {
    pub fn uninit<C: Config<F = F>>(builder: &mut Builder<C>) -> Self {
        Self {
            app_commit: array::from_fn(|_| builder.uninit()),
            connector: VmConnectorPvs {
                initial_pc: builder.uninit(),
                final_pc: builder.uninit(),
                exit_code: builder.uninit(),
                is_terminate: builder.uninit(),
            },
            memory: MemoryMerklePvs {
                initial_root: array::from_fn(|_| builder.uninit()),
                final_root: array::from_fn(|_| builder.uninit()),
            },
            public_values_commit: array::from_fn(|_| builder.uninit()),
        }
    }
}

impl<F: Default + Clone> VmVerifierPvs<Felt<F>> {
    pub fn flatten(self) -> Vec<Felt<F>> {
        let mut v = vec![Felt(0, Default::default()); VmVerifierPvs::<u8>::width()];
        *v.as_mut_slice().borrow_mut() = self;
        v
    }
}
