use std::iter;

use afs_derive::AlignedBorrow;
use afs_primitives::ecc::{EcAirConfig, EcAuxCols as EcPrimitivesAuxCols};

use crate::{
    arch::columns::ExecutionState,
    memory::{
        offline_checker::{MemoryHeapReadAuxCols, MemoryHeapWriteAuxCols},
        MemoryHeapDataIoCols,
    },
    modular_arithmetic::TWO_NUM_LIMBS,
};

pub struct EcAddUnequalCols<T: Clone> {
    pub io: EcAddUnequalIoCols<T>,
    pub aux: EcAddUnequalAuxCols<T>,
}

impl<T: Clone> EcAddUnequalCols<T> {
    pub fn width(config: &EcAirConfig) -> usize {
        EcAddUnequalIoCols::<T>::width() + EcAddUnequalAuxCols::<T>::width(config)
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>, config: &EcAirConfig) -> Self {
        let io = EcAddUnequalIoCols::from_iterator(iter.by_ref());
        let aux = EcAddUnequalAuxCols::from_iterator(iter.by_ref(), config);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }
}

#[derive(AlignedBorrow, Clone)]
pub struct EcAddUnequalIoCols<T: Clone> {
    pub from_state: ExecutionState<T>,
    pub p1: MemoryHeapDataIoCols<T, TWO_NUM_LIMBS>,
    pub p2: MemoryHeapDataIoCols<T, TWO_NUM_LIMBS>,
    pub p3: MemoryHeapDataIoCols<T, TWO_NUM_LIMBS>,
}

impl<T: Clone> EcAddUnequalIoCols<T> {
    pub const fn width() -> usize {
        2 + 3 * (TWO_NUM_LIMBS + 5)
    }

    pub fn from_iterator(mut iter: impl Iterator<Item = T>) -> Self {
        let from_state = ExecutionState::from_iter(iter.by_ref());
        let p1 = MemoryHeapDataIoCols::from_iterator(iter.by_ref());
        let p2 = MemoryHeapDataIoCols::from_iterator(iter.by_ref());
        let p3 = MemoryHeapDataIoCols::from_iterator(iter.by_ref());
        Self {
            from_state,
            p1,
            p2,
            p3,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        iter::once(&self.from_state.pc)
            .chain(iter::once(&self.from_state.timestamp))
            .chain(self.p1.flatten())
            .chain(self.p2.flatten())
            .chain(self.p3.flatten())
            .cloned()
            .collect()
    }
}

pub struct EcAddUnequalAuxCols<T: Clone> {
    pub read_p1_aux_cols: MemoryHeapReadAuxCols<T, TWO_NUM_LIMBS>,
    pub read_p2_aux_cols: MemoryHeapReadAuxCols<T, TWO_NUM_LIMBS>,
    pub write_p3_aux_cols: MemoryHeapWriteAuxCols<T, TWO_NUM_LIMBS>,

    pub aux: EcPrimitivesAuxCols<T>,
}

impl<T: Clone> EcAddUnequalAuxCols<T> {
    pub fn from_iterator(mut iter: impl Iterator<Item = T>, config: &EcAirConfig) -> Self {
        let aux_width = EcPrimitivesAuxCols::<T>::width(config);
        Self {
            read_p1_aux_cols: MemoryHeapReadAuxCols::from_iterator(iter.by_ref()),
            read_p2_aux_cols: MemoryHeapReadAuxCols::from_iterator(iter.by_ref()),
            write_p3_aux_cols: MemoryHeapWriteAuxCols::from_iterator(iter.by_ref()),
            aux: EcPrimitivesAuxCols::from_slice(
                &iter.by_ref().take(aux_width).collect::<Vec<_>>(),
                config.num_limbs,
            ),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mem = [
            self.read_p1_aux_cols.clone().flatten(),
            self.read_p2_aux_cols.clone().flatten(),
            self.write_p3_aux_cols.clone().flatten(),
        ]
        .concat();
        let aux = self.aux.flatten();

        [mem, aux].concat()
    }

    pub fn width(config: &EcAirConfig) -> usize {
        MemoryHeapReadAuxCols::<T, TWO_NUM_LIMBS>::width() * 2
            + MemoryHeapWriteAuxCols::<T, TWO_NUM_LIMBS>::width()
            + EcPrimitivesAuxCols::<T>::width(config)
    }
}
