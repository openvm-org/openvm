mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;
use openvm_circuit_primitives::is_less_than::LessThanAuxCols;
use openvm_stark_backend::p3_field::PrimeField32;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryBaseAuxRecord {
    pub prev_timestamp: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct MemoryExtendedAuxRecord {
    pub prev_timestamp: u32,
    pub timestamp_lt_aux: [u32; AUX_LEN],
}

impl MemoryExtendedAuxRecord {
    pub fn from_aux_cols<F: PrimeField32>(aux_cols: MemoryBaseAuxCols<F>) -> Self {
        Self {
            prev_timestamp: aux_cols.prev_timestamp.as_canonical_u32(),
            timestamp_lt_aux: aux_cols
                .timestamp_lt_aux
                .lower_decomp
                .map(|x| x.as_canonical_u32()),
        }
    }

    pub fn to_aux_cols<F: PrimeField32>(&self, aux_cols: &mut MemoryBaseAuxCols<F>) {
        aux_cols.prev_timestamp = F::from_canonical_u32(self.prev_timestamp);
        aux_cols.timestamp_lt_aux.lower_decomp =
            self.timestamp_lt_aux.map(|x| F::from_canonical_u32(x));
    }
}

pub type MemoryReadAuxRecord = MemoryBaseAuxRecord;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryExtendedAuxRecord {
    pub prev_timestamp: u32,
    pub timestamp_lt_aux: [u32; AUX_LEN],
}

impl MemoryExtendedAuxRecord {
    pub fn from_aux_cols<F: PrimeField32>(aux_cols: MemoryBaseAuxCols<F>) -> Self {
        Self {
            prev_timestamp: aux_cols.prev_timestamp.as_canonical_u32(),
            timestamp_lt_aux: aux_cols
                .timestamp_lt_aux
                .lower_decomp
                .map(|x| x.as_canonical_u32()),
        }
    }

    pub fn to_aux_cols<F: PrimeField32>(&self) -> MemoryBaseAuxCols<F> {
        MemoryBaseAuxCols {
            prev_timestamp: F::from_canonical_u32(self.prev_timestamp),
            timestamp_lt_aux: LessThanAuxCols {
                lower_decomp: self.timestamp_lt_aux.map(|x| F::from_canonical_u32(x)),
            },
        }
    }
}

pub type MemoryReadAuxRecord = MemoryBaseAuxRecord;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemoryWriteAuxRecord<T, const NUM_LIMBS: usize> {
    pub prev_timestamp: u32,
    pub prev_data: [T; NUM_LIMBS],
}

pub type MemoryWriteBytesAuxRecord<const NUM_LIMBS: usize> = MemoryWriteAuxRecord<u8, NUM_LIMBS>;
