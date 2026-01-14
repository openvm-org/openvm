use openvm_circuit_primitives_derive::AlignedBorrow;
use p3_keccak_air::KeccakCols as KeccakPermCols;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
pub struct KeccakfPeripheryCols<T> {
    pub inner: KeccakPermCols<T>,
    /// The AIR **assumes** but does not constrain that the timestamp should be unique for each
    /// distinct preimage state.
    pub timestamp: T,
}

impl<T: Copy> KeccakfPeripheryCols<T> {
    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        self.inner.a_prime_prime_prime(y, x, limb)
    }

    pub fn is_first_round(&self) -> T {
        *self.inner.step_flags.first().unwrap()
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

pub const NUM_KECCAKF_PERI_COLS: usize = size_of::<KeccakfPeripheryCols<u8>>();
