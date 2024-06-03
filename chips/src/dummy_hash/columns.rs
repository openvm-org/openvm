use afs_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DummyHashCols<T, const N: usize, const R: usize> {
    pub io: DummyHashIOCols<T, N, R>,
    pub aux: DummyHashAuxCols,
}

#[derive(Copy, Clone)]
pub struct DummyHashIOCols<F, const N: usize, const R: usize> {
    pub curr_state: [F; N],
    pub to_absorb: [F; R],
}

pub struct DummyHashAuxCols {}

impl<F: Clone, const N: usize, const R: usize> DummyHashCols<F, N, R> {
    pub const fn new(curr_state: [F; N], to_absorb: [F; R]) -> DummyHashCols<F, N, R> {
        DummyHashCols {
            io: DummyHashIOCols {
                curr_state,
                to_absorb,
            },
            aux: DummyHashAuxCols {},
        }
    }

    pub fn flatten(&self) -> Vec<F> {
        vec![]
    }

    pub fn get_width() -> usize {
        N + R
    }
}
