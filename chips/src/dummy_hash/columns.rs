pub struct DummyHashCols<T, const N: usize, const R: usize> {
    pub io: DummyHashIOCols<T, N, R>,
    pub aux: DummyHashAuxCols,
}

#[derive(Clone)]
pub struct DummyHashIOCols<F, const N: usize, const R: usize> {
    pub curr_state: Vec<F>,
    pub to_absorb: Vec<F>,
    pub new_state: Vec<F>,
}

#[derive(Copy, Clone)]
pub struct DummyHashAuxCols {}

impl<F: Clone, const N: usize, const R: usize> DummyHashCols<F, N, R> {
    pub const fn new(
        curr_state: Vec<F>,
        to_absorb: Vec<F>,
        new_state: Vec<F>,
    ) -> DummyHashCols<F, N, R> {
        DummyHashCols {
            io: DummyHashIOCols {
                curr_state,
                to_absorb,
                new_state,
            },
            aux: DummyHashAuxCols {},
        }
    }

    pub fn flatten(&self) -> Vec<F> {
        let mut result = Vec::with_capacity(N + R + N);
        result.extend_from_slice(&self.io.curr_state);
        result.extend_from_slice(&self.io.to_absorb);
        result.extend_from_slice(&self.io.new_state);
        result
    }

    pub fn get_width() -> usize {
        2 * N + R
    }

    pub fn from_slice(slc: &[F]) -> Self {
        let curr_state = slc[0..N].to_vec();
        let to_absorb = slc[N..N + R].to_vec();
        let new_state = slc[N + R..2 * N + R].to_vec();

        Self {
            io: DummyHashIOCols {
                curr_state,
                to_absorb,
                new_state,
            },
            aux: DummyHashAuxCols {},
        }
    }
}
