pub struct DummyHashCols<T> {
    pub io: DummyHashIoCols<T>,
    pub aux: DummyHashAuxCols,
    pub width: usize,
    pub rate: usize,
}

#[derive(Clone)]
pub struct DummyHashIoCols<F> {
    pub is_alloc: F,
    pub curr_state: Vec<F>,
    pub to_absorb: Vec<F>,
    pub new_state: Vec<F>,
}

#[derive(Copy, Clone)]
pub struct DummyHashAuxCols {}

impl<F: Copy> DummyHashCols<F> {
    pub fn new(
        is_alloc: F,
        curr_state: Vec<F>,
        to_absorb: Vec<F>,
        new_state: Vec<F>,
        width: usize,
        rate: usize,
    ) -> DummyHashCols<F> {
        DummyHashCols {
            io: DummyHashIoCols {
                is_alloc,
                curr_state,
                to_absorb,
                new_state,
            },
            aux: DummyHashAuxCols {},
            width,
            rate,
        }
    }

    pub fn flatten(&self) -> Vec<F> {
        let mut result = Vec::with_capacity(2 * self.width + self.rate + 1);
        result.push(self.io.is_alloc);
        result.extend_from_slice(&self.io.curr_state);
        result.extend_from_slice(&self.io.to_absorb);
        result.extend_from_slice(&self.io.new_state);
        result
    }

    pub fn get_width(&self) -> usize {
        2 * self.width + self.rate + 1
    }

    pub fn from_slice(slc: &[F], width: usize, rate: usize) -> Self {
        let is_alloc = slc[0];
        let curr_state = slc[1..width + 1].to_vec();
        let to_absorb = slc[width + 1..width + rate + 1].to_vec();
        let new_state = slc[width + rate + 1..2 * width + rate + 1].to_vec();

        Self {
            io: DummyHashIoCols {
                is_alloc,
                curr_state,
                to_absorb,
                new_state,
            },
            aux: DummyHashAuxCols {},
            width,
            rate,
        }
    }
}
