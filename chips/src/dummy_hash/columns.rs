pub struct DummyHashCols<T> {
    pub io: DummyHashIOCols<T>,
    pub aux: DummyHashAuxCols,
    pub width: usize,
    pub rate: usize,
}

#[derive(Clone)]
pub struct DummyHashIOCols<F> {
    pub curr_state: Vec<F>,
    pub to_absorb: Vec<F>,
    pub new_state: Vec<F>,
}

#[derive(Copy, Clone)]
pub struct DummyHashAuxCols {}

impl<F: Clone> DummyHashCols<F> {
    pub const fn new(
        curr_state: Vec<F>,
        to_absorb: Vec<F>,
        new_state: Vec<F>,
        width: usize,
        rate: usize,
    ) -> DummyHashCols<F> {
        DummyHashCols {
            io: DummyHashIOCols {
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
        let mut result = Vec::with_capacity(2 * self.width + self.rate);
        result.extend_from_slice(&self.io.curr_state);
        result.extend_from_slice(&self.io.to_absorb);
        result.extend_from_slice(&self.io.new_state);
        result
    }

    pub fn get_width(&self) -> usize {
        2 * self.width + self.rate
    }

    pub fn from_slice(slc: &[F], width: usize, rate: usize) -> Self {
        let curr_state = slc[0..width].to_vec();
        let to_absorb = slc[width..width + rate].to_vec();
        let new_state = slc[width + rate..2 * width + rate].to_vec();

        Self {
            io: DummyHashIOCols {
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
