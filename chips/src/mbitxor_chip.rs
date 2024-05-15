use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

pub struct MBitXorChip<const M: u32> {
    pub x: Vec<Arc<AtomicU32>>,
    pub y: Vec<Arc<AtomicU32>>,
    pub z: Vec<Arc<AtomicU32>>,
}

impl<const M: u32> MBitXorChip<M> {
    pub fn new() -> Self {
        let mut x = vec![];
        let mut y = vec![];
        let mut z = vec![];
        for _ in 0..(1 << M) {
            x.push(Arc::new(AtomicU32::new(0)));
            y.push(Arc::new(AtomicU32::new(0)));
            z.push(Arc::new(AtomicU32::new(0)));
        }
        Self { x, y, z }
    }

    pub fn compute_xor(&self, x_val: u32, y_val: u32) -> u32 {
        x_val ^ y_val
    }

    pub fn add_count(&self, x_val: u32, y_val: u32) {
        let z_val = self.compute_xor(x_val, y_val);
        self.x[x_val as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.y[y_val as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.z[z_val as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<F: Field, const M: u32> BaseAir<F> for MBitXorChip<M> {
    fn width(&self) -> usize {
        3 // x, y, z
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let mut columns = vec![];
        for i in 0..(1 << M) {
            columns.push(F::from_canonical_u32(i as u32));
        }
        Some(RowMajorMatrix::new_col(columns))
    }
}

impl<AB, const M: u32> Air<AB> for MBitXorChip<M>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
