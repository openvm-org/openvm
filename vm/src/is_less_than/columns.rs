use afs_primitives::is_less_than::columns::IsLessThanCols;

pub struct IsLessThanVmCols<T> {
    pub is_enabled: T,
    pub internal: IsLessThanCols<T>,
}

impl<T: Clone> IsLessThanVmCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            is_enabled: slc[0].clone(),
            internal: IsLessThanCols::from_slice(&slc[1..]),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.is_enabled.clone()];
        result.extend(self.internal.flatten());
        result
    }
}
