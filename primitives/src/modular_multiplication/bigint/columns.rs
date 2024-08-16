use crate::modular_multiplication::{
    bigint::air::ModularMultiplicationBigIntAir, columns::ModularMultiplicationCols,
};

// a * b = (p * q) + r

pub struct ModularMultiplicationBigIntCols<T> {
    pub general: ModularMultiplicationCols<T>,
    pub carries: Vec<T>,
}

impl<T: Clone> ModularMultiplicationBigIntCols<T> {
    pub fn from_slice(slc: &[T], air: &ModularMultiplicationBigIntAir) -> Self {
        let mut start = 0;
        let mut end = 0;

        end += ModularMultiplicationCols::<T>::get_width(&air.limb_dimensions);
        let general = ModularMultiplicationCols::from_slice(&slc[start..end], &air.limb_dimensions);
        start = end;

        end += air.num_carries;
        let carries = slc[start..end].to_vec();

        Self { general, carries }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        result.extend(self.general.flatten());
        result.extend(self.carries.clone());
        result
    }

    pub fn get_width(air: &ModularMultiplicationBigIntAir) -> usize {
        ModularMultiplicationCols::<T>::get_width(&air.limb_dimensions) + air.num_carries
    }
}
