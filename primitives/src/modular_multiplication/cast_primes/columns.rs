use p3_field::Field;

use crate::modular_multiplication::{
    cast_primes::air::ModularMultiplicationPrimesAir, columns::ModularMultiplicationCols,
};

// a * b = (p * q) + r

pub struct ModularMultiplicationPrimesCols<T> {
    pub general: ModularMultiplicationCols<T>,
    pub system_cols: Vec<SmallModulusSystemCols<T>>,
}

pub struct SmallModulusSystemCols<T> {
    pub a_quotient: T,
    pub b_quotient: T,
}

impl<T: Clone> SmallModulusSystemCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        SmallModulusSystemCols {
            a_quotient: slc[0].clone(),
            b_quotient: slc[1].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.a_quotient.clone(), self.b_quotient.clone()]
    }

    pub fn get_width() -> usize {
        2
    }
}

impl<T: Clone> ModularMultiplicationPrimesCols<T> {
    pub fn from_slice<F: Field>(slc: &[T], air: &ModularMultiplicationPrimesAir<F>) -> Self {
        let mut start = 0;
        let mut end = 0;

        end += ModularMultiplicationCols::<T>::get_width(&air.limb_dimensions);
        let general = ModularMultiplicationCols::from_slice(&slc[start..end], &air.limb_dimensions);
        start = end;

        let system_cols = (0..air.small_moduli_systems.len())
            .map(|_| {
                end += SmallModulusSystemCols::<T>::get_width();
                let result = SmallModulusSystemCols::from_slice(&slc[start..end]);
                start = end;
                result
            })
            .collect();

        Self {
            general,
            system_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        result.extend(self.general.flatten());
        for system_cols in self.system_cols.iter() {
            result.extend(system_cols.flatten());
        }
        result
    }

    pub fn get_width<F: Field>(air: &ModularMultiplicationPrimesAir<F>) -> usize {
        ModularMultiplicationCols::<T>::get_width(&air.limb_dimensions)
            + (air.small_moduli_systems.len() * SmallModulusSystemCols::<T>::get_width())
    }
}
