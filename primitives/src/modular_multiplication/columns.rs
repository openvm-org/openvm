use itertools::Itertools;
use p3_field::Field;

use crate::modular_multiplication::air::{LimbDimensions, ModularMultiplicationAir};

// a * b = (p * q) + r

pub struct ModularMultiplicationPrimesCols<T> {
    pub io: ModularMultiplicationIoCols<T>,
    pub aux: ModularMultiplicationAuxCols<T>,
    pub system_cols: Vec<SmallModulusSystemCols<T>>,
}

pub struct ModularMultiplicationIoCols<T> {
    pub a_elems: Vec<T>,
    pub b_elems: Vec<T>,
    pub r_elems: Vec<T>,
}

pub struct ModularMultiplicationAuxCols<T> {
    pub a_limbs_without_first: Vec<Vec<T>>,
    pub b_limbs_without_first: Vec<Vec<T>>,
    pub r_limbs_without_first: Vec<Vec<T>>,
    pub q_limbs: Vec<T>,
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

impl<T: Clone> ModularMultiplicationAuxCols<T> {
    pub fn from_slice(slc: &[T], limb_dimensions: &LimbDimensions) -> Self {
        let mut start = 0;
        let mut end = 0;

        let mut take_io_limbs = || {
            limb_dimensions
                .io_limb_sizes
                .iter()
                .map(|limbs| {
                    end += limbs.len() - 1;
                    let result = slc[start..end].to_vec();
                    start = end;
                    result
                })
                .collect_vec()
        };

        let a_limbs = take_io_limbs();
        let b_limbs = take_io_limbs();
        let r_limbs = take_io_limbs();

        end += limb_dimensions.q_limb_sizes.len();
        let q_limbs = slc[start..end].to_vec();

        Self {
            a_limbs_without_first: a_limbs,
            b_limbs_without_first: b_limbs,
            r_limbs_without_first: r_limbs,
            q_limbs,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        for limbs in self.a_limbs_without_first.iter() {
            result.extend(limbs.clone());
        }
        for limbs in self.b_limbs_without_first.iter() {
            result.extend(limbs.clone());
        }
        for limbs in self.r_limbs_without_first.iter() {
            result.extend(limbs.clone());
        }
        result.extend(self.q_limbs.clone());
        result
    }

    pub fn get_width(limb_dimensions: &LimbDimensions) -> usize {
        (3 * limb_dimensions.num_materialized_io_limbs) + limb_dimensions.q_limb_sizes.len()
    }
}

impl<T: Clone> ModularMultiplicationIoCols<T> {
    pub fn from_slice(slc: &[T], limb_dimensions: &LimbDimensions) -> Self {
        let mut start = 0;
        let mut end = 0;

        end += limb_dimensions.io_limb_sizes.len();
        let a_elems = slc[start..end].to_vec();
        start = end;

        end += limb_dimensions.io_limb_sizes.len();
        let b_elems = slc[start..end].to_vec();
        start = end;

        end += limb_dimensions.io_limb_sizes.len();
        let r_elems = slc[start..end].to_vec();

        Self {
            a_elems,
            b_elems,
            r_elems,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        result.extend(self.a_elems.clone());
        result.extend(self.b_elems.clone());
        result.extend(self.r_elems.clone());
        result
    }

    pub fn get_width(limb_dimensions: &LimbDimensions) -> usize {
        3 * limb_dimensions.io_limb_sizes.len()
    }
}

impl<T: Clone> ModularMultiplicationPrimesCols<T> {
    pub fn from_slice<F: Field>(slc: &[T], air: &ModularMultiplicationAir<F>) -> Self {
        let mut start = 0;
        let mut end = 0;

        end += ModularMultiplicationIoCols::<T>::get_width(&air.limb_dimensions);
        let io = ModularMultiplicationIoCols::from_slice(&slc[start..end], &air.limb_dimensions);
        start = end;

        end += ModularMultiplicationAuxCols::<T>::get_width(&air.limb_dimensions);
        let aux = ModularMultiplicationAuxCols::from_slice(&slc[start..end], &air.limb_dimensions);
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
            io,
            aux,
            system_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        result.extend(self.io.flatten());
        result.extend(self.aux.flatten());
        for system_cols in self.system_cols.iter() {
            result.extend(system_cols.flatten());
        }
        result
    }

    pub fn get_width<F: Field>(air: &ModularMultiplicationAir<F>) -> usize {
        ModularMultiplicationIoCols::<T>::get_width(&air.limb_dimensions)
            + ModularMultiplicationAuxCols::<T>::get_width(&air.limb_dimensions)
            + (air.small_moduli_systems.len() * SmallModulusSystemCols::<T>::get_width())
    }
}
