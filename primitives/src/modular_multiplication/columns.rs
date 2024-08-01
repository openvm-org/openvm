use crate::modular_multiplication::air::ModularMultiplicationAir;

pub const NUM_COLS: usize = 3;

// a * b = (p * q) + r

pub struct ModularMultiplicationCols<T> {
    pub io: ModularMultiplicationIoCols<T>,
    pub aux: ModularMultiplicationAuxCols<T>,
}

pub struct ModularMultiplicationIoCols<T> {
    pub a_elems: Vec<T>,
    pub b_elems: Vec<T>,
    pub r_elems: Vec<T>,
}

pub struct ModularMultiplicationAuxCols<T> {
    pub a_limbs: Vec<Vec<T>>,
    pub b_limbs: Vec<Vec<T>>,
    pub r_limbs: Vec<Vec<T>>,
    pub q_limbs: Vec<T>,
    pub system_cols: Vec<SmallModulusSystemCols<T>>,
}

pub struct SmallModulusSystemCols<T> {
    pub a_residue: T,
    pub a_quotient: T,
    pub b_residue: T,
    pub b_quotient: T,
    pub total_quotient: T,
}

impl<T: Clone> SmallModulusSystemCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        SmallModulusSystemCols {
            a_residue: slc[0].clone(),
            a_quotient: slc[1].clone(),
            b_residue: slc[2].clone(),
            b_quotient: slc[3].clone(),
            total_quotient: slc[4].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.a_residue.clone(),
            self.a_quotient.clone(),
            self.b_residue.clone(),
            self.b_quotient.clone(),
            self.total_quotient.clone(),
        ]
    }

    pub fn get_width() -> usize {
        5
    }
}

impl<T: Clone> ModularMultiplicationAuxCols<T> {
    pub fn from_slice(slc: &[T], air: &ModularMultiplicationAir) -> Self {
        let mut start = 0;
        let mut end = 0;

        let a_limbs = air
            .io_limb_sizes
            .iter()
            .map(|limbs| {
                end += limbs.len();
                let result = slc[start..end].to_vec();
                start = end;
                result
            })
            .collect();

        let b_limbs = air
            .io_limb_sizes
            .iter()
            .map(|limbs| {
                end += limbs.len();
                let result = slc[start..end].to_vec();
                start = end;
                result
            })
            .collect();

        let r_limbs = air
            .io_limb_sizes
            .iter()
            .map(|limbs| {
                end += limbs.len();
                let result = slc[start..end].to_vec();
                start = end;
                result
            })
            .collect();

        end += air.q_limb_sizes.len();
        let q_limbs = slc[start..end].to_vec();
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
            a_limbs,
            b_limbs,
            r_limbs,
            q_limbs,
            system_cols,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![];
        for limbs in self.a_limbs.iter() {
            result.extend(limbs.clone());
        }
        for limbs in self.b_limbs.iter() {
            result.extend(limbs.clone());
        }
        for limbs in self.r_limbs.iter() {
            result.extend(limbs.clone());
        }
        result.extend(self.q_limbs.clone());
        for system_cols in self.system_cols.iter() {
            result.extend(system_cols.flatten());
        }
        result
    }

    pub fn get_width(air: &ModularMultiplicationAir) -> usize {
        (3 * air.num_io_limbs)
            + air.q_limb_sizes.len()
            + (air.small_moduli_systems.len() * SmallModulusSystemCols::<T>::get_width())
    }
}

impl<T: Clone> ModularMultiplicationIoCols<T> {
    pub fn from_slice(slc: &[T], air: &ModularMultiplicationAir) -> Self {
        let mut start = 0;
        let mut end = 0;

        end += air.io_limb_sizes.len();
        let a_elems = slc[start..end].to_vec();
        start = end;

        end += air.io_limb_sizes.len();
        let b_elems = slc[start..end].to_vec();
        start = end;

        end += air.io_limb_sizes.len();
        let r_elems = slc[start..end].to_vec();
        start = end;

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

    pub fn get_width(air: &ModularMultiplicationAir) -> usize {
        3 * air.io_limb_sizes.len()
    }
}

impl<T: Clone> ModularMultiplicationCols<T> {
    pub fn from_slice(slc: &[T], air: &ModularMultiplicationAir) -> Self {
        let io_width = ModularMultiplicationIoCols::<T>::get_width(air);
        let aux_width = ModularMultiplicationAuxCols::<T>::get_width(air);
        Self {
            io: ModularMultiplicationIoCols::from_slice(&slc[0..io_width], air),
            aux: ModularMultiplicationAuxCols::from_slice(
                &slc[io_width..io_width + aux_width],
                air,
            ),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io
            .flatten()
            .into_iter()
            .chain(self.aux.flatten())
            .collect()
    }

    pub fn get_width(air: &ModularMultiplicationAir) -> usize {
        ModularMultiplicationIoCols::<T>::get_width(air)
            + ModularMultiplicationAuxCols::<T>::get_width(air)
    }
}
