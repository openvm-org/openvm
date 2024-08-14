pub struct LongAdditionCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
    pub carry: Vec<T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongAdditionCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn num_limbs() -> usize {
        (ARG_SIZE + LIMB_SIZE - 1) / LIMB_SIZE
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = Self::num_limbs();

        let x_limbs = slc[0..num_limbs].to_vec();
        let y_limbs = slc[num_limbs..2 * num_limbs].to_vec();
        let z_limbs = slc[2 * num_limbs..3 * num_limbs].to_vec();
        let carry = slc[3 * num_limbs..4 * num_limbs].to_vec();

        Self {
            x_limbs,
            y_limbs,
            z_limbs,
            carry,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut flattened = vec![];
        flattened.extend(self.x_limbs.iter().cloned());
        flattened.extend(self.y_limbs.iter().cloned());
        flattened.extend(self.z_limbs.iter().cloned());
        flattened.extend(self.carry.iter().cloned());

        flattened
    }

    pub const fn get_width() -> usize {
        4 * Self::num_limbs() // TODO: discard the last carry limb
    }
}
