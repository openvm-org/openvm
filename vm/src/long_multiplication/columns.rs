pub struct LongMultiplicationCols<T> {
    pub rcv_count: T,
    pub opcode: T,
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
    pub carry: Vec<T>,
}

impl<T: Clone> LongMultiplicationCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        assert!(slc.len() % 4 == 2);
        let num_limbs = slc.len() / 4;
        let rcv_count = slc[0].clone();
        let opcode = slc[1].clone();
        let x_limbs = slc[2..num_limbs + 2].to_vec();
        let y_limbs = slc[num_limbs + 2..2 * num_limbs + 2].to_vec();
        let z_limbs = slc[2 * num_limbs + 2..3 * num_limbs + 2].to_vec();
        let carry = slc[3 * num_limbs + 2..].to_vec();
        Self {
            rcv_count,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
            carry,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            vec![self.rcv_count.clone(), self.opcode.clone()],
            self.x_limbs.clone(),
            self.y_limbs.clone(),
            self.z_limbs.clone(),
            self.carry.clone(),
        ]
        .concat()
    }
}
