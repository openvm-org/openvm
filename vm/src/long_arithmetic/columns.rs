use super::num_limbs;

pub struct LongArithmeticCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub io: LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>,
    pub aux: LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>,
}

pub struct LongArithmeticIoCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub rcv_count: T,
    pub opcode: T,
    pub x_limbs: Vec<T>,
    pub y_limbs: Vec<T>,
    pub z_limbs: Vec<T>,
    pub cmp_result: T,
}

pub struct LongArithmeticAuxCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub opcode_add_flag: T, // 1 if z_limbs should contain the result of addition
    pub opcode_sub_flag: T, // 1 if z_limbs should contain the result of subtraction (means that opcode is SUB or LT)
    pub opcode_lt_flag: T,  // 1 if opcode is LT
    pub opcode_eq_flag: T,  // 1 if opcode is EQ
    // buffer is the carry of the addition/subtraction,
    // or may serve as a single-nonzero-inverse helper vector for EQ256.
    // Refer to air.rs for more details.
    pub buffer: Vec<T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let io =
            LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[..3 * num_limbs + 3]);
        let aux =
            LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[3 * num_limbs + 3..]);

        Self { io, aux }
    }

    pub fn flatten(&self) -> Vec<T> {
        [self.io.flatten(), self.aux.flatten()].concat()
    }

    pub const fn get_width() -> usize {
        LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::get_width()
            + LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::get_width()
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticIoCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        3 * num_limbs::<ARG_SIZE, LIMB_SIZE>() + 3
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let rcv_count = slc[0].clone();
        let opcode = slc[1].clone();
        let x_limbs = slc[2..2 + num_limbs].to_vec();
        let y_limbs = slc[2 + num_limbs..2 + 2 * num_limbs].to_vec();
        let z_limbs = slc[2 + 2 * num_limbs..2 + 3 * num_limbs].to_vec();
        let cmp_result = slc[2 + 3 * num_limbs].clone();

        Self {
            rcv_count,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
            cmp_result,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            vec![self.rcv_count.clone(), self.opcode.clone()],
            self.x_limbs.clone(),
            self.y_limbs.clone(),
            self.z_limbs.clone(),
            vec![self.cmp_result.clone()],
        ]
        .concat()
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        4 + num_limbs::<ARG_SIZE, LIMB_SIZE>()
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let opcode_add_flag = slc[0].clone();
        let opcode_sub_flag = slc[1].clone();
        let opcode_lt_flag = slc[2].clone();
        let opcode_eq_flag = slc[3].clone();
        let buffer = slc[4..4 + num_limbs].to_vec();

        Self {
            opcode_add_flag,
            opcode_sub_flag,
            opcode_lt_flag,
            opcode_eq_flag,
            buffer,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            vec![
                self.opcode_add_flag.clone(),
                self.opcode_sub_flag.clone(),
                self.opcode_lt_flag.clone(),
                self.opcode_eq_flag.clone(),
            ],
            self.buffer.clone(),
        ]
        .concat()
    }
}
