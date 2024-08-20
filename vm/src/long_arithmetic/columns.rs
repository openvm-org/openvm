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
}

pub struct LongArithmeticAuxCols<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    // This flag is 1 if the opcode is SUB, and 0 otherwise (if ADD).
    // Will probably evolve into an array of indicators for all supported
    // opcodes by the chip.
    pub opcode_sub_flag: T,
    // Note: this "carry" vector may serve as a "borrow" vector in the case of
    // subtraction. However, I decided to call it just "carry", because:
    // 1. "borrow" may cause confusion in rust,
    // 2. it is more often carry than borrow among [ADD, SUB, MUL],
    // 3. even in case of subtraction, this is technically a "carry" of the
    //    expression x = y + z.
    pub carry: Vec<T>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let io =
            LongArithmeticIoCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[..3 * num_limbs + 2]);
        let aux =
            LongArithmeticAuxCols::<ARG_SIZE, LIMB_SIZE, T>::from_slice(&slc[3 * num_limbs + 2..]);

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
        3 * num_limbs::<ARG_SIZE, LIMB_SIZE>() + 2
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let rcv_count = slc[0].clone();
        let opcode = slc[1].clone();
        let x_limbs = slc[2..2 + num_limbs].to_vec();
        let y_limbs = slc[2 + num_limbs..2 + 2 * num_limbs].to_vec();
        let z_limbs = slc[2 + 2 * num_limbs..2 + 3 * num_limbs].to_vec();

        Self {
            rcv_count,
            opcode,
            x_limbs,
            y_limbs,
            z_limbs,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [
            vec![self.rcv_count.clone(), self.opcode.clone()],
            self.x_limbs.clone(),
            self.y_limbs.clone(),
            self.z_limbs.clone(),
        ]
        .concat()
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Clone>
    LongArithmeticAuxCols<ARG_SIZE, LIMB_SIZE, T>
{
    pub const fn get_width() -> usize {
        1 + num_limbs::<ARG_SIZE, LIMB_SIZE>()
    }

    pub fn from_slice(slc: &[T]) -> Self {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let opcode_sub_flag = slc[0].clone();
        let carry = slc[1..1 + num_limbs].to_vec();

        Self {
            opcode_sub_flag,
            carry,
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        [vec![self.opcode_sub_flag.clone()], self.carry.clone()].concat()
    }
}
