use super::Poseidon2Chip;
// use afs_derive::AlignedBorrow;
use super::Poseidon2Query;
use p3_field::Field;
use poseidon2::poseidon2::columns::Poseidon2Cols;
/// Columns for field arithmetic chip.
///
/// Five IO columns for rcv_count, opcode, x, y, result.
/// Eight aux columns for interpreting opcode, evaluating indicators, inverse, and explicit computations.
// #[derive(AlignedBorrow)]
// #[repr(C)]
pub struct Poseidon2ChipCols<const WIDTH: usize, T> {
    pub io: Poseidon2ChipIoCols<T>,
    pub aux: Poseidon2Cols<WIDTH, T>,
}

pub struct Poseidon2ChipIoCols<T> {
    pub clk: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub cmp: T,
}

impl<const WIDTH: usize, T> Poseidon2ChipCols<WIDTH, T>
where
    T: Field,
{
    pub fn get_width(poseidon2_chip: &Poseidon2Chip<WIDTH, T>) -> usize {
        Poseidon2ChipIoCols::<T>::get_width()
            + Poseidon2Cols::<WIDTH, T>::get_width(&poseidon2_chip.air)
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = self.io.flatten();
        result.extend(self.aux.flatten());
        result
    }

    pub fn from_slice(
        slice: &[T],
        poseidon2_chip: &Poseidon2Chip<WIDTH, T>,
    ) -> Poseidon2ChipCols<WIDTH, T> {
        let io_width = Poseidon2ChipIoCols::<T>::get_width();
        Self {
            io: Poseidon2ChipIoCols::<T>::from_slice(&slice[..io_width]),
            aux: Poseidon2Cols::<WIDTH, T>::from_slice(
                &slice[io_width..],
                &Poseidon2Cols::<WIDTH, T>::index_map(&poseidon2_chip.air),
            ),
        }
    }
}

impl<T: Field> Poseidon2ChipIoCols<T> {
    pub fn get_width() -> usize {
        7
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![self.clk, self.a, self.b, self.c, self.d, self.e, self.cmp]
    }

    pub fn from_slice(slice: &[T]) -> Self {
        Self {
            clk: slice[0],
            a: slice[1],
            b: slice[2],
            c: slice[3],
            d: slice[4],
            e: slice[5],
            cmp: slice[6],
        }
    }
}
