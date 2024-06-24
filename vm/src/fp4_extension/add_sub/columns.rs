use afs_derive::AlignedBorrow;

/// Columns for field arithmetic chip.
///
/// Four IO columns for opcode, x, y, result.
/// Seven aux columns for interpreting opcode, evaluating indicators, and explicit computations.
#[derive(AlignedBorrow)]
pub struct FieldExtensionAddSubCols<T> {
    pub io: FieldExtensionAddSubIOCols<T>,
}

pub struct FieldExtensionAddSubIOCols<T> {
    pub opcode: T,
    pub x: [T; 4],
    pub y: [T; 4],
    pub z: [T; 4],
}

impl<T: Clone> FieldExtensionAddSubCols<T> {
    pub const NUM_COLS: usize = 13;

    pub fn get_width() -> usize {
        FieldExtensionAddSubIOCols::<T>::get_width()
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let opcode = slice[0].clone();
        let x: [T; 4] = [
            slice[1].clone(),
            slice[2].clone(),
            slice[3].clone(),
            slice[4].clone(),
        ];
        let y: [T; 4] = [
            slice[5].clone(),
            slice[6].clone(),
            slice[7].clone(),
            slice[8].clone(),
        ];
        let z: [T; 4] = [
            slice[9].clone(),
            slice[10].clone(),
            slice[11].clone(),
            slice[12].clone(),
        ];

        Self {
            io: FieldExtensionAddSubIOCols { opcode, x, y, z },
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        self.io.flatten()
    }
}

impl<T: Clone> FieldExtensionAddSubIOCols<T> {
    pub fn get_width() -> usize {
        13
    }

    pub fn flatten(&self) -> Vec<T> {
        let mut result = vec![self.opcode.clone()];
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }
}
