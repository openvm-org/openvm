use afs_derive::AlignedBorrow;

#[derive(Copy, Clone, Debug, AlignedBorrow, PartialEq, Eq)]
#[repr(C)]
pub struct ProgramPreprocessedCols<T> {
    pub pc: T,

    pub opcode: T,
    pub op_a: T,
    pub op_b: T,
    pub op_c: T,
    pub as_b: T,
    pub as_c: T,
    pub op_f: T,
    pub op_g: T,
}

impl<T: Clone> ProgramPreprocessedCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            pc: slc[0].clone(),
            opcode: slc[1].clone(),
            op_a: slc[2].clone(),
            op_b: slc[3].clone(),
            op_c: slc[4].clone(),
            as_b: slc[5].clone(),
            as_c: slc[6].clone(),
            op_f: slc[7].clone(),
            op_g: slc[8].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.pc.clone(),
            self.opcode.clone(),
            self.op_a.clone(),
            self.op_b.clone(),
            self.op_c.clone(),
            self.as_b.clone(),
            self.as_c.clone(),
            self.op_f.clone(),
            self.op_g.clone(),
        ]
    }

    pub fn get_width() -> usize {
        9
    }
}
