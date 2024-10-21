use afs_derive::AlignedBorrow;

#[derive(Copy, Clone, Debug, AlignedBorrow, PartialEq, Eq)]
#[repr(C)]
pub struct ProgramCols<T> {
    pub exec: ProgramExecutionCols<T>,
    pub exec_freq: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow, PartialEq, Eq)]
#[repr(C)]
pub struct ProgramExecutionCols<T> {
    pub pc: T,

    pub opcode: T,
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    pub g: T,
}

// Straightforward implementation for from_slice, flatten, and width functions.

impl<T: Clone> ProgramCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            exec: ProgramExecutionCols::from_slice(&slc[0..slc.len() - 1]),
            exec_freq: slc[slc.len() - 1].clone(),
        }
    }

    pub fn flatten(self) -> Vec<T> {
        self.exec
            .flatten()
            .into_iter()
            .chain(vec![self.exec_freq])
            .collect()
    }
}

impl<T: Clone> ProgramExecutionCols<T> {
    pub fn from_slice(slc: &[T]) -> Self {
        Self {
            pc: slc[0].clone(),
            opcode: slc[1].clone(),
            a: slc[2].clone(),
            b: slc[3].clone(),
            c: slc[4].clone(),
            d: slc[5].clone(),
            e: slc[6].clone(),
            f: slc[7].clone(),
            g: slc[8].clone(),
        }
    }

    pub fn flatten(&self) -> Vec<T> {
        vec![
            self.pc.clone(),
            self.opcode.clone(),
            self.a.clone(),
            self.b.clone(),
            self.c.clone(),
            self.d.clone(),
            self.e.clone(),
            self.f.clone(),
            self.g.clone(),
        ]
    }
}
