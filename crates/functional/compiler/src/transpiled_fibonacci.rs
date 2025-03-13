use std::ops::Neg;

use openvm_stark_sdk::openvm_stark_backend::p3_field::FieldAlgebra;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;
#[derive(Default, Debug)]
pub struct Tracker {}
#[derive(Clone, Copy, Default, Debug)]
pub struct TLRef(usize);
#[derive(Clone, Copy, Default, Debug)]
pub struct TLArray(usize);
#[derive(Default, Debug)]
pub struct Memory<T: Copy + Clone> {
    pub references: Vec<T>,
    pub reference_timestamps: Vec<usize>,
    pub arrays: Vec<Vec<T>>,
    pub array_timestamps: Vec<usize>,
}
impl<T: Copy + Clone> Memory<T> {
    pub fn create_ref(&mut self, value: T) -> TLRef {
        let index = self.references.len();
        self.references.push(value);
        self.reference_timestamps.push(0);
        TLRef(index)
    }
    pub fn dereference(&self, reference: TLRef) -> T {
        self.references[reference.0]
    }
    pub fn create_empty_under_construction_array(&mut self) -> TLArray {
        let index = self.arrays.len();
        self.arrays.push(vec![]);
        self.array_timestamps.push(0);
        TLArray(index)
    }
    pub fn append_under_construction_array(&mut self, array: TLArray, value: T) {
        self.arrays[array.0].push(value);
    }
    pub fn finalize_array(&mut self, array: TLArray) -> TLArray {
        array
    }
    pub fn array_access(&self, array: TLArray, index: usize) -> T {
        self.arrays[array.0][index]
    }
}
pub fn isize_to_field_elem(x: isize) -> F {
    let base = F::from_canonical_usize(x.unsigned_abs());
    if x >= 0 {
        base
    } else {
        base.neg()
    }
}
#[derive(Default, Debug)]
pub struct TLFunction_fibonacci {
    pub a: F,
    pub y_0_False: F,
    pub n: F,
    pub x_0_False: F,
    pub b: F,
    pub scope_0_True: bool,
    pub scope_0_False: bool,
    pub callee_0: Box<Option<TLFunction_fibonacci>>,
}
impl TLFunction_fibonacci {
    pub fn stage_0(&mut self, tracker: &mut Tracker) {
        match self.n == isize_to_field_elem(0isize) {
            true => self.scope_0_True = true,
            false => self.scope_0_False = true,
        }
        if self.scope_0_True {
            assert_eq!(isize_to_field_elem(0isize), self.n);
        }
        if self.scope_0_True {
            self.a = isize_to_field_elem(0isize);
        }
        if self.scope_0_True {
            self.b = isize_to_field_elem(1isize);
        }
        if self.scope_0_False {
            self.callee_0 = Box::new(Some(TLFunction_fibonacci::default()));
            self.callee_0.as_mut().as_mut().unwrap().n = self.n - isize_to_field_elem(1isize);
            self.callee_0.as_mut().as_mut().unwrap().stage_0(tracker);
            self.x_0_False = self.callee_0.as_ref().as_ref().unwrap().a;
            self.y_0_False = self.callee_0.as_ref().as_ref().unwrap().b;
        }
        if self.scope_0_False {
            self.a = self.y_0_False;
        }
        if self.scope_0_False {
            self.b = self.x_0_False + self.y_0_False;
        }
    }
}
