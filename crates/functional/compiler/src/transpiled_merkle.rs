use std::ops::Neg;

use openvm_stark_sdk::openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;
#[derive(Default, Debug)]
pub struct Tracker {
    pub memory_ConstArray8_F: Memory<[F; 8usize]>,
    pub memory_Bool: Memory<bool>,
}
#[derive(Clone, Copy, Default, Debug)]
pub struct TLRef(usize);
#[derive(Clone, Copy, Default, Debug)]
pub struct TLArray(usize, usize);
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
        TLArray(index, 0)
    }
    pub fn append_under_construction_array(&mut self, array: TLArray, value: T) -> TLArray {
        self.arrays[array.0].push(value);
        TLArray(array.0, array.1 + 1)
    }
    pub fn array_access(&self, array: TLArray, index: F) -> T {
        self.arrays[array.0][index.as_canonical_u32() as usize]
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
pub struct TLFunction_merkle_verify {
    pub bit_0_False: bool,
    pub inline0_left: [F; 8usize],
    pub hash_result: [F; 8usize],
    pub child_0_False: [F; 8usize],
    pub length: F,
    pub inline0_result: [F; 8usize],
    pub commit: [F; 8usize],
    pub siblings: TLArray,
    pub i_0_False: F,
    pub inline0_right: [F; 8usize],
    pub bits: TLArray,
    pub left: [F; 8usize],
    pub sibling_0_False: [F; 8usize],
    pub leaf: [F; 8usize],
    pub right: [F; 8usize],
    pub scope_0_True: bool,
    pub scope_0_False: bool,
    pub scope_0_False_0_True: bool,
    pub scope_0_False_0_False: bool,
    pub callee_0: Box<Option<TLFunction_merkle_verify>>,
}
impl TLFunction_merkle_verify {
    pub fn stage_0(&mut self, tracker: &mut Tracker) {
        match self.length == isize_to_field_elem(0isize) {
            true => self.scope_0_True = true,
            false => self.scope_0_False = true,
        }
        if self.scope_0_True {
            assert_eq!(isize_to_field_elem(0isize), self.length);
        }
        if self.scope_0_True {
            self.left = [isize_to_field_elem(0isize); 8usize];
        }
        if self.scope_0_True {
            self.right = [isize_to_field_elem(0isize); 8usize];
        }
        if self.scope_0_True {
            self.commit = self.leaf;
        }
        if self.scope_0_False {
            self.i_0_False = self.length - isize_to_field_elem(1isize);
        }
        if self.scope_0_False {
            self.bit_0_False = tracker.memory_Bool.array_access(self.bits, self.i_0_False);
        }
        if self.scope_0_False {
            self.sibling_0_False = tracker
                .memory_ConstArray8_F
                .array_access(self.siblings, self.i_0_False);
        }
        if self.scope_0_False {
            match self.bit_0_False {
                true => self.scope_0_False_0_True = true,
                false => self.scope_0_False_0_False = true,
            }
        }
        if self.scope_0_False_0_True {
            self.left = self.sibling_0_False;
        }
        if self.scope_0_False_0_False {
            self.right = self.sibling_0_False;
        }
        if self.scope_0_False {
            self.callee_0 = Box::new(Some(TLFunction_merkle_verify::default()));
            self.callee_0.as_mut().as_mut().unwrap().leaf = self.leaf;
            self.callee_0.as_mut().as_mut().unwrap().length =
                self.length - isize_to_field_elem(1isize);
            self.callee_0.as_mut().as_mut().unwrap().bits = self.bits;
            self.callee_0.as_mut().as_mut().unwrap().siblings = self.siblings;
            self.callee_0.as_mut().as_mut().unwrap().stage_0(tracker);
            self.child_0_False = self.callee_0.as_ref().as_ref().unwrap().commit;
        }
        if self.scope_0_False_0_True {
            self.right = self.child_0_False;
        }
        if self.scope_0_False_0_False {
            self.left = self.child_0_False;
        }
        self.inline0_right = self.right;
        self.inline0_left = self.left;
        self.inline0_result = [
            self.inline0_left[0usize] + self.inline0_right[0usize],
            self.inline0_left[1usize] * self.inline0_right[1usize],
            self.inline0_left[2usize] - self.inline0_right[2usize],
            self.inline0_left[3usize],
            self.inline0_right[4usize],
            isize_to_field_elem(115isize),
            self.inline0_left[6usize] * self.inline0_left[7usize],
            self.inline0_right[6usize] * self.inline0_right[7usize],
        ];
        self.hash_result = self.inline0_result;
        if self.scope_0_False {
            self.commit = self.hash_result;
        }
    }
}
#[derive(Default, Debug)]
pub struct TLFunction_main {
    pub root: [F; 8usize],
    pub siblings: TLArray,
    pub y: [F; 8usize],
    pub bits: TLArray,
    pub bits0: TLArray,
    pub inline0_left: [F; 8usize],
    pub leaf: [F; 8usize],
    pub siblings3: TLArray,
    pub bits3: TLArray,
    pub siblings0: TLArray,
    pub bits1: TLArray,
    pub inline0_result: [F; 8usize],
    pub x: [F; 8usize],
    pub inline0_right: [F; 8usize],
    pub inline1_result: [F; 8usize],
    pub inline1_left: [F; 8usize],
    pub should_fail: bool,
    pub inline1_right: [F; 8usize],
    pub bits2: TLArray,
    pub c: [F; 8usize],
    pub inline2_result: [F; 8usize],
    pub inline2_right: [F; 8usize],
    pub inline2_left: [F; 8usize],
    pub siblings2: TLArray,
    pub b: [F; 8usize],
    pub siblings1: TLArray,
    pub a: [F; 8usize],
    pub scope_0_False: bool,
    pub scope_0_True: bool,
    pub appended_array_5: TLArray,
    pub appended_array_6: TLArray,
    pub appended_array_7: TLArray,
    pub appended_array_10: TLArray,
    pub appended_array_11: TLArray,
    pub appended_array_12: TLArray,
    pub callee_0: Box<Option<TLFunction_merkle_verify>>,
    pub callee_1: Box<Option<TLFunction_merkle_verify>>,
}
impl TLFunction_main {
    pub fn stage_0(&mut self, tracker: &mut Tracker) {
        self.leaf = [isize_to_field_elem(0isize); 8usize];
        self.a = [isize_to_field_elem(1isize); 8usize];
        self.b = [isize_to_field_elem(2isize); 8usize];
        self.c = [isize_to_field_elem(3isize); 8usize];
        self.siblings0 = tracker
            .memory_ConstArray8_F
            .create_empty_under_construction_array();
        self.appended_array_5 = self.siblings0;
        self.siblings1 = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_5, self.a);
        self.appended_array_6 = self.siblings1;
        self.siblings2 = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_6, self.b);
        self.appended_array_7 = self.siblings2;
        self.siblings3 = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_7, self.c);
        self.siblings = self.siblings3;
        self.bits0 = tracker.memory_Bool.create_empty_under_construction_array();
        self.appended_array_10 = self.bits0;
        self.bits1 = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_10, false);
        self.appended_array_11 = self.bits1;
        self.bits2 = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_11, true);
        self.appended_array_12 = self.bits2;
        self.bits3 = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_12, false);
        self.bits = self.bits3;
        match self.should_fail {
            false => self.scope_0_False = true,
            true => self.scope_0_True = true,
        }
        self.inline0_right = self.a;
        self.inline0_left = self.leaf;
        self.inline0_result = [
            self.inline0_left[0usize] + self.inline0_right[0usize],
            self.inline0_left[1usize] * self.inline0_right[1usize],
            self.inline0_left[2usize] - self.inline0_right[2usize],
            self.inline0_left[3usize],
            self.inline0_right[4usize],
            isize_to_field_elem(115isize),
            self.inline0_left[6usize] * self.inline0_left[7usize],
            self.inline0_right[6usize] * self.inline0_right[7usize],
        ];
        self.x = self.inline0_result;
        self.inline1_right = self.x;
        self.inline1_left = self.b;
        self.inline1_result = [
            self.inline1_left[0usize] + self.inline1_right[0usize],
            self.inline1_left[1usize] * self.inline1_right[1usize],
            self.inline1_left[2usize] - self.inline1_right[2usize],
            self.inline1_left[3usize],
            self.inline1_right[4usize],
            isize_to_field_elem(115isize),
            self.inline1_left[6usize] * self.inline1_left[7usize],
            self.inline1_right[6usize] * self.inline1_right[7usize],
        ];
        self.y = self.inline1_result;
        self.inline2_right = self.c;
        self.inline2_left = self.y;
        self.inline2_result = [
            self.inline2_left[0usize] + self.inline2_right[0usize],
            self.inline2_left[1usize] * self.inline2_right[1usize],
            self.inline2_left[2usize] - self.inline2_right[2usize],
            self.inline2_left[3usize],
            self.inline2_right[4usize],
            isize_to_field_elem(115isize),
            self.inline2_left[6usize] * self.inline2_left[7usize],
            self.inline2_right[6usize] * self.inline2_right[7usize],
        ];
        self.root = self.inline2_result;
        if self.scope_0_False {
            self.callee_0 = Box::new(Some(TLFunction_merkle_verify::default()));
            self.callee_0.as_mut().as_mut().unwrap().leaf = self.leaf;
            self.callee_0.as_mut().as_mut().unwrap().length = isize_to_field_elem(3isize);
            self.callee_0.as_mut().as_mut().unwrap().bits = self.bits;
            self.callee_0.as_mut().as_mut().unwrap().siblings = self.siblings;
            self.callee_0.as_mut().as_mut().unwrap().stage_0(tracker);
            assert_eq!(self.callee_0.as_ref().as_ref().unwrap().commit, self.root);
        }
        if self.scope_0_True {
            self.callee_1 = Box::new(Some(TLFunction_merkle_verify::default()));
            self.callee_1.as_mut().as_mut().unwrap().leaf = self.leaf;
            self.callee_1.as_mut().as_mut().unwrap().length = isize_to_field_elem(2isize);
            self.callee_1.as_mut().as_mut().unwrap().bits = self.bits;
            self.callee_1.as_mut().as_mut().unwrap().siblings = self.siblings;
            self.callee_1.as_mut().as_mut().unwrap().stage_0(tracker);
            assert_eq!(self.callee_1.as_ref().as_ref().unwrap().commit, self.root);
        }
    }
}
