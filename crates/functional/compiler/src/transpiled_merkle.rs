use openvm_stark_sdk::openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use std::ops::Neg;
type F = openvm_stark_sdk::p3_baby_bear::BabyBear;
#[derive(Default, Debug)]
pub struct Tracker {
    pub merkle_verify_call_counter: usize,
    pub main_call_counter: usize,
    pub memory_ConstArray8_F: Memory<[F; 8usize]>,
    pub memory_Bool: Memory<bool>,
}
#[derive(Clone, Copy, Default, Debug)]
pub struct TLRef {
    execution_index: usize,
    zk_identifier: usize,
}
#[derive(Clone, Copy, Default, Debug)]
pub struct TLArray {
    execution_index: usize,
    zk_identifier: usize,
}
#[derive(Default, Debug)]
pub struct Memory<T: Copy + Clone> {
    pub references: Vec<T>,
    pub reference_num_accesses: Vec<usize>,
    pub arrays: Vec<Vec<T>>,
    pub array_num_accesses: Vec<Vec<usize>>,
}
impl<T: Copy + Clone> Memory<T> {
    pub fn create_ref(&mut self, value: T, zk_identifier: usize) -> TLRef {
        let index = self.references.len();
        self.references.push(value);
        self.reference_num_accesses.push(0);
        TLRef {
            execution_index: index,
            zk_identifier,
        }
    }
    pub fn dereference(&mut self, reference: TLRef) -> T {
        self.reference_num_accesses[reference.execution_index] += 1;
        self.references[reference.execution_index]
    }
    pub fn create_empty_under_construction_array(&mut self, zk_identifier: usize) -> TLArray {
        let index = self.arrays.len();
        self.arrays.push(vec![]);
        self.array_num_accesses.push(vec![]);
        TLArray {
            execution_index: index,
            zk_identifier,
        }
    }
    pub fn append_under_construction_array(
        &mut self,
        array: TLArray,
        value: T,
    ) -> (usize, TLArray) {
        self.arrays[array.execution_index].push(value);
        self.array_num_accesses[array.execution_index].push(0);
        (
            self.array_num_accesses[array.execution_index].len() - 1,
            array,
        )
    }
    pub fn array_access(&mut self, array: TLArray, index: F) -> T {
        let index = index.as_canonical_u32() as usize;
        self.array_num_accesses[array.execution_index][index] += 1;
        self.arrays[array.execution_index][index]
    }
    pub fn get_reference_multiplicity(&self, reference: TLRef) -> usize {
        self.reference_num_accesses[reference.execution_index]
    }
    pub fn get_array_multiplicity(&self, array: TLArray, index: usize) -> usize {
        self.array_num_accesses[array.execution_index][index]
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
    pub materialized: bool,
    pub call_index: usize,
    pub child_0_False: [F; 8usize],
    pub hash_result: [F; 8usize],
    pub right: [F; 8usize],
    pub commit: [F; 8usize],
    pub left: [F; 8usize],
    pub inline0_left: [F; 8usize],
    pub inline0_result: [F; 8usize],
    pub siblings: TLArray,
    pub i_0_False: F,
    pub inline0_right: [F; 8usize],
    pub leaf: [F; 8usize],
    pub bits: TLArray,
    pub sibling_0_False: [F; 8usize],
    pub length: F,
    pub bit_0_False: bool,
    pub scope_0_True: bool,
    pub scope_0_False: bool,
    pub scope_0_False_0_True: bool,
    pub scope_0_False_0_False: bool,
    pub callee_0: Box<Option<TLFunction_merkle_verify>>,
}
impl TLFunction_merkle_verify {
    const FUNCTION_ID: usize = 0usize;
    pub fn stage_0(&mut self, tracker: &mut Tracker) {
        if self.materialized {
            self.call_index = tracker.merkle_verify_call_counter;
            tracker.merkle_verify_call_counter += 1;
        }
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
            self.callee_0.as_mut().as_mut().unwrap().materialized = self.materialized;
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
    fn calc_zk_identifier(&self, _: usize) -> usize {
        0
    }
}
#[derive(Default, Debug)]
pub struct TLFunction_main {
    pub materialized: bool,
    pub call_index: usize,
    pub bits2: TLArray,
    pub bits3: TLArray,
    pub inline1_right: [F; 8usize],
    pub c: [F; 8usize],
    pub bits1: TLArray,
    pub should_fail: bool,
    pub inline1_result: [F; 8usize],
    pub bits: TLArray,
    pub a: [F; 8usize],
    pub inline0_result: [F; 8usize],
    pub inline2_result: [F; 8usize],
    pub inline2_right: [F; 8usize],
    pub siblings2: TLArray,
    pub siblings3: TLArray,
    pub siblings: TLArray,
    pub x: [F; 8usize],
    pub inline0_right: [F; 8usize],
    pub inline2_left: [F; 8usize],
    pub bits0: TLArray,
    pub root: [F; 8usize],
    pub siblings1: TLArray,
    pub siblings0: TLArray,
    pub leaf: [F; 8usize],
    pub y: [F; 8usize],
    pub inline0_left: [F; 8usize],
    pub b: [F; 8usize],
    pub inline1_left: [F; 8usize],
    pub scope_0_False: bool,
    pub scope_0_True: bool,
    pub appended_array_5: TLArray,
    pub appended_index_5: usize,
    pub appended_array_6: TLArray,
    pub appended_index_6: usize,
    pub appended_array_7: TLArray,
    pub appended_index_7: usize,
    pub appended_array_10: TLArray,
    pub appended_index_10: usize,
    pub appended_array_11: TLArray,
    pub appended_index_11: usize,
    pub appended_array_12: TLArray,
    pub appended_index_12: usize,
    pub callee_0: Box<Option<TLFunction_merkle_verify>>,
    pub callee_1: Box<Option<TLFunction_merkle_verify>>,
}
impl TLFunction_main {
    const FUNCTION_ID: usize = 1usize;
    pub fn stage_0(&mut self, tracker: &mut Tracker) {
        if self.materialized {
            self.call_index = tracker.main_call_counter;
            tracker.main_call_counter += 1;
        }
        self.leaf = [isize_to_field_elem(0isize); 8usize];
        self.a = [isize_to_field_elem(1isize); 8usize];
        self.b = [isize_to_field_elem(2isize); 8usize];
        self.c = [isize_to_field_elem(3isize); 8usize];
        self.siblings0 = tracker
            .memory_ConstArray8_F
            .create_empty_under_construction_array(self.calc_zk_identifier(4usize));
        self.appended_array_5 = self.siblings0;
        let (temp_2, temp_1) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_5, self.a);
        self.appended_index_5 = temp_2;
        self.siblings1 = temp_1;
        self.appended_array_6 = self.siblings1;
        let (temp_4, temp_3) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_6, self.b);
        self.appended_index_6 = temp_4;
        self.siblings2 = temp_3;
        self.appended_array_7 = self.siblings2;
        let (temp_6, temp_5) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(self.appended_array_7, self.c);
        self.appended_index_7 = temp_6;
        self.siblings3 = temp_5;
        self.siblings = self.siblings3;
        self.bits0 = tracker
            .memory_Bool
            .create_empty_under_construction_array(self.calc_zk_identifier(9usize));
        self.appended_array_10 = self.bits0;
        let (temp_8, temp_7) = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_10, false);
        self.appended_index_10 = temp_8;
        self.bits1 = temp_7;
        self.appended_array_11 = self.bits1;
        let (temp_10, temp_9) = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_11, true);
        self.appended_index_11 = temp_10;
        self.bits2 = temp_9;
        self.appended_array_12 = self.bits2;
        let (temp_12, temp_11) = tracker
            .memory_Bool
            .append_under_construction_array(self.appended_array_12, false);
        self.appended_index_12 = temp_12;
        self.bits3 = temp_11;
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
            self.callee_0.as_mut().as_mut().unwrap().materialized = self.materialized;
            self.callee_0.as_mut().as_mut().unwrap().leaf = self.leaf;
            self.callee_0.as_mut().as_mut().unwrap().length = isize_to_field_elem(3isize);
            self.callee_0.as_mut().as_mut().unwrap().bits = self.bits;
            self.callee_0.as_mut().as_mut().unwrap().siblings = self.siblings;
            self.callee_0.as_mut().as_mut().unwrap().stage_0(tracker);
            assert_eq!(self.callee_0.as_ref().as_ref().unwrap().commit, self.root);
        }
        if self.scope_0_True {
            self.callee_1 = Box::new(Some(TLFunction_merkle_verify::default()));
            self.callee_1.as_mut().as_mut().unwrap().materialized = self.materialized;
            self.callee_1.as_mut().as_mut().unwrap().leaf = self.leaf;
            self.callee_1.as_mut().as_mut().unwrap().length = isize_to_field_elem(2isize);
            self.callee_1.as_mut().as_mut().unwrap().bits = self.bits;
            self.callee_1.as_mut().as_mut().unwrap().siblings = self.siblings;
            self.callee_1.as_mut().as_mut().unwrap().stage_0(tracker);
            assert_eq!(self.callee_1.as_ref().as_ref().unwrap().commit, self.root);
        }
    }
    fn calc_zk_identifier(&self, _: usize) -> usize {
        0
    }
}
