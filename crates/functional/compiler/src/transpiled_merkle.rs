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
    pub siblings: TLArray,
    pub inline0_right: [F; 8usize],
    pub inline0_left: [F; 8usize],
    pub child_0_False: [F; 8usize],
    pub bit_0_False: bool,
    pub inline0_result: [F; 8usize],
    pub left: [F; 8usize],
    pub i_0_False: F,
    pub sibling_0_False: [F; 8usize],
    pub leaf: [F; 8usize],
    pub bits: TLArray,
    pub commit: [F; 8usize],
    pub right: [F; 8usize],
    pub length: F,
    pub hash_result: [F; 8usize],
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
        if self.scope_0_False_0_True {
            assert_eq!(self.right, self.child_0_False);
        }
        if self.scope_0_False_0_False {
            self.left = self.child_0_False;
        }
        if self.scope_0_False_0_False {
            assert_eq!(self.left, self.child_0_False);
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
    pub materialized: bool,
    pub call_index: usize,
    pub inline0_left: [F; 8usize],
    pub bits2: TLArray,
    pub bits0: TLArray,
    pub inline0_right: [F; 8usize],
    pub bits1: TLArray,
    pub a: [F; 8usize],
    pub inline1_left: [F; 8usize],
    pub inline1_right: [F; 8usize],
    pub siblings1: TLArray,
    pub c: [F; 8usize],
    pub siblings2: TLArray,
    pub siblings: TLArray,
    pub b: [F; 8usize],
    pub siblings3: TLArray,
    pub should_fail: bool,
    pub y: [F; 8usize],
    pub root: [F; 8usize],
    pub leaf: [F; 8usize],
    pub inline2_result: [F; 8usize],
    pub inline2_left: [F; 8usize],
    pub inline0_result: [F; 8usize],
    pub siblings0: TLArray,
    pub inline2_right: [F; 8usize],
    pub bits: TLArray,
    pub inline1_result: [F; 8usize],
    pub bits3: TLArray,
    pub x: [F; 8usize],
    pub scope_0_False: bool,
    pub scope_0_True: bool,
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
            .create_empty_under_construction_array(0);
        let temp_1 = self.siblings0;
        let (_, temp_2) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(temp_1, self.a);
        self.siblings1 = temp_2;
        let temp_3 = self.siblings1;
        let (_, temp_4) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(temp_3, self.b);
        self.siblings2 = temp_4;
        let temp_5 = self.siblings2;
        let (_, temp_6) = tracker
            .memory_ConstArray8_F
            .append_under_construction_array(temp_5, self.c);
        self.siblings3 = temp_6;
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
}
const MAX_TRACE_HEIGHT: usize = 16777216usize;
#[derive(Default, Debug)]
pub struct TraceSet {
    pub merkle_verify_trace: Vec<F>,
    pub main_trace: Vec<F>,
}
impl TraceSet {
    pub fn new(tracker: &Tracker) -> Self {
        Self {
            merkle_verify_trace: Self::init_trace(
                tracker.merkle_verify_call_counter,
                TLFunction_merkle_verify::TRACE_WIDTH,
            ),
            main_trace: Self::init_trace(tracker.main_call_counter, TLFunction_main::TRACE_WIDTH),
        }
    }
    pub fn init_trace(num_calls: usize, width: usize) -> Vec<F> {
        let height = num_calls.next_power_of_two();
        vec![F::ZERO; height * width]
    }
}
impl TLFunction_merkle_verify {
    const TRACE_WIDTH: usize = 29usize;
    const NUM_REFERENCES: usize = 0usize;
    pub fn generate_trace(&self, tracker: &Tracker, trace_set: &mut TraceSet) {
        if self.scope_0_False {
            self.callee_0
                .as_ref()
                .as_ref()
                .unwrap()
                .generate_trace(tracker, trace_set);
        }
        let row = &mut trace_set.merkle_verify_trace
            [self.call_index * Self::TRACE_WIDTH..(self.call_index + 1) * Self::TRACE_WIDTH];
        if self.scope_0_True {
            row[0usize] = F::ONE;
        }
        if self.scope_0_False_0_True {
            row[1usize] = F::ONE;
        }
        if self.scope_0_False_0_False {
            row[2usize] = F::ONE;
        }
        let mut as_cells = [F::ZERO; 1usize];
        to_cells_F(self.length, &mut as_cells);
        row[11usize] = as_cells[0usize];
        let mut as_cells = [F::ZERO; 8usize];
        to_cells_ConstArray8_F(self.left, &mut as_cells);
        row[13usize] = as_cells[0usize];
        row[14usize] = as_cells[1usize];
        row[15usize] = as_cells[2usize];
        row[16usize] = as_cells[3usize];
        row[17usize] = as_cells[4usize];
        row[18usize] = as_cells[5usize];
        row[19usize] = as_cells[6usize];
        row[20usize] = as_cells[7usize];
        let mut as_cells = [F::ZERO; 8usize];
        to_cells_ConstArray8_F(self.leaf, &mut as_cells);
        row[3usize] = as_cells[0usize];
        row[4usize] = as_cells[1usize];
        row[5usize] = as_cells[2usize];
        row[6usize] = as_cells[3usize];
        row[7usize] = as_cells[4usize];
        row[8usize] = as_cells[5usize];
        row[9usize] = as_cells[6usize];
        row[10usize] = as_cells[7usize];
        let mut as_cells = [F::ZERO; 8usize];
        to_cells_ConstArray8_F(self.right, &mut as_cells);
        row[21usize] = as_cells[0usize];
        row[22usize] = as_cells[1usize];
        row[23usize] = as_cells[2usize];
        row[24usize] = as_cells[3usize];
        row[25usize] = as_cells[4usize];
        row[26usize] = as_cells[5usize];
        row[27usize] = as_cells[6usize];
        row[28usize] = as_cells[7usize];
        let mut as_cells = [F::ZERO; 1usize];
        to_cells_Array_Bool(self.bits, &mut as_cells);
        row[12usize] = as_cells[0usize];
    }
}
impl TLFunction_main {
    const TRACE_WIDTH: usize = 7usize;
    const NUM_REFERENCES: usize = 1usize;
    pub fn generate_trace(&self, tracker: &Tracker, trace_set: &mut TraceSet) {
        if self.scope_0_False {
            self.callee_0
                .as_ref()
                .as_ref()
                .unwrap()
                .generate_trace(tracker, trace_set);
        }
        if self.scope_0_True {
            self.callee_1
                .as_ref()
                .as_ref()
                .unwrap()
                .generate_trace(tracker, trace_set);
        }
        let row = &mut trace_set.main_trace
            [self.call_index * Self::TRACE_WIDTH..(self.call_index + 1) * Self::TRACE_WIDTH];
        row[2usize] = F::from_canonical_usize(self.call_index);
        if self.scope_0_True {
            row[1usize] = F::ONE;
        }
        if self.scope_0_False {
            row[0usize] = F::ONE;
        }
        let mut as_cells = [F::ZERO; 1usize];
        to_cells_Bool(self.should_fail, &mut as_cells);
        row[3usize] = as_cells[0usize];
        row[6usize] = F::from_canonical_usize(
            tracker
                .memory_Bool
                .get_array_multiplicity(self.appended_array_12, self.appended_index_12),
        );
        row[5usize] = F::from_canonical_usize(
            tracker
                .memory_Bool
                .get_array_multiplicity(self.appended_array_11, self.appended_index_11),
        );
        row[4usize] = F::from_canonical_usize(
            tracker
                .memory_Bool
                .get_array_multiplicity(self.appended_array_10, self.appended_index_10),
        );
    }
    pub fn calc_zk_identifier(&self, i: usize) -> usize {
        let offset = match i {
            9usize => 0usize,
            _ => unreachable!(),
        };
        (Self::FUNCTION_ID * MAX_TRACE_HEIGHT) + (self.call_index * Self::NUM_REFERENCES) + offset
    }
}
fn to_cells_ConstArray8_F(value: [F; 8usize], result: &mut [F]) {
    to_cells_F(value[0usize], &mut result[0usize..1usize]);
    to_cells_F(value[1usize], &mut result[1usize..2usize]);
    to_cells_F(value[2usize], &mut result[2usize..3usize]);
    to_cells_F(value[3usize], &mut result[3usize..4usize]);
    to_cells_F(value[4usize], &mut result[4usize..5usize]);
    to_cells_F(value[5usize], &mut result[5usize..6usize]);
    to_cells_F(value[6usize], &mut result[6usize..7usize]);
    to_cells_F(value[7usize], &mut result[7usize..8usize]);
}
fn to_cells_Bool(value: bool, result: &mut [F]) {
    result[0] = F::from_bool(value)
}
fn to_cells_F(value: F, result: &mut [F]) {
    result[0] = value;
}
fn to_cells_Array_Bool(value: TLArray, result: &mut [F]) {
    result[0] = F::from_canonical_usize(value.zk_identifier);
}
fn to_cells_AppendablePrefix_Bool(value: TLArray, result: &mut [F]) {
    result[0] = F::from_canonical_usize(value.zk_identifier);
}
