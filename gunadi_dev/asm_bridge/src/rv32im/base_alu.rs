use crate::c_void;

use openvm_rv32im_circuit::{
    AluOp,
    AddOp,
    SubOp,
    XorOp,
    OrOp,
    AndOp,
};

// base, split_pre_compute_ptr, pc, instret
// instret only increase by 1

