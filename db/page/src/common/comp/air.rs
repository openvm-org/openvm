use afs_primitives::{is_equal_vec::IsEqualVecAir, is_less_than_tuple::IsLessThanTupleAir};

pub struct StrictCompAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
}

pub struct StrictInvCompAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
    pub inv: usize,
}

// TODO[optimization]: <= is same as not >
pub struct NonStrictCompAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
    pub is_equal_vec_air: IsEqualVecAir,
}

pub struct EqCompAir {
    pub is_equal_vec_air: IsEqualVecAir,
}
