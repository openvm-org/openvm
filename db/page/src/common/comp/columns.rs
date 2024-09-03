use afs_primitives::{
    is_equal_vec::columns::IsEqualVecAuxCols, is_less_than_tuple::columns::IsLessThanTupleAuxCols,
};

pub struct StrictCompAuxCols<T> {
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
}

pub struct StrictInvCompAuxCols<T> {
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    pub inv: T,
}

pub struct NonStrictCompAuxCols<T> {
    pub satisfies_strict_comp: T,
    pub satisfies_eq_comp: T,
    pub is_less_than_tuple_aux: IsLessThanTupleAuxCols<T>,
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}

pub struct EqCompAuxCols<T> {
    pub is_equal_vec_aux: IsEqualVecAuxCols<T>,
}
