use afs_middleware::interaction::{Chip, Interaction};
use afs_middleware_derive::AlignedBorrow;
use core::mem::size_of;
use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder, VirtualPairCol};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::indices_arr;
use std::mem::transmute;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DummyInteractionCols<F> {
    pub count: F,
    pub val: F,
}

const NUM_DUMMY_INTERACTION_COLS: usize = size_of::<DummyInteractionCols<u8>>();
const DUMMY_INTERACTION_COL_MAP: DummyInteractionCols<usize> = make_col_map();

const fn make_col_map() -> DummyInteractionCols<usize> {
    let indices_arr = indices_arr::<NUM_DUMMY_INTERACTION_COLS>();
    unsafe {
        transmute::<[usize; NUM_DUMMY_INTERACTION_COLS], DummyInteractionCols<usize>>(indices_arr)
    }
}

pub struct DummyInteractionAir {
    // Send if true. Receive if false.
    pub is_send: bool,
}

impl<F: Field> Chip<F> for DummyInteractionAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        if self.is_send {
            vec![Interaction::<F> {
                fields: vec![VirtualPairCol::<F>::single_main(
                    DUMMY_INTERACTION_COL_MAP.val,
                )],
                count: VirtualPairCol::<F>::single_main(DUMMY_INTERACTION_COL_MAP.count),
                argument_index: 0,
            }]
        } else {
            vec![]
        }
    }
    fn receives(&self) -> Vec<Interaction<F>> {
        if !self.is_send {
            vec![Interaction::<F> {
                fields: vec![VirtualPairCol::<F>::single_main(
                    DUMMY_INTERACTION_COL_MAP.val,
                )],
                count: VirtualPairCol::<F>::single_main(DUMMY_INTERACTION_COL_MAP.count),
                argument_index: 0,
            }]
        } else {
            vec![]
        }
    }
}

impl<F: Field> BaseAir<F> for DummyInteractionAir {
    fn width(&self) -> usize {
        2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for DummyInteractionAir {
    fn eval(&self, _builder: &mut AB) {}
}
