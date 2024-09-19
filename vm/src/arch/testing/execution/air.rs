use std::{borrow::Borrow, mem::size_of};

use afs_derive::AlignedBorrow;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::arch::{bus::ExecutionBus, columns::ExecutionState};

#[derive(Clone, Copy, Debug, AlignedBorrow, derive_new::new)]
#[repr(C)]
pub struct DummyExecutionInteractionCols<T> {
    /// The receive frequency. To send, set to negative.
    pub count: T,
    pub initial_state: ExecutionState<T>,
    pub final_state: ExecutionState<T>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ExecutionDummyAir {
    pub bus: ExecutionBus,
}

impl<F: Field> BaseAir<F> for ExecutionDummyAir {
    fn width(&self) -> usize {
        size_of::<DummyExecutionInteractionCols<u8>>()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ExecutionDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &DummyExecutionInteractionCols<AB::Var> = (*local).borrow();
        self.bus
            .execute(builder, local.count, local.initial_state, local.final_state);
    }
}
