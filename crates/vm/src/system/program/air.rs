use std::borrow::Borrow;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use super::ProgramBus;

#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection, PartialEq, Eq)]
#[repr(C)]
pub struct ProgramCols<T> {
    pub cached: ProgramCachedCols<T>,
    pub exec_freq: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection, PartialEq, Eq)]
#[repr(C)]
pub struct ProgramCachedCols<T> {
    pub exec_end: T,
    pub exec: ProgramExecutionCols<T>,
    pub exec_start: T,
}

#[derive(Copy, Clone, Debug, AlignedBorrow, StructReflection, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(ProgramCols<u8>)]
pub struct ProgramAir {
    pub bus: ProgramBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for ProgramAir {}
impl<F: Field> PartitionedBaseAir<F> for ProgramAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![ProgramCachedCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        1
    }
}
impl<F: Field> BaseAir<F> for ProgramAir {
    fn width(&self) -> usize {
        ProgramCols::<F>::width()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for ProgramAir {
    fn eval(&self, builder: &mut AB) {
        let common_trace = builder.common_main();
        let cached_trace = &builder.cached_mains()[0];

        let exec_freq = common_trace.row_slice(0).expect("row 0 present")[0];
        let cached_row = cached_trace.row_slice(0).expect("row 0 present").to_vec();
        let cached_cols: &ProgramCachedCols<AB::Var> = cached_row.as_slice().borrow();

        // These constraints also apply to padding and unexecuted instructions, ensuring
        // that all rows are accounted for in the cached trace commit.
        builder.assert_eq(cached_cols.exec_end, AB::Expr::ONE + builder.is_last_row());
        builder.assert_eq(cached_cols.exec_start, builder.is_first_row());

        self.bus.inner.add_key_with_lookups(
            builder,
            cached_row[1..cached_row.len() - 1].iter().copied(),
            exec_freq,
        );
    }
}
