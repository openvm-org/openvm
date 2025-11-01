use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::F;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{ExpBitsLenBus, ExpBitsLenMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
struct ExpBitsLenCols<T> {
    is_valid: T,
    base: T,
    bit_src: T,
    num_bits: T,
    result: T,
}

#[derive(Debug)]
pub struct ExpBitsLenAir {
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub(crate) records: std::sync::Mutex<Vec<ExpBitsLenRecord>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ExpBitsLenRecord {
    base: F,
    bit_src: F,
    num_bits: F,
    result: F,
}

impl BaseAirWithPublicValues<F> for ExpBitsLenAir {}
impl PartitionedBaseAir<F> for ExpBitsLenAir {}

impl<F> BaseAir<F> for ExpBitsLenAir {
    fn width(&self) -> usize {
        ExpBitsLenCols::<F>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &ExpBitsLenCols<AB::Var> = (*local).borrow();

        // TODO: add constraints

        self.exp_bits_len_bus.add_key_with_lookups(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.is_valid,
        );
    }
}

impl ExpBitsLenAir {
    pub fn new(exp_bits_len_bus: ExpBitsLenBus) -> Self {
        Self {
            exp_bits_len_bus,
            records: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn add_exp_bits_len(&self, base: F, bit_src: F, num_bits: F, result: F) {
        let record = ExpBitsLenRecord {
            base,
            bit_src,
            num_bits,
            result,
        };
        self.records.lock().unwrap().push(record);
    }

    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let records = self.records.lock().unwrap();
        let num_valid_rows = records.len();
        let num_rows = num_valid_rows.next_power_of_two();
        let width = ExpBitsLenCols::<F>::width();

        let mut trace = vec![F::ZERO; num_rows * width];

        for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
            let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
            let record = &records[i];

            cols.is_valid = F::ONE;
            cols.base = record.base;
            cols.bit_src = record.bit_src;
            cols.num_bits = record.num_bits;
            cols.result = record.result;
        }

        RowMajorMatrix::new(trace, BaseAir::<F>::width(self))
    }
}
