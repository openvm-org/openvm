use core::{borrow::Borrow, mem::size_of};
use std::{array::from_fn, sync::Arc};

use openvm_circuit_primitives::ColumnsAir;
use openvm_poseidon2_air::{
    Poseidon2SubAir, Poseidon2SubCols, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
    BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder, interaction::InteractionBuilder, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::bus::{
    Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
};

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2Cols<T, const SBOX_REGISTERS: usize> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub permute_mult: T,
    pub compress_mult: T,
}

pub struct Poseidon2Air<F: Field, const SBOX_REGISTERS: usize> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2Air<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        Poseidon2Cols::<F, SBOX_REGISTERS>::width()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}

impl<AB: AirBuilder + InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2Air<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should have at least one row");
        let local: &Poseidon2Cols<AB::Var, SBOX_REGISTERS> = (*local).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        self.poseidon2_permute_bus.add_key_with_lookups(
            builder,
            Poseidon2PermuteMessage {
                input: local.inner.inputs,
                output: local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1]
                    .post,
            },
            local.permute_mult,
        );

        self.poseidon2_compress_bus.add_key_with_lookups(
            builder,
            Poseidon2CompressMessage {
                input: local.inner.inputs,
                output: from_fn(|i| {
                    local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
                }),
            },
            local.compress_mult,
        );
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> ColumnsAir for Poseidon2Air<F, SBOX_REGISTERS> {
    fn columns(&self) -> Option<Vec<String>> {
        let mut columns = poseidon2_sub_columns::<SBOX_REGISTERS>();
        columns.push("permute_mult".to_string());
        columns.push("compress_mult".to_string());
        debug_assert_eq!(columns.len(), Poseidon2Cols::<u8, SBOX_REGISTERS>::width());
        Some(columns)
    }
}

pub(crate) fn poseidon2_sub_columns<const SBOX_REGISTERS: usize>() -> Vec<String> {
    let mut columns = Vec::with_capacity(size_of::<Poseidon2SubCols<u8, SBOX_REGISTERS>>());
    columns.push("export_col".to_string());
    push_array_columns(&mut columns, "inputs", POSEIDON2_WIDTH);
    for round in 0..BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS {
        for lane in 0..POSEIDON2_WIDTH {
            push_array_columns(
                &mut columns,
                &format!("beginning_full_rounds_{round}_sbox_{lane}_registers"),
                SBOX_REGISTERS,
            );
        }
        push_array_columns(
            &mut columns,
            &format!("beginning_full_rounds_{round}_post"),
            POSEIDON2_WIDTH,
        );
    }
    for round in 0..BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS {
        push_array_columns(
            &mut columns,
            &format!("partial_rounds_{round}_sbox_registers"),
            SBOX_REGISTERS,
        );
        columns.push(format!("partial_rounds_{round}_post"));
    }
    for round in 0..BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS {
        for lane in 0..POSEIDON2_WIDTH {
            push_array_columns(
                &mut columns,
                &format!("ending_full_rounds_{round}_sbox_{lane}_registers"),
                SBOX_REGISTERS,
            );
        }
        push_array_columns(
            &mut columns,
            &format!("ending_full_rounds_{round}_post"),
            POSEIDON2_WIDTH,
        );
    }
    debug_assert_eq!(
        columns.len(),
        size_of::<Poseidon2SubCols<u8, SBOX_REGISTERS>>()
    );
    columns
}

fn push_array_columns(columns: &mut Vec<String>, field: &str, len: usize) {
    for i in 0..len {
        columns.push(format!("{field}_{i}"));
    }
}
