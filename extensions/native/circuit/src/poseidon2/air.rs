use std::{array::from_fn, borrow::Borrow, sync::Arc};

use derive_new::new;
use itertools::izip;
use openvm_circuit::{
    arch::ExecutionBridge,
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};
use openvm_poseidon2_air::{Poseidon2SubAir, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{AbstractField, Field},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use super::{NativePoseidon2Cols, NATIVE_POSEIDON2_CHUNK_SIZE};

#[derive(Debug, new)]
pub struct NativePoseidon2Air<F: Field, const SBOX_REGISTERS: usize> {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
    pub(super) subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub(super) offset: usize,
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for NativePoseidon2Air<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        NativePoseidon2Cols::<F, SBOX_REGISTERS>::width()
    }
}
impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for NativePoseidon2Air<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for NativePoseidon2Air<F, SBOX_REGISTERS>
{
}

impl<AB: InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for NativePoseidon2Air<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);
        self.eval_memory_and_execution(builder);
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> NativePoseidon2Air<F, SBOX_REGISTERS> {
    fn eval_memory_and_execution<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let cols: &NativePoseidon2Cols<AB::Var, SBOX_REGISTERS> = (*local).borrow();

        let timestamp = cols.memory.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Because there are only two opcodes this adapter needs to handle, we make it so that
        // is_valid == 1 if PERMUTE, is_valid == 2 if COMPRESS, and is_valid != 0 if valid.
        let permute = -cols.memory.opcode_flag.into() * (cols.memory.opcode_flag - AB::F::TWO);
        let compress = cols.memory.opcode_flag.into()
            * (cols.memory.opcode_flag - AB::F::ONE)
            * (AB::F::TWO.inverse());
        let is_valid = permute.clone() + compress.clone();
        builder.assert_zero(
            cols.memory.opcode_flag
                * (cols.memory.opcode_flag - AB::F::ONE)
                * (cols.memory.opcode_flag - AB::F::TWO),
        );
        builder
            .when_ne(permute.clone(), AB::F::ONE)
            .assert_eq(cols.memory.c, cols.memory.rs_ptr[1]);

        for (ptr, val, aux, count) in izip!(
            [
                cols.memory.rd_ptr,
                cols.memory.rs_ptr[0],
                cols.memory.rs_ptr[1]
            ],
            [
                cols.memory.rd_val,
                cols.memory.rs_val[0],
                cols.memory.rs_val[1]
            ],
            &[
                cols.memory.rd_read_aux,
                cols.memory.rs_read_aux[0],
                cols.memory.rs_read_aux[1],
            ],
            [is_valid.clone(), is_valid.clone(), compress],
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(cols.memory.ptr_as, ptr),
                    [val],
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, count);
        }

        let read_chunk_1: [_; NATIVE_POSEIDON2_CHUNK_SIZE] = from_fn(|i| cols.inner.inputs[i]);
        let read_chunk_2: [_; NATIVE_POSEIDON2_CHUNK_SIZE] =
            from_fn(|i| cols.inner.inputs[i + NATIVE_POSEIDON2_CHUNK_SIZE]);

        for (ptr, data, aux, count) in izip!(
            [cols.memory.rs_val[0], cols.memory.rs_val[1]],
            [read_chunk_1, read_chunk_2],
            &[cols.memory.chunk_read_aux[0], cols.memory.chunk_read_aux[1]],
            [is_valid.clone(), is_valid.clone()],
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(cols.memory.chunk_as, ptr),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, count);
        }

        let write_chunk_1: [_; NATIVE_POSEIDON2_CHUNK_SIZE] = from_fn(|i| {
            cols.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
        });
        let write_chunk_2: [_; NATIVE_POSEIDON2_CHUNK_SIZE] = from_fn(|i| {
            cols.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post
                [i + NATIVE_POSEIDON2_CHUNK_SIZE]
        });

        for (ptr, data, aux, count) in izip!(
            [
                cols.memory.rd_val.into(),
                cols.memory.rd_val + AB::F::from_canonical_usize(NATIVE_POSEIDON2_CHUNK_SIZE)
            ],
            [write_chunk_1, write_chunk_2],
            &[
                cols.memory.chunk_write_aux[0],
                cols.memory.chunk_write_aux[1]
            ],
            [is_valid.clone(), permute],
        ) {
            self.memory_bridge
                .write(
                    MemoryAddress::new(cols.memory.chunk_as, ptr),
                    data,
                    timestamp_pp(),
                    aux,
                )
                .eval(builder, count);
        }

        self.execution_bridge
            .execute_and_increment_pc(
                cols.memory.opcode_flag - AB::F::ONE + AB::F::from_canonical_usize(self.offset),
                [
                    cols.memory.rd_ptr,
                    cols.memory.rs_ptr[0],
                    cols.memory.c,
                    cols.memory.ptr_as,
                    cols.memory.chunk_as,
                ],
                cols.memory.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, is_valid);
    }
}
