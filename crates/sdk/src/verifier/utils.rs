use std::array;

use openvm_circuit::{
    arch::MemoryConfig, system::memory::tree::public_values::PUBLIC_VALUES_ADDRESS_SPACE_OFFSET,
};
use openvm_native_compiler::prelude::*;
use openvm_native_recursion::{hints::Hintable, types::InnerConfig};
use openvm_stark_backend::p3_util::log2_strict_usize;
use openvm_stark_sdk::{openvm_stark_backend::p3_field::FieldAlgebra, p3_baby_bear::BabyBear};

use super::leaf::types::UserPublicValuesRootProof;
use crate::{C, F};

pub(crate) fn assign_array_to_slice<C: Config>(
    builder: &mut Builder<C>,
    dst_slice: &[Felt<C::F>],
    src: &Array<C, Felt<C::F>>,
    src_offset: usize,
) {
    for (i, dst) in dst_slice.iter().enumerate() {
        let pv = builder.get(src, i + src_offset);
        builder.assign(dst, pv);
    }
}

pub(crate) fn assign_slice_to_array<C: Config>(
    builder: &mut Builder<C>,
    dst: &Array<C, Felt<C::F>>,
    src_slice: &[Felt<C::F>],
) {
    for (i, &src) in src_slice.iter().enumerate() {
        builder.set_value(dst, i, src);
    }
}

pub(crate) fn write_field_slice(arr: &[BabyBear; DIGEST_SIZE]) -> Vec<Vec<BabyBear>> {
    arr.iter()
        .flat_map(Hintable::<InnerConfig>::write)
        .collect()
}

/// Returns 1 if lhs == rhs, 0 otherwise.
pub(crate) fn eq_felt_slice<C: Config, const N: usize>(
    builder: &mut Builder<C>,
    lhs: &[Felt<C::F>; N],
    rhs: &[Felt<C::F>; N],
) -> Var<C::N> {
    let sub_res: [Felt<C::F>; N] = array::from_fn(|i| builder.eval(lhs[i] - rhs[i]));
    let var_res = sub_res.map(|f| builder.cast_felt_to_var(f));
    let ret: Var<C::N> = builder.eval(C::N::ONE);
    var_res.into_iter().for_each(|v| {
        builder
            .if_ne(v, C::N::ZERO)
            .then(|builder| builder.assign(&ret, C::N::ZERO))
    });
    ret
}

#[derive(Clone)]
pub(crate) struct VariableP2Compressor<C: Config> {
    state: Array<C, Felt<C::F>>,
    lhs: Array<C, Felt<C::F>>,
    rhs: Array<C, Felt<C::F>>,
}

impl<C: Config> VariableP2Compressor<C> {
    pub fn new(builder: &mut Builder<C>) -> Self {
        Self {
            state: builder.array(PERMUTATION_WIDTH),
            lhs: builder.array(DIGEST_SIZE),
            rhs: builder.array(DIGEST_SIZE),
        }
    }

    pub fn compress(
        &self,
        builder: &mut Builder<C>,
        lhs: &[Felt<C::F>; DIGEST_SIZE],
        rhs: &[Felt<C::F>; DIGEST_SIZE],
    ) -> [Felt<C::F>; DIGEST_SIZE] {
        assign_slice_to_array(builder, &self.lhs, lhs);
        assign_slice_to_array(builder, &self.rhs, rhs);
        builder.poseidon2_compress_x(&self.state, &self.lhs, &self.rhs);
        array::from_fn(|i| builder.get(&self.state, i))
    }

    pub fn compress_array(
        &self,
        builder: &mut Builder<C>,
        lhs: &Array<C, Felt<C::F>>,
        rhs: &Array<C, Felt<C::F>>,
    ) -> Array<C, Felt<C::F>> {
        let ret = builder.array(DIGEST_SIZE);
        builder.poseidon2_compress_x(&self.state, lhs, rhs);
        for i in 0..DIGEST_SIZE {
            let v = builder.get(&self.state, i);
            builder.set_value(&ret, i, v);
        }
        ret
    }
}

#[derive(Clone)]
pub(crate) struct VariableP2Hasher<C: Config> {
    pub compressor: VariableP2Compressor<C>,
    pub const_zeros: Array<C, Felt<C::F>>,
    pub const_zero: Felt<C::F>,
}

impl<C: Config> VariableP2Hasher<C> {
    pub fn new(builder: &mut Builder<C>) -> Self {
        let const_zero: Felt<C::F> = builder.eval(C::F::ZERO);
        let const_zeros = builder.array(DIGEST_SIZE);
        for i in 0..DIGEST_SIZE {
            builder.set_value(&const_zeros, i, const_zero);
        }
        Self {
            compressor: VariableP2Compressor::new(builder),
            const_zeros,
            const_zero,
        }
    }
    pub fn hash(
        &self,
        builder: &mut Builder<C>,
        payload: &[Felt<C::F>; DIGEST_SIZE],
    ) -> [Felt<C::F>; DIGEST_SIZE] {
        self.compressor
            .compress(builder, payload, &[self.const_zero; DIGEST_SIZE])
    }
    pub fn hash_array(
        &self,
        builder: &mut Builder<C>,
        payload: &Array<C, Felt<C::F>>,
    ) -> Array<C, Felt<C::F>> {
        self.compressor
            .compress_array(builder, payload, &self.const_zeros)
    }

    pub fn merkle_root(
        &self,
        builder: &mut Builder<C>,
        values: &[Felt<C::F>],
    ) -> [Felt<C::F>; DIGEST_SIZE] {
        assert_eq!(values.len() % DIGEST_SIZE, 0);
        assert!((values.len() / DIGEST_SIZE).is_power_of_two());
        let buffer = builder.array(DIGEST_SIZE);
        let mut leaves: Vec<_> = values
            .chunks_exact(DIGEST_SIZE)
            .map(|chunk| {
                assign_slice_to_array(builder, &buffer, chunk);
                self.hash_array(builder, &buffer)
            })
            .collect();
        while leaves.len() > 1 {
            leaves = leaves
                .chunks_exact(2)
                .map(|chunk| {
                    self.compressor
                        .compress_array(builder, &chunk[0], &chunk[1])
                })
                .collect();
        }
        array::from_fn(|i| builder.get(&leaves[0], i))
    }
}

pub(crate) fn heights_le(a: &[usize], b: &[usize]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).all(|(a, b)| a <= b)
}

/// Read the public values root proof from the input stream and verify it.
/// This verification must be consistent `openvm_circuit::system::memory::tree::public_values`.
/// Returns the public values commit and the corresponding memory state root.
pub(crate) fn verify_user_public_values_root(
    builder: &mut Builder<C>,
    num_public_values: usize,
    memory_config: MemoryConfig,
) -> ([Felt<F>; DIGEST_SIZE], [Felt<F>; DIGEST_SIZE]) {
    let memory_dimensions = memory_config.memory_dimensions();
    let pv_as = PUBLIC_VALUES_ADDRESS_SPACE_OFFSET + memory_dimensions.as_offset;
    let pv_start_idx = memory_dimensions.label_to_index((pv_as, 0));
    let pv_height = log2_strict_usize(num_public_values / DIGEST_SIZE);
    let proof_len = memory_dimensions.overall_height() - pv_height;
    let idx_prefix = pv_start_idx >> pv_height;

    // Read the public values root proof from the input stream.
    let root_proof = UserPublicValuesRootProof::<F>::read(builder);
    builder.assert_eq::<Usize<_>>(root_proof.sibling_hashes.len(), Usize::from(proof_len));
    let mut curr_commit = root_proof.public_values_commit;
    // Share the same state array to avoid unnecessary allocations.
    let compressor = VariableP2Compressor::new(builder);
    for i in 0..proof_len {
        let sibling_hash = builder.get(&root_proof.sibling_hashes, i);
        let (l_hash, r_hash) = if idx_prefix & (1 << i) != 0 {
            (sibling_hash, curr_commit)
        } else {
            (curr_commit, sibling_hash)
        };
        curr_commit = compressor.compress(builder, &l_hash, &r_hash);
    }
    (root_proof.public_values_commit, curr_commit)
}
