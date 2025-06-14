use std::array;

use openvm_circuit::arch::instructions::program::Program;
use openvm_native_compiler::{asm::HEAP_START_ADDRESS, conversion::CompilerOptions, prelude::*};
use openvm_native_recursion::{
    fri::TwoAdicFriPcsVariable, hints::Hintable, types::new_from_inner_multi_vk,
    utils::const_fri_config,
};
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::{
    config::FriParameters,
    openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, p3_field::FieldAlgebra},
};

use crate::{
    verifier::{
        common::non_leaf::NonLeafVerifierVariables,
        root::{
            types::{RootVmVerifierInput, RootVmVerifierPvs},
            vars::RootVmVerifierInputVariable,
        },
        utils::VariableP2Hasher,
    },
    C, F, SC,
};

pub mod types;
mod vars;

/// Config to generate Root VM verifier program.
pub struct RootVmVerifierConfig {
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub num_user_public_values: usize,
    pub internal_vm_verifier_commit: [F; DIGEST_SIZE],
    pub compiler_options: CompilerOptions,
}
impl RootVmVerifierConfig {
    pub fn build_program(
        &self,
        leaf_vm_vk: &MultiStarkVerifyingKey<SC>,
        internal_vm_vk: &MultiStarkVerifyingKey<SC>,
    ) -> Program<F> {
        let mut builder = Builder::<C>::default();

        builder.cycle_tracker_start("ReadProofsFromInput");
        let root_verifier_input = RootVmVerifierInput::<SC>::read(&mut builder);
        builder.cycle_tracker_end("ReadProofsFromInput");
        let pvs = self.verifier_impl(
            &mut builder,
            leaf_vm_vk,
            internal_vm_vk,
            root_verifier_input,
        );
        pvs.flatten()
            .into_iter()
            .for_each(|v| builder.commit_public_value(v));
        builder.halt();
        builder.compile_isa_with_options(self.compiler_options)
    }

    /// Build instructions which can be called as a kernel function in RISC-V guest programs.
    /// Inputs for generated instructions:
    /// - expected `app_exe_commit`, `app_vm_commit` and user public values should be stored from
    ///   `HEAP_START_ADDRESS` in the native address space .
    ///
    /// These instructions take a proof from the input stream and verify the proof. Then these
    /// instructions check if the public values are consistent with the expected public values
    /// from RISC-V guest programs.
    pub fn build_kernel_asm(
        &self,
        leaf_vm_vk: &MultiStarkVerifyingKey<SC>,
        internal_vm_vk: &MultiStarkVerifyingKey<SC>,
    ) -> Program<F> {
        let mut builder = Builder::<C>::default();

        const BYTE_PER_WORD: usize = 4;
        let num_public_values = self.num_user_public_values + DIGEST_SIZE * 2;
        let num_bytes = num_public_values * BYTE_PER_WORD;
        // Move heap pointer in order to keep input arguments from address space 2.
        let heap_addr: Var<F> = builder.eval(F::from_canonical_u32(
            HEAP_START_ADDRESS as u32 + num_bytes as u32,
        ));
        builder.store_heap_ptr(Ptr { address: heap_addr });
        let expected_pvs: Vec<Felt<_>> = (0..num_public_values)
            .map(|i| {
                let fs: [Felt<_>; BYTE_PER_WORD] = array::from_fn(|j| {
                    let ptr = Ptr {
                        address: builder.eval(F::from_canonical_u32(
                            HEAP_START_ADDRESS as u32 + (i * 4) as u32,
                        )),
                    };
                    let idx = MemIndex {
                        index: RVar::from(j),
                        offset: 0,
                        size: 1,
                    };
                    let f = Felt::uninit(&mut builder);
                    f.load(ptr, idx, &mut builder);
                    f
                });
                builder.eval(
                    fs[0]
                        + fs[1] * F::from_canonical_u32(1 << 8)
                        + fs[2] * F::from_canonical_u32(1 << 16)
                        + fs[3] * F::from_canonical_u32(1 << 24),
                )
            })
            .collect();
        let expected_pvs = RootVmVerifierPvs::<Felt<F>>::from_flatten(expected_pvs);
        let user_pvs = builder.array(self.num_user_public_values);
        for (i, &pv) in expected_pvs.public_values.iter().enumerate() {
            builder.set(&user_pvs, i, pv);
        }

        builder.cycle_tracker_start("ReadFromStdin");
        let proof = Proof::<SC>::read(&mut builder);
        builder.cycle_tracker_end("ReadFromStdin");
        let proofs = builder.array(1);
        builder.set(&proofs, 0, proof);
        let pvs = self.verifier_impl(
            &mut builder,
            leaf_vm_vk,
            internal_vm_vk,
            RootVmVerifierInputVariable {
                proofs,
                public_values: user_pvs,
            },
        );
        builder.assert_eq::<[Felt<_>; DIGEST_SIZE]>(pvs.exe_commit, expected_pvs.exe_commit);
        builder.assert_eq::<[Felt<_>; DIGEST_SIZE]>(
            pvs.leaf_verifier_commit,
            expected_pvs.leaf_verifier_commit,
        );

        builder.compile_isa_with_options(self.compiler_options)
    }

    fn verifier_impl(
        &self,
        builder: &mut Builder<C>,
        leaf_vm_vk: &MultiStarkVerifyingKey<SC>,
        internal_vm_vk: &MultiStarkVerifyingKey<SC>,
        root_verifier_input: RootVmVerifierInputVariable<C>,
    ) -> RootVmVerifierPvs<Felt<F>> {
        let leaf_advice = new_from_inner_multi_vk(leaf_vm_vk);
        let internal_advice = new_from_inner_multi_vk(internal_vm_vk);
        let RootVmVerifierInputVariable {
            proofs,
            public_values,
        } = root_verifier_input;

        builder.cycle_tracker_start("InitializePcsConst");
        let leaf_pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(builder, &self.leaf_fri_params),
        };
        let internal_pcs = TwoAdicFriPcsVariable {
            config: const_fri_config(builder, &self.internal_fri_params),
        };
        builder.cycle_tracker_end("InitializePcsConst");
        builder.cycle_tracker_start("VerifyProofs");
        let internal_program_commit =
            array::from_fn(|i| builder.eval(self.internal_vm_verifier_commit[i]));
        let non_leaf_verifier = NonLeafVerifierVariables {
            internal_program_commit,
            leaf_pcs,
            leaf_advice,
            internal_pcs,
            internal_advice,
        };
        let (merged_pvs, expected_leaf_commit) =
            non_leaf_verifier.verify_internal_or_leaf_verifier_proofs(builder, &proofs);
        builder.cycle_tracker_end("VerifyProofs");

        // App Program should terminate
        builder.assert_felt_eq(merged_pvs.connector.is_terminate, F::ONE);
        // App Program should exit successfully
        builder.assert_felt_eq(merged_pvs.connector.exit_code, F::ZERO);

        builder.cycle_tracker_start("ExtractPublicValues");
        builder.assert_usize_eq(public_values.len(), RVar::from(self.num_user_public_values));
        let public_values_vec: Vec<Felt<F>> = (0..self.num_user_public_values)
            .map(|i| builder.get(&public_values, i))
            .collect();
        let hasher = VariableP2Hasher::new(builder);
        let pv_commit = hasher.merkle_root(builder, &public_values_vec);
        builder.assert_eq::<[_; DIGEST_SIZE]>(merged_pvs.public_values_commit, pv_commit);
        builder.cycle_tracker_end("ExtractPublicValues");

        RootVmVerifierPvs {
            exe_commit: compute_exe_commit(
                builder,
                &hasher,
                merged_pvs.app_commit,
                merged_pvs.memory.initial_root,
                merged_pvs.connector.initial_pc,
            ),
            leaf_verifier_commit: expected_leaf_commit,
            public_values: public_values_vec,
        }
    }
}

fn compute_exe_commit<C: Config>(
    builder: &mut Builder<C>,
    hasher: &VariableP2Hasher<C>,
    app_commit: [Felt<C::F>; DIGEST_SIZE],
    init_memory: [Felt<C::F>; DIGEST_SIZE],
    pc_start: Felt<C::F>,
) -> [Felt<C::F>; DIGEST_SIZE] {
    let app_commit_hash = hasher.hash(builder, &app_commit);
    let init_memory_hash = hasher.hash(builder, &init_memory);
    let const_zero = hasher.const_zero;
    let padded_pc_start = array::from_fn(|i| if i == 0 { pc_start } else { const_zero });
    let pc_start_hash = hasher.hash(builder, &padded_pc_start);
    let compress_1 = hasher
        .compressor
        .compress(builder, &app_commit_hash, &init_memory_hash);
    hasher
        .compressor
        .compress(builder, &compress_1, &pc_start_hash)
}
