use std::sync::Arc;

use openvm_circuit::arch::{
    instructions::exe::VmExe, ContinuationVmProof, PreflightExecutor, SingleSegmentVmProver,
    VirtualMachineError, VmBuilder, VmExecutionConfig, VmInstance,
};
#[cfg(feature = "evm-prove")]
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
use openvm_continuations::verifier::{
    internal::types::{InternalVmVerifierInput, VmStarkProof},
    leaf::types::LeafVmVerifierInput,
};
use openvm_native_circuit::{NativeConfig, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::{engine::StarkFriEngine, openvm_stark_backend::proof::Proof};
use tracing::{info_span, instrument};

use crate::{
    config::AggregationTreeConfig, keygen::AggProvingKey, prover::vm::new_local_prover,
    util::check_max_constraint_degrees, F, SC,
};
#[cfg(feature = "evm-prove")]
use crate::{prover::RootVerifierLocalProver, RootSC};

pub struct AggStarkProver<E, NativeBuilder>
where
    E: StarkFriEngine<SC = SC>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
{
    leaf_prover: VmInstance<E, NativeBuilder>,
    leaf_controller: LeafProvingController,

    pub internal_prover: VmInstance<E, NativeBuilder>,
    #[cfg(feature = "evm-prove")]
    root_prover: RootVerifierLocalProver,
    pub num_children_internal: usize,
    pub max_internal_wrapper_layers: usize,
}

pub struct LeafProvingController {
    /// Each leaf proof aggregations `<= num_children` App VM proofs
    pub num_children: usize,
}

impl<E, NativeBuilder> AggStarkProver<E, NativeBuilder>
where
    E: StarkFriEngine<SC = SC>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone,
    <NativeConfig as VmExecutionConfig<F>>::Executor:
        PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
{
    pub fn new(
        native_builder: NativeBuilder,
        agg_pk: &AggProvingKey,
        leaf_verifier_exe: Arc<VmExe<F>>,
        tree_config: AggregationTreeConfig,
    ) -> Result<Self, VirtualMachineError> {
        let leaf_prover = new_local_prover(
            native_builder.clone(),
            &agg_pk.leaf_vm_pk,
            leaf_verifier_exe,
        )?;
        let internal_prover = new_local_prover(
            native_builder,
            &agg_pk.internal_vm_pk,
            agg_pk.internal_committed_exe.exe.clone(),
        )?;
        #[cfg(feature = "evm-prove")]
        let root_prover = RootVerifierLocalProver::new(&agg_pk.root_verifier_pk)?;
        Ok(Self::new_from_instances(
            leaf_prover,
            internal_prover,
            #[cfg(feature = "evm-prove")]
            root_prover,
            tree_config,
        ))
    }

    pub fn new_from_instances(
        leaf_instance: VmInstance<E, NativeBuilder>,
        internal_instance: VmInstance<E, NativeBuilder>,
        #[cfg(feature = "evm-prove")] root_instance: RootVerifierLocalProver,
        tree_config: AggregationTreeConfig,
    ) -> Self {
        let leaf_controller = LeafProvingController {
            num_children: tree_config.num_children_leaf,
        };
        Self {
            leaf_prover: leaf_instance,
            leaf_controller,
            internal_prover: internal_instance,
            #[cfg(feature = "evm-prove")]
            root_prover: root_instance,
            num_children_internal: tree_config.num_children_internal,
            max_internal_wrapper_layers: tree_config.max_internal_wrapper_layers,
        }
    }

    pub fn with_num_children_leaf(mut self, num_children_leaf: usize) -> Self {
        self.leaf_controller.num_children = num_children_leaf;
        self
    }

    pub fn with_num_children_internal(mut self, num_children_internal: usize) -> Self {
        self.num_children_internal = num_children_internal;
        self
    }

    pub fn with_max_internal_wrapper_layers(mut self, max_internal_wrapper_layers: usize) -> Self {
        self.max_internal_wrapper_layers = max_internal_wrapper_layers;
        self
    }

    /// Generate the root proof for outer recursion.
    #[cfg(feature = "evm-prove")]
    pub fn generate_root_proof(
        &mut self,
        app_proofs: ContinuationVmProof<SC>,
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        let root_verifier_input = self.generate_root_verifier_input(app_proofs)?;
        self.generate_root_proof_impl(root_verifier_input)
    }

    pub fn generate_leaf_proofs(
        &mut self,
        app_proofs: &ContinuationVmProof<SC>,
    ) -> Result<Vec<Proof<SC>>, VirtualMachineError> {
        check_max_constraint_degrees(
            self.leaf_prover.vm.config().as_ref(),
            &self.leaf_prover.vm.engine.fri_params(),
        );
        self.leaf_controller
            .generate_proof(&mut self.leaf_prover, app_proofs)
    }

    /// This is typically only used for the halo2 verifier.
    #[cfg(feature = "evm-prove")]
    pub fn generate_root_verifier_input(
        &mut self,
        app_proofs: ContinuationVmProof<SC>,
    ) -> Result<RootVmVerifierInput<SC>, VirtualMachineError> {
        let leaf_proofs = self.generate_leaf_proofs(&app_proofs)?;
        let public_values = app_proofs.user_public_values.public_values;
        let e2e_stark_proof = self.aggregate_leaf_proofs(leaf_proofs, public_values)?;
        let wrapped_stark_proof = self.wrap_e2e_stark_proof(e2e_stark_proof)?;
        Ok(wrapped_stark_proof)
    }

    pub fn aggregate_leaf_proofs(
        &mut self,
        leaf_proofs: Vec<Proof<SC>>,
        public_values: Vec<F>,
    ) -> Result<VmStarkProof<SC>, VirtualMachineError> {
        check_max_constraint_degrees(
            self.internal_prover.vm.config().as_ref(),
            &self.internal_prover.vm.engine.fri_params(),
        );

        let mut internal_node_idx = -1;
        let mut internal_node_height = 0;
        let mut proofs = leaf_proofs;
        // We will always generate at least one internal proof, even if there is only one leaf
        // proof, in order to shrink the proof size
        while proofs.len() > 1 || internal_node_height == 0 {
            let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
                (*self.internal_prover.program_commitment()).into(),
                &proofs,
                self.num_children_internal,
            );
            proofs = info_span!(
                "agg_layer",
                group = format!("internal.{internal_node_height}")
            )
            .in_scope(|| {
                #[cfg(feature = "metrics")]
                {
                    metrics::counter!("fri.log_blowup")
                        .absolute(self.internal_prover.vm.engine.fri_params().log_blowup as u64);
                    metrics::counter!("num_children").absolute(self.num_children_internal as u64);
                }
                internal_inputs
                    .into_iter()
                    .map(|input| {
                        internal_node_idx += 1;
                        info_span!("single_internal_agg", idx = internal_node_idx,).in_scope(|| {
                            SingleSegmentVmProver::prove(
                                &mut self.internal_prover,
                                input.write(),
                                NATIVE_MAX_TRACE_HEIGHTS,
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?;
            internal_node_height += 1;
        }
        let proof = proofs.pop().unwrap();
        Ok(VmStarkProof {
            inner: proof,
            user_public_values: public_values,
        })
    }

    /// Wrap the e2e stark proof until its heights meet the requirements of the root verifier.
    #[cfg(feature = "evm-prove")]
    fn wrap_e2e_stark_proof(
        &mut self,
        e2e_stark_proof: VmStarkProof<SC>,
    ) -> Result<RootVmVerifierInput<SC>, VirtualMachineError> {
        let internal_commit = (*self.internal_prover.program_commitment()).into();
        let internal_prover = &mut self.internal_prover;
        let root_prover = &mut self.root_prover;
        let max_internal_wrapper_layers = self.max_internal_wrapper_layers;
        fn heights_le(a: &[u32], b: &[u32]) -> bool {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).all(|(a, b)| a <= b)
        }

        let VmStarkProof {
            inner: mut proof,
            user_public_values,
        } = e2e_stark_proof;
        let mut wrapper_layers = 0;
        loop {
            let input = RootVmVerifierInput {
                proofs: vec![proof.clone()],
                public_values: user_public_values.clone(),
            };
            let actual_air_heights = root_prover.execute_for_air_heights(input)?;
            // Root verifier can handle the internal proof. We can stop here.
            if heights_le(&actual_air_heights, root_prover.fixed_air_heights()) {
                break;
            }
            if wrapper_layers >= max_internal_wrapper_layers {
                panic!("The heights of the root verifier still exceed the required heights after {} wrapper layers", max_internal_wrapper_layers);
            }
            wrapper_layers += 1;
            let input = InternalVmVerifierInput {
                self_program_commit: internal_commit,
                proofs: vec![proof.clone()],
            };
            proof = info_span!(
                "wrapper_layer",
                group = format!("internal_wrapper.{wrapper_layers}")
            )
            .in_scope(|| {
                #[cfg(feature = "metrics")]
                {
                    metrics::counter!("fri.log_blowup")
                        .absolute(internal_prover.vm.engine.fri_params().log_blowup as u64);
                }
                SingleSegmentVmProver::prove(
                    internal_prover,
                    input.write(),
                    NATIVE_MAX_TRACE_HEIGHTS,
                )
            })?;
        }
        Ok(RootVmVerifierInput {
            proofs: vec![proof],
            public_values: user_public_values,
        })
    }

    #[cfg(feature = "evm-prove")]
    #[instrument(name = "agg_layer", skip_all, fields(group = "root", idx = 0))]
    fn generate_root_proof_impl(
        &mut self,
        root_input: RootVmVerifierInput<SC>,
    ) -> Result<Proof<RootSC>, VirtualMachineError> {
        check_max_constraint_degrees(
            self.root_prover.vm_config().as_ref(),
            self.root_prover.fri_params(),
        );
        let input = root_input.write();
        #[cfg(feature = "metrics")]
        metrics::counter!("fri.log_blowup")
            .absolute(self.root_prover.fri_params().log_blowup as u64);
        SingleSegmentVmProver::prove(&mut self.root_prover, input, NATIVE_MAX_TRACE_HEIGHTS)
    }
}

impl LeafProvingController {
    pub fn with_num_children(mut self, num_children_leaf: usize) -> Self {
        self.num_children = num_children_leaf;
        self
    }

    #[instrument(name = "agg_layer", skip_all, fields(group = "leaf"))]
    pub fn generate_proof<E, NativeBuilder>(
        &self,
        prover: &mut VmInstance<E, NativeBuilder>,
        app_proofs: &ContinuationVmProof<SC>,
    ) -> Result<Vec<Proof<SC>>, VirtualMachineError>
    where
        E: StarkFriEngine<SC = SC>,
        NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
        <NativeConfig as VmExecutionConfig<F>>::Executor:
            PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
    {
        #[cfg(feature = "metrics")]
        {
            metrics::counter!("fri.log_blowup")
                .absolute(prover.vm.engine.fri_params().log_blowup as u64);
            metrics::counter!("num_children").absolute(self.num_children as u64);
        }
        let leaf_inputs =
            LeafVmVerifierInput::chunk_continuation_vm_proof(app_proofs, self.num_children);
        tracing::info!("num_leaf_proofs={}", leaf_inputs.len());
        leaf_inputs
            .into_iter()
            .enumerate()
            .map(|(leaf_node_idx, input)| {
                info_span!("single_leaf_agg", idx = leaf_node_idx).in_scope(|| {
                    SingleSegmentVmProver::prove(
                        prover,
                        input.write_to_stream(),
                        NATIVE_MAX_TRACE_HEIGHTS,
                    )
                })
            })
            .collect()
    }
}
