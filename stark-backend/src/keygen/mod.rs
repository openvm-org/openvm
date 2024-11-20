use std::sync::Arc;

use itertools::Itertools;
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use tracing::instrument;

use crate::{
    air_builders::symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    config::{RapPhaseSeqProvingKey, StarkGenericConfig, Val},
    interaction::{HasInteractionChunkSize, RapPhaseSeq, RapPhaseSeqKind},
    keygen::types::{
        MultiStarkProvingKey, ProverOnlySinglePreprocessedData, StarkProvingKey, StarkVerifyingKey,
        TraceWidth, VerifierSinglePreprocessedData,
    },
    prover::types::TraceCommitter,
    rap::AnyRap,
};

pub mod types;
pub(crate) mod view;

struct AirKeygenBuilder<SC: StarkGenericConfig> {
    air: Arc<dyn AnyRap<SC>>,
    rap_phase_seq_kind: RapPhaseSeqKind,
    prep_keygen_data: PrepKeygenData<SC>,
}

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    /// Information for partitioned AIRs.
    partitioned_airs: Vec<AirKeygenBuilder<SC>>,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            partitioned_airs: vec![],
        }
    }

    /// Default way to add a single Interactive AIR.
    /// Returns `air_id`
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: Arc<dyn AnyRap<SC>>) -> usize {
        self.partitioned_airs.push(AirKeygenBuilder::new(
            self.config.pcs(),
            SC::RapPhaseSeq::ID,
            air,
        ));
        self.partitioned_airs.len() - 1
    }

    /// Consume the builder and generate proving key.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(self) -> MultiStarkProvingKey<SC> {
        let global_max_constraint_degree = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| {
                let max_constraint_degree = keygen_builder.max_constraint_degree();
                tracing::debug!(
                    "{} has constraint degree {}",
                    keygen_builder.air.name(),
                    max_constraint_degree
                );
                max_constraint_degree
            })
            .max()
            .unwrap();
        tracing::info!(
            "Max constraint (excluding logup constraints) degree across all AIRs: {}",
            global_max_constraint_degree
        );

        let symbolic_constraints_per_air = self
            .partitioned_airs
            .iter()
            .map(|keygen_builder| keygen_builder.get_symbolic_builder(None).constraints())
            .collect();
        let rap_phase_seq_pk_per_air = self
            .config
            .rap_phase_seq()
            .generate_pk_per_air(symbolic_constraints_per_air);

        let pk_per_air: Vec<_> = self
            .partitioned_airs
            .into_iter()
            .zip_eq(rap_phase_seq_pk_per_air)
            .map(|(keygen_builder, params)| keygen_builder.generate_pk(params))
            .collect();

        for pk in pk_per_air.iter() {
            let width = &pk.vk.params.width;
            tracing::info!("{:<20} | Quotient Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | Perm Cols = {:<4} | {:4} Constraints | {:3} Interactions On Buses {:?}",
                pk.air_name,
                pk.vk.quotient_degree,
                width.preprocessed.unwrap_or(0),
                format!("{:?}",width.main_widths()),
                format!("{:?}",width.after_challenge.iter().map(|&x| x * <SC::Challenge as AbstractExtensionField<Val<SC>>>::D).collect_vec()),
                pk.vk.symbolic_constraints.constraints.len(),
                pk.vk.symbolic_constraints.interactions.len(),
                pk.vk
                    .symbolic_constraints
                    .interactions
                    .iter()
                    .map(|i| i.bus_index)
                    .collect_vec()
            );
            #[cfg(feature = "bench-metrics")]
            {
                let labels = [("air_name", pk.air_name.clone())];
                metrics::counter!("quotient_deg", &labels).absolute(pk.vk.quotient_degree as u64);
                // column info will be logged by prover later
                metrics::counter!("constraints", &labels)
                    .absolute(pk.vk.symbolic_constraints.constraints.len() as u64);
                metrics::counter!("interactions", &labels)
                    .absolute(pk.vk.symbolic_constraints.interactions.len() as u64);
            }
        }

        MultiStarkProvingKey {
            per_air: pk_per_air,
            max_constraint_degree: global_max_constraint_degree,
        }
    }
}

impl<SC: StarkGenericConfig> AirKeygenBuilder<SC> {
    fn new(pcs: &SC::Pcs, rap_phase_seq_kind: RapPhaseSeqKind, air: Arc<dyn AnyRap<SC>>) -> Self {
        let prep_keygen_data = compute_prep_data_for_air(pcs, air.as_ref());
        AirKeygenBuilder {
            air,
            rap_phase_seq_kind,
            prep_keygen_data,
        }
    }

    fn max_constraint_degree(&self) -> usize {
        self.get_symbolic_builder(None)
            .constraints()
            .max_constraint_degree()
    }

    fn generate_pk(self, rap_phase_seq_pk: RapPhaseSeqProvingKey<SC>) -> StarkProvingKey<SC> {
        let air_name = self.air.name();

        let interaction_chunk_size = rap_phase_seq_pk.interaction_chunk_size();
        let symbolic_builder = self.get_symbolic_builder(Some(interaction_chunk_size));
        let params = symbolic_builder.params();
        let symbolic_constraints = symbolic_builder.constraints();
        let log_quotient_degree = symbolic_constraints.get_log_quotient_degree();
        let quotient_degree = 1 << log_quotient_degree;

        let Self {
            prep_keygen_data:
                PrepKeygenData {
                    verifier_data: prep_verifier_data,
                    prover_data: prep_prover_data,
                },
            ..
        } = self;

        let vk = StarkVerifyingKey {
            preprocessed_data: prep_verifier_data,
            params,
            symbolic_constraints,
            quotient_degree,
            rap_phase_seq_kind: self.rap_phase_seq_kind,
        };
        StarkProvingKey {
            air_name,
            vk,
            preprocessed_data: prep_prover_data,
            rap_phase_seq_pk,
        }
    }

    fn get_symbolic_builder(
        &self,
        interaction_chunk_size: Option<usize>,
    ) -> SymbolicRapBuilder<Val<SC>> {
        let width = TraceWidth {
            preprocessed: self.prep_keygen_data.width(),
            cached_mains: self.air.cached_main_widths(),
            common_main: self.air.common_main_width(),
            after_challenge: vec![],
        };
        get_symbolic_builder(
            self.air.as_ref(),
            &width,
            &[],
            &[],
            SC::RapPhaseSeq::ID,
            interaction_chunk_size.unwrap_or(1),
        )
    }
}

pub(super) struct PrepKeygenData<SC: StarkGenericConfig> {
    pub verifier_data: Option<VerifierSinglePreprocessedData<SC>>,
    pub prover_data: Option<ProverOnlySinglePreprocessedData<SC>>,
}

impl<SC: StarkGenericConfig> PrepKeygenData<SC> {
    pub fn width(&self) -> Option<usize> {
        self.prover_data.as_ref().map(|d| d.trace.width())
    }
}

fn compute_prep_data_for_air<SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    air: &dyn AnyRap<SC>,
) -> PrepKeygenData<SC> {
    let preprocessed_trace = air.preprocessed_trace();
    let vpdata_opt = preprocessed_trace.map(|trace| {
        let trace_committer = TraceCommitter::<SC>::new(pcs);
        let data = trace_committer.commit(vec![trace.clone()]);
        let vdata = VerifierSinglePreprocessedData {
            commit: data.commit,
        };
        let pdata = ProverOnlySinglePreprocessedData {
            trace,
            data: data.data,
        };
        (vdata, pdata)
    });
    if let Some((vdata, pdata)) = vpdata_opt {
        PrepKeygenData {
            prover_data: Some(pdata),
            verifier_data: Some(vdata),
        }
    } else {
        PrepKeygenData {
            prover_data: None,
            verifier_data: None,
        }
    }
}
