use std::sync::Arc;

use itertools::Itertools;
use p3_field::AbstractExtensionField;
use p3_matrix::Matrix;
use tracing::instrument;

use crate::{
    air_builders::symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    config::{StarkGenericConfig, Val},
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
    prep_keygen_data: PrepKeygenData<SC>,
    interaction_chunk_size: Option<usize>,
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
        self.add_air_with_interaction_chunk_size(air, None)
    }

    /// Add a single Interactive AIR with a specified interaction chunk size.
    /// Returns `air_id`
    pub fn add_air_with_interaction_chunk_size(
        &mut self,
        air: Arc<dyn AnyRap<SC>>,
        interaction_chunk_size: Option<usize>,
    ) -> usize {
        self.partitioned_airs.push(AirKeygenBuilder::new(
            self.config.pcs(),
            air,
            interaction_chunk_size,
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

        let pk_per_air: Vec<_> = self
            .partitioned_airs
            .into_iter()
            .map(|keygen_builder| keygen_builder.generate_pk(global_max_constraint_degree))
            .collect();

        for pk in pk_per_air.iter() {
            let width = &pk.vk.params.width;
            tracing::info!("{:<20} | Quotient Deg = {:<2} | Prep Cols = {:<2} | Main Cols = {:<8} | Perm Cols = {:<4} | {:<4} Constraints | {:<3} Interactions On Buses {:?}",
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
    fn new(pcs: &SC::Pcs, air: Arc<dyn AnyRap<SC>>, interaction_chunk_size: Option<usize>) -> Self {
        let prep_keygen_data = compute_prep_data_for_air(pcs, air.as_ref());
        AirKeygenBuilder {
            air,
            prep_keygen_data,
            interaction_chunk_size,
        }
    }

    fn max_constraint_degree(&self) -> usize {
        self.get_symbolic_builder()
            .constraints()
            .max_constraint_degree()
    }

    fn generate_pk(mut self, max_constraint_degree: usize) -> StarkProvingKey<SC> {
        let air_name = self.air.name();
        self.find_interaction_chunk_size(max_constraint_degree);

        let symbolic_builder = self.get_symbolic_builder();
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
            interaction_chunk_size,
            ..
        } = self;
        let interaction_chunk_size = interaction_chunk_size
            .expect("Interaction chunk size should be set before generating proving key");

        let vk = StarkVerifyingKey {
            preprocessed_data: prep_verifier_data,
            params,
            symbolic_constraints,
            quotient_degree,
        };
        StarkProvingKey {
            air_name,
            vk,
            preprocessed_data: prep_prover_data,
            interaction_chunk_size,
        }
    }

    /// Finds the interaction chunk size for the AIR if it is not provided.
    /// `global_max_constraint_degree` is the maximum constraint degree across all AIRs.
    /// The degree of the dominating logup constraint is bounded by
    /// logup_degree = max(1 + max_field_degree * interaction_chunk_size,
    /// max_count_degree + max_field_degree * (interaction_chunk_size - 1))
    /// More details about this can be found in the function eval_permutation_constraints
    ///
    /// The goal is to pick interaction_chunk_size so that logup_degree does not
    /// exceed max_constraint_degree (if possible), while maximizing interaction_chunk_size
    fn find_interaction_chunk_size(&mut self, global_max_constraint_degree: usize) {
        if self.interaction_chunk_size.is_some() {
            return;
        }

        let (max_field_degree, max_count_degree) = self
            .get_symbolic_builder()
            .constraints()
            .max_interaction_degrees();

        let interaction_chunk_size = if max_field_degree == 0 {
            1
        } else {
            let mut interaction_chunk_size = (global_max_constraint_degree - 1) / max_field_degree;
            interaction_chunk_size = interaction_chunk_size.min(
                (global_max_constraint_degree - max_count_degree + max_field_degree)
                    / max_field_degree,
            );
            interaction_chunk_size = interaction_chunk_size.max(1);
            interaction_chunk_size
        };

        self.interaction_chunk_size = Some(interaction_chunk_size);
    }

    fn get_symbolic_builder(&self) -> SymbolicRapBuilder<Val<SC>> {
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
            self.interaction_chunk_size.unwrap_or(1),
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
