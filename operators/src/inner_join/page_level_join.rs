use std::{marker::PhantomData, sync::Arc};

use afs_chips::inner_join::controller::{FKInnerJoinController, IJBuses, T2Format, TableFormat};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkPartialProvingKey, MultiStarkPartialVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder},
        types::Proof,
        MultiTraceStarkProver,
    },
};
use afs_test_utils::engine::StarkEngine;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use parking_lot::Mutex;

use crate::{common::Commitment, dataframe::DataFrame, page_db::PageDb};

// TODO: this doesn't look good. Just write the whole struct around a mutex instead
// of wrapping each field in a mutex

pub struct PageLevelJoin<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub pis: Mutex<PageLevelJoinPis<COMMIT_LEN>>,
    ij_controller: Mutex<FKInnerJoinController<SC>>,
    page_prover_data: Mutex<Option<Vec<ProverTraceData<SC>>>>,
    proof: Mutex<Option<Proof<SC>>>,
    partial_pk: Mutex<MultiStarkPartialProvingKey<SC>>,

    _marker2: PhantomData<E>,
}

// TODO: think about the public values for this the page-level circuit
// I think a lot of those public values can be removed?
// Actually, if we ma
#[derive(Clone, derive_new::new, Default)]
pub struct PageLevelJoinPis<const COMMIT_LEN: usize> {
    pub init_running_df_commit: Commitment<COMMIT_LEN>,
    pub pairs_list_index: u32,
    pub parent_page_commit: Commitment<COMMIT_LEN>,
    pub child_page_commit: Commitment<COMMIT_LEN>,
    pub pairs_commit: Commitment<COMMIT_LEN>,
    pub final_running_df_commit: Commitment<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    PageLevelJoin<COMMIT_LEN, SC, E>
{
    pub fn new(
        parent_page_commit: Commitment<COMMIT_LEN>,
        child_page_commit: Commitment<COMMIT_LEN>,
        t1_format: TableFormat,
        t2_format: T2Format,
        decomp: usize,
    ) -> Self {
        Self {
            // Note that all public values except the input page commitments
            // are updated in generate_trace. Placeholders are used here.
            pis: Mutex::new(PageLevelJoinPis::new(
                Commitment::<COMMIT_LEN>::default(),
                0,
                parent_page_commit,
                child_page_commit,
                Commitment::<COMMIT_LEN>::default(),
                Commitment::<COMMIT_LEN>::default(),
            )),
            ij_controller: Mutex::new(FKInnerJoinController::<SC>::new(
                IJBuses::default(),
                t1_format,
                t2_format,
                decomp,
            )),
            proof: Mutex::new(None),
            page_prover_data: Mutex::new(None),
            partial_pk: Mutex::new(MultiStarkPartialProvingKey::<SC>::default()),
            _marker2: PhantomData,
        }
    }

    /// Note that, currently, this function does not only trace generation
    /// This function does part of the proof (specifically, caching input and output pages)
    /// TODO: maybe provide functionality to allow it to be purely trace generation
    pub fn generate_trace(
        &self,
        page_db: Arc<PageDb<COMMIT_LEN>>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        pairs_list_index: &mut u32,
        engine: &E,
    ) where
        Val<SC>: PrimeField,
    {
        let mut pis = self.pis.lock();

        let parent_page = page_db.get_page(&pis.parent_page_commit).unwrap();
        let child_page = page_db.get_page(&pis.child_page_commit).unwrap();

        pis.init_running_df_commit = output_df.commit.clone();
        pis.pairs_list_index = *pairs_list_index;

        let _output_page = self
            .ij_controller
            .lock()
            .inner_join(&parent_page, &child_page);

        let output_page_commit = Commitment::<COMMIT_LEN>::default(); // TODO: update this to be the correct commitment
        output_df.push(output_page_commit.clone());
        *pairs_list_index += 1;

        pis.final_running_df_commit = output_df.commit.clone();

        let mut ij_controller = self.ij_controller.lock();

        let intersector_trace_degree = 2 * parent_page.height().max(child_page.height());

        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        let prover_data = ij_controller.load_tables(
            &parent_page,
            &child_page,
            intersector_trace_degree,
            &mut trace_builder.committer,
        );
        *self.page_prover_data.lock() = Some(prover_data);
    }

    pub fn set_up_keygen_builder(&self, engine: &E)
    where
        Val<SC>: PrimeField,
    {
        let mut keygen_builder: MultiStarkKeygenBuilder<SC> =
            MultiStarkKeygenBuilder::new(&engine.config());
        let ij_controller = self.ij_controller.lock();
        ij_controller.set_up_keygen_builder(&mut keygen_builder);
        *self.partial_pk.lock() = keygen_builder.generate_partial_pk();
    }

    pub fn prove(&self, engine: &E)
    where
        Val<SC>: PrimeField,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let ij_controller = self.ij_controller.lock();
        let prover_data = self.page_prover_data.lock().take().unwrap();

        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let proof = ij_controller.prove(
            engine,
            &self.partial_pk.lock(),
            &mut trace_builder,
            prover_data,
        );
        *self.proof.lock() = Some(proof);
    }

    pub fn verify(&self, engine: &E, output_df: &mut DataFrame<COMMIT_LEN>)
    where
        Val<SC>: PrimeField,
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        // Note that this consumes the proof stored in self.proof
        // In the future we might want to load the proof directly from disk anyway
        let proof = self.proof.lock().take().unwrap();

        let pis = self.pis.lock();

        let partial_pk = self.partial_pk.lock();
        let partial_vk = partial_pk.partial_vk();

        let (parent_page_commit, child_page_commit, output_page_commit) =
            Self::get_page_commits_from_proof(&proof, &partial_vk);

        assert_eq!(pis.parent_page_commit, parent_page_commit);
        assert_eq!(pis.child_page_commit, child_page_commit);

        assert_eq!(output_df.commit, pis.init_running_df_commit);
        output_df.push(output_page_commit.clone());
        assert_eq!(output_df.commit, pis.final_running_df_commit);

        let ij_controller = self.ij_controller.lock();
        ij_controller
            .verify(engine, partial_pk.partial_vk(), proof)
            .expect("proof failed to verify");
    }

    fn get_page_commits_from_proof(
        proof: &Proof<SC>,
        partial_vk: &MultiStarkPartialVerifyingKey<SC>,
    ) -> (
        Commitment<COMMIT_LEN>,
        Commitment<COMMIT_LEN>,
        Commitment<COMMIT_LEN>,
    ) {
        let _parent_page_commit = proof.commitments.main_trace[partial_vk.per_air[0]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone();

        let _child_page_commit = proof.commitments.main_trace[partial_vk.per_air[1]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone();

        let _output_page_commit = proof.commitments.main_trace[partial_vk.per_air[2]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone();

        // TODO: figure this out
        let placeholder = Commitment::<COMMIT_LEN>::default();

        (
            placeholder.clone(),
            placeholder.clone(),
            placeholder.clone(),
        )
    }
}
