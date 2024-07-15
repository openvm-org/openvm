use std::marker::PhantomData;

use afs_chips::{
    common::page::Page,
    inner_join::controller::{FKInnerJoinController, IJBuses, T2Format, TableFormat},
};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{types::MultiStarkPartialVerifyingKey, MultiStarkKeygenBuilder},
    prover::{trace::TraceCommitmentBuilder, types::Proof, MultiTraceStarkProver},
};
use afs_test_utils::{engine::StarkEngine, utils::create_seeded_rng};
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use parking_lot::Mutex;

use crate::{common::Commitment, dataframe::DataFrame};

#[derive(derive_new::new)]
pub struct PageLevelJoin<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    parent_page: Page,
    child_page: Page,

    pub pis: Mutex<PageLevelJoinPis<COMMIT_LEN>>,

    ij_controller: Mutex<FKInnerJoinController<SC>>,

    proof: Mutex<Option<Proof<SC>>>, // TODO: might be a good idea to store this to disk and load it for verification

    _marker2: PhantomData<E>,
}

// TODO: think about the public values for this the page-level circuit
// I think a lot of those public values can be removed?
// Actually, if we ma
#[derive(Clone, derive_new::new)]
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
    pub fn load_pages_from_commits(
        parent_page_commit: Commitment<COMMIT_LEN>,
        child_page_commit: Commitment<COMMIT_LEN>,
    ) -> Self {
        let mut rng = create_seeded_rng();
        let idx_len = 5;
        let data_len = 10;
        let max_idx = 128;
        let max_data = 100;
        let height = 1024;
        let fkey_start = 0;
        let fkey_end = 10;

        let idx_limb_bits = 7;
        let decomp = 3;

        // TODO: replace this with actual page loading
        let parent_page = Page::random(
            &mut rng, idx_len, data_len, max_idx, max_data, height, height,
        );

        Self::new(
            parent_page.clone(),
            parent_page,
            Mutex::new(PageLevelJoinPis::new(
                Commitment::<COMMIT_LEN>::default(), // This should be updated in trace generation
                u32::MAX,                            // This should be updated in trace generation
                parent_page_commit,
                child_page_commit,
                Commitment::<COMMIT_LEN>::default(), // This should be updated in trace generation
                Commitment::<COMMIT_LEN>::default(), // This should be updated in trace generation
            )),
            Mutex::new(FKInnerJoinController::<SC>::new(
                IJBuses::default(),
                TableFormat::new(idx_len, data_len, idx_limb_bits),
                T2Format::new(
                    TableFormat::new(idx_len, data_len, idx_limb_bits),
                    fkey_start,
                    fkey_end,
                ),
                decomp,
            )),
            Mutex::new(None),
        )
    }

    pub fn generate_trace(&self, output_df: &mut DataFrame<COMMIT_LEN>, pairs_list_index: u32) {
        let mut pis = self.pis.lock();

        pis.init_running_df_commit = output_df.commit.clone();
        pis.pairs_list_index = pairs_list_index;

        // TODO: this is bad design -- I'm calling inner_join twice
        // Note: we need to store data from here on disk to parallelize trace generation
        let _output_page = self
            .ij_controller
            .lock()
            .inner_join(&self.parent_page, &self.child_page);

        let output_page_commit = Commitment::<COMMIT_LEN>::default(); // TODO: update this to be the correct commitment
        output_df.push(output_page_commit.clone());

        pis.final_running_df_commit = output_df.commit.clone();
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
        let mut ij_controller = self.ij_controller.lock();

        let mut keygen_builder: MultiStarkKeygenBuilder<SC> =
            MultiStarkKeygenBuilder::new(&engine.config());
        ij_controller.set_up_keygen_builder(&mut keygen_builder);

        let intersector_trace_degree = 2 * self.parent_page.height().max(self.child_page.height());

        let partial_pk = keygen_builder.generate_partial_pk();
        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        let prover_data = ij_controller.load_tables(
            &self.parent_page,
            &self.child_page,
            intersector_trace_degree,
            &mut trace_builder.committer,
        );
        let proof = ij_controller.prove(engine, &partial_pk, &mut trace_builder, prover_data);
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

        let mut keygen_builder: MultiStarkKeygenBuilder<SC> =
            MultiStarkKeygenBuilder::new(&engine.config());

        let ij_controller = self.ij_controller.lock();

        ij_controller.set_up_keygen_builder(&mut keygen_builder);
        let partial_pk = keygen_builder.generate_partial_pk();
        let partial_vk = partial_pk.partial_vk();

        let (parent_page_commit, child_page_commit, output_page_commit) =
            Self::get_page_commits_from_proof(&proof, &partial_vk);

        assert_eq!(pis.parent_page_commit, parent_page_commit);
        assert_eq!(pis.child_page_commit, child_page_commit);

        assert_eq!(output_df.commit, pis.init_running_df_commit);
        output_df.push(output_page_commit.clone());
        assert_eq!(output_df.commit, pis.final_running_df_commit);

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
