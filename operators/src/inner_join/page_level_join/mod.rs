use std::{any::Any, marker::PhantomData};

use afs_chips::{
    common::page::Page,
    inner_join::controller::{FKInnerJoinController, IJBuses, T2Format, TableFormat},
};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::MultiStarkKeygenBuilder,
    prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver},
    verifier::VerificationError,
};
use afs_test_utils::{engine::StarkEngine, utils::create_seeded_rng};
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    common::{Commitment, Verifiable},
    dataframe::DataFrame,
};

#[derive(derive_new::new)]
pub struct PageLevelJoin<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    parent_page: Page,
    child_page: Page,

    output_df: DataFrame<COMMIT_LEN>,

    pub pis: PageLevelJoinPis<COMMIT_LEN>,

    ij_controller: FKInnerJoinController<SC>,

    _marker1: PhantomData<SC>, // This should be removed eventually
    _marker2: PhantomData<E>,
}

// TODO: think about the public values for this the page-level circuit
// I think a lot of those public values can be removed?
// Actually, if we ma
#[derive(Clone, derive_new::new)]
pub struct PageLevelJoinPis<const COMMIT_LEN: usize> {
    pub init_running_df_commit: Commitment<COMMIT_LEN>,
    pub df_cur_page: u32,
    pub pairs_list_index: u32,
    pub parent_page_commit: Commitment<COMMIT_LEN>,
    pub child_page_commit: Commitment<COMMIT_LEN>,
    // TODO: this hasn't been used yet. should be used to verify the page added to the dataframe is correct
    pub output_page_commit: Commitment<COMMIT_LEN>,
    pub pairs_commit: Commitment<COMMIT_LEN>,
    pub final_running_df_commit: Commitment<COMMIT_LEN>,
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    PageLevelJoin<COMMIT_LEN, SC, E>
{
    pub fn load_pages_from_commits(
        parent_page_commit: Commitment<COMMIT_LEN>,
        child_page_commit: Commitment<COMMIT_LEN>,
        output_df: &DataFrame<COMMIT_LEN>,
        df_cur_page: u32,
        pairs_commit: Commitment<COMMIT_LEN>,
        pairs_list_index: u32,
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
            output_df.clone(), // TODO: add note here why cloning actually makes sense
            PageLevelJoinPis::new(
                output_df.commit.clone(),
                df_cur_page,
                pairs_list_index,
                parent_page_commit,
                child_page_commit,
                Commitment::<COMMIT_LEN>::default(), // This should be updated in trace generation
                pairs_commit,
                Commitment::<COMMIT_LEN>::default(), // This should be updated in trace generation
            ),
            FKInnerJoinController::<SC>::new(
                IJBuses::default(),
                TableFormat::new(idx_len, data_len, idx_limb_bits),
                T2Format::new(
                    TableFormat::new(idx_len, data_len, idx_limb_bits),
                    fkey_start,
                    fkey_end,
                ),
                decomp,
            ),
        )
    }

    pub fn generate_trace(&mut self, output_df: &mut DataFrame<COMMIT_LEN>, df_cur_page: u32) {
        // Note: we need to store data from here on disk to parallelize trace generation
        let _output_page = self
            .ij_controller
            .inner_join(&self.parent_page, &self.child_page);

        let output_page_commit = Commitment::<COMMIT_LEN>::default(); // TODO: update this to be the correct commitment
        output_df.edit_page_commit(df_cur_page as usize, output_page_commit.clone());

        self.pis.output_page_commit = output_page_commit;
        self.pis.final_running_df_commit = output_df.commit.clone();
    }
}

impl<const COMMIT_LEN: usize, SC: StarkGenericConfig + 'static, E: StarkEngine<SC> + 'static>
    Verifiable<SC, E> for PageLevelJoin<COMMIT_LEN, SC, E>
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
    fn verify(&mut self, engine: &E) -> Result<(), VerificationError> {
        // TODO: implement the following logic
        // assert_eq!(self.parent_page.commit(), self.pis.parent_page_commit);
        // assert_eq!(self.child_page.commit(), self.pis.child_page_commit);

        assert_eq!(self.output_df.commit, self.pis.init_running_df_commit);
        self.output_df.edit_page_commit(
            self.pis.df_cur_page as usize,
            self.pis.output_page_commit.clone(),
        );
        assert_eq!(self.output_df.commit, self.pis.final_running_df_commit);

        let intersector_trace_degree = 2 * self.parent_page.height().max(self.child_page.height());

        let mut keygen_builder: MultiStarkKeygenBuilder<SC> =
            MultiStarkKeygenBuilder::new(&engine.config());

        self.ij_controller.set_up_keygen_builder(
            &mut keygen_builder,
            self.parent_page.height(),
            self.child_page.height(),
            intersector_trace_degree,
        );

        let partial_pk = keygen_builder.generate_partial_pk();

        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let prover_data = self.ij_controller.load_tables(
            &self.parent_page,
            &self.child_page,
            intersector_trace_degree,
            &mut trace_builder.committer,
        );

        let proof = self
            .ij_controller
            .prove(engine, &partial_pk, &mut trace_builder, prover_data);
        self.ij_controller
            .verify(engine, partial_pk.partial_vk(), proof)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
