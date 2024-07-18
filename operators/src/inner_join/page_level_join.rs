use std::{cell::RefCell, marker::PhantomData};

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
use p3_baby_bear::BabyBear;
use p3_field::PrimeField;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    common::{
        hash_struct,
        provider::{DataProvider, PageDataLoader},
        Commitment,
    },
    dataframe::DataFrame,
};

pub struct PageLevelJoin<const COMMIT_LEN: usize, SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub pis: RefCell<PageLevelJoinPis<COMMIT_LEN>>,
    ij_controller: RefCell<FKInnerJoinController<SC>>,
    page_prover_data: RefCell<Option<Vec<ProverTraceData<SC>>>>,
    proof: RefCell<Option<Proof<SC>>>,
    partial_pk: RefCell<MultiStarkPartialProvingKey<SC>>,

    _marker2: PhantomData<E>,
}

#[derive(Clone, derive_new::new, Default)]
pub struct PageLevelJoinPis<const COMMIT_LEN: usize> {
    pub init_running_df_commit: Commitment<COMMIT_LEN>,
    pub pairs_list_index: u32,
    pub parent_page_commit: Commitment<COMMIT_LEN>,
    pub child_page_commit: Commitment<COMMIT_LEN>,
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
            pis: RefCell::new(PageLevelJoinPis::new(
                Commitment::<COMMIT_LEN>::default(),
                0,
                parent_page_commit,
                child_page_commit,
                // Commitment::<COMMIT_LEN>::default(),
                Commitment::<COMMIT_LEN>::default(),
            )),
            ij_controller: RefCell::new(FKInnerJoinController::<SC>::new(
                IJBuses::default(),
                t1_format,
                t2_format,
                decomp,
            )),
            proof: RefCell::new(None),
            page_prover_data: RefCell::new(None),
            partial_pk: RefCell::new(MultiStarkPartialProvingKey::<SC>::default()),
            _marker2: PhantomData,
        }
    }

    /// Note: for this function to do purely trace generation, the commitments and PoverTraceData
    /// for the input and output pages have to be stored in page_loader. Otherwise, if the commitment
    /// or the ProverTraceData is not present for a page, they will be generated in this function and
    /// stored in the page_loader.
    pub fn generate_trace(
        &self,
        page_loader: &mut PageDataLoader<SC, COMMIT_LEN>,
        parent_page_commit: Commitment<COMMIT_LEN>,
        child_page_commit: Commitment<COMMIT_LEN>,
        output_df: &mut DataFrame<COMMIT_LEN>,
        pairs_list_index: &mut u32,
        engine: &E,
    ) where
        Val<SC>: PrimeField,
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
        ProverTraceData<SC>: DeserializeOwned + Serialize,
    {
        let mut pis = self.pis.borrow_mut();

        pis.parent_page_commit = parent_page_commit;
        pis.child_page_commit = child_page_commit;

        let parent_page = page_loader
            .get_page_by_commitment(&pis.parent_page_commit)
            .unwrap();
        let child_page = page_loader
            .get_page_by_commitment(&pis.child_page_commit)
            .unwrap();

        pis.init_running_df_commit = hash_struct(&output_df);
        pis.pairs_list_index = *pairs_list_index;

        let mut ij_controller = self.ij_controller.borrow_mut();

        let output_page = ij_controller.inner_join(&parent_page, &child_page);

        let output_page_commit = page_loader.add_page(&output_page, engine);
        output_df.push_unindexed_page(output_page_commit.clone());
        *pairs_list_index += 1;

        pis.final_running_df_commit = hash_struct(&output_df);

        let intersector_trace_degree = 2 * parent_page.height().max(child_page.height());

        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        let prover_data = ij_controller.load_tables(
            &parent_page,
            &child_page,
            page_loader.get_pdata_by_commitment(&pis.parent_page_commit),
            page_loader.get_pdata_by_commitment(&pis.child_page_commit),
            page_loader.get_pdata_by_commitment(&pis.final_running_df_commit),
            intersector_trace_degree,
            &mut trace_builder.committer,
        );
        *self.page_prover_data.borrow_mut() = Some(prover_data);
    }

    pub fn set_up_keygen_builder(&self, engine: &E)
    where
        Val<SC>: PrimeField,
    {
        let mut keygen_builder: MultiStarkKeygenBuilder<SC> =
            MultiStarkKeygenBuilder::new(&engine.config());
        let ij_controller = self.ij_controller.borrow_mut();
        ij_controller.set_up_keygen_builder(&mut keygen_builder);
        *self.partial_pk.borrow_mut() = keygen_builder.generate_partial_pk();
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
        let ij_controller = self.ij_controller.borrow();
        let prover_data = self.page_prover_data.borrow_mut().take().unwrap();

        let prover = MultiTraceStarkProver::new(engine.config());
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let proof = ij_controller.prove(
            engine,
            &self.partial_pk.borrow(),
            &mut trace_builder,
            prover_data,
        );
        *self.proof.borrow_mut() = Some(proof);
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
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
    {
        // Note that this consumes the proof stored in self.proof
        // In the future we might want to load the proof directly from disk anyway
        let proof = self.proof.borrow_mut().take().unwrap();

        let pis = self.pis.borrow();

        let partial_pk = self.partial_pk.borrow();
        let partial_vk = partial_pk.partial_vk();

        let (parent_page_commit, child_page_commit, output_page_commit) =
            Self::get_page_commits_from_proof(&proof, &partial_vk);

        assert_eq!(pis.parent_page_commit, parent_page_commit);
        assert_eq!(pis.child_page_commit, child_page_commit);

        assert_eq!(hash_struct(&output_df), pis.init_running_df_commit);
        output_df.push_unindexed_page(output_page_commit.clone());
        assert_eq!(hash_struct(&output_df), pis.final_running_df_commit);

        let ij_controller = self.ij_controller.borrow();
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
    )
    where
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
    {
        let parent_page_commit = proof.commitments.main_trace[partial_vk.per_air[0]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone()
            .into();

        let child_page_commit = proof.commitments.main_trace[partial_vk.per_air[1]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone()
            .into();

        let output_page_commit = proof.commitments.main_trace[partial_vk.per_air[2]
            .main_graph
            .matrix_ptrs
            .get(0)
            .unwrap()
            .commit_index]
            .clone()
            .into();

        (
            parent_page_commit.into(),
            child_page_commit.into(),
            output_page_commit.into(),
        )
    }
}
