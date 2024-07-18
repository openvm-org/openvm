use std::collections::HashMap;

use afs_chips::common::page::Page;
use afs_stark_backend::{
    config::Com,
    prover::{
        trace::{ProverTraceData, TraceCommitmentBuilder},
        MultiTraceStarkProver,
    },
};
use afs_test_utils::engine::StarkEngine;
use bimap::BiMap;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::common::get_commit_from_pdata;

use super::Commitment;

pub trait DataProvider<SC: StarkGenericConfig, const COMMIT_LEN: usize> {
    fn get_page_by_commitment(&self, commitment: &Commitment<COMMIT_LEN>) -> Option<Page>;

    fn get_pdata_by_commitment(
        &self,
        commitment: &Commitment<COMMIT_LEN>,
    ) -> Option<ProverTraceData<SC>>;

    fn remove_page_and_pdata_by_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>);

    fn add_page_and_pdata_with_commitment(
        &mut self,
        commitment: &Commitment<COMMIT_LEN>,
        page: &Page,
        pdata: ProverTraceData<SC>,
    );

    fn add_page(&mut self, page: &Page, engine: &impl StarkEngine<SC>) -> Commitment<COMMIT_LEN>
    where
        Val<SC>: PrimeField,
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>;
}
pub struct PageDataLoader<SC: StarkGenericConfig, const COMMIT_LEN: usize> {
    pub page_map: BiMap<Commitment<COMMIT_LEN>, Page>,
    pub pdata_map: HashMap<Commitment<COMMIT_LEN>, ProverTraceData<SC>>,
}

impl<SC: StarkGenericConfig, const COMMIT_LEN: usize> PageDataLoader<SC, COMMIT_LEN> {
    pub fn empty() -> Self {
        Self {
            page_map: BiMap::new(),
            pdata_map: HashMap::new(),
        }
    }

    fn gen_pdata_and_commitment(
        &mut self,
        page: &Page,
        engine: &impl StarkEngine<SC>,
    ) -> (ProverTraceData<SC>, Commitment<COMMIT_LEN>)
    where
        Val<SC>: PrimeField,
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
    {
        let prover = MultiTraceStarkProver::<SC>::new(&engine.config());
        let trace_builder = TraceCommitmentBuilder::<SC>::new(prover.pcs());
        let trace_committer = trace_builder.committer;

        let page_trace = page.gen_trace::<Val<SC>>();
        let pdata = trace_committer.commit(vec![page_trace.clone()]);

        let commit = get_commit_from_pdata(&pdata);

        (pdata, commit)
    }
}

impl<SC: StarkGenericConfig, const COMMIT_LEN: usize> DataProvider<SC, COMMIT_LEN>
    for PageDataLoader<SC, COMMIT_LEN>
{
    fn get_page_by_commitment(&self, commitment: &Commitment<COMMIT_LEN>) -> Option<Page> {
        self.page_map.get_by_left(commitment).cloned()
    }

    fn get_pdata_by_commitment(
        &self,
        commitment: &Commitment<COMMIT_LEN>,
    ) -> Option<ProverTraceData<SC>> {
        self.pdata_map.get(commitment).cloned()
    }

    fn remove_page_and_pdata_by_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>) {
        self.page_map.remove_by_left(commitment);
        self.pdata_map.remove(commitment);
    }

    fn add_page_and_pdata_with_commitment(
        &mut self,
        commitment: &Commitment<COMMIT_LEN>,
        page: &Page,
        pdata: ProverTraceData<SC>,
    ) {
        self.page_map.insert(commitment.clone(), page.clone());
        self.pdata_map.insert(commitment.clone(), pdata.clone());
    }

    fn add_page(&mut self, page: &Page, engine: &impl StarkEngine<SC>) -> Commitment<COMMIT_LEN>
    where
        Val<SC>: PrimeField,
        Com<SC>: Into<[BabyBear; COMMIT_LEN]>,
    {
        if let Some(commitment) = self.page_map.get_by_right(page) {
            // We already have the ProverTraceData and commitment for this page
            return commitment.clone();
        }

        let (pdata, commitment) = self.gen_pdata_and_commitment(page, engine);
        self.add_page_and_pdata_with_commitment(&commitment, page, pdata);

        commitment
    }
}
