use std::{
    collections::BTreeMap,
    fs::{create_dir_all, remove_file, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::Arc,
};

use afs_chips::{common::page::Page, page_btree::commit_u32_to_str};
use afs_stark_backend::prover::trace::ProverTraceData;
use p3_uni_stark::StarkGenericConfig;
use serde::{de::DeserializeOwned, Serialize};

use super::Commitment;

pub trait PageProvider<const COMMIT_LEN: usize> {
    fn load_page_by_commitment(&self, commitment: &Commitment<COMMIT_LEN>) -> Option<Page>;
    fn remove_page_by_commitment(&mut self, _commitment: &Commitment<COMMIT_LEN>) {}
    fn add_page_with_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>, page: &Page);
}

pub trait ProverTraceDataProvider<SC: StarkGenericConfig, const COMMIT_LEN: usize> {
    fn load_pdata_by_commitment(
        &self,
        commitment: &Commitment<COMMIT_LEN>,
    ) -> Option<Arc<ProverTraceData<SC>>>;
    fn remove_pdata_by_commitment(&mut self, _commitment: &Commitment<COMMIT_LEN>) {}
    fn add_pdata_with_commitment(
        &mut self,
        commitment: &Commitment<COMMIT_LEN>,
        pdata: Arc<ProverTraceData<SC>>,
    ) -> Arc<ProverTraceData<SC>>;
}

// impl PageProvider for BTreeMap<Vec<u32>, Page> {
//     fn load_page_by_commitment(&self, commitment: &[u32]) -> Option<Page> {
//         self.get(&commitment.to_vec()).cloned()
//     }
//     fn remove_page_by_commitment(&mut self, commitment: &[u32]) {
//         self.remove(&commitment.to_vec()).unwrap();
//     }
// }

// impl<SC: StarkGenericConfig> ProverTraceDataProvider<SC>
//     for BTreeMap<Vec<u32>, Arc<ProverTraceData<SC>>>
// {
//     fn load_pdata_by_commitment(&self, commitment: &[u32]) -> Option<Arc<ProverTraceData<SC>>> {
//         self.get(&commitment.to_vec()).cloned()
//     }
//     fn remove_pdata_by_commitment(&mut self, commitment: &[u32]) {
//         self.remove(&commitment.to_vec()).unwrap();
//     }
// }

// impl<SC: StarkGenericConfig> ProverTraceDataProvider<SC>
//     for BTreeMap<Vec<u32>, (Page, Arc<ProverTraceData<SC>>)>
// {
//     fn load_pdata_by_commitment(&self, commitment: &[u32]) -> Option<Arc<ProverTraceData<SC>>> {
//         self.get(&commitment.to_vec())
//             .cloned()
//             .map(|(_, pdata)| pdata)
//     }
//     fn remove_pdata_by_commitment(&mut self, commitment: &[u32]) {
//         self.remove(&commitment.to_vec()).unwrap();
//     }
// }

// impl<SC: StarkGenericConfig> PageProvider for BTreeMap<Vec<u32>, (Page, Arc<ProverTraceData<SC>>)> {
//     fn load_page_by_commitment(&self, commitment: &[u32]) -> Option<Page> {
//         self.get(&commitment.to_vec())
//             .cloned()
//             .map(|(page, _)| page)
//     }
//     fn remove_page_by_commitment(&mut self, commitment: &[u32]) {
//         self.remove(&commitment.to_vec()).unwrap();
//     }
// }

pub struct DiskPageLoader<const COMMIT_LEN: usize> {
    pub db_path: String,
    pub idx_len: usize,
    pub data_len: usize,
    pub blank_commit: Commitment<COMMIT_LEN>,
}

pub struct BTreeMapPageLoader<SC: StarkGenericConfig, const COMMIT_LEN: usize> {
    pub page_map: BTreeMap<Commitment<COMMIT_LEN>, Page>,
    pub pdata_map: BTreeMap<Commitment<COMMIT_LEN>, Arc<ProverTraceData<SC>>>,
    pub blank_commit: Commitment<COMMIT_LEN>,
}

impl<SC: StarkGenericConfig, const COMMIT_LEN: usize> BTreeMapPageLoader<SC, COMMIT_LEN> {
    pub fn new(blank_commit: Commitment<COMMIT_LEN>) -> Self {
        BTreeMapPageLoader {
            page_map: BTreeMap::new(),
            pdata_map: BTreeMap::new(),
            blank_commit,
        }
    }
}

impl<const COMMIT_LEN: usize> DiskPageLoader<COMMIT_LEN> {
    pub fn new(
        db_path: String,
        idx_len: usize,
        data_len: usize,
        blank_commit: Commitment<COMMIT_LEN>,
    ) -> Self {
        create_dir_all(db_path.clone()).unwrap();
        DiskPageLoader {
            db_path,
            idx_len,
            data_len,
            blank_commit,
        }
    }
}

impl<const COMMIT_LEN: usize> PageProvider<COMMIT_LEN> for DiskPageLoader<COMMIT_LEN> {
    fn load_page_by_commitment(&self, commitment: &Commitment<COMMIT_LEN>) -> Option<Page> {
        let s = commit_u32_to_str(&commitment.commit);
        match File::open(self.db_path.clone() + "/" + &s + ".trace") {
            Err(_) => None,
            Ok(file) => {
                let mut reader = BufReader::new(file);
                let mut encoded_trace = vec![];
                reader.read_to_end(&mut encoded_trace).unwrap();
                let trace: Vec<Vec<u32>> = bincode::deserialize(&encoded_trace).unwrap();
                Some(Page::from_2d_vec(&trace, self.idx_len, self.data_len))
            }
        }
    }
    fn remove_page_by_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>) {
        if !self.blank_commit.eq(commitment) {
            let s = commit_u32_to_str(&commitment.commit);
            let s = self.db_path.clone() + "/" + &s + ".trace";
            let path = Path::new(&s);
            if path.is_file() {
                remove_file(path).unwrap();
            }
        }
    }
    fn add_page_with_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>, page: &Page) {
        let s = commit_u32_to_str(&commitment.commit);
        let file = File::create(self.db_path.clone() + "/" + &s + ".trace").unwrap();
        let mut writer = BufWriter::new(file);
        let trace = page.to_2d_vec();
        let encoded_trace = bincode::serialize(&trace).unwrap();
        writer.write_all(&encoded_trace).unwrap();
    }
}

impl<SC: StarkGenericConfig, const COMMIT_LEN: usize> PageProvider<COMMIT_LEN>
    for BTreeMapPageLoader<SC, COMMIT_LEN>
{
    fn load_page_by_commitment(&self, commitment: &Commitment<COMMIT_LEN>) -> Option<Page> {
        self.page_map.get(commitment).cloned()
    }
    fn remove_page_by_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>) {
        if !self.blank_commit.eq(commitment) {
            self.page_map.remove(commitment);
        }
    }
    fn add_page_with_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>, page: &Page) {
        self.page_map.insert(commitment.clone(), page.clone());
    }
}

impl<SC: StarkGenericConfig, const COMMIT_LEN: usize> ProverTraceDataProvider<SC, COMMIT_LEN>
    for DiskPageLoader<COMMIT_LEN>
where
    ProverTraceData<SC>: DeserializeOwned + Serialize,
{
    fn load_pdata_by_commitment(
        &self,
        commitment: &Commitment<COMMIT_LEN>,
    ) -> Option<Arc<ProverTraceData<SC>>> {
        let s = commit_u32_to_str(&commitment.commit);
        match File::open(self.db_path.clone() + "/" + &s + ".cache.bin") {
            Err(_) => None,
            Ok(file) => {
                let mut reader = BufReader::new(file);
                let mut encoded_trace = vec![];
                reader.read_to_end(&mut encoded_trace).unwrap();
                let pdata: ProverTraceData<SC> = bincode::deserialize(&encoded_trace).unwrap();
                Some(Arc::new(pdata))
            }
        }
    }
    fn remove_pdata_by_commitment(&mut self, commitment: &Commitment<COMMIT_LEN>) {
        if !self.blank_commit.eq(commitment) {
            let s = commit_u32_to_str(&commitment.commit);
            let s = self.db_path.clone() + "/" + &s + ".cache.bin";
            let path = Path::new(&s);
            if path.is_file() {
                remove_file(path).unwrap();
            }
        }
    }
    fn add_pdata_with_commitment(
        &mut self,
        commitment: &Commitment<COMMIT_LEN>,
        pdata: Arc<ProverTraceData<SC>>,
    ) -> Arc<ProverTraceData<SC>> {
        let s = commit_u32_to_str(&commitment.commit);
        let file = File::create(self.db_path.clone() + "/" + &s + ".cache.bin").unwrap();
        let mut writer = BufWriter::new(file);
        let pdata = match Arc::try_unwrap(pdata) {
            Err(_) => panic!(),
            Ok(pdata) => pdata,
        };
        let encoded_pdata = bincode::serialize(&pdata).unwrap();
        writer.write_all(&encoded_pdata).unwrap();
        Arc::new(pdata)
    }
}

// impl<SC: StarkGenericConfig> ProverTraceDataProvider<SC> for BTreeMapPageLoader<SC>
// where
//     ProverTraceData<SC>: DeserializeOwned + Serialize,
// {
//     fn load_pdata_by_commitment(&self, commitment: &[u32]) -> Option<Arc<ProverTraceData<SC>>> {
//         self.pdata_map.get(commitment).cloned()
//     }
//     fn remove_pdata_by_commitment(&mut self, commitment: &[u32]) {
//         let mut is_blank = true;
//         for (x, y) in self.blank_commit.iter().zip(commitment.iter()) {
//             if *x != *y {
//                 is_blank = false;
//                 break;
//             }
//         }
//         if !is_blank {
//             self.pdata_map.remove(commitment);
//         }
//     }
//     fn add_pdata_with_commitment(&mut self, commitment: &[u32], pdata: Arc<ProverTraceData<SC>>) {
//         self.pdata_map.insert(commitment.to_vec(), pdata);
//     }
// }
