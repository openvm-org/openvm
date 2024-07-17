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

pub trait PageProvider {
    fn load_page_by_commitment(&self, commitment: &[u32]) -> Option<Page>;
    fn remove_page_by_commitment(&mut self, _commitment: &[u32]) {}
    fn add_page_with_commitment(&mut self, commitment: &[u32], page: &Page);
}

pub trait ProverTraceDataProvider<SC: StarkGenericConfig> {
    fn load_pdata_by_commitment(&self, commitment: &[u32]) -> Option<Arc<ProverTraceData<SC>>>;
    fn remove_pdata_by_commitment(&mut self, _commitment: &[u32]) {}
    fn add_pdata_with_commitment(
        &mut self,
        commitment: &[u32],
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

pub struct DiskPageLoader {
    pub db_path: String,
    pub idx_len: usize,
    pub data_len: usize,
    pub blank_commit: Vec<u32>,
}

pub struct BTreeMapPageLoader<SC: StarkGenericConfig> {
    pub page_map: BTreeMap<Vec<u32>, Page>,
    pub pdata_map: BTreeMap<Vec<u32>, Arc<ProverTraceData<SC>>>,
    pub blank_commit: Vec<u32>,
}

impl<SC: StarkGenericConfig> BTreeMapPageLoader<SC> {
    pub fn new(blank_commit: Vec<u32>) -> Self {
        BTreeMapPageLoader {
            page_map: BTreeMap::new(),
            pdata_map: BTreeMap::new(),
            blank_commit,
        }
    }
}

impl DiskPageLoader {
    pub fn new(db_path: String, idx_len: usize, data_len: usize, blank_commit: Vec<u32>) -> Self {
        create_dir_all(db_path.clone()).unwrap();
        DiskPageLoader {
            db_path,
            idx_len,
            data_len,
            blank_commit,
        }
    }
}

impl PageProvider for DiskPageLoader {
    fn load_page_by_commitment(&self, commitment: &[u32]) -> Option<Page> {
        let s = commit_u32_to_str(commitment);
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
    fn remove_page_by_commitment(&mut self, commitment: &[u32]) {
        let mut is_blank = true;
        for (x, y) in self.blank_commit.iter().zip(commitment.iter()) {
            if *x != *y {
                is_blank = false;
                break;
            }
        }
        if !is_blank {
            let s = commit_u32_to_str(commitment);
            let s = self.db_path.clone() + "/" + &s + ".trace";
            let path = Path::new(&s);
            if path.is_file() {
                remove_file(path).unwrap();
            }
        }
    }
    fn add_page_with_commitment(&mut self, commitment: &[u32], page: &Page) {
        let s = commit_u32_to_str(commitment);
        let file = File::create(self.db_path.clone() + "/" + &s + ".trace").unwrap();
        let mut writer = BufWriter::new(file);
        let trace = page.to_2d_vec();
        let encoded_trace = bincode::serialize(&trace).unwrap();
        writer.write_all(&encoded_trace).unwrap();
    }
}

impl<SC: StarkGenericConfig> PageProvider for BTreeMapPageLoader<SC> {
    fn load_page_by_commitment(&self, commitment: &[u32]) -> Option<Page> {
        self.page_map.get(commitment).cloned()
    }
    fn remove_page_by_commitment(&mut self, commitment: &[u32]) {
        let mut is_blank = true;
        for (x, y) in self.blank_commit.iter().zip(commitment.iter()) {
            if *x != *y {
                is_blank = false;
                break;
            }
        }
        if !is_blank {
            self.page_map.remove(commitment);
        }
    }
    fn add_page_with_commitment(&mut self, commitment: &[u32], page: &Page) {
        self.page_map.insert(commitment.to_vec(), page.clone());
    }
}

impl<SC: StarkGenericConfig> ProverTraceDataProvider<SC> for DiskPageLoader
where
    ProverTraceData<SC>: DeserializeOwned + Serialize,
{
    fn load_pdata_by_commitment(&self, commitment: &[u32]) -> Option<Arc<ProverTraceData<SC>>> {
        let s = commit_u32_to_str(commitment);
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
    fn remove_pdata_by_commitment(&mut self, commitment: &[u32]) {
        let mut is_blank = true;
        for (x, y) in self.blank_commit.iter().zip(commitment.iter()) {
            if *x != *y {
                is_blank = false;
                break;
            }
        }
        if !is_blank {
            let s = commit_u32_to_str(commitment);
            let s = self.db_path.clone() + "/" + &s + ".cache.bin";
            let path = Path::new(&s);
            if path.is_file() {
                remove_file(path).unwrap();
            }
        }
    }
    fn add_pdata_with_commitment(
        &mut self,
        commitment: &[u32],
        pdata: Arc<ProverTraceData<SC>>,
    ) -> Arc<ProverTraceData<SC>> {
        let s = commit_u32_to_str(commitment);
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
