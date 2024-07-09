use std::{
    cell::RefCell,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use afs_stark_backend::{config::Com, prover::trace::TraceCommitter};
use itertools::Itertools;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

#[cfg(test)]
pub mod tests;

#[derive(Debug)]
pub enum PageBTreeNode<const COMMITMENT_LEN: usize> {
    Leaf(PageBTreeLeafNode),
    Internal(PageBTreeInternalNode<COMMITMENT_LEN>),
    Unloaded(PageBTreeUnloadedNode<COMMITMENT_LEN>),
}
#[derive(Debug)]
pub struct PageBTreeLeafNode {
    kv_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    leaf_page_height: usize,
    trace: Option<Vec<Vec<u32>>>,
}

#[derive(Debug)]
/// placeholder for nodes that we have not opened, so that we don't have
/// the entire tree in memory at the same time
pub struct PageBTreeUnloadedNode<const COMMITMENT_LEN: usize> {
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    commit: Vec<u32>,
}

#[derive(Debug)]
pub struct PageBTreeInternalNode<const COMMITMENT_LEN: usize> {
    keys: Vec<Vec<u32>>,
    children: Vec<PageBTreeNode<COMMITMENT_LEN>>,
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    internal_page_height: usize,
    trace: Option<Vec<Vec<u32>>>,
}
#[derive(Debug)]
pub struct PageBTree<const COMMITMENT_LEN: usize> {
    limb_bits: usize,
    key_len: usize,
    val_len: usize,
    leaf_page_height: usize,
    internal_page_height: usize,
    root: RefCell<PageBTreeNode<COMMITMENT_LEN>>,
    // records the pages of a tree that are loaded from disk
    // makes it easy to record the initial "touched" state of a tree
    loaded_pages: PageBTreePages,
    depth: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PageBTreePages {
    pub leaf_pages: Vec<Vec<Vec<u32>>>,
    pub internal_pages: Vec<Vec<Vec<u32>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageBTreeRootInfo {
    max_internal: usize,
    max_leaf: usize,
    commitment_len: usize,
    limb_bits: usize,
    key_len: usize,
    val_len: usize,
    leaf_page_height: usize,
    internal_page_height: usize,
    root_commitment: Vec<u32>,
    depth: usize,
    max_key: Vec<u32>,
    min_key: Vec<u32>,
}

pub fn matrix_usize_to_u32(mat: Vec<Vec<usize>>) -> Vec<Vec<u32>> {
    mat.into_iter()
        .map(|row| row.into_iter().map(|u| u as u32).collect_vec())
        .collect_vec()
}

impl<const COMMITMENT_LEN: usize> PageBTreeUnloadedNode<COMMITMENT_LEN> {
    fn load_leaf(&self, key_len: usize) -> Option<PageBTreeLeafNode> {
        let s = commit_u32_to_str(&self.commit);
        let file = match File::open("src/pagebtree/leaf/".to_owned() + &s + ".trace") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let mut encoded_trace = vec![];
        reader.read_to_end(&mut encoded_trace).unwrap();
        let trace: Vec<Vec<u32>> = bincode::deserialize(&encoded_trace).unwrap();
        let mut kv_pairs = vec![];
        if trace[0][0] > 1 {
            panic!();
        }
        for row in &trace {
            if row[0] == 1 {
                kv_pairs.push((row[1..1 + key_len].to_vec(), row[1 + key_len..].to_vec()));
            }
        }
        Some(PageBTreeLeafNode {
            kv_pairs,
            min_key: self.min_key.clone(),
            max_key: self.max_key.clone(),
            leaf_page_height: trace.len(),
            trace: Some(trace),
        })
    }
    fn load_internal(&self, key_len: usize) -> Option<PageBTreeInternalNode<COMMITMENT_LEN>> {
        let s = commit_u32_to_str(&self.commit);
        let file = match File::open("src/pagebtree/internal/".to_owned() + &s + ".trace") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let mut encoded_trace = vec![];
        reader.read_to_end(&mut encoded_trace).unwrap();
        let trace: Vec<Vec<u32>> = bincode::deserialize(&encoded_trace).unwrap();
        if trace[0][0] != 2 {
            panic!();
        }
        let mut keys = vec![];
        let mut children = vec![];
        for (i, row) in trace.iter().enumerate() {
            if row[1] == 1 {
                let min_key = row[2..2 + key_len].to_vec();
                children.push(PageBTreeNode::Unloaded(PageBTreeUnloadedNode {
                    min_key: min_key.clone(),
                    max_key: row[2 + key_len..2 + 2 * key_len].to_vec(),
                    commit: row[2 + 2 * key_len..2 + 2 * key_len + COMMITMENT_LEN].to_vec(),
                }));
                if i > 0 {
                    keys.push(min_key);
                }
            }
        }
        Some(PageBTreeInternalNode {
            keys,
            children,
            min_key: self.min_key.clone(),
            max_key: self.max_key.clone(),
            internal_page_height: trace.len(),
            trace: Some(trace),
        })
    }

    fn load(
        &self,
        key_len: usize,
        loaded_pages: &mut PageBTreePages,
    ) -> Option<PageBTreeNode<COMMITMENT_LEN>> {
        let leaf = self.load_leaf(key_len);
        if let Some(leaf) = leaf {
            loaded_pages.leaf_pages.push(leaf.trace.clone().unwrap());
            return Some(PageBTreeNode::Leaf(leaf));
        };
        let internal = self.load_internal(key_len);
        if let Some(internal) = internal {
            loaded_pages
                .internal_pages
                .push(internal.trace.clone().unwrap());
            return Some(PageBTreeNode::Internal(internal));
        };
        None
    }
}

impl PageBTreePages {
    pub fn new() -> Self {
        PageBTreePages {
            leaf_pages: vec![],
            internal_pages: vec![],
        }
    }
}

impl PageBTreeLeafNode {
    /// assume that kv_pairs is sorted
    fn new(kv_pairs: Vec<(Vec<u32>, Vec<u32>)>, leaf_page_height: usize) -> Self {
        debug_assert!(leaf_page_height >= kv_pairs.len());
        if kv_pairs.is_empty() {
            Self {
                kv_pairs: Vec::new(),
                min_key: vec![],
                max_key: vec![],
                leaf_page_height,
                trace: None,
            }
        } else {
            let min_key = kv_pairs[0].0.clone();
            let max_key = kv_pairs[kv_pairs.len() - 1].0.clone();
            Self {
                kv_pairs,
                min_key,
                max_key,
                leaf_page_height,
                trace: None,
            }
        }
    }

    fn search(&self, key: &[u32]) -> Option<Vec<u32>> {
        let (i, is_eq) = binsearch_kv(&self.kv_pairs, key);
        if is_eq {
            Some(self.kv_pairs[i - 1].1.clone())
        } else {
            None
        }
    }

    /// updates a leaf node with a new entry. If there is overflow, this function returns an index along with
    /// a new node to be placed in the parent node. We trigger this overflow only when the page is full
    fn update(&mut self, key: &[u32], val: &[u32]) -> Option<(Vec<u32>, PageBTreeLeafNode)> {
        self.trace = None;
        self.add_kv(key, val);
        if self.kv_pairs.len() == self.leaf_page_height + 1 {
            let mididx = self.leaf_page_height / 2;
            let mid = self.kv_pairs[mididx].clone();
            let new_kv_pairs = self.kv_pairs.split_off(mididx);
            let l2 = Self::new(new_kv_pairs, self.leaf_page_height);
            self.max_key = self.kv_pairs[mididx - 1].clone().0;
            Some((mid.0, l2))
        } else {
            None
        }
    }
    // assumes we have space
    fn add_kv(&mut self, key: &[u32], val: &[u32]) {
        let (i, is_eq) = binsearch_kv(&self.kv_pairs, key);
        if is_eq {
            self.kv_pairs[i - 1].1 = val.to_vec();
        } else {
            if i == 0 {
                self.min_key = key.to_vec();
            }
            if i == self.kv_pairs.len() {
                self.max_key = key.to_vec();
            }
            self.kv_pairs.insert(i, (key.to_vec(), val.to_vec()));
        }
    }

    fn consistency_check(&self) {
        for i in 0..self.kv_pairs.len() - 1 {
            assert!(cmp(&self.kv_pairs[i].0, &self.kv_pairs[i + 1].0) < 0);
        }
        assert!(cmp(&self.min_key, &self.kv_pairs[0].0) == 0);
        assert!(cmp(&self.max_key, &self.kv_pairs[self.kv_pairs.len() - 1].0) == 0);
    }

    fn gen_trace(&mut self, key_len: usize, val_len: usize) -> Vec<Vec<u32>> {
        if let Some(t) = &self.trace {
            return t.clone();
        }
        let mut trace = Vec::new();
        for i in 0..self.kv_pairs.len() {
            let mut row = Vec::new();
            row.push(1);
            for k in &self.kv_pairs[i].0 {
                row.push(*k);
            }
            for v in &self.kv_pairs[i].1 {
                row.push(*v);
            }
            trace.push(row);
        }
        trace.resize(self.leaf_page_height, vec![]);
        for t in &mut trace {
            t.resize(1 + key_len + val_len, 0);
        }
        self.trace = Some(trace.clone());
        trace
    }

    fn gen_all_trace(&mut self, key_len: usize, val_len: usize, pages: &mut PageBTreePages) {
        pages.leaf_pages.push(self.gen_trace(key_len, val_len));
    }

    fn commit<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        if self.trace.is_none() {
            self.gen_trace(key_len, val_len);
        }
        let commitment = committer.commit(vec![RowMajorMatrix::new(
            self.trace
                .clone()
                .unwrap()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            1 + key_len + val_len,
        )]);
        let commit: [Val<SC>; COMMITMENT_LEN] = commitment.commit.into();
        let s = commit_to_str::<SC>(&commit);
        if !Path::new(&("src/pagebtree/leaf/".to_owned() + &s + ".trace")).is_file() {
            let file = File::create("src/pagebtree/leaf/".to_owned() + &s + ".trace").unwrap();
            let mut writer = BufWriter::new(file);
            let encoded_trace = bincode::serialize(&self.trace.as_ref().unwrap()).unwrap();
            writer.write_all(&encoded_trace).unwrap();
        }
    }
}

impl<const COMMITMENT_LEN: usize> PageBTreeNode<COMMITMENT_LEN> {
    fn min_key(&self) -> Vec<u32> {
        match self {
            PageBTreeNode::Leaf(l) => l.min_key.clone(),
            PageBTreeNode::Internal(i) => i.min_key.clone(),
            PageBTreeNode::Unloaded(u) => u.min_key.clone(),
        }
    }
    fn max_key(&self) -> Vec<u32> {
        match self {
            PageBTreeNode::Leaf(l) => l.max_key.clone(),
            PageBTreeNode::Internal(i) => i.max_key.clone(),
            PageBTreeNode::Unloaded(u) => u.max_key.clone(),
        }
    }
    fn search(&mut self, key: &Vec<u32>, loaded_pages: &mut PageBTreePages) -> Option<Vec<u32>> {
        match self {
            PageBTreeNode::Leaf(l) => l.search(key),
            PageBTreeNode::Internal(i) => i.search(key, loaded_pages),
            PageBTreeNode::Unloaded(_) => panic!(),
        }
    }
    fn update(
        &mut self,
        key: &[u32],
        val: &[u32],
        loaded_pages: &mut PageBTreePages,
    ) -> Option<(Vec<u32>, PageBTreeNode<COMMITMENT_LEN>)> {
        match self {
            PageBTreeNode::Leaf(l) => l
                .update(key, val)
                .map(|(mid, l2)| (mid, PageBTreeNode::Leaf(l2))),
            PageBTreeNode::Internal(i) => i.update(key, val, loaded_pages),
            PageBTreeNode::Unloaded(_) => panic!(),
        }
    }
    fn consistency_check(&self) {
        match self {
            PageBTreeNode::Leaf(l) => l.consistency_check(),
            PageBTreeNode::Internal(i) => i.consistency_check(),
            PageBTreeNode::Unloaded(u) => assert!(cmp(&u.min_key, &u.max_key) < 0),
        }
    }
    fn gen_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) -> Vec<Vec<u32>>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.gen_trace(key_len, val_len),
            PageBTreeNode::Internal(i) => i.gen_trace(committer, key_len, val_len),
            PageBTreeNode::Unloaded(_) => panic!(),
        }
    }

    fn gen_commitment<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) -> Vec<u32>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => {
                let trace = l.gen_trace(key_len, val_len);
                let width = trace[0].len();
                let commitment = committer.commit(vec![RowMajorMatrix::new(
                    trace
                        .into_iter()
                        .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                        .collect(),
                    width,
                )]);
                let commit: [Val<SC>; COMMITMENT_LEN] = commitment.commit.into();
                commit.into_iter().map(|u| u.as_canonical_u32()).collect()
            }
            PageBTreeNode::Internal(i) => {
                let trace = i.gen_trace(committer, key_len, val_len);
                let width = trace[0].len();
                let commitment = committer.commit(vec![RowMajorMatrix::new(
                    trace
                        .into_iter()
                        .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                        .collect(),
                    width,
                )]);
                let commit: [Val<SC>; COMMITMENT_LEN] = commitment.commit.into();
                commit.into_iter().map(|u| u.as_canonical_u32()).collect()
            }
            PageBTreeNode::Unloaded(u) => u.commit.to_vec(),
        }
    }

    fn gen_all_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
        pages: &mut PageBTreePages,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.gen_all_trace(key_len, val_len, pages),
            PageBTreeNode::Internal(i) => i.gen_all_trace(committer, key_len, val_len, pages),
            PageBTreeNode::Unloaded(_) => (),
        }
    }

    fn commit_all<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.commit(committer, key_len, val_len),
            PageBTreeNode::Internal(i) => i.commit_all(committer, key_len, val_len),
            PageBTreeNode::Unloaded(_) => (),
        }
    }
}

impl<const COMMITMENT_LEN: usize> PageBTreeInternalNode<COMMITMENT_LEN> {
    fn new(
        keys: Vec<Vec<u32>>,
        children: Vec<PageBTreeNode<COMMITMENT_LEN>>,
        internal_page_height: usize,
    ) -> Self {
        assert!(!keys.is_empty());
        assert!(children.len() == keys.len() + 1);
        let min_key = children[0].min_key();
        let max_key = children[children.len() - 1].max_key();
        Self {
            keys,
            children,
            min_key,
            max_key,
            internal_page_height,
            trace: None,
        }
    }

    fn search(&mut self, key: &Vec<u32>, loaded_pages: &mut PageBTreePages) -> Option<Vec<u32>> {
        let i = binsearch(&self.keys, key);
        if let PageBTreeNode::Unloaded(u) = &self.children[i] {
            self.children[i] = u.load(key.len(), loaded_pages).unwrap();
        }
        self.children[i].search(key, loaded_pages)
    }

    /// updates an internal node with a new entry. If there is overflow, this function returns an index along with
    /// a new node to be placed in the parent node. We trigger this overflow only when the page is full
    fn update(
        &mut self,
        key: &[u32],
        val: &[u32],
        loaded_pages: &mut PageBTreePages,
    ) -> Option<(Vec<u32>, PageBTreeNode<COMMITMENT_LEN>)> {
        self.trace = None;
        let mut ret = None;
        let i = binsearch(&self.keys, key);
        if let PageBTreeNode::Unloaded(u) = &self.children[i] {
            self.children[i] = u.load(key.len(), loaded_pages).unwrap();
        }
        if let Some((k, node)) = self.children[i].update(key, val, loaded_pages) {
            ret = self.add_key(&k, node, i + 1);
        };
        self.min_key = self.children[0].min_key();
        self.max_key = self.children[self.children.len() - 1].max_key();
        ret
    }

    fn add_key(
        &mut self,
        key: &[u32],
        node: PageBTreeNode<COMMITMENT_LEN>,
        idx: usize,
    ) -> Option<(Vec<u32>, PageBTreeNode<COMMITMENT_LEN>)> {
        if self.children.len() == self.internal_page_height {
            // let mut new_children = vec![];
            self.keys.insert(idx - 1, key.to_vec());
            self.children.insert(idx, node);
            let mididx = self.internal_page_height / 2;
            let new_children = self.children.split_off(mididx + 1);
            let new_keys = self.keys.split_off(mididx + 1);
            let l2 = Self::new(new_keys, new_children, self.internal_page_height);
            let mid = self.keys.pop().unwrap();
            self.max_key = self.children[self.children.len() - 1].max_key();
            Some((mid, PageBTreeNode::Internal(l2)))
        } else {
            if idx < self.children.len() {
                self.keys.insert(idx - 1, key.to_vec());
                self.children.insert(idx, node);
                return None;
            }
            self.keys.push(key.to_vec());
            self.children.push(node);
            self.max_key = self.children[self.children.len() - 1].max_key();
            None
        }
    }

    fn consistency_check(&self) {
        for child in &self.children {
            child.consistency_check();
        }
        for i in 0..self.keys.len() {
            assert!(cmp(&self.keys[i], &self.children[i].max_key()) > 0);
            assert!(cmp(&self.keys[i], &self.children[i + 1].min_key()) == 0)
        }
        assert!(cmp(&self.min_key, &self.children[0].min_key()) == 0);
        assert!(
            cmp(
                &self.max_key,
                &self.children[self.children.len() - 1].max_key()
            ) == 0
        );
    }

    fn gen_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) -> Vec<Vec<u32>>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        if let Some(t) = &self.trace {
            return t.clone();
        }
        let mut trace = Vec::new();
        for i in 0..self.children.len() {
            let mut row = Vec::new();
            row.push(2);
            row.push(1);
            for k in self.children[i].min_key() {
                row.push(k);
            }
            for v in self.children[i].max_key() {
                row.push(v);
            }
            let child_commit = self.children[i].gen_commitment(committer, key_len, val_len);
            row.extend(child_commit.clone());
            trace.push(row);
        }
        trace.resize(self.internal_page_height, vec![2]);
        for t in &mut trace {
            t.resize(2 + 2 * key_len + COMMITMENT_LEN, 0);
        }
        self.trace = Some(trace.clone());
        trace
    }

    fn gen_all_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
        pages: &mut PageBTreePages,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        for i in 0..self.children.len() {
            self.children[i].gen_all_trace(committer, key_len, val_len, pages);
        }
        pages
            .internal_pages
            .push(self.gen_trace(committer, key_len, val_len));
    }

    fn commit<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        if self.trace.is_none() {
            self.gen_trace(committer, key_len, val_len);
        }
        let commitment = committer.commit(vec![RowMajorMatrix::new(
            self.trace
                .clone()
                .unwrap()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            2 + 2 * key_len + COMMITMENT_LEN,
        )]);
        let commit: [Val<SC>; COMMITMENT_LEN] = commitment.commit.into();
        let s = commit_to_str::<SC>(&commit);
        if !Path::new(&("src/pagebtree/internal/".to_owned() + &s + ".trace")).is_file() {
            let file = File::create("src/pagebtree/internal/".to_owned() + &s + ".trace").unwrap();
            let mut writer = BufWriter::new(file);
            let encoded_trace = bincode::serialize(&self.trace.as_ref().unwrap()).unwrap();
            writer.write_all(&encoded_trace).unwrap();
        }
    }

    fn commit_all<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        self.commit(committer, key_len, val_len);
        for child in &mut self.children {
            child.commit_all(committer, key_len, val_len);
        }
    }
}

impl<const COMMITMENT_LEN: usize> PageBTree<COMMITMENT_LEN> {
    pub fn new(
        limb_bits: usize,
        key_len: usize,
        val_len: usize,
        leaf_page_height: usize,
        internal_page_height: usize,
    ) -> Self {
        let leaf = PageBTreeLeafNode {
            kv_pairs: Vec::new(),
            min_key: vec![0; key_len],
            max_key: vec![(1 << limb_bits) - 1; key_len],
            leaf_page_height,
            trace: None,
        };
        let leaf = PageBTreeNode::Leaf(leaf);
        PageBTree {
            limb_bits,
            key_len,
            val_len,
            root: RefCell::new(leaf),
            loaded_pages: PageBTreePages::new(),
            leaf_page_height,
            internal_page_height,
            depth: 1,
        }
    }

    pub fn load(root_commit: Vec<u32>) -> Option<Self> {
        let s = commit_u32_to_str(&root_commit);
        let file = match File::open("src/pagebtree/root/".to_owned() + &s + ".trace") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let mut encoded_info = vec![];
        reader.read_to_end(&mut encoded_info).unwrap();
        let info: PageBTreeRootInfo = bincode::deserialize(&encoded_info).unwrap();
        debug_assert!(info.commitment_len == COMMITMENT_LEN);
        let root = PageBTreeNode::Unloaded(PageBTreeUnloadedNode {
            min_key: info.min_key,
            max_key: info.max_key,
            commit: root_commit,
        });
        Some(PageBTree {
            limb_bits: info.limb_bits,
            key_len: info.key_len,
            val_len: info.val_len,
            leaf_page_height: info.leaf_page_height,
            internal_page_height: info.leaf_page_height,
            root: RefCell::new(root),
            loaded_pages: PageBTreePages::new(),
            depth: info.depth,
        })
    }
    pub fn min_key(&self) -> Vec<u32> {
        self.root.borrow().min_key()
    }
    pub fn max_key(&self) -> Vec<u32> {
        self.root.borrow().max_key()
    }
    pub fn search(&mut self, key: &Vec<u32>) -> Option<Vec<u32>> {
        for k in key {
            assert!(*k < 1 << self.limb_bits);
        }
        assert!(key.len() == self.key_len);
        if let PageBTreeNode::Unloaded(u) = self.root.get_mut() {
            self.root = RefCell::new(u.load(key.len(), &mut self.loaded_pages).unwrap());
        }
        self.root.get_mut().search(key, &mut self.loaded_pages)
    }

    // Updates the tree with a new key value pair.
    pub fn update(&mut self, key: &[u32], val: &[u32]) {
        for k in key {
            assert!(*k < 1 << self.limb_bits);
        }
        assert!(key.len() == self.key_len);
        assert!(val.len() == self.val_len);
        if let PageBTreeNode::Unloaded(u) = self.root.get_mut() {
            self.root = RefCell::new(u.load(key.len(), &mut self.loaded_pages).unwrap());
        }
        let ret = self.root.get_mut().update(key, val, &mut self.loaded_pages);
        if let Some((k, node)) = ret {
            let root = self
                .root
                .replace(PageBTreeNode::Leaf(PageBTreeLeafNode::new(vec![], 0)));
            let min_key = root.min_key();
            let max_key = node.max_key();
            let internal = PageBTreeInternalNode {
                keys: vec![k],
                children: vec![root, node],
                min_key,
                max_key,
                internal_page_height: self.internal_page_height,
                trace: None,
            };
            self.depth += 1;
            self.root = RefCell::new(PageBTreeNode::Internal(internal));
        }
    }
    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn consistency_check(&self) {
        self.root.borrow().consistency_check()
    }

    pub fn page_min_width(&self) -> usize {
        self.key_len + self.val_len + 1
    }

    pub fn gen_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
    ) -> Vec<Vec<u32>>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        self.root
            .get_mut()
            .gen_trace(committer, self.key_len, self.val_len)
    }

    pub fn gen_all_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &TraceCommitter<SC>,
    ) -> PageBTreePages
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        let mut pages = PageBTreePages::new();
        self.root
            .get_mut()
            .gen_all_trace(committer, self.key_len, self.val_len, &mut pages);
        pages.leaf_pages.reverse();
        pages.internal_pages.reverse();
        pages
    }

    pub fn gen_loaded_trace(&self) -> PageBTreePages {
        self.loaded_pages.clone()
    }

    pub fn commit<SC: StarkGenericConfig>(&mut self, committer: &TraceCommitter<SC>)
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        let root_trace = self
            .root
            .get_mut()
            .gen_trace(committer, self.key_len, self.val_len);
        let width = root_trace[0].len();
        let commitment: [Val<SC>; COMMITMENT_LEN] = committer
            .commit(vec![RowMajorMatrix::new(
                root_trace
                    .into_iter()
                    .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                    .collect(),
                width,
            )])
            .commit
            .into();
        let commit: Vec<u32> = commitment
            .into_iter()
            .map(|c| c.as_canonical_u32())
            .collect();
        let s = commit_u32_to_str(&commit);
        let file = File::create("src/pagebtree/root/".to_owned() + &s + ".trace").unwrap();
        let root_info = PageBTreeRootInfo {
            max_internal: self.internal_page_height,
            max_leaf: self.leaf_page_height,
            commitment_len: COMMITMENT_LEN,
            limb_bits: self.limb_bits,
            key_len: self.key_len,
            val_len: self.val_len,
            leaf_page_height: self.leaf_page_height,
            internal_page_height: self.internal_page_height,
            root_commitment: commit,
            depth: self.depth,
            max_key: self.max_key(),
            min_key: self.min_key(),
        };
        let mut writer = BufWriter::new(file);
        let encoded_info = bincode::serialize(&root_info).unwrap();
        writer.write_all(&encoded_info).unwrap();
        self.root
            .get_mut()
            .commit_all(committer, self.key_len, self.val_len);
    }
}

pub fn cmp(key1: &[u32], key2: &[u32]) -> i32 {
    assert!(key1.len() == key2.len());
    let mut i = 0;
    while i < key1.len() && key1[i] == key2[i] {
        i += 1;
    }
    if i == key1.len() {
        0
    } else {
        2 * ((key1[i] > key2[i]) as i32) - 1
    }
}

fn binsearch(keys: &[Vec<u32>], k: &[u32]) -> usize {
    let mut hi = keys.len() + 1;
    let mut lo = 0;
    // invariant is lo <= ans < hi
    while hi > lo + 1 {
        let mid = (hi + lo) / 2 - 1;
        if cmp(&keys[mid], k) > 0 {
            hi = mid + 1;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn binsearch_kv(kv_pairs: &[(Vec<u32>, Vec<u32>)], k: &[u32]) -> (usize, bool) {
    let mut hi = kv_pairs.len() + 1;
    let mut lo = 0;
    // invariant is lo <= ans < hi
    while hi > lo + 1 {
        let mid = (hi + lo) / 2 - 1;
        if cmp(&kv_pairs[mid].0, k) > 0 {
            hi = mid + 1;
        } else {
            lo = mid + 1;
        }
    }
    if lo == 0 {
        (lo, false)
    } else {
        (lo, cmp(&kv_pairs[lo - 1].0, k) == 0)
    }
}

fn commit_to_str<SC: StarkGenericConfig>(commit: &[Val<SC>]) -> String
where
    Val<SC>: PrimeField32 + AbstractField,
{
    commit.iter().fold("".to_owned(), |acc, x| {
        acc.to_owned() + &format!("{:08x}", x.as_canonical_u32())
    })
}

fn commit_u32_to_str(commit: &[u32]) -> String {
    commit.iter().fold("".to_owned(), |acc, x| {
        acc.to_owned() + &format!("{:08x}", x)
    })
}
