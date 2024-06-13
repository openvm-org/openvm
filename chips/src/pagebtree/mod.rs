use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
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
pub enum PageBTreeNode<
    const MAX_INTERNAL: usize,
    const MAX_LEAF: usize,
    const COMMITMENT_LEN: usize,
> {
    Leaf(PageBTreeLeafNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>),
    Internal(PageBTreeInternalNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>),
    Unloaded(PageBTreeUnloadedNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>),
}
#[derive(Debug)]
pub struct PageBTreeLeafNode<
    const MAX_INTERNAL: usize,
    const MAX_LEAF: usize,
    const COMMITMENT_LEN: usize,
> {
    kv_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    trace: Option<Vec<Vec<u32>>>,
}

#[derive(Debug)]
pub struct PageBTreeUnloadedNode<
    const MAX_INTERNAL: usize,
    const MAX_LEAF: usize,
    const COMMITMENT_LEN: usize,
> {
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    commit: Vec<u32>,
}

#[derive(Debug)]
pub struct PageBTreeInternalNode<
    const MAX_INTERNAL: usize,
    const MAX_LEAF: usize,
    const COMMITMENT_LEN: usize,
> {
    keys: Vec<Vec<u32>>,
    children: Vec<PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>>,
    min_key: Vec<u32>,
    max_key: Vec<u32>,
    trace: Option<Vec<Vec<u32>>>,
}
#[derive(Debug)]
pub struct PageBTree<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
{
    limb_bits: usize,
    key_len: usize,
    val_len: usize,
    leaf_page_height: usize,
    internal_page_height: usize,
    root: Vec<PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>>,
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

impl<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
    PageBTreeUnloadedNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>
{
    fn load_leaf(
        &self,
        key_len: usize,
    ) -> Option<PageBTreeLeafNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>> {
        let s = self.commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x)
        });
        let file = match File::open("src/pagebtree/leaf/".to_owned() + &s + ".json") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let trace: Vec<Vec<u32>> = serde_json::from_reader(&mut reader).unwrap();
        let mut kv_pairs = vec![];
        if trace[0][0] == 0 {
            panic!();
        }
        for row in &trace {
            if row[1] == 1 {
                kv_pairs.push((row[2..2 + key_len].to_vec(), row[2 + key_len..].to_vec()));
            }
        }
        Some(PageBTreeLeafNode {
            kv_pairs,
            min_key: self.min_key.clone(),
            max_key: self.max_key.clone(),
            trace: Some(trace),
        })
    }
    fn load_internal(
        &self,
        key_len: usize,
    ) -> Option<PageBTreeInternalNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>> {
        let s = self.commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x)
        });
        let file = match File::open("src/pagebtree/internal/".to_owned() + &s + ".json") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let trace: Vec<Vec<u32>> = serde_json::from_reader(&mut reader).unwrap();
        if trace[0][0] == 1 {
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
            trace: Some(trace),
        })
    }

    fn load(
        &self,
        key_len: usize,
        loaded_pages: &mut PageBTreePages,
    ) -> Option<PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>> {
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

impl<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
    PageBTreeLeafNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>
{
    /// assume that kv_pairs is sorted
    fn new(kv_pairs: Vec<(Vec<u32>, Vec<u32>)>) -> Self {
        if kv_pairs.is_empty() {
            Self {
                kv_pairs: Vec::new(),
                min_key: vec![],
                max_key: vec![],
                trace: None,
            }
        } else {
            let min_key = kv_pairs[0].0.clone();
            let max_key = kv_pairs[kv_pairs.len() - 1].0.clone();
            // for (k, _) in &kv_pairs {
            //     if cmp(&min_key, k) > 0 {
            //         min_key = k.to_vec();
            //     }
            //     if cmp(k, &max_key) > 0 {
            //         max_key = k.to_vec();
            //     }
            // }
            Self {
                kv_pairs,
                min_key,
                max_key,
                trace: None,
            }
        }
    }

    fn search(&self, key: &[u32]) -> Option<Vec<u32>> {
        for (k, v) in &self.kv_pairs {
            let c = cmp(k, key);
            if c > 0 {
                return None;
            } else if c == 0 {
                return Some(v.clone());
            }
        }
        None
    }

    fn update(
        &mut self,
        key: &Vec<u32>,
        val: &Vec<u32>,
    ) -> Option<(
        Vec<u32>,
        PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>,
    )> {
        self.trace = None;
        self.add_kv(key, val);
        if self.kv_pairs.len() == MAX_LEAF + 1 {
            let mididx = MAX_LEAF / 2;
            let mid = self.kv_pairs[mididx].clone();
            let l2 = Self::new(self.kv_pairs[mididx..MAX_LEAF + 1].to_vec());
            self.kv_pairs = self.kv_pairs[0..mididx].to_vec();
            self.max_key = self.kv_pairs[mididx - 1].clone().0;
            Some((mid.0, PageBTreeNode::Leaf(l2)))
        } else {
            None
        }
    }
    // assumes we have space
    fn add_kv(&mut self, key: &Vec<u32>, val: &Vec<u32>) {
        if self.kv_pairs.is_empty() {
            self.min_key.clone_from(key);
        }
        for (i, (k, _)) in self.kv_pairs.iter().enumerate() {
            let c = cmp(k, key);
            if c > 0 {
                self.kv_pairs.insert(i, (key.clone(), val.to_vec()));
                if i == 0 {
                    self.min_key.clone_from(key);
                }
                return;
            } else if c == 0 {
                self.kv_pairs[i].1.clone_from(val);
                return;
            }
        }
        self.kv_pairs.push((key.to_vec(), val.to_vec()));
        self.max_key.clone_from(key);
    }

    fn consistency_check(&self) {
        for i in 0..self.kv_pairs.len() - 1 {
            assert!(cmp(&self.kv_pairs[i].0, &self.kv_pairs[i + 1].0) < 0);
        }
        assert!(cmp(&self.min_key, &self.kv_pairs[0].0) == 0);
        assert!(cmp(&self.max_key, &self.kv_pairs[self.kv_pairs.len() - 1].0) == 0);
    }

    fn gen_trace(&mut self, page_height: usize, key_len: usize, val_len: usize) -> Vec<Vec<u32>> {
        if let Some(t) = &self.trace {
            return t.clone();
        }
        let mut trace = Vec::new();
        for i in 0..self.kv_pairs.len() {
            let mut row = Vec::new();
            row.push(1);
            row.push(1);
            // row.push(1);
            for k in &self.kv_pairs[i].0 {
                row.push(*k);
            }
            for v in &self.kv_pairs[i].1 {
                row.push(*v);
            }
            trace.push(row);
        }
        trace.resize(page_height, vec![1]);
        for t in &mut trace {
            t.resize(2 + key_len + val_len, 0);
        }
        // println!("THIS IS THE TRACE OF A LEAF NODE: {:?}", trace.clone());
        self.trace = Some(trace.clone());
        trace
    }

    fn gen_all_trace(
        &mut self,
        page_height: usize,
        key_len: usize,
        val_len: usize,
        pages: &mut PageBTreePages,
    ) {
        pages
            .leaf_pages
            .push(self.gen_trace(page_height, key_len, val_len));
    }

    fn commit<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        page_height: usize,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        if self.trace.is_none() {
            self.gen_trace(page_height, key_len, val_len);
        }
        let commitment = committer.commit(vec![RowMajorMatrix::new(
            self.trace
                .clone()
                .unwrap()
                .into_iter()
                .flat_map(|row| row.into_iter().map(Val::<SC>::from_wrapped_u32))
                .collect(),
            2 + key_len + val_len,
        )]);
        let commit: [Val<SC>; COMMITMENT_LEN] = commitment.commit.into();
        let s = commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x.as_canonical_u32())
        });
        if Path::new(&("src/pagebtree/leaf/".to_owned() + &s + ".json")).is_file() {
            let file = File::create("src/pagebtree/leaf/".to_owned() + &s + ".json").unwrap();
            let mut writer = BufWriter::new(file);
            let _ = serde_json::to_writer(&mut writer, &self.trace.clone().unwrap());
            let _ = writer.flush();
        }
    }
}

impl<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
    PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>
{
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
        key: &Vec<u32>,
        val: &Vec<u32>,
        loaded_pages: &mut PageBTreePages,
    ) -> Option<(
        Vec<u32>,
        PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>,
    )> {
        match self {
            PageBTreeNode::Leaf(l) => l.update(key, val),
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
        committer: &mut TraceCommitter<SC>,
        leaf_page_height: usize,
        internal_page_height: usize,
        key_len: usize,
        val_len: usize,
    ) -> Vec<Vec<u32>>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.gen_trace(leaf_page_height, key_len, val_len),
            PageBTreeNode::Internal(i) => {
                i.gen_trace(committer, internal_page_height, key_len, val_len)
            }
            PageBTreeNode::Unloaded(_) => panic!(),
        }
    }

    fn gen_commit<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        page_height: usize,
        key_len: usize,
        val_len: usize,
    ) -> Vec<u32>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => {
                let trace = l.gen_trace(page_height, key_len, val_len);
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
                let trace = i.gen_trace(committer, page_height, key_len, val_len);
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
        committer: &mut TraceCommitter<SC>,
        leaf_page_height: usize,
        internal_page_height: usize,
        key_len: usize,
        val_len: usize,
        pages: &mut PageBTreePages,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.gen_all_trace(leaf_page_height, key_len, val_len, pages),
            PageBTreeNode::Internal(i) => i.gen_all_trace(
                committer,
                leaf_page_height,
                internal_page_height,
                key_len,
                val_len,
                pages,
            ),
            PageBTreeNode::Unloaded(_) => (),
        }
    }

    fn commit_all<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        leaf_page_height: usize,
        internal_page_height: usize,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        match self {
            PageBTreeNode::Leaf(l) => l.commit(committer, leaf_page_height, key_len, val_len),
            PageBTreeNode::Internal(i) => i.commit_all(
                committer,
                leaf_page_height,
                internal_page_height,
                key_len,
                val_len,
            ),
            PageBTreeNode::Unloaded(_) => (),
        }
    }
}

impl<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
    PageBTreeInternalNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>
{
    fn new(
        keys: Vec<Vec<u32>>,
        children: Vec<PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>>,
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
            trace: None,
        }
    }

    fn search(&mut self, key: &Vec<u32>, loaded_pages: &mut PageBTreePages) -> Option<Vec<u32>> {
        for (i, k) in self.keys.iter().enumerate() {
            let c = cmp(k, key);
            if c > 0 {
                if let PageBTreeNode::Unloaded(u) = &self.children[i] {
                    self.children[i] = u.load(key.len(), loaded_pages).unwrap();
                }
                return self.children[i].search(key, loaded_pages);
            }
        }
        let last_idx = self.keys.len();
        if let PageBTreeNode::Unloaded(u) = &self.children[last_idx] {
            self.children[last_idx] = u.load(key.len(), loaded_pages).unwrap();
        }
        self.children[self.keys.len()].search(key, loaded_pages)
    }

    fn update(
        &mut self,
        key: &Vec<u32>,
        val: &Vec<u32>,
        loaded_pages: &mut PageBTreePages,
    ) -> Option<(
        Vec<u32>,
        PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>,
    )> {
        self.trace = None;
        for (i, k) in self.keys.iter().enumerate() {
            let c = cmp(k, key);
            if c > 0 {
                let mut ret = None;
                if let PageBTreeNode::Unloaded(u) = &self.children[i] {
                    self.children[i] = u.load(key.len(), loaded_pages).unwrap();
                }
                if let Some((k, node)) = self.children[i].update(key, val, loaded_pages) {
                    ret = self.add_key(&k, node, i + 1);
                };
                self.min_key = self.children[0].min_key();
                self.max_key = self.children[self.children.len() - 1].max_key();
                return ret;
            }
        }
        let mut ret = None;
        let last_idx = self.children.len() - 1;
        if let PageBTreeNode::Unloaded(u) = &self.children[last_idx] {
            self.children[last_idx] = u.load(key.len(), loaded_pages).unwrap();
        }
        if let Some((k, node)) = self.children[last_idx].update(key, val, loaded_pages) {
            ret = self.add_key(&k, node, last_idx + 1);
        };
        self.min_key = self.children[0].min_key();
        self.max_key = self.children[self.children.len() - 1].max_key();
        ret
    }

    fn add_key(
        &mut self,
        key: &[u32],
        node: PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>,
        idx: usize,
    ) -> Option<(
        Vec<u32>,
        PageBTreeNode<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>,
    )> {
        if self.children.len() == MAX_INTERNAL {
            let mut new_children = vec![];
            self.keys.insert(idx - 1, key.to_vec());
            self.children.insert(idx, node);
            let mididx = MAX_INTERNAL / 2;
            for _ in mididx + 1..MAX_INTERNAL + 1 {
                let last_node = self.children.pop().unwrap();
                new_children.push(last_node);
            }
            new_children.reverse();
            let l2 = Self::new(self.keys[mididx + 1..MAX_INTERNAL].to_vec(), new_children);
            let mid = self.keys[mididx].clone();
            self.keys = self.keys[0..mididx].to_vec();
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
        committer: &mut TraceCommitter<SC>,
        page_height: usize,
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
            row.push(0);
            row.push(1);
            for k in self.children[i].min_key() {
                row.push(k);
            }
            for v in self.children[i].max_key() {
                row.push(v);
            }
            let child_commit =
                self.children[i].gen_commit(committer, page_height, key_len, val_len);
            row.extend(child_commit.clone());
            trace.push(row);
        }
        trace.resize(page_height, vec![]);
        for t in &mut trace {
            t.resize(2 + 2 * key_len + COMMITMENT_LEN, 0);
        }
        // println!("THIS IS THE TRACE OF AN INTERNAL NODE: {:?}", trace);
        self.trace = Some(trace.clone());
        trace
    }

    fn gen_all_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        leaf_page_height: usize,
        internal_page_height: usize,
        key_len: usize,
        val_len: usize,
        pages: &mut PageBTreePages,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        for i in 0..self.children.len() {
            self.children[i].gen_all_trace(
                committer,
                leaf_page_height,
                internal_page_height,
                key_len,
                val_len,
                pages,
            );
        }
        pages.internal_pages.push(self.gen_trace(
            committer,
            internal_page_height,
            key_len,
            val_len,
        ));
    }

    fn commit<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        page_height: usize,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        if self.trace.is_none() {
            self.gen_trace(committer, page_height, key_len, val_len);
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
        let s = commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x.as_canonical_u32())
        });
        if !Path::new(&("src/pagebtree/internal/".to_owned() + &s + ".json")).is_file() {
            let s: String = "src/pagebtree/internal/".to_owned() + &s + ".json";
            let path = PathBuf::from(&s);
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            let mut writer = BufWriter::new(file);
            let _ = serde_json::to_writer(&mut writer, &self.trace.clone().unwrap());
            let _ = writer.flush();
        }
    }

    fn commit_all<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
        leaf_page_height: usize,
        internal_page_height: usize,
        key_len: usize,
        val_len: usize,
    ) where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        self.commit(committer, internal_page_height, key_len, val_len);
        for child in &mut self.children {
            child.commit_all(
                committer,
                leaf_page_height,
                internal_page_height,
                key_len,
                val_len,
            );
        }
    }
}

impl<const MAX_INTERNAL: usize, const MAX_LEAF: usize, const COMMITMENT_LEN: usize>
    PageBTree<MAX_INTERNAL, MAX_LEAF, COMMITMENT_LEN>
{
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
            trace: None,
        };
        let leaf = PageBTreeNode::Leaf(leaf);
        let tree = PageBTree {
            limb_bits,
            key_len,
            val_len,
            root: vec![leaf],
            loaded_pages: PageBTreePages::new(),
            leaf_page_height,
            internal_page_height,
            depth: 1,
        };
        assert!(internal_page_height >= MAX_INTERNAL);
        assert!(leaf_page_height >= MAX_LEAF);
        tree
    }
    pub fn load(root_commit: Vec<u32>) -> Option<Self> {
        let s = root_commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x)
        });
        let file = match File::open("src/pagebtree/root/".to_owned() + &s + ".json") {
            Err(_) => return None,
            Ok(file) => file,
        };
        let mut reader = BufReader::new(file);
        let info: PageBTreeRootInfo = serde_json::from_reader(&mut reader).unwrap();
        assert!(info.commitment_len == COMMITMENT_LEN);
        assert!(info.max_internal == MAX_INTERNAL);
        assert!(info.max_leaf == MAX_LEAF);
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
            root: vec![root],
            loaded_pages: PageBTreePages::new(),
            depth: info.depth,
        })
    }
    pub fn min_key(&self) -> Vec<u32> {
        self.root[0].min_key()
    }
    pub fn max_key(&self) -> Vec<u32> {
        self.root[0].max_key()
    }
    pub fn search(&mut self, key: &Vec<u32>) -> Option<Vec<u32>> {
        for k in key {
            assert!(*k < 1 << self.limb_bits);
        }
        assert!(key.len() == self.key_len);
        if let PageBTreeNode::Unloaded(u) = &self.root[0] {
            self.root[0] = u.load(key.len(), &mut self.loaded_pages).unwrap();
        }
        self.root[0].search(key, &mut self.loaded_pages)
    }
    pub fn update(&mut self, key: &Vec<u32>, val: &Vec<u32>) {
        for k in key {
            assert!(*k < 1 << self.limb_bits);
        }
        assert!(key.len() == self.key_len);
        assert!(val.len() == self.val_len);
        if let PageBTreeNode::Unloaded(u) = &self.root[0] {
            self.root[0] = u.load(key.len(), &mut self.loaded_pages).unwrap();
        }
        let ret = self.root[0].update(key, val, &mut self.loaded_pages);
        if let Some((k, node)) = ret {
            let root = self.root.pop().unwrap();
            let min_key = root.min_key();
            let max_key = node.max_key();
            let internal = PageBTreeInternalNode {
                keys: vec![k],
                children: vec![root, node],
                min_key,
                max_key,
                trace: None,
            };
            self.depth += 1;
            self.root.push(PageBTreeNode::Internal(internal));
        }
    }
    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn consistency_check(&self) {
        self.root[0].consistency_check()
    }

    pub fn page_min_width(&self) -> usize {
        self.key_len + self.val_len + 1
    }

    pub fn gen_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
    ) -> Vec<Vec<u32>>
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        self.root[0].gen_trace(
            committer,
            self.leaf_page_height,
            self.internal_page_height,
            self.key_len,
            self.val_len,
        )
    }

    pub fn gen_all_trace<SC: StarkGenericConfig>(
        &mut self,
        committer: &mut TraceCommitter<SC>,
    ) -> PageBTreePages
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        let mut pages = PageBTreePages::new();
        self.root[0].gen_all_trace(
            committer,
            self.leaf_page_height,
            self.internal_page_height,
            self.key_len,
            self.val_len,
            &mut pages,
        );
        pages.leaf_pages.reverse();
        pages.internal_pages.reverse();
        pages
    }

    pub fn gen_loaded_trace(&self) -> PageBTreePages {
        self.loaded_pages.clone()
    }

    pub fn commit<SC: StarkGenericConfig>(&mut self, committer: &mut TraceCommitter<SC>)
    where
        Val<SC>: PrimeField32 + AbstractField,
        Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
    {
        let root_trace = self.root[0].gen_trace(
            committer,
            self.leaf_page_height,
            self.internal_page_height,
            self.key_len,
            self.val_len,
        );
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
        let s = commit.iter().fold("".to_owned(), |acc, x| {
            acc.to_owned() + &format!("{:08x}", x)
        });
        let file = File::create("src/pagebtree/root/".to_owned() + &s + ".json").unwrap();
        let root_info = PageBTreeRootInfo {
            max_internal: MAX_INTERNAL,
            max_leaf: MAX_LEAF,
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
        let _ = serde_json::to_writer(&mut writer, &root_info);
        let _ = writer.flush();
        self.root[0].commit_all(
            committer,
            self.leaf_page_height,
            self.internal_page_height,
            self.key_len,
            self.val_len,
        );
    }
}

fn cmp(key1: &[u32], key2: &[u32]) -> i32 {
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
