use afs_stark_backend::config::Com;
use p3_field::{AbstractField, PrimeField64};
use std::collections::BTreeMap;

use p3_uni_stark::{StarkGenericConfig, Val};

use crate::page_btree::cmp;

#[derive(Clone)]
/// A pointer to a page within the given set of pages - (is_leaf, idx)
pub(crate) struct Node(bool, usize);

pub(crate) struct PageTreeGraph<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub _root: Node,
    pub mults: Vec<Vec<u32>>,
    pub child_ids: Vec<Vec<u32>>,
    pub leaf_ranges: Vec<(Vec<u32>, Vec<u32>)>,
    pub internal_ranges: Vec<(Vec<u32>, Vec<u32>)>,
    pub root_range: (Vec<u32>, Vec<u32>),
    pub root_mult: u32,
    pub _commitment_to_node: BTreeMap<Vec<Val<SC>>, Node>,
    pub mega_page: Vec<Vec<u32>>,
}

impl<SC: StarkGenericConfig, const COMMITMENT_LEN: usize> PageTreeGraph<SC, COMMITMENT_LEN>
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn dfs(
        leaf_pages: &[Vec<Vec<u32>>],
        internal_pages: &[Vec<Vec<u32>>],
        leaf_ranges: &mut Vec<(Vec<u32>, Vec<u32>)>,
        internal_ranges: &mut Vec<(Vec<u32>, Vec<u32>)>,
        commitment_to_node: &BTreeMap<Vec<Val<SC>>, Node>,
        mega_page: &mut Vec<Vec<u32>>,
        mults: &mut Vec<Vec<u32>>,
        child_ids: &mut Vec<Vec<u32>>,
        idx_len: usize,
        cur_node: Node,
    ) -> u32 {
        if !cur_node.0 {
            let mut ans = 0;
            let mut mult = vec![];
            let mut child_id = vec![];
            for row in &internal_pages[cur_node.1] {
                if row[1] == 0 {
                    mult.push(0);
                    child_id.push(0);
                } else {
                    let f_row: Vec<Val<SC>> = row
                        .iter()
                        .map(|r| Val::<SC>::from_canonical_u32(*r))
                        .collect();
                    let next_node = commitment_to_node
                        .get(&f_row[2 + 2 * idx_len..2 + 2 * idx_len + COMMITMENT_LEN].to_vec());
                    match next_node {
                        None => {
                            mult.push(1);
                            child_id.push(0);
                            ans += 1;
                        }
                        Some(n) => {
                            if n.0 {
                                leaf_ranges[n.1].0 = row[2..2 + idx_len].to_vec();
                                leaf_ranges[n.1].1 = row[2 + idx_len..2 + 2 * idx_len].to_vec();
                            } else {
                                internal_ranges[n.1].0 = row[2..2 + idx_len].to_vec();
                                internal_ranges[n.1].1 = row[2 + idx_len..2 + 2 * idx_len].to_vec();
                            }
                            let m = PageTreeGraph::<SC, COMMITMENT_LEN>::dfs(
                                leaf_pages,
                                internal_pages,
                                leaf_ranges,
                                internal_ranges,
                                commitment_to_node,
                                mega_page,
                                mults,
                                child_ids,
                                idx_len,
                                n.clone(),
                            );
                            ans += m;
                            mult.push(m);
                            child_id.push(n.1 as u32);
                        }
                    }
                }
            }
            mults[cur_node.1] = mult;
            child_ids[cur_node.1] = child_id;
            ans + 1
        } else {
            let mut ans = 0;
            for row in &leaf_pages[cur_node.1] {
                if row[0] == 1 {
                    mega_page.push(row.clone());
                    ans += 1;
                }
            }
            ans + 1
        }
    }

    pub fn new(
        leaf_pages: &[Vec<Vec<u32>>],
        internal_pages: &[Vec<Vec<u32>>],
        leaf_commits: &[Com<SC>],
        internal_commits: &[Com<SC>],
        root: (bool, usize),
        idx_len: usize,
    ) -> Self {
        let root = Node(root.0, root.1);
        let leaf_commits: Vec<[Val<SC>; COMMITMENT_LEN]> = leaf_commits
            .iter()
            .map(|c| {
                let c: [Val<SC>; COMMITMENT_LEN] = c.clone().into();
                c
            })
            .collect();
        let internal_commits: Vec<[Val<SC>; COMMITMENT_LEN]> = internal_commits
            .iter()
            .map(|c| {
                let c: [Val<SC>; COMMITMENT_LEN] = c.clone().into();
                c
            })
            .collect();
        let mut commitment_to_node = BTreeMap::<Vec<Val<SC>>, Node>::new();
        let root_commitment = if root.0 {
            leaf_commits[root.1]
        } else {
            internal_commits[root.1]
        };
        let mut leaf_ranges = vec![(vec![], vec![]); leaf_pages.len()];
        let mut internal_ranges = vec![(vec![], vec![]); internal_pages.len()];
        if root.0 {
            for row in &leaf_pages[root.1] {
                if row[0] == 1 {
                    if leaf_ranges[root.1].0.is_empty() {
                        leaf_ranges[root.1] =
                            (row[1..1 + idx_len].to_vec(), row[1..1 + idx_len].to_vec());
                    } else {
                        let idx = row[1..1 + idx_len].to_vec();
                        if cmp(&leaf_ranges[root.1].0, &idx) > 0 {
                            leaf_ranges[root.1].0.clone_from(&idx);
                        }
                        if cmp(&leaf_ranges[root.1].1, &idx) < 0 {
                            leaf_ranges[root.1].1 = idx;
                        }
                    }
                }
            }
        } else {
            for row in &internal_pages[root.1] {
                if row[1] == 1 {
                    let idx1 = row[2..2 + idx_len].to_vec();
                    let idx2 = row[2 + idx_len..2 + 2 * idx_len].to_vec();
                    if internal_ranges[root.1].0.is_empty() {
                        internal_ranges[root.1] = (idx1, idx2);
                    } else {
                        if cmp(&internal_ranges[root.1].0, &idx1) > 0 {
                            internal_ranges[root.1].0 = idx1;
                        }
                        if cmp(&internal_ranges[root.1].1, &idx2) < 0 {
                            internal_ranges[root.1].1 = idx2;
                        }
                    }
                }
            }
        }
        for (i, c) in leaf_commits.iter().enumerate() {
            commitment_to_node.insert(c.clone().to_vec(), Node(true, i));
        }
        for (i, c) in internal_commits.iter().enumerate() {
            commitment_to_node.insert(c.clone().to_vec(), Node(false, i));
        }
        let mut mults = vec![vec![0; internal_pages[0].len()]; internal_pages.len()];
        let mut child_ids = vec![vec![0; internal_pages[0].len()]; internal_pages.len()];
        commitment_to_node.insert(root_commitment.clone().to_vec(), root.clone());
        let mut mega_page = vec![];
        let root_mult = PageTreeGraph::<SC, COMMITMENT_LEN>::dfs(
            leaf_pages,
            internal_pages,
            &mut leaf_ranges,
            &mut internal_ranges,
            &commitment_to_node,
            &mut mega_page,
            &mut mults,
            &mut child_ids,
            idx_len,
            root.clone(),
        );
        for range in &mut leaf_ranges {
            if range.0.is_empty() {
                *range = (vec![0; idx_len], vec![0; idx_len]);
            }
        }
        for range in &mut internal_ranges {
            if range.0.is_empty() {
                *range = (vec![0; idx_len], vec![0; idx_len]);
            }
        }
        let root_range = if root.0 {
            leaf_ranges[root.1].clone()
        } else {
            internal_ranges[root.1].clone()
        };
        PageTreeGraph {
            _root: root,
            root_range,
            mults,
            child_ids,
            root_mult,
            _commitment_to_node: commitment_to_node,
            leaf_ranges,
            internal_ranges,
            mega_page,
        }
    }
}
