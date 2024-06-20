use std::collections::BTreeMap;

use afs_stark_backend::prover::{trace::TraceCommitmentBuilder, MultiTraceStarkProver};
use afs_test_utils::{
    config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config},
    utils::create_seeded_rng,
};
use rand::Rng;

use super::PageBTree;

#[test]
pub fn make_new_tree() {
    let key_len = 1;
    let val_len = 1;
    let limb_bits = 20;
    PageBTree::<8>::new(limb_bits, key_len, val_len, 16, 16, "".to_owned());
}

#[test]
pub fn update_tree() {
    let key_len = 1;
    let val_len = 1;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 8, 8, "".to_owned());
    let mut rng = create_seeded_rng();
    let mut truth = BTreeMap::<u32, u32>::new();
    for _ in 0..10 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        tree.update(&vec![i], &vec![j]);
        truth.insert(i, j);
    }
    for _ in 0..10 {
        let i = rng.gen::<u32>() % 100;
        let my_ans = tree.search(&vec![i]);
        let real_ans = truth.get(&i);
        if my_ans == None {
            assert!(real_ans == None);
        } else {
            assert!(my_ans.unwrap()[0] == *real_ans.unwrap());
        }
    }
    println!("{:?}", tree);
}

#[test]
pub fn consistency_check() {
    let key_len = 1;
    let val_len = 1;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 8, 8, "".to_owned());
    let mut rng = create_seeded_rng();
    let mut truth = BTreeMap::<u32, u32>::new();
    for _ in 0..400 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        tree.update(&vec![i], &vec![j]);
        truth.insert(i, j);
    }
    tree.consistency_check()
}

#[test]
pub fn update_tree_key_len() {
    let key_len = 2;
    let val_len = 2;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 20, 20, "".to_owned());
    let mut rng = create_seeded_rng();
    let mut truth = BTreeMap::<Vec<u32>, Vec<u32>>::new();
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let k = rng.gen::<u32>() % 100;
        let l = rng.gen::<u32>() % 100;
        tree.update(&vec![i, j], &vec![k, l]);
        truth.insert(vec![i, j], vec![k, l]);
    }
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let my_ans = tree.search(&vec![i, j]);
        let real_ans = truth.get(&vec![i, j]);
        if my_ans == None {
            assert!(real_ans == None);
        } else {
            let my_ans = my_ans.unwrap();
            let real_ans = real_ans.unwrap();
            assert!(my_ans[0] == real_ans[0]);
            assert!(my_ans[1] == real_ans[1]);
        }
    }
    tree.consistency_check();
}

#[test]
pub fn benchmark_pagebtree() {
    let key_len = 2;
    let val_len = 2;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 20, 20, "".to_owned());
    let mut rng = create_seeded_rng();
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let k = rng.gen::<u32>() % 100;
        let l = rng.gen::<u32>() % 100;
        tree.update(&vec![i, j], &vec![k, l]);
    }
}

#[test]
pub fn benchmark_btreemap() {
    let mut truth = BTreeMap::<Vec<u32>, Vec<u32>>::new();
    let mut rng = create_seeded_rng();
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let k = rng.gen::<u32>() % 100;
        let l = rng.gen::<u32>() % 100;
        truth.insert(vec![i, j], vec![k, l]);
    }
}

#[test]
pub fn wide_tree() {
    let key_len = 2;
    let val_len = 2;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 32, 32, "".to_owned());
    let mut rng = create_seeded_rng();
    let mut truth = BTreeMap::<Vec<u32>, Vec<u32>>::new();
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let k = rng.gen::<u32>() % 100;
        let l = rng.gen::<u32>() % 100;
        tree.update(&vec![i, j], &vec![k, l]);
        truth.insert(vec![i, j], vec![k, l]);
    }
    for _ in 0..1000000 {
        let i = rng.gen::<u32>() % 100;
        let j = rng.gen::<u32>() % 100;
        let my_ans = tree.search(&vec![i, j]);
        let real_ans = truth.get(&vec![i, j]);
        if my_ans == None {
            assert!(real_ans == None);
        } else {
            let my_ans = my_ans.unwrap();
            let real_ans = real_ans.unwrap();
            assert!(my_ans[0] == real_ans[0]);
            assert!(my_ans[1] == real_ans[1]);
        }
    }
    println!("DEPTH IS: {:?}", tree.depth());
    tree.consistency_check();
}

#[test]
pub fn gen_trace_test() {
    let key_len = 2;
    let val_len = 2;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 8, 8, "".to_owned());
    let mut rng = create_seeded_rng();
    for _ in 0..100 {
        let i = rng.gen::<u32>() % 10;
        let j = rng.gen::<u32>() % 10;
        let k = rng.gen::<u32>() % 10;
        let l = rng.gen::<u32>() % 10;
        tree.update(&vec![i, j], &vec![k, l]);
    }
    let log_page_height = 3;
    let log_num_requests = 32;
    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(log_num_requests));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder: TraceCommitmentBuilder<BabyBearPoseidon2Config> =
        TraceCommitmentBuilder::new(prover.pcs());
    tree.gen_trace(&mut trace_builder.committer, None);
    tree.consistency_check();
}

#[test]
pub fn commit_test() {
    let key_len = 2;
    let val_len = 2;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 8, 8, "commit".to_owned());
    let mut rng = create_seeded_rng();
    for _ in 0..100 {
        let i = rng.gen::<u32>() % 10;
        let j = rng.gen::<u32>() % 10;
        let k = rng.gen::<u32>() % 10;
        let l = rng.gen::<u32>() % 10;
        tree.update(&vec![i, j], &vec![k, l]);
    }
    let log_page_height = 3;
    let log_num_requests = 32;
    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(log_num_requests));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder: TraceCommitmentBuilder<BabyBearPoseidon2Config> =
        TraceCommitmentBuilder::new(prover.pcs());
    tree.commit(&mut trace_builder.committer, "src/pagebtree".to_owned());
    tree.consistency_check();
}

#[test]
pub fn load_test() {
    let mut tree = PageBTree::<8>::load(
        "src/pagebtree".to_owned(),
        "commit".to_owned(),
        "".to_owned(),
    )
    .unwrap();
    let mut rng = create_seeded_rng();
    for _ in 0..100 {
        let i = rng.gen::<u32>() % 10;
        let j = rng.gen::<u32>() % 10;
        tree.search(&vec![i, j]);
    }
    println!("{:?}", tree);
    tree.consistency_check();
}

#[test]
pub fn make_a_large_tree() {
    let key_len = 2;
    let val_len = 3;
    let limb_bits = 20;
    let mut tree = PageBTree::<8>::new(limb_bits, key_len, val_len, 32, 32, "large".to_owned());
    let mut rng = create_seeded_rng();
    const BIG_TREE_SIZE: usize = 1000_000;
    const BIG_TREE_MAX_KEY: u32 = 1000_000;
    for i in 0..BIG_TREE_SIZE {
        if i % 10000 == 0 {
            println!("Processed {:?} entries...", i);
        }
        let i = rng.gen::<u32>() % BIG_TREE_MAX_KEY;
        let j = rng.gen::<u32>() % BIG_TREE_MAX_KEY;
        let k = rng.gen::<u32>() % 100;
        let l = rng.gen::<u32>() % 100;
        tree.update(&vec![i, j], &vec![k, l, l]);
    }
    let log_page_height = 3;
    let log_num_requests = 32;
    let engine = config::baby_bear_poseidon2::default_engine(log_page_height.max(log_num_requests));

    let prover = MultiTraceStarkProver::new(&engine.config);
    let mut trace_builder: TraceCommitmentBuilder<BabyBearPoseidon2Config> =
        TraceCommitmentBuilder::new(prover.pcs());
    tree.commit(&mut trace_builder.committer, "src/pagebtree".to_owned());
    // tree.consistency_check();
}

#[test]
pub fn load_and_read_a_large_tree() {
    let mut tree = PageBTree::<8>::load(
        "src/pagebtree".to_owned(),
        "large".to_owned(),
        "".to_owned(),
    )
    .unwrap();
    let mut rng = create_seeded_rng();
    for _ in 0..100 {
        let i = rng.gen::<u32>() % 10;
        let j = rng.gen::<u32>() % 10;
        println!("{:?}", tree.search(&vec![i, j]));
    }
    tree.consistency_check();
}
