use afs_chips::{
    common::page::Page,
    inner_join::controller::{T2Format, TableFormat},
};
use afs_test_utils::{config, utils::create_seeded_rng};
use p3_uni_stark::StarkGenericConfig;
use rand::Rng;

use crate::{
    common::{
        hash_struct,
        provider::{BTreeMapPageLoader, PageProvider},
        Commitment,
    },
    dataframe::{DataFrame, DataFrameType, IndexRange},
};

use super::TableJoinController;

#[test]
pub fn one_page_table_inner_join_test() {
    let decomp = 4;
    let log_page_height = 3;
    let num_parent_pages = 4;
    let num_child_pages = 4;

    let (parent_page_format, child_page_format, parent_df, child_df, mut page_loader) =
        generate_two_tables(log_page_height, num_parent_pages, num_child_pages);

    let mut output_df = DataFrame::new(vec![], DataFrameType::Unindexed);

    let engine = config::baby_bear_poseidon2::default_engine(decomp.max(log_page_height + 1));

    let mut ij_table_controller = TableJoinController::new(
        parent_df,
        child_df,
        &parent_page_format,
        &child_page_format,
        decomp,
        &engine,
    );

    ij_table_controller.generate_trace(&engine, &mut page_loader);
    ij_table_controller.set_up_keygen_builder(&engine);
    ij_table_controller.prove(&engine);
    ij_table_controller.verify(&engine, &mut output_df);
}

fn generate_two_tables<SC: StarkGenericConfig>(
    log_page_height: usize,
    num_parent_pages: usize,
    num_child_pages: usize,
) -> (
    TableFormat,
    T2Format,
    DataFrame<8>,
    DataFrame<8>,
    BTreeMapPageLoader<SC, 8>,
) {
    let mut rng = create_seeded_rng();

    const COMMIT_LEN: usize = 8;
    const MAX_VAL: u32 = 0x78000001 / 2; // The prime used by BabyBear / 2

    let parent_idx_len = rng.gen::<usize>() % 2 + 2;
    let parent_data_len = rng.gen::<usize>() % 2 + 2;

    let child_idx_len = rng.gen::<usize>() % 2 + 2;
    let child_data_len = rng.gen::<usize>() % 2 + parent_idx_len;

    let page_height = 1 << log_page_height;

    let fkey_start = rng.gen::<usize>() % (child_data_len - parent_idx_len);
    let fkey_end = fkey_start + parent_idx_len;

    let idx_limb_bits = 10;
    let max_idx = 1 << idx_limb_bits;

    let parent_page_format = TableFormat::new(parent_idx_len, parent_data_len, idx_limb_bits);
    let child_page_format = T2Format::new(
        TableFormat::new(child_idx_len, child_data_len, idx_limb_bits),
        fkey_start,
        fkey_end,
    );

    let parent_table_as_page = Page::random(
        &mut rng,
        parent_idx_len,
        parent_data_len,
        max_idx,
        MAX_VAL,
        page_height * num_parent_pages,
        page_height * num_parent_pages,
    );

    let mut child_table_as_page = Page::random(
        &mut rng,
        child_idx_len,
        child_data_len,
        max_idx,
        MAX_VAL,
        page_height * num_child_pages,
        page_height * num_child_pages,
    );

    // Assigning foreign key in t2 rows
    for row in child_table_as_page.iter_mut() {
        row.data[fkey_start..fkey_end]
            .clone_from_slice(&parent_table_as_page.get_random_idx(&mut rng));
    }

    let mut page_loader = BTreeMapPageLoader::new(Commitment::<COMMIT_LEN>::default());

    // Building the parent table Indexed DataFrame with ranges coming from idx
    let mut parent_df = DataFrame::empty_indexed();
    for page_idx in 0..num_parent_pages {
        let cur_page = Page::from_2d_vec(
            &parent_table_as_page
                .iter()
                .skip(page_idx * page_height)
                .take(page_height)
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<u32>>>(),
            parent_idx_len,
            parent_data_len,
        );

        let cur_page_commit = hash_struct(&cur_page); // TODO: replace this
        let page_index_range = cur_page.get_index_range();
        parent_df.push_indexed_page(
            cur_page_commit.clone(),
            IndexRange::new(page_index_range.0, page_index_range.1),
        );
        page_loader.add_page_with_commitment(&cur_page_commit, &cur_page);
    }

    // Building the child table Indexed DataFrame with ranges coming from the foreign key
    let mut child_df = DataFrame::empty_indexed();
    for page_idx in 0..num_child_pages {
        let cur_page = Page::from_2d_vec(
            &child_table_as_page
                .iter()
                .skip(page_idx * page_height)
                .take(page_height)
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<u32>>>(),
            child_idx_len,
            child_data_len,
        );

        let cur_page_commit = hash_struct(&cur_page); // TODO: replace this
        let page_index_range = fkey_page_range(&cur_page, fkey_start, fkey_end);
        child_df.push_indexed_page(cur_page_commit.clone(), page_index_range);
        page_loader.add_page_with_commitment(&cur_page_commit, &cur_page);
    }

    (
        parent_page_format,
        child_page_format,
        parent_df,
        child_df,
        page_loader,
    )
}

fn fkey_page_range(page: &Page, fkey_start: usize, fkey_end: usize) -> IndexRange {
    if page[0].is_alloc == 0 {
        return IndexRange::new(
            vec![0; fkey_end - fkey_start],
            vec![0; fkey_end - fkey_start],
        );
    }

    let mut smallest_fkey = page[0].data[fkey_start..fkey_end].to_vec();
    let mut largest_fkey = smallest_fkey.clone();
    for row in page.iter() {
        let cur_fkey = row.data[fkey_start..fkey_end].to_vec();
        if cur_fkey < smallest_fkey {
            smallest_fkey = cur_fkey.clone();
        }
        if cur_fkey > largest_fkey {
            largest_fkey = cur_fkey;
        }
    }

    IndexRange::new(smallest_fkey, largest_fkey)
}
