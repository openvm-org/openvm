use afs_chips::{
    common::{page::Page, page_cols::PageCols},
    inner_join::controller::{T2Format, TableFormat},
};
use afs_test_utils::{
    config::{
        self,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    utils::create_seeded_rng,
};
use rand::Rng;

use crate::{
    common::provider::{DataProvider, PageDataLoader},
    dataframe::{DataFrame, DataFrameType, IndexRange},
};

use super::TableJoinController;

const COMMIT_LEN: usize = 8;

#[test]
pub fn one_page_table_inner_join_test() {
    let decomp = 4;
    let log_page_height = 3;
    let num_parent_pages = 3;
    let num_child_pages = 4;

    let engine = config::baby_bear_poseidon2::default_engine(decomp.max(log_page_height + 1));

    let (parent_page_format, child_page_format, parent_df, child_df, mut page_loader) =
        generate_two_tables(log_page_height, num_parent_pages, num_child_pages, &engine);

    let mut output_df = DataFrame::new(vec![], DataFrameType::Unindexed);

    let mut ij_table_controller = TableJoinController::new(
        parent_df,
        child_df,
        &parent_page_format,
        &child_page_format,
        decomp,
    );

    ij_table_controller.generate_trace(&engine, &mut page_loader);
    ij_table_controller.set_up_keygen_builder(&engine);
    ij_table_controller.prove(&engine);
    ij_table_controller.verify(&engine, &mut output_df);
}

fn generate_two_tables(
    log_page_height: usize,
    num_parent_pages: usize,
    num_child_pages: usize,
    engine: &BabyBearPoseidon2Engine,
) -> (
    TableFormat,
    T2Format,
    DataFrame<COMMIT_LEN>,
    DataFrame<COMMIT_LEN>,
    PageDataLoader<BabyBearPoseidon2Config, COMMIT_LEN>,
) {
    let mut rng = create_seeded_rng();

    const MAX_VAL: u32 = 0x78000001 / 2; // The prime used by BabyBear / 2

    let parent_idx_len = rng.gen::<usize>() % 2 + 2;
    let parent_data_len = rng.gen::<usize>() % 2 + 2;

    let child_idx_len = rng.gen::<usize>() % 2 + 2;
    let child_data_len = rng.gen::<usize>() % 2 + parent_idx_len;

    let page_height = 1 << log_page_height;

    let fkey_start = rng.gen::<usize>() % (child_data_len - parent_idx_len + 1);
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

    // Now, sort the rows in child table by the foreign key
    let mut child_rows = child_table_as_page.iter().collect::<Vec<&PageCols<u32>>>();
    child_rows.sort_by(|a, b| a.data[fkey_start..fkey_end].cmp(&b.data[fkey_start..fkey_end]));
    let child_table_as_page = Page::from_page_cols(child_rows.into_iter().cloned().collect());

    let mut page_loader = PageDataLoader::empty();

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

        let cur_page_commit = page_loader.add_page(&cur_page, engine);
        let page_index_range = cur_page.get_index_range();
        parent_df.push_indexed_page(
            cur_page_commit.clone(),
            IndexRange::new(page_index_range.0, page_index_range.1),
        );
    }

    // Building the child table Indexed DataFrame with ranges coming from the foreign key
    let mut child_df = DataFrame::empty_indexed();
    for page_idx in 0..num_child_pages {
        let mut cur_page_cols = child_table_as_page
            .iter()
            .skip(page_idx * page_height)
            .take(page_height)
            .cloned()
            .collect::<Vec<PageCols<u32>>>();

        // Every individual child page should be sorted by index
        cur_page_cols.sort_by(|a, b| a.idx.cmp(&b.idx));

        let cur_page = Page::from_page_cols(cur_page_cols);

        let cur_page_commit = page_loader.add_page(&cur_page, engine);
        let page_index_range = fkey_page_range(&cur_page, fkey_start, fkey_end);
        child_df.push_indexed_page(cur_page_commit.clone(), page_index_range);
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
            smallest_fkey.clone_from(&cur_fkey);
        }
        if cur_fkey > largest_fkey {
            largest_fkey = cur_fkey;
        }
    }

    IndexRange::new(smallest_fkey, largest_fkey)
}
