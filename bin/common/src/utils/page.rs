use afs_page::common::page::Page;
use itertools::{izip, Itertools};
use logical_interface::utils::u16_vec_to_hex_string;

fn wrap_string(s: String, width: usize) -> Vec<String> {
    s.chars()
        .chunks(width)
        .into_iter()
        .map(|c| c.collect::<String>())
        .collect::<Vec<String>>()
}

pub fn print_page(p: &Page, idx_wrap_bytes: usize, data_wrap_bytes: usize, max_rows: usize) {
    let idx_wrap_chars = if idx_wrap_bytes == 0 {
        p.idx_len() * 4
    } else {
        idx_wrap_bytes * 2
    };
    let data_wrap_chars = if data_wrap_bytes == 0 {
        p.data_len() * 4
    } else {
        data_wrap_bytes * 2
    };
    for row in p.iter() {
        let idx = u16_vec_to_hex_string(row.idx.clone())
            .strip_prefix("0x")
            .unwrap()
            .to_string();
        let data = u16_vec_to_hex_string(row.data.clone())
            .strip_prefix("0x")
            .unwrap()
            .to_string();
        let mut idx_rows = if idx_wrap_chars == 0 {
            vec![idx]
        } else {
            wrap_string(idx, idx_wrap_chars)
        };
        let mut data_rows = if data_wrap_chars == 0 {
            vec![data]
        } else {
            wrap_string(data, data_wrap_chars)
        };
        let num_rows = std::cmp::max(idx_rows.len(), data_rows.len());
        let mut is_alloc_rows = vec![row.is_alloc];
        is_alloc_rows.resize(num_rows, 0);
        idx_rows.resize(num_rows, " ".repeat(idx_wrap_chars));
        data_rows.resize(num_rows, " ".repeat(data_wrap_chars));

        for (i, (is_alloc, idx, data)) in izip!(is_alloc_rows, idx_rows, data_rows).enumerate() {
            if i != 0 && i >= max_rows {
                break;
            } else if i == 0 {
                println!("{}|0x{}|0x{}", is_alloc, idx, data);
            } else {
                println!("{}|  {}|  {}", is_alloc, idx, data);
            }
        }
    }
    println!("Height: {}", p.height());
}

pub fn print_page_nowrap_upto(p: &Page, max_rows: usize) {
    print_page(p, 0, 0, max_rows);
}

pub fn print_page_nowrap(p: &Page) {
    print_page(p, 0, 0, 0);
}
