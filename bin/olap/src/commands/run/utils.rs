use afs_chips::common::page::Page;
use logical_interface::utils::u16_vec_to_hex_string;

pub fn pretty_print_page(p: &Page) {
    for row in &p.rows {
        println!(
            "{}|{}|{}",
            row.is_alloc,
            u16_vec_to_hex_string(row.idx.clone()),
            u16_vec_to_hex_string(row.data.clone())
        );
    }
}
