use self::columns::PageReadCols;
use afs_stark_backend::interaction::Interaction;
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

pub mod air;
pub mod chip;
pub mod columns;
pub mod trace;

pub struct PageReadChip {
    bus_index: usize,

    val_len: usize,
    page_size: usize,
    page_data: Vec<Vec<u32>>,
}

impl PageReadChip {
    pub fn new(bus_index: usize, page: Vec<Vec<u32>>) -> Self {
        assert!(page.len() > 0);

        Self {
            bus_index,
            val_len: page[0].len(),
            page_size: page.len(),
            page_data: page,
        }
    }

    pub fn bus_index(&self) -> usize {
        self.bus_index
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn get_width(&self) -> usize {
        2 + self.val_len
    }

    // receives: ([index] | [page] ) mult times
    pub fn receives_custom<F: PrimeField64>(
        &self,
        cols: PageReadCols<usize>,
    ) -> Vec<Interaction<F>> {
        println!("here I'm in receives_custom");
        println!("index: {:?}", cols.index);
        println!("page_row: {:?}", cols.page_row);
        println!("mult: {:?}", cols.mult);

        let mut virtual_cols: Vec<VirtualPairCol<F>> =
            vec![VirtualPairCol::single_main(cols.index)];
        for page_col in cols.page_row {
            virtual_cols.push(VirtualPairCol::single_main(page_col));
        }

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols.mult),
            argument_index: self.bus_index(),
        }]
    }
}
