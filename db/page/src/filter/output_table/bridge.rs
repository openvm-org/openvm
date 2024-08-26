use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use itertools::Itertools;
use p3_air::Air;
use p3_matrix::Matrix;

use super::air::FilterOutputTableAir;

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for FilterOutputTableAir {
    fn eval(&self, builder: &mut AB) {
        // Making sure the page is in the proper format
        self.inner.eval(builder);

        let page = &builder.partitioned_main()[0];
        let page_local = page.row_slice(0);
        let page_blob = page_local.iter().skip(1).copied().collect_vec();
        let is_alloc = page_local[0];
        drop(page_local);

        builder.push_receive(self.page_bus_index, page_blob, is_alloc);
    }
}
