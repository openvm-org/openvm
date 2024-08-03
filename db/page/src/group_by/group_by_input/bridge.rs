use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::GroupByCols, GroupByAir};

// impl<F: PrimeField64> AirBridge<F> for GroupByAir {
impl GroupByAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: GroupByCols<AB::Var>,
    ) {
        // Sends desired columns (group_by and to_aggregate) from input page internally with count
        // `is_alloc`, and sends answer columns with count `is_final`.
        assert_eq!(local.aux.grouped.is_some(), !self.sorted);

        let group_by_col_indices = if let Some(grouped) = local.aux.grouped.clone() {
            grouped.group_by
        } else {
            self.group_by_cols
                .iter()
                .map(|&i| local.page.data[i])
                .collect()
        };
        let output_sent_fields = group_by_col_indices
            .into_iter()
            .chain(iter::once(local.aux.partial_aggregated))
            .collect::<Vec<_>>();
        let output_count = local.aux.is_final;
        builder.push_send(self.output_bus, output_sent_fields, output_count);

        if !self.sorted {
            // Must do internal grouping of page based on group_by columns
            // Sends from columns in input page on internal bus to grouped page
            let page = local.page;
            let internal_sent_fields = self
                .group_by_cols
                .iter()
                .chain(iter::once(&self.aggregated_col))
                .map(|&i| page.data[i])
                .collect::<Vec<_>>();
            let internal_count = page.is_alloc;

            builder.push_send(self.internal_bus, internal_sent_fields, internal_count);

            // Receives from columns in internal bus to grouped page
            let grouped = local.aux.grouped.clone().unwrap();
            let internal_received_fields = grouped
                .group_by
                .into_iter()
                .chain(iter::once(grouped.to_aggregate))
                .collect::<Vec<_>>();
            let internal_count = grouped.is_alloc;

            builder.push_receive(self.internal_bus, internal_received_fields, internal_count);
        }
    }
}
