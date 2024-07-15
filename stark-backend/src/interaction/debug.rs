use std::collections::{BTreeMap, HashMap};

use itertools::Itertools;
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrixView, Matrix};

use super::{AirBridge, InteractionType};

/// The actual interactions that are sent/received during a single run
/// of trace generation. For debugging purposes only.
#[derive(Default, Clone, Debug)]
pub struct LogicalInteractions<F: Field> {
    /// Bus index => (fields => (air_idx, interaction_type, count))
    #[allow(clippy::type_complexity)]
    pub at_bus: BTreeMap<usize, HashMap<Vec<F>, Vec<(usize, InteractionType, F)>>>,
}

pub fn generate_logical_interactions<F, A>(
    air_idx: usize,
    air: &A,
    preprocessed: &Option<RowMajorMatrixView<F>>,
    partitioned_main: &[RowMajorMatrixView<F>],
    logical_interactions: &mut LogicalInteractions<F>,
) where
    F: Field,
    A: AirBridge<F> + ?Sized,
{
    let all_interactions = air.all_interactions();
    if all_interactions.is_empty() {
        return;
    }

    let height = partitioned_main[0].height();

    for n in 0..height {
        let preprocessed_row = preprocessed
            .as_ref()
            .map(|preprocessed| {
                // manual implementation of row_slice because of a drop issue
                &preprocessed.values[n * preprocessed.width..(n + 1) * preprocessed.width]
            })
            .unwrap_or(&[]);
        let main_row: Vec<F> = partitioned_main
            .iter()
            .flat_map(|main_part| main_part.row_slice(n).to_vec())
            .collect();
        for (interaction, interaction_type) in &all_interactions {
            let fields = interaction
                .fields
                .iter()
                .map(|columns| columns.apply::<F, F>(preprocessed_row, &main_row))
                .collect_vec();
            let count = interaction.count.apply::<F, F>(preprocessed_row, &main_row);
            if count.is_zero() {
                continue;
            }
            logical_interactions
                .at_bus
                .entry(interaction.argument_index)
                .or_default()
                .entry(fields)
                .or_default()
                .push((air_idx, *interaction_type, count));
        }
    }
}
