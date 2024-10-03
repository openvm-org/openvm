use stark_vm::vm::cycle_tracker::{CanDiff, CycleTracker};
use tracing::info;

pub(crate) type Halo2CellTracker = CycleTracker<Halo2Stats>;

#[derive(Default, Clone)]
pub(crate) struct Halo2Stats {
    pub total_gate_cell: usize,
    pub total_fixed: usize,
    pub total_lookup_cell: usize,
}

impl Halo2Stats {
    pub fn add_assign(&mut self, b: &Self) {
        self.total_gate_cell += b.total_gate_cell;
        self.total_fixed += b.total_fixed;
        self.total_lookup_cell += b.total_lookup_cell;
    }
}

impl CanDiff for Halo2Stats {
    fn diff(&mut self, another: &Self) {
        *self = Self {
            total_gate_cell: self.total_gate_cell - another.total_gate_cell,
            total_fixed: self.total_fixed - another.total_fixed,
            total_lookup_cell: self.total_lookup_cell - another.total_lookup_cell,
        };
    }
}

pub(crate) fn print(
    cell_tracker: &Halo2CellTracker,
    babybear_stats: &Halo2Stats,
    num2bits_metrics: &Halo2Stats,
) {
    if cell_tracker.instances.is_empty() {
        return;
    }
    for name in &cell_tracker.order {
        let spans = cell_tracker.instances.get(name).unwrap();
        let num_spans = spans.len();

        if num_spans == 0 {
            continue;
        }

        let agg_stats = spans.iter().fold(Halo2Stats::default(), |mut total, span| {
            total.add_assign(&span.metrics);
            total
        });

        info!("span [{}] ({}):", name, num_spans);
        info!("  - total_gate_cell: {}", agg_stats.total_gate_cell);
        info!("  - total_fixed: {}", agg_stats.total_fixed);
        info!("  - total_lookup_cell: {}", agg_stats.total_lookup_cell);
    }
    info!("Babybear:");
    info!("  - total_gate_cell: {}", babybear_stats.total_gate_cell);
    info!("  - total_fixed: {}", babybear_stats.total_fixed);
    info!(
        "  - total_lookup_cell: {}",
        babybear_stats.total_lookup_cell
    );
    info!("Num2Bits:");
    info!("  - total_gate_cell: {}", num2bits_metrics.total_gate_cell);
    info!("  - total_fixed: {}", num2bits_metrics.total_fixed);
    info!(
        "  - total_lookup_cell: {}",
        num2bits_metrics.total_lookup_cell
    );
}
