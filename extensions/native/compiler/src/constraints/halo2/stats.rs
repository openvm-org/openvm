#[derive(Default, Clone)]
pub(crate) struct Halo2Stats {
    pub total_gate_cell: usize,
    pub total_fixed: usize,
    pub total_lookup_cell: usize,
}

impl Halo2Stats {
    #[allow(dead_code)]
    pub fn add_assign(&mut self, b: &Self) {
        self.total_gate_cell += b.total_gate_cell;
        self.total_fixed += b.total_fixed;
        self.total_lookup_cell += b.total_lookup_cell;
    }
}

#[cfg(feature = "metrics")]
mod emit {
    use metrics::counter;

    use super::Halo2Stats;

    impl Halo2Stats {
        pub fn diff(&mut self, another: &Self) {
            *self = Self {
                total_gate_cell: self.total_gate_cell - another.total_gate_cell,
                total_fixed: self.total_fixed - another.total_fixed,
                total_lookup_cell: self.total_lookup_cell - another.total_lookup_cell,
            };
        }
        pub fn increment(&self, span_name: String) {
            let labels = [("cell_tracker_span", span_name)];
            counter!("simple_advice_cells", &labels).increment(self.total_gate_cell as u64);
            counter!("fixed_cells", &labels).increment(self.total_fixed as u64);
            counter!("lookup_advice_cells", &labels).increment(self.total_lookup_cell as u64);
        }
    }
}
