use std::collections::BTreeMap;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct CycleTrackerData {
    pub cpu_rows: usize,
    pub clock_cycles: usize,
    pub time_elapsed: usize,
    pub vm_metrics: BTreeMap<String, usize>,
}

#[derive(Debug, Clone)]
pub struct CycleTrackerSpan {
    pub is_active: bool,
    pub start: CycleTrackerData,
    pub end: CycleTrackerData,
}

impl CycleTrackerSpan {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        start_cpu_rows: usize,
        start_clock_cycle: usize,
        start_timestamp: usize,
        vm_metrics: &BTreeMap<String, usize>,
    ) -> Self {
        let vm_metrics_zero = vm_metrics.iter().map(|(k, _)| (k.clone(), 0)).collect();
        Self {
            is_active: true,
            start: CycleTrackerData {
                cpu_rows: start_cpu_rows,
                clock_cycles: start_clock_cycle,
                time_elapsed: start_timestamp,
                vm_metrics: vm_metrics.clone(),
            },
            end: CycleTrackerData {
                cpu_rows: 0,
                clock_cycles: 0,
                time_elapsed: 0,
                vm_metrics: vm_metrics_zero,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(
        &mut self,
        end_cpu_rows: usize,
        end_clock_cycle: usize,
        end_timestamp: usize,
        vm_metrics: &BTreeMap<String, usize>,
    ) {
        self.is_active = false;
        self.end.cpu_rows = end_cpu_rows - self.start.cpu_rows;
        self.end.clock_cycles = end_clock_cycle - self.start.clock_cycles;
        self.end.time_elapsed = end_timestamp - self.start.time_elapsed;
        for (key, value) in vm_metrics {
            let diff = value - self.start.vm_metrics.get(key).unwrap();
            self.end.vm_metrics.insert(key.clone(), diff);
        }
    }
}

impl Display for CycleTrackerSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  - cpu_rows: {}", self.end.cpu_rows)?;
        writeln!(f, "  - clock_cycles: {}", self.end.clock_cycles)?;
        writeln!(f, "  - time_elapsed: {}", self.end.time_elapsed)?;
        for (key, value) in &self.end.vm_metrics {
            writeln!(f, "  - {}: {}", key, value)?;
        }
        Ok(())
    }
}
