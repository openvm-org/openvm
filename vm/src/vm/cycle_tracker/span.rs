use std::{collections::BTreeMap, fmt::Display};

#[derive(Debug, Clone)]
pub struct CycleTrackerData {
    pub vm_metrics: BTreeMap<String, usize>,
    pub opcode_counts: BTreeMap<String, usize>,
    pub dsl_counts: BTreeMap<String, usize>,
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
        vm_metrics: &BTreeMap<String, usize>,
        opcode_counts: &BTreeMap<String, usize>,
        dsl_counts: &BTreeMap<String, usize>,
    ) -> Self {
        let vm_metrics_zero = vm_metrics.iter().map(|(k, _)| (k.clone(), 0)).collect();
        let opcode_counts_zero = opcode_counts.iter().map(|(k, _)| (k.clone(), 0)).collect();
        let dsl_counts_zero = dsl_counts.iter().map(|(k, _)| (k.clone(), 0)).collect();
        Self {
            is_active: true,
            start: CycleTrackerData {
                vm_metrics: vm_metrics.clone(),
                opcode_counts: opcode_counts.clone(),
                dsl_counts: dsl_counts.clone(),
            },
            end: CycleTrackerData {
                vm_metrics: vm_metrics_zero,
                opcode_counts: opcode_counts_zero,
                dsl_counts: dsl_counts_zero,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(
        &mut self,
        vm_metrics: &BTreeMap<String, usize>,
        opcode_counts: &BTreeMap<String, usize>,
        dsl_counts: &BTreeMap<String, usize>,
    ) {
        self.is_active = false;
        for (key, value) in vm_metrics {
            let diff = value - self.start.vm_metrics.get(key).unwrap();
            self.end.vm_metrics.insert(key.clone(), diff);
        }
        for (key, value) in opcode_counts {
            let diff = value - self.start.opcode_counts.get(key).unwrap_or(&0);
            self.end.opcode_counts.insert(key.clone(), diff);
        }
        for (key, value) in dsl_counts {
            let diff = value - self.start.dsl_counts.get(key).unwrap_or(&0);
            self.end.dsl_counts.insert(key.clone(), diff);
        }
    }
}

impl Display for CycleTrackerSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.end.vm_metrics {
            writeln!(f, "  - {}: {}", key, value)?;
        }

        let mut sorted_opcode_counts: Vec<(&String, &usize)> =
            self.end.opcode_counts.iter().collect();
        sorted_opcode_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_opcode_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }

        let mut sorted_dsl_counts: Vec<(&String, &usize)> = self.end.dsl_counts.iter().collect();
        sorted_dsl_counts.sort_by(|a, b| a.1.cmp(b.1)); // Sort ascending by value

        for (key, value) in sorted_dsl_counts {
            if *value > 0 {
                writeln!(f, "  - {}: {}", key, value)?;
            }
        }
        Ok(())
    }
}
