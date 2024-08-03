use std::{collections::BTreeMap, fmt::Display};

#[derive(Debug, Clone)]
pub struct CycleTrackerData {
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
    pub fn start(vm_metrics: &BTreeMap<String, usize>) -> Self {
        let vm_metrics_zero = vm_metrics.iter().map(|(k, _)| (k.clone(), 0)).collect();
        Self {
            is_active: true,
            start: CycleTrackerData {
                vm_metrics: vm_metrics.clone(),
            },
            end: CycleTrackerData {
                vm_metrics: vm_metrics_zero,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(&mut self, vm_metrics: &BTreeMap<String, usize>) {
        self.is_active = false;
        for (key, value) in vm_metrics {
            let diff = value - self.start.vm_metrics.get(key).unwrap();
            self.end.vm_metrics.insert(key.clone(), diff);
        }
    }
}

impl Display for CycleTrackerSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.end.vm_metrics {
            writeln!(f, "  - {}: {}", key, value)?;
        }
        Ok(())
    }
}
