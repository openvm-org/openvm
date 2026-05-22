use std::time::Instant;

#[derive(Clone, Copy)]
pub(crate) enum ExecutionMetric {
    Pure,
    Metered,
    MeteredCost,
}

impl ExecutionMetric {
    fn counter_name(self) -> &'static str {
        match self {
            Self::Pure => "execute_pure_insns",
            Self::Metered => "execute_metered_insns",
            Self::MeteredCost => "execute_metered_cost_insns",
        }
    }

    fn gauge_name(self) -> &'static str {
        match self {
            Self::Pure => "execute_pure_insn_mi/s",
            Self::Metered => "execute_metered_insn_mi/s",
            Self::MeteredCost => "execute_metered_cost_insn_mi/s",
        }
    }
}

pub(crate) struct ExecutionMetricTimer {
    counter_name: &'static str,
    gauge_name: &'static str,
    start: Instant,
}

impl ExecutionMetricTimer {
    pub(crate) fn start(metric: ExecutionMetric) -> Self {
        Self {
            counter_name: metric.counter_name(),
            gauge_name: metric.gauge_name(),
            start: Instant::now(),
        }
    }

    pub(crate) fn start_custom(counter_name: &'static str, gauge_name: &'static str) -> Self {
        Self {
            counter_name,
            gauge_name,
            start: Instant::now(),
        }
    }

    pub(crate) fn record(self, insns: u64) {
        let elapsed = self.start.elapsed();
        let elapsed_micros = elapsed.as_nanos().max(1) as f64 / 1_000.0;
        tracing::info!("instructions_executed={insns}");
        metrics::counter!(self.counter_name).absolute(insns);
        metrics::gauge!(self.gauge_name).set(insns as f64 / elapsed_micros);
    }
}
