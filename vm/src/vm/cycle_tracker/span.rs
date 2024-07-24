#[derive(Debug)]
pub struct CycleTrackerData {
    pub cpu_rows: usize,
    pub clock_cycles: usize,
    pub time_elapsed: usize,
    pub mem_accesses: usize,
    pub field_arithmetic_ops: usize,
    pub field_extension_ops: usize,
    pub range_checker_count: usize,
    pub poseidon2_rows: usize,
    pub input_stream_len: usize,
}

#[derive(Debug)]
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
        start_mem_accesses: usize,
        start_field_arithmetic_ops: usize,
        start_field_extension_ops: usize,
        start_range_checker_count: usize,
        start_poseidon2_rows: usize,
        start_input_stream_len: usize,
    ) -> Self {
        Self {
            is_active: true,
            start: CycleTrackerData {
                cpu_rows: start_cpu_rows,
                clock_cycles: start_clock_cycle,
                time_elapsed: start_timestamp,
                mem_accesses: start_mem_accesses,
                field_arithmetic_ops: start_field_arithmetic_ops,
                field_extension_ops: start_field_extension_ops,
                range_checker_count: start_range_checker_count,
                poseidon2_rows: start_poseidon2_rows,
                input_stream_len: start_input_stream_len,
            },
            end: CycleTrackerData {
                cpu_rows: 0,
                clock_cycles: 0,
                time_elapsed: 0,
                mem_accesses: 0,
                field_arithmetic_ops: 0,
                field_extension_ops: 0,
                range_checker_count: 0,
                poseidon2_rows: 0,
                input_stream_len: 0,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn end(
        &mut self,
        end_cpu_rows: usize,
        end_clock_cycle: usize,
        end_timestamp: usize,
        end_mem_accesses: usize,
        end_field_arithmetic_ops: usize,
        end_field_extension_ops: usize,
        end_range_checker_count: usize,
        end_poseidon2_rows: usize,
        end_input_stream_len: usize,
    ) {
        self.is_active = false;
        self.end.cpu_rows = end_cpu_rows - self.start.cpu_rows;
        self.end.clock_cycles = end_clock_cycle - self.start.clock_cycles;
        self.end.time_elapsed = end_timestamp - self.start.time_elapsed;
        self.end.mem_accesses = end_mem_accesses - self.start.mem_accesses;
        self.end.field_arithmetic_ops = end_field_arithmetic_ops - self.start.field_arithmetic_ops;
        self.end.field_extension_ops = end_field_extension_ops - self.start.field_extension_ops;
        self.end.range_checker_count = end_range_checker_count - self.start.range_checker_count;
        self.end.poseidon2_rows = end_poseidon2_rows - self.start.poseidon2_rows;
        self.end.input_stream_len = end_input_stream_len - self.start.input_stream_len;
    }
}
