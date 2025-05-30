mod normal;
mod segmentation;

pub use normal::TracegenExecutionControl;
pub use segmentation::TracegenExecutionControlWithSegmentation;

pub struct TracegenCtx<F> {
    pub trace_buffers: Vec<Vec<F>>,
    pub buffer_indices: Vec<usize>,
    // TODO(ayush): see if i need this here, can get from airs
    pub trace_widths: Vec<usize>,
    // TODO(ayush): for segmentation, remove from tracegen
    pub since_last_segment_check: usize,
}

impl<F> TracegenCtx<F> {
    pub fn new(trace_widths: Vec<usize>, trace_heights: Vec<usize>) -> Self {
        let trace_buffers = trace_widths
            .iter()
            .zip(trace_heights.iter())
            .map(|(&width, &height)| Vec::with_capacity(width * height))
            .collect();

        let buffer_indices = vec![0; trace_widths.len()];

        Self {
            trace_buffers,
            buffer_indices,
            trace_widths,
            since_last_segment_check: 0,
        }
    }
}
