mod normal;
mod segmentation;

pub use normal::TracegenExecutionControl;
pub use segmentation::TracegenExecutionControlWithSegmentation;

pub struct TracegenCtx {
    pub since_last_segment_check: usize,
}
