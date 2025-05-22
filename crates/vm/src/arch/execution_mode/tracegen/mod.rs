mod normal;
mod segmentation;

pub use normal::TracegenExecutionControl;
pub use segmentation::TracegenExecutionControlWithSegmentation;

pub type TracegenCtx = ();
