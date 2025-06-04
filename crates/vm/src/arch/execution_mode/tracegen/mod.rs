mod normal;
mod segmentation;

pub use normal::TracegenExecutionControl;
use openvm_stark_backend::{p3_field::Field, p3_matrix::dense::RowMajorMatrix};
pub use segmentation::TracegenExecutionControlWithSegmentation;

// TODO(ayush): this should be generic over arena
#[derive(Debug)]
pub struct TracegenCtx<F> {
    pub main_widths: Vec<usize>,
    pub trace_buffers: Vec<Vec<F>>,
    // TODO(ayush): for segmentation, remove from tracegen
    pub since_last_segment_check: usize,
}

impl<F: Field> TracegenCtx<F> {
    pub fn new(main_widths: &[usize]) -> Self {
        let main_widths = main_widths.to_vec();
        let trace_buffers = vec![Vec::new(); main_widths.len()];
        Self {
            main_widths,
            trace_buffers,
            since_last_segment_check: 0,
        }
    }

    // TODO(ayush): this should also handle all the overriden heights stuff
    pub fn new_with_capacity(main_widths: &[usize], trace_heights: Vec<usize>) -> Self {
        let main_widths = main_widths.to_vec();
        let trace_buffers = main_widths
            .iter()
            .zip(trace_heights.iter())
            .map(|(&width, &height)| Vec::with_capacity(width * height))
            .collect();

        Self {
            main_widths,
            trace_buffers,
            since_last_segment_check: 0,
        }
    }

    pub fn alloc(&mut self, chip_index: usize) -> &mut [F] {
        let width = self.main_widths[chip_index];
        let start_index = self.trace_buffers[chip_index].len();
        self.trace_buffers[chip_index].resize(start_index + width, F::ZERO);
        &mut self.trace_buffers[chip_index][start_index..start_index + width]
    }

    pub fn last(&mut self, chip_index: usize) -> &mut [F] {
        let width = self.main_widths[chip_index];
        let start_index = self.trace_buffers[chip_index].len() - width;
        &mut self.trace_buffers[chip_index][start_index..]
    }

    pub fn width(&self, chip_index: usize) -> usize {
        self.main_widths[chip_index]
    }

    pub fn height(&self, chip_index: usize) -> usize {
        self.trace_buffers[chip_index].len() / self.width(chip_index)
    }

    pub fn resize(&mut self, chip_index: usize, new_height: usize) {
        debug_assert!(self.height(chip_index) <= new_height);
        let width = self.width(chip_index);
        self.trace_buffers[chip_index].resize(width * new_height, F::ZERO);
    }

    pub fn into_matrices(self) -> Vec<RowMajorMatrix<F>> {
        self.trace_buffers
            .into_iter()
            .zip(self.main_widths.iter())
            .map(|(mut buffer, &width)| {
                let current_height = buffer.len() / width;
                let next_power_of_two_height = current_height.next_power_of_two();
                buffer.resize(width * next_power_of_two_height, F::ZERO);
                RowMajorMatrix::new(buffer, width)
            })
            .collect()
    }
}
