use crate::arch::Arena;

pub struct PreflightCtx<RA> {
    pub arenas: Vec<RA>,
    pub instret_end: Option<u64>,
}

impl<RA: Arena> PreflightCtx<RA> {
    /// `capacities` is list of `(height, width)` dimensions for each arena, indexed by AIR index.
    /// The length of `capacities` must equal the number of AIRs.
    /// Here `height` will always mean an overestimate of the trace height for that AIR, while
    /// `width` may have different meanings depending on the `RA` type.
    pub fn new_with_capacity(capacities: &[(usize, usize)], instret_end: Option<u64>) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, main_width)| RA::with_capacity(height, main_width))
            .collect();

        Self {
            arenas,
            instret_end,
        }
    }
}
