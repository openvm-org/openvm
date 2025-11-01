use stark_backend_v2::F;

#[repr(C)]
#[derive(Debug, Default)]
pub struct TraceMetadata {
    pub air_idx: usize,
    pub hypercube_dim: usize,
    pub cached_idx: usize,
    pub starting_cidx: usize,
    pub total_interactions: usize,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct PublicValueData {
    pub air_idx: usize,
    pub air_num_pvs: usize,
    pub pv_idx: usize,
    pub value: F,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct AirData {
    pub num_cached: usize,
    pub num_interactions_per_row: usize,
    pub has_preprocessed: bool,
}
