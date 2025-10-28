use stark_backend_v2::F;

#[repr(C)]
#[derive(Debug, Default)]
pub struct TraceMetadata {
    pub hypercube_dim: usize,
    pub is_present: bool,
    pub num_cached: usize,
    pub cached_idx: usize,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct PublicValueData {
    pub air_idx: usize,
    pub pv_idx: usize,
    pub value: F,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct AirData {
    pub num_interactions: usize,
    pub has_preprocessed: bool,
}
