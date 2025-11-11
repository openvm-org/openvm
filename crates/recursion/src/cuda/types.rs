use stark_backend_v2::F;

#[repr(C)]
#[derive(Debug, Default)]
pub struct TraceMetadata {
    pub air_idx: usize,
    pub cached_idx: usize,
    pub starting_cidx: usize,
    pub total_interactions: usize,
    pub num_air_id_lookups: usize,
    pub log_height: u8,
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct PublicValueData {
    pub air_idx: usize,
    pub air_num_pvs: usize,
    pub num_airs: usize,
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
