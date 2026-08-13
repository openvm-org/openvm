use super::*;
use crate::arch::Postflight;

impl GpuPostflightProgram {
    /// Uploads an interpreter-produced history for CPU/GPU trace comparison.
    ///
    /// Production interpreter proving derives chronology on the device from
    /// the segment-start memory image. Tests already carry exact first-write
    /// seeds, so they can upload the validated predecessor index directly.
    pub fn upload_history_for_test<F: PrimeField32>(
        &self,
        program: &Program<F>,
        history: &PreflightHistory,
        exit_code: Option<u32>,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let postflight = Postflight::new(program, history, &self.memory_config, exit_code)
            .map_err(|error| GpuPostflightError::InvalidTranscript(error.to_string()))?;
        self.upload_validated_history_for_test(history, postflight)
    }

    /// Uploads an isolated chip fixture whose final sentinel need not resolve
    /// to another instruction in the fixture program.
    pub fn upload_isolated_history_for_test<F: PrimeField32>(
        &self,
        program: &Program<F>,
        history: &PreflightHistory,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let postflight = Postflight::new_for_test(program, history, &self.memory_config)
            .map_err(|error| GpuPostflightError::InvalidTranscript(error.to_string()))?;
        self.upload_validated_history_for_test(history, postflight)
    }

    fn upload_validated_history_for_test<F: PrimeField32>(
        &self,
        history: &PreflightHistory,
        postflight: Postflight<'_, F>,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let replay_steps = postflight
            .replay_steps_for_test()
            .map(|(program_index, memory_start)| GpuReplayStep {
                program_index,
                memory_start,
            })
            .collect::<Vec<_>>();
        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log: upload(&history.program, &self.device_ctx)?,
            memory_log: upload(&history.memory.accesses, &self.device_ctx)?,
            initial_write_log: upload(&history.memory.initial_writes, &self.device_ctx)?,
            field_values: upload(&history.memory.field_values, &self.device_ctx)?,
            field_initial_values: upload(&history.memory.field_initial_values, &self.device_ctx)?,
            memory_predecessors: upload(
                postflight.memory_predecessors_for_test(),
                &self.device_ctx,
            )?,
            touched_blocks: DeviceBuffer::new(),
            num_touched_blocks: 0,
            error: [0u32].to_device_on(&self.device_ctx)?,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = GpuPostflightPlan {
            steps: upload(&replay_steps, &self.device_ctx)?,
            program_frequencies: upload(postflight.filtered_exec_frequencies(), &self.device_ctx)?,
            opcode_ranges: postflight.opcode_ranges_for_test().clone(),
            from_state: postflight.from_state(),
            to_state: postflight.to_state(),
            exit_code: postflight.exit_code(),
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity,
        };
        Ok((transcript, plan))
    }

    #[cfg(feature = "rvr")]
    #[cfg(test)]
    pub(super) fn upload_history_with_initial_memory_for_test(
        &self,
        history: &PreflightHistory,
        boundary: GpuPostflightBoundary,
        initial_memory_images: &[DeviceBufferView],
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        self.upload_history(
            history,
            boundary.connector_boundary(),
            initial_memory_images,
        )
    }

    #[cfg(feature = "rvr")]
    #[cfg(test)]
    pub(super) fn synthetic_for_test(
        opcodes: &[u32],
        pc_base: u32,
        timestamp_max_bits: u32,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        let instructions = opcodes
            .iter()
            .map(|&opcode| GpuReplayInstruction {
                words: [opcode, 0, 0, 0, 0, 0, 0, 0],
            })
            .collect::<Vec<_>>();
        let mut active_opcodes = opcodes
            .iter()
            .copied()
            .filter(|&opcode| opcode != u32::MAX)
            .collect::<Vec<_>>();
        active_opcodes.sort_unstable();
        active_opcodes.dedup();
        let mut next_program_row = 0u32;
        let dense_program_rows = opcodes
            .iter()
            .map(|&opcode| {
                if opcode == u32::MAX {
                    u32::MAX
                } else {
                    let row = next_program_row;
                    next_program_row += 1;
                    row
                }
            })
            .collect::<Vec<_>>();
        let memory_config = MemoryConfig::default();
        let memory_address_spaces = memory_config
            .addr_spaces
            .iter()
            .map(|config| GpuMemoryAddressSpace {
                num_cells: config.num_cells as u64,
                cell_type: gpu_memory_cell_type(config.layout),
                _padding: 0,
            })
            .collect::<Vec<_>>();
        Ok(Self {
            instructions: upload(&instructions, device_ctx)?,
            dense_program_rows: upload(&dense_program_rows, device_ctx)?,
            num_program_rows: next_program_row as usize,
            d_active_opcodes: upload(&active_opcodes, device_ctx)?,
            active_opcodes,
            memory_address_spaces: upload(&memory_address_spaces, device_ctx)?,
            memory_config,
            address_space_height: MemoryConfig::default().addr_space_height as u32,
            cell_pointer_max_bits: MemoryConfig::default().pointer_max_bits as u32,
            timestamp_max_bits,
            pc_base,
            device_ctx: device_ctx.clone(),
            identity: Arc::new(()),
        })
    }

    #[cfg(feature = "rvr")]
    #[cfg(test)]
    pub(super) fn index_program_log_for_test(
        &self,
        program_log: &[PreflightProgramEvent],
        boundary: GpuPostflightBoundary,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let segment_identity = Arc::new(());
        let transcript = GpuPostflightTranscript {
            program_log: upload(program_log, &self.device_ctx)?,
            memory_log: DeviceBuffer::new(),
            initial_write_log: DeviceBuffer::new(),
            field_values: DeviceBuffer::new(),
            field_initial_values: DeviceBuffer::new(),
            memory_predecessors: DeviceBuffer::new(),
            touched_blocks: DeviceBuffer::new(),
            num_touched_blocks: 0,
            error: [0u32].to_device_on(&self.device_ctx)?,
            device_ctx: self.device_ctx.clone(),
            program_identity: self.identity.clone(),
            segment_identity: segment_identity.clone(),
        };
        let plan = GpuPostflightPlan::build(
            self,
            &transcript,
            boundary.connector_boundary(),
            self.identity.clone(),
            segment_identity,
        )?;
        Ok((transcript, plan))
    }
}

impl GpuPostflightTranscript {
    pub fn program_log_host(&self) -> Result<Vec<PreflightProgramEvent>, MemCopyError> {
        self.program_log.to_host_on(&self.device_ctx)
    }

    pub fn replace_program_log_for_test(
        &mut self,
        program_log: &[PreflightProgramEvent],
    ) -> Result<(), MemCopyError> {
        assert_eq!(
            program_log.len(),
            self.program_log.len(),
            "a replay plan is valid only for its original program-log length"
        );
        self.program_log = program_log.to_device_on(&self.device_ctx)?;
        Ok(())
    }

    pub fn memory_log_host(&self) -> Result<Vec<PreflightMemoryEvent>, MemCopyError> {
        self.memory_log.to_host_on(&self.device_ctx)
    }

    pub fn initial_write_log_host(&self) -> Result<Vec<PreflightInitialWrite>, MemCopyError> {
        self.initial_write_log.to_host_on(&self.device_ctx)
    }

    pub fn field_values_host(&self) -> Result<Vec<[u32; BLOCK_FE_WIDTH]>, MemCopyError> {
        Ok(self
            .field_values
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|block| block.values)
            .collect())
    }

    #[cfg(feature = "rvr")]
    #[cfg(test)]
    pub(super) fn memory_predecessors_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.memory_predecessors.to_host_on(&self.device_ctx)
    }
}

impl GpuPostflightPlan {
    pub fn steps_host(&self) -> Result<Vec<[u32; 2]>, MemCopyError> {
        Ok(self
            .steps
            .to_host_on(&self.device_ctx)?
            .into_iter()
            .map(|step| [step.program_index, step.memory_start])
            .collect())
    }

    #[cfg(test)]
    #[allow(dead_code)] // Used by CUDA integration tests, which are not built in every feature set.
    pub(super) const fn connector_boundary_for_test(&self) -> GpuPostflightBoundary {
        let (from, to, exit_code) = self.connector_boundary();
        GpuPostflightBoundary::new(from, to, exit_code)
    }

    #[cfg(feature = "rvr")]
    #[cfg(test)]
    pub(super) fn program_frequencies_host(&self) -> Result<Vec<u32>, MemCopyError> {
        self.program_frequencies.to_host_on(&self.device_ctx)
    }
}

#[cfg(all(test, feature = "rvr"))]
pub type ChronologyOutputForTest = (
    Vec<PreflightMemoryEvent>,
    Vec<PreflightInitialWrite>,
    Vec<PreflightFieldBlock>,
    Vec<PreflightFieldBlock>,
    Vec<u32>,
    Vec<TouchedBlock<BabyBear>>,
);

#[cfg(all(test, feature = "rvr"))]
pub(super) fn build_memory_chronology_for_test(
    memory: &[PreflightMemoryEvent],
    write_masks: &[u8],
    field_values: &[PreflightFieldBlock],
    initial_memory: &[Vec<u8>],
    config: &MemoryConfig,
) -> Result<ChronologyOutputForTest, GpuPostflightError> {
    let device_ctx = GpuDeviceCtx::for_current_device()?;
    let memory = upload(memory, &device_ctx)?;
    let write_masks = upload(write_masks, &device_ctx)?;
    let field_values = upload(field_values, &device_ctx)?;
    let initial_memory = initial_memory
        .iter()
        .map(|image| upload(image, &device_ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let initial_memory_views = initial_memory
        .iter()
        .map(|image| image.view())
        .collect::<Vec<_>>();
    let address_spaces = config
        .addr_spaces
        .iter()
        .map(|config| GpuMemoryAddressSpace {
            num_cells: config.num_cells as u64,
            cell_type: gpu_memory_cell_type(config.layout),
            _padding: 0,
        })
        .collect::<Vec<_>>();
    let address_spaces = upload(&address_spaces, &device_ctx)?;
    let error = [0u32].to_device_on(&device_ctx)?;
    let (seeds, field_seeds, index) = build_gpu_memory_chronology(GpuMemoryChronologyInput {
        memory: &memory,
        write_masks: &write_masks,
        field_values: &field_values,
        initial_memory: &initial_memory_views,
        address_space_height: config.addr_space_height as u32,
        pointer_max_bits: config.pointer_max_bits as u32,
        address_spaces: address_spaces.view(),
        error: &error,
        device_ctx: &device_ctx,
    })?;
    let mut touched = index.touched_blocks.to_host_on(&device_ctx)?;
    touched.truncate(index.num_touched_blocks);
    Ok((
        memory.to_host_on(&device_ctx)?,
        seeds.to_host_on(&device_ctx)?,
        field_values.to_host_on(&device_ctx)?,
        field_seeds.to_host_on(&device_ctx)?,
        index.predecessors.to_host_on(&device_ctx)?,
        touched,
    ))
}

#[cfg(all(test, feature = "rvr"))]
pub(super) fn empty_chronology_counts_for_test(
    count_field_metadata: bool,
) -> Result<Vec<u32>, GpuPostflightError> {
    let device_ctx = GpuDeviceCtx::for_current_device()?;
    let memory = DeviceBuffer::<PreflightMemoryEvent>::new();
    let write_masks = DeviceBuffer::<u8>::new();
    let field_values = DeviceBuffer::<PreflightFieldBlock>::new();
    let address_spaces = DeviceBuffer::<GpuMemoryAddressSpace>::new();
    let workspace = DeviceBuffer::<u64>::new();
    let counts_len = if count_field_metadata { 7 } else { 3 };
    let counts = upload(&vec![u32::MAX; counts_len], &device_ctx)?;
    let temp_storage = DeviceBuffer::<u8>::new();
    let error = [0u32].to_device_on(&device_ctx)?;
    let memory_config = MemoryConfig::default();
    // SAFETY: every view belongs to `device_ctx`, output buffers have the lengths required by the
    // ABI, and the empty input does not dereference the zero-length event buffers.
    unsafe {
        postflight::memory_chronology_sort_and_count(
            memory.view(),
            write_masks.view(),
            field_values.view(),
            address_spaces.view(),
            ADDR_SPACE_OFFSET,
            memory_config.addr_space_height as u32,
            memory_config.pointer_max_bits as u32,
            DEFERRAL_AS,
            count_field_metadata,
            &workspace,
            &workspace,
            &counts,
            &temp_storage,
            0,
            &error,
            device_ctx.stream.as_raw(),
        )?;
    }
    Ok(counts.to_host_on(&device_ctx)?)
}
