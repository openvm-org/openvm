#[cfg(feature = "metrics")]
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use openvm_circuit::{
    primitives::Chip, system::poseidon2::columns::Poseidon2PeripheryCols,
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};

use crate::cuda_abi::poseidon2;

#[derive(Clone)]
pub struct SharedBuffer<T> {
    buffer: Arc<Mutex<Option<Arc<DeviceBuffer<T>>>>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

impl<T> SharedBuffer<T> {
    pub fn records(&self) -> Arc<DeviceBuffer<T>> {
        let records = self.buffer.lock().unwrap();
        records
            .clone()
            .expect("Poseidon2 records buffer must be prepared before tracegen")
    }
}

pub struct Poseidon2ChipGPU<const SBOX_REGISTERS: usize> {
    pub device_ctx: GpuDeviceCtx,
    pub records: Arc<Mutex<Option<Arc<DeviceBuffer<F>>>>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    #[cfg(feature = "metrics")]
    pub(crate) current_trace_height: Arc<AtomicUsize>,
}

impl<const SBOX_REGISTERS: usize> Poseidon2ChipGPU<SBOX_REGISTERS> {
    pub fn new(device_ctx: GpuDeviceCtx) -> Self {
        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity_on(1, &device_ctx));
        idx.fill_zero_on(&device_ctx).unwrap();
        Self {
            device_ctx: device_ctx.clone(),
            records: Arc::new(Mutex::new(None)),
            idx,
            #[cfg(feature = "metrics")]
            current_trace_height: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Prepare an exact one-segment scratch buffer for Poseidon2 records.
    ///
    /// Each Poseidon2 record occupies `POSEIDON2_WIDTH` field elements.
    pub fn prepare_records(&self, num_records: usize) {
        self.idx.fill_zero_on(&self.device_ctx).unwrap();
        let mut records = self.records.lock().unwrap();
        assert!(
            records.is_none(),
            "Poseidon2 records buffer already prepared"
        );
        if num_records == 0 {
            return;
        }
        let num_elements = num_records
            .checked_mul(POSEIDON2_WIDTH)
            .expect("Poseidon2 records buffer size overflow");
        records.replace(Arc::new(DeviceBuffer::<F>::with_capacity_on(
            num_elements,
            &self.device_ctx,
        )));
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        SharedBuffer {
            buffer: self.records.clone(),
            idx: self.idx.clone(),
        }
    }

    pub fn trace_width() -> usize {
        Poseidon2PeripheryCols::<F, SBOX_REGISTERS>::width()
    }
}

impl<RA, const SBOX_REGISTERS: usize> Chip<RA, GpuBackend> for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        let Some(records) = self.records.lock().unwrap().take() else {
            self.idx.fill_zero_on(&self.device_ctx).unwrap();
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        };
        debug_assert_eq!(records.len() % POSEIDON2_WIDTH, 0);
        let capacity_records = records.len() / POSEIDON2_WIDTH;
        let mut num_records = self.idx.to_host_on(&self.device_ctx).unwrap()[0] as usize;
        assert!(
            num_records <= capacity_records,
            "Poseidon2 records buffer overflow: pushed {num_records} records into capacity {capacity_records}"
        );
        if num_records == 0 {
            self.idx.fill_zero_on(&self.device_ctx).unwrap();
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let counts = DeviceBuffer::<u32>::with_capacity_on(num_records, &self.device_ctx);
        let dedup_records =
            DeviceBuffer::<F>::with_capacity_on(num_records * POSEIDON2_WIDTH, &self.device_ctx);
        let dedup_counts = DeviceBuffer::<u32>::with_capacity_on(num_records, &self.device_ctx);
        unsafe {
            let d_num_records = [num_records].to_device_on(&self.device_ctx).unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &records,
                &counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to get temp bytes");
            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.device_ctx)
            };
            poseidon2::deduplicate_records(
                &records,
                &counts,
                &dedup_records,
                &dedup_counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to deduplicate records");
            num_records = *d_num_records
                .to_host_on(&self.device_ctx)
                .unwrap()
                .first()
                .unwrap();
        }
        drop(records);
        drop(counts);
        #[cfg(feature = "metrics")]
        self.current_trace_height
            .store(num_records, std::sync::atomic::Ordering::Relaxed);
        let trace_height = next_power_of_two_or_zero(num_records);
        let trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            Self::trace_width(),
            &self.device_ctx,
        );
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &dedup_records,
                &dedup_counts,
                num_records,
                SBOX_REGISTERS,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate trace");
        }
        // Reset state of this chip.
        self.idx.fill_zero_on(&self.device_ctx).unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}

pub enum Poseidon2PeripheryChipGPU {
    Register0(Poseidon2ChipGPU<0>),
    Register1(Poseidon2ChipGPU<1>),
}

impl Poseidon2PeripheryChipGPU {
    pub fn new(sbox_registers: usize, device_ctx: GpuDeviceCtx) -> Self {
        match sbox_registers {
            0 => Self::Register0(Poseidon2ChipGPU::new(device_ctx)),
            1 => Self::Register1(Poseidon2ChipGPU::new(device_ctx)),
            _ => panic!("Invalid number of sbox registers: {sbox_registers}"),
        }
    }

    pub fn prepare_records(&self, num_records: usize) {
        match self {
            Self::Register0(chip) => chip.prepare_records(num_records),
            Self::Register1(chip) => chip.prepare_records(num_records),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        match self {
            Self::Register0(chip) => chip.shared_buffer(),
            Self::Register1(chip) => chip.shared_buffer(),
        }
    }

    pub fn device_ctx(&self) -> &GpuDeviceCtx {
        match self {
            Self::Register0(chip) => &chip.device_ctx,
            Self::Register1(chip) => &chip.device_ctx,
        }
    }
}

impl<RA> Chip<RA, GpuBackend> for Poseidon2PeripheryChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        match self {
            Self::Register0(chip) => chip.generate_proving_ctx(()),
            Self::Register1(chip) => chip.generate_proving_ctx(()),
        }
    }
}
