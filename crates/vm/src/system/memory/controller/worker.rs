use std::{
    sync::{Arc, Mutex},
    thread,
};

use crossbeam_channel;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::system::memory::{
    adapter::AccessAdapterInventory, interface::MemoryInterface, OfflineMemory,
};

#[derive(Debug)]
pub enum MemoryTask<T> {
    Read {
        address_space: u32,
        pointer: u32,
        len: usize,
    },
    Write {
        address_space: u32,
        pointer: u32,
        data: Vec<T>,
    },
    IncrementTimestampBy(u32),
}

pub(super) struct MemoryBackgroundWorker<F> {
    tx: Option<crossbeam_channel::Sender<MemoryTask<F>>>,
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl<F: PrimeField32> MemoryBackgroundWorker<F> {
    pub(super) fn spawn(
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
        interface_chip: Arc<Mutex<MemoryInterface<F>>>,
        access_adapters: Arc<Mutex<AccessAdapterInventory<F>>>,
    ) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded::<MemoryTask<F>>();

        let worker_handle = thread::spawn(move || {
            const MAX_BATCH_SIZE: usize = 1 << 10;

            while let Ok(record) = rx.recv() {
                let mut access_adapters = access_adapters.lock().unwrap();
                let mut interface_chip = interface_chip.lock().unwrap();
                let mut offline_memory = offline_memory.lock().unwrap();

                Self::process_task(
                    record,
                    &mut offline_memory,
                    &mut interface_chip,
                    &mut access_adapters,
                );

                // While we have the locks, process up to MAX_BATCH_SIZE total records.
                for record in rx.try_iter().take(MAX_BATCH_SIZE - 1) {
                    Self::process_task(
                        record,
                        &mut offline_memory,
                        &mut interface_chip,
                        &mut access_adapters,
                    );
                }
            }
        });

        Self {
            tx: Some(tx),
            worker_handle: Some(worker_handle),
        }
    }

    pub(super) fn send_task(&self, task: MemoryTask<F>) {
        if let Some(tx) = &self.tx {
            tx.send(task)
                .expect("Failed to send task to background thread");
        }
    }

    pub(super) fn wait_for_completion(self) {
        drop(self.tx);

        if let Some(handle) = self.worker_handle {
            handle.join().expect("Failed to join background thread");
        }
    }

    fn process_task(
        entry: MemoryTask<F>,
        offline_memory: &mut OfflineMemory<F>,
        interface_chip: &mut MemoryInterface<F>,
        adapter_records: &mut AccessAdapterInventory<F>,
    ) {
        match entry {
            MemoryTask::Read {
                address_space,
                pointer,
                len,
            } => {
                if address_space != 0 {
                    interface_chip.touch_range(address_space, pointer, len as u32);
                }
                offline_memory.read(address_space, pointer, len, adapter_records);
            }
            MemoryTask::Write {
                address_space,
                pointer,
                data,
            } => {
                if address_space != 0 {
                    interface_chip.touch_range(address_space, pointer, data.len() as u32);
                }
                offline_memory.write(address_space, pointer, data, adapter_records);
            }
            MemoryTask::IncrementTimestampBy(amount) => {
                offline_memory.increment_timestamp_by(amount);
            }
        }
    }
}
