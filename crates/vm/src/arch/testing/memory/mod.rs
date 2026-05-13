use air::{MemoryDummyAir, MemoryDummyChip};
use rand::Rng;

use crate::{
    arch::{MemoryCellType, VmField, BLOCK_FE_WIDTH, BUS_PTR_SCALE, MEMORY_BLOCK_BYTES},
    system::memory::{online::TracingMemory, MemoryController},
};

pub mod air;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

/// A dummy testing chip that sends/receives unconstrained messages on the [MemoryBus].
/// All memory accesses use `BLOCK_FE_WIDTH`.
pub struct MemoryTester<F: VmField> {
    pub(crate) chip: MemoryDummyChip<F>,
    pub memory: TracingMemory,
    pub(super) controller: MemoryController<F>,
}

impl<F: VmField> MemoryTester<F> {
    pub fn new(controller: MemoryController<F>, memory: TracingMemory) -> Self {
        let chip = MemoryDummyChip::new(MemoryDummyAir::new(controller.memory_bus));
        Self {
            chip,
            memory,
            controller,
        }
    }

    /// Dispatches one memory-bus message. The interface accepts two values of
    /// `N`:
    ///
    /// - `N = BLOCK_FE_WIDTH`: `ptr` is a cell index. The chip's permutation message uses the
    ///   normalized bus pointer `BUS_PTR_SCALE * ptr`. Storage type per AS: DEFERRAL_AS reads F
    ///   cells, u16-celled ASes read u16 cells lifted to F via `F::from_u16`.
    /// - `N = MEMORY_BLOCK_BYTES` (u16 ASes only): `ptr` is the **byte pointer** (= bus pointer for
    ///   u16 ASes). Performs a byte-view read and returns the raw bytes lifted to F via
    ///   `F::from_u8`. The chip's permutation message packs the bytes pairwise into
    ///   `BLOCK_FE_WIDTH` field elements via base-256: `out[i] = byte[2i] + 256 * byte[2i+1]`.
    pub fn read<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            self.read_cells::<N>(addr_space, ptr)
        } else {
            self.read_bytes::<N>(addr_space, ptr)
        }
    }

    /// Dispatches one memory-bus write. See [`Self::read`] for the dual N
    /// semantics.
    pub fn write<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        const { assert!(N == BLOCK_FE_WIDTH || N == MEMORY_BLOCK_BYTES) };
        if N == BLOCK_FE_WIDTH {
            self.write_cells::<N>(addr_space, ptr, data);
        } else {
            self.write_bytes::<N>(addr_space, ptr, data);
        }
    }

    fn read_cells<const N: usize>(&mut self, addr_space: usize, ptr: usize) -> [F; N] {
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        let (t_prev, data) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                memory.read::<F, N>(addr_space as u32, ptr as u32)
            },
            MemoryCellType::U16 => {
                let (t_prev, data) =
                    unsafe { memory.read::<u16, N>(addr_space as u32, ptr as u32) };
                (t_prev, data.map(F::from_u16))
            }
            other => panic!("MemoryTester::read_cells unsupported cell type {other:?}"),
        };
        let bus_ptr = (ptr * BUS_PTR_SCALE) as u32;
        self.chip.receive(addr_space as u32, bus_ptr, &data, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
        data
    }

    fn write_cells<const N: usize>(&mut self, addr_space: usize, ptr: usize, data: [F; N]) {
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        let (t_prev, data_prev) = match cell_layout {
            MemoryCellType::F { .. } => unsafe {
                memory.write::<F, N>(addr_space as u32, ptr as u32, data)
            },
            MemoryCellType::U16 => {
                let (t_prev, data_prev) = unsafe {
                    memory.write::<u16, N>(
                        addr_space as u32,
                        ptr as u32,
                        data.map(|x| {
                            let v = x.as_canonical_u32();
                            assert!(
                                v <= u16::MAX as u32,
                                "MemoryTester::write got F value {v} outside u16 range",
                            );
                            v as u16
                        }),
                    )
                };
                (t_prev, data_prev.map(F::from_u16))
            }
            other => panic!("MemoryTester::write_cells unsupported cell type {other:?}"),
        };
        let bus_ptr = (ptr * BUS_PTR_SCALE) as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &data_prev, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &data, t);
    }

    fn read_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize) -> [F; N] {
        assert_eq!(N, MEMORY_BLOCK_BYTES);
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "MemoryTester::read byte-view requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes) = unsafe { memory.read::<u8, N>(addr_space as u32, byte_ptr as u32) };
        let data = bytes.map(F::from_u8);
        let packed = pack_bytes_for_bus::<F>(&data);
        let bus_ptr = byte_ptr as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &packed, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &packed, t);
        data
    }

    fn write_bytes<const N: usize>(&mut self, addr_space: usize, byte_ptr: usize, data: [F; N]) {
        assert_eq!(N, MEMORY_BLOCK_BYTES);
        let memory = &mut self.memory;
        let t = memory.timestamp();
        let cell_layout = memory.data().memory.config[addr_space].layout;
        assert!(
            matches!(cell_layout, MemoryCellType::U16),
            "MemoryTester::write byte-view requires a u16-celled AS, got {cell_layout:?}",
        );
        let (t_prev, bytes_prev) = unsafe {
            memory.write::<u8, N>(
                addr_space as u32,
                byte_ptr as u32,
                data.map(|x| {
                    let v = x.as_canonical_u32();
                    assert!(
                        v <= u8::MAX as u32,
                        "MemoryTester::write byte-view got F value {v} outside u8 range",
                    );
                    v as u8
                }),
            )
        };
        let data_prev = bytes_prev.map(F::from_u8);
        let packed_prev = pack_bytes_for_bus::<F>(&data_prev);
        let packed_new = pack_bytes_for_bus::<F>(&data);
        let bus_ptr = byte_ptr as u32;
        self.chip
            .receive(addr_space as u32, bus_ptr, &packed_prev, t_prev);
        self.chip.send(addr_space as u32, bus_ptr, &packed_new, t);
    }
}

pub fn gen_pointer<R>(rng: &mut R, len: usize) -> usize
where
    R: Rng + ?Sized,
{
    const MAX_MEMORY: usize = 1 << 29;
    rng.random_range(0..MAX_MEMORY - len) / len * len
}

/// Packs `MEMORY_BLOCK_BYTES` u8-typed F values into `BLOCK_FE_WIDTH` packed F
/// values via base-256: `out[i] = data[ratio*i] + 256 * data[ratio*i+1] + …`.
/// Matches `bridge::pack_for_bus` so the testing chip's bus message exactly
/// equals the bus message a real chip would emit through the legacy bridge.
fn pack_bytes_for_bus<F: VmField>(data: &[F]) -> [F; BLOCK_FE_WIDTH] {
    assert_eq!(data.len(), MEMORY_BLOCK_BYTES);
    let ratio = MEMORY_BLOCK_BYTES / BLOCK_FE_WIDTH;
    let mut packed = [F::ZERO; BLOCK_FE_WIDTH];
    for i in 0..BLOCK_FE_WIDTH {
        let mut acc = F::ZERO;
        let mut mult = 1u64;
        for k in 0..ratio {
            acc += data[i * ratio + k] * F::from_u64(mult);
            mult *= 256;
        }
        packed[i] = acc;
    }
    packed
}
