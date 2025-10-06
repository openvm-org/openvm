use openvm_circuit::arch::VmState;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use std::sync::OnceLock;
use std::pin::Pin;
use openvm_circuit::arch::MemoryConfig;
use openvm_circuit::arch::SystemConfig;
use std::collections::BTreeMap;
use std::io::Write;

use openvm_rv32im_circuit::{
    LoadStoreCoreAir,
    LoadStoreExecutor, 
    LoadStoreFiller,
    Rv32LoadStoreAir,
    Rv32LoadStoreExecutor,
    Rv32LoadStoreChip,
    LoadStoreCoreRecord,
    AluOp,
    AddOp,
    SubOp,
    XorOp,
    OrOp,
    AndOp,
};

use libc::{
    shm_open, 
    shm_unlink, 
    mmap, 
    munmap, 
    ftruncate, 
    close,
    O_CREAT, 
    O_RDWR, 
    O_RDONLY, 
    PROT_READ, 
    PROT_WRITE, 
    MAP_SHARED,
    MAP_FAILED
};

use std::ffi::CString;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MemoryUpdate {
    address_space: u32, 
    pointer: u32, 
    value: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryLog {
    count: usize,
    updates: [MemoryUpdate; 10],
}

pub fn update_memory_log(update: MemoryUpdate) {
    let shmem_name = "/shmem";
    let c_name = CString::new(shmem_name).unwrap();

    unsafe {
        let size = std::mem::size_of::<MemoryLog>();

        let fd = shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, 0o666);
        ftruncate(fd, size as i64);

        let ptr = mmap(
            std::ptr::null_mut(), 
            size, 
            PROT_READ | PROT_WRITE, 
            MAP_SHARED, 
            fd, 
            0
        );

        let log_ptr = ptr as *mut MemoryLog;
        (*log_ptr).updates[(*log_ptr).count] = update;
        (*log_ptr).count += 1;

        println!("Count: {}", (*log_ptr).count);
        println!("Updates: {:?}", (*log_ptr).updates);

        munmap(ptr, size);
        close(fd);
    }
}

#[no_mangle]
pub fn ADD_RV32(a: u32, b: u32, c: u32, e: u32) -> u32 {
    println!("ADD_RV32 called with a: {}, b: {}, c: {}, e: {}", a, b, c, e);
    let result = AddOp::compute(b, c);
    println!("Result: {}", result);
    
    let update = MemoryUpdate {
        address_space: 1,
        pointer: a,
        value: result
    };
    
    update_memory_log(update);
    result
}