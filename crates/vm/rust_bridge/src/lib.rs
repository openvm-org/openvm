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
};

#[no_mangle]
pub extern "C" fn print_message() {
    println!("Hello world!");
}

static VM_STATE: OnceLock<Pin<Box<VmState<BabyBear>>>> = OnceLock::new();

#[no_mangle]
pub extern "C" fn initialize_vmstate() -> usize {
    println!("initialize_vmstate() called");

    let vm_state = VM_STATE.get_or_init(|| {
        println!("Creating VmState...");
        
        let mut memory_config = MemoryConfig::default();

        let system_config = SystemConfig::default_from_memory(memory_config);
        let init_memory = BTreeMap::new();

        println!("Calling VmState::initial");
        let state = VmState::initial(&system_config, &init_memory, 0, vec![]);
        println!("Done calling VmState::initial");    
        Box::pin(state)
    });

    let ptr = vm_state.as_ref().get_ref() as *const VmState<BabyBear> as usize;
    println!("the address of the vm_state is: 0x{:x}", ptr);
    ptr
}

#[no_mangle]
pub extern "C" fn write_to_vmstate(vm_state_address: usize) {
    println!("write_to_vmstate() called");
    type T = u8;
    type F = BabyBear;

    let vm_state_ptr = vm_state_address as *mut VmState<F>;
    let memory = unsafe { 
        &mut (*vm_state_ptr).memory    
    };

    const BLOCK_SIZE : usize = 4;
    let address_space = 2;
    let pointer = 0;
    let data: [T; BLOCK_SIZE] = [1, 2, 3, 4];
    
    unsafe {
        memory.write::<T, BLOCK_SIZE>(address_space, pointer, data);
    }

    println!("write to vmstate done succesfully!");
}