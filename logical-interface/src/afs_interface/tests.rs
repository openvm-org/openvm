use alloy_primitives::U32;

use super::AfsInterface;

#[test]
pub fn test_initialize_interface() {
    let interface = AfsInterface::<U32, U32, 32, 1024>::new();
}
