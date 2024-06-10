use super::AfsInterface;

#[test]
pub fn test_initialize_interface() {
    let interface = AfsInterface::<u32, u64, 4, 8, 32, 32>::new();
}
