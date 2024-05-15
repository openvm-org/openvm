#[cfg(test)]
mod tests {
    use afs_chips::mbitxor_chip::MBitXorChip;

    #[test]
    fn test_mbitxor_chip() {
        const M: u32 = 3; // Example with 3-bit values
        let chip = MBitXorChip::<M>::new();

        // Test XOR computation
        assert_eq!(chip.compute_xor(0b000, 0b000), 0b000);
        assert_eq!(chip.compute_xor(0b001, 0b010), 0b011);
        assert_eq!(chip.compute_xor(0b111, 0b111), 0b000);

        // Test add_count
        chip.add_count(0b001, 0b010);
        assert_eq!(chip.x[0b001].load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(chip.y[0b010].load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(chip.z[0b011].load(std::sync::atomic::Ordering::Relaxed), 1);
    }
}
