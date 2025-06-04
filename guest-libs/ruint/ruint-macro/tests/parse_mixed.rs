use ruint::aliases::{B256, U256};
use ruint_macro::uint;

#[test]
fn test_non_literal() {
    uint! {
        assert_eq!(0xBBBB_B432_B245_B323_u64, 13527604035569693475);
    };
}

#[test]
fn test_parse_mixed() {
    uint! {
        assert_eq!(0x10U256, "0x10".parse::<U256>().unwrap());
        assert_eq!(0o10U256, "0o10".parse::<U256>().unwrap());
        assert_eq!(0b10U256, "0b10".parse::<U256>().unwrap());

        assert_eq!(0x10_U256, "0x10".parse::<U256>().unwrap());
        assert_eq!(0o10_U256, "0o10".parse::<U256>().unwrap());
        assert_eq!(0b10_U256, "0b10".parse::<U256>().unwrap());

        assert_eq!(0x10_B256, "0x10".parse::<B256>().unwrap());
        assert_eq!(0o10B256, "0o10".parse::<B256>().unwrap());
        assert_eq!(0b10B256, "0b10".parse::<B256>().unwrap());

        assert_eq!(0o10_B256, "0o10".parse::<B256>().unwrap());
        assert_eq!(0b10_B256, "0b10".parse::<B256>().unwrap());

        assert_eq!(2, 2);
    }
}
