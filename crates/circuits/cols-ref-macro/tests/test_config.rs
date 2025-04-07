pub trait TestConfig {
    const N: usize;
}
pub struct TestConfigImpl;
impl TestConfig for TestConfigImpl {
    const N: usize = 4;
}
