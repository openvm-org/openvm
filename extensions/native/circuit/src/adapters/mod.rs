pub mod alu_native_adapter;
// 2 reads, 0 writes, imm support, jump support
pub mod branch_native_adapter;
// 1 read, 1 write, arbitrary read size, arbitrary write size, no imm support
pub mod convert_adapter;
pub mod loadstore_native_adapter;
// 2 reads, 1 write, read size = write size = N, no imm support, read/write to address space d
pub mod native_vectorized_adapter;
