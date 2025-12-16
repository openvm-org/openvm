#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/bitwise_op_lookup.cuh"
#include "primitives/trace_access.h"
#include "xorin/xorin.cuh"

using namespace xorin;

#define COL_SPONGE_IS_PADDING_BYTES(i) (i)
#define COL_SPONGE_PREIMAGE_BUFFER_BYTES(i) (34 + i)
#define COL_SPONGE_INPUT_BYTES(i) (34 + 136 + i)
#define COL_SPONGE_POSTIMAGE_BUFFER_BYTES(i) (34 + 136 + 136 + i)
#define COL_INSTRUCTION_PC (34 + 136 * 3)
#define COL_INSTRUCTION_IS_ENABLED (COL_INSTRUCTION_PC + 1)
#define COL_INSTRUCTION_BUFFER_PTR (COL_INSTRUCTION_IS_ENABLED + 1)
#define COL_INSTRUCTION_INPUT_PTR (COL_INSTRUCTION_BUFFER_PTR + 1)
#define COL_INSTRUCTION_LEN_PTR (COL_INSTRUCTION_INPUT_PTR + 1)
#define COL_INSTRUCTION_BUFFER (COL_INSTRUCTION_LEN_PTR + 1)
#define COL_INSTRUCTION_BUFFER_LIMBS(i) (COL_INSTRUCTION_BUFFER + 1 + i)
#define COL_INSTRUCTION_INPUT (COL_INSTRUCTION_BUFFER_LIMBS(0) + 4)
#define COL_INSTRUCTION_INPUT_LIMBS(i) (COL_INSTRUCTION_INPUT + 1 + i)
#define COL_INSTRUCTION_LEN (COL_INSTRUCTION_INPUT_LIMBS(0) + 4)
#define COL_INSTRUCTION_LEN_LIMBS(i) (COL_INSTRUCTION_LEN + 1 + i)
#define COL_INSTRUCTION_START_TIMESTAMP (COL_INSTRUCTION_LEN_LIMBS(0) + 4)

// Memory OC columns start after instruction columns
#define COL_MEM_OC_START (COL_INSTRUCTION_START_TIMESTAMP + 1)

__global__ void xorin_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<XorinVmRecord> d_records,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);
    
    if (idx < d_records.len()) {
        auto const &rec = d_records[idx];
        
        // Fill instruction columns
        COL_WRITE_VALUE(row, COL_INSTRUCTION_PC, rec.from_pc);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_IS_ENABLED, 1);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_BUFFER_PTR, rec.rd_ptr);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_INPUT_PTR, rec.rs1_ptr);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_LEN_PTR, rec.rs2_ptr);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_BUFFER, rec.buffer);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_INPUT, rec.input);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_LEN, rec.len);
        COL_WRITE_VALUE(row, COL_INSTRUCTION_START_TIMESTAMP, rec.timestamp);
        
        // Fill buffer limbs
        uint8_t buffer_bytes[4];
        memcpy(buffer_bytes, &rec.buffer, 4);
        for (int i = 0; i < 4; i++) {
            COL_WRITE_VALUE(row, COL_INSTRUCTION_BUFFER_LIMBS(i), buffer_bytes[i]);
        }
        
        // Fill input limbs
        uint8_t input_bytes[4];
        memcpy(input_bytes, &rec.input, 4);
        for (int i = 0; i < 4; i++) {
            COL_WRITE_VALUE(row, COL_INSTRUCTION_INPUT_LIMBS(i), input_bytes[i]);
        }
        
        // Fill len limbs
        uint8_t len_bytes[4];
        memcpy(len_bytes, &rec.len, 4);
        for (int i = 0; i < 4; i++) {
            COL_WRITE_VALUE(row, COL_INSTRUCTION_LEN_LIMBS(i), len_bytes[i]);
        }
        
        // Fill is_padding_bytes
        uint32_t num_words = rec.len / 4;
        for (uint32_t i = 0; i < num_words && i < 34; i++) {
            COL_WRITE_VALUE(row, COL_SPONGE_IS_PADDING_BYTES(i), 0);
        }
        for (uint32_t i = num_words; i < 34; i++) {
            COL_WRITE_VALUE(row, COL_SPONGE_IS_PADDING_BYTES(i), 1);
        }
        
        // Fill sponge columns and request bitwise operations
        BitwiseOperationLookup bitwise_lookup(d_bitwise_lookup_ptr, bitwise_num_bits);
        
        for (uint32_t i = 0; i < rec.len && i < 136; i++) {
            uint8_t buffer_byte = rec.buffer_limbs[i];
            uint8_t input_byte = rec.input_limbs[i];
            uint8_t result_byte = buffer_byte ^ input_byte;
            
            COL_WRITE_VALUE(row, COL_SPONGE_PREIMAGE_BUFFER_BYTES(i), buffer_byte);
            COL_WRITE_VALUE(row, COL_SPONGE_INPUT_BYTES(i), input_byte);
            COL_WRITE_VALUE(row, COL_SPONGE_POSTIMAGE_BUFFER_BYTES(i), result_byte);
            
            // Request bitwise XOR operation
            bitwise_lookup.request_xor(buffer_byte, input_byte);
        }
        
        // Zero-fill remaining sponge columns
        for (uint32_t i = rec.len; i < 136; i++) {
            COL_WRITE_VALUE(row, COL_SPONGE_PREIMAGE_BUFFER_BYTES(i), 0);
            COL_WRITE_VALUE(row, COL_SPONGE_INPUT_BYTES(i), 0);
            COL_WRITE_VALUE(row, COL_SPONGE_POSTIMAGE_BUFFER_BYTES(i), 0);
        }
        
        // TODO: Fill memory auxiliary columns properly
        // For now, we'll leave them as zeros to get compilation working
        
        // Range check for pointer bounds
        uint32_t limb_shift = 1 << (32 - pointer_max_bits);
        bitwise_lookup.request_range(buffer_bytes[3] * limb_shift, input_bytes[3] * limb_shift);
        bitwise_lookup.request_range(len_bytes[3] * limb_shift, len_bytes[3] * limb_shift);
        
    } else {
        // Zero-fill padding rows
        row.fill_zero(0, sizeof(XorinVmCols<uint8_t>));
    }
}

extern "C" int _xorin_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<XorinVmRecord> d_records,
    uint32_t *d_bitwise_lookup_ptr,
    size_t bitwise_num_bits,
    uint32_t pointer_max_bits
) {
    assert((height & (height - 1)) == 0);
    
    dim3 threads_per_block(128);
    dim3 blocks(cdiv(height, threads_per_block.x));
    
    xorin_tracegen<<<blocks, threads_per_block>>>(
        d_trace,
        height,
        d_records,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        pointer_max_bits
    );
    
    return gpu_check_error();
}