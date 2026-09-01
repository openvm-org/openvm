// Stable single-function raw-buffer adapter around the Lean-generated VM verifier.
//
// This is OpenVM-owned integration code, not vendored Lean output. Keeping
// Lean object construction and ownership here lets Rust call one ordinary C
// function without reproducing the internal layout from lean.h. Calls,
// including lazy initialization, are serialized by the Rust wrapper.

#include <lean/lean.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dest, const void *src, size_t count);

lean_object *initialize_VmVerifier_Spec_Runtime(uint8_t builtin);
lean_object *l_VmVerifier_verifyVmStarkProof(lean_object *vk, lean_object *baseline,
                                             lean_object *proof, lean_object *public_values,
                                             lean_object *user_public_values);

char **lean_setup_args(int argc, char **argv);
void lean_initialize_runtime_module(void);

enum {
    OPENVM_VM_INIT_ERROR = -1,
    OPENVM_VM_INVALID_ARGUMENT = -2,
    OPENVM_VM_INVALID_RESULT = -3,
};

static bool initialized = false;
static bool initialization_failed = false;
static char program_name[] = "openvm-certified-verifier";
static char *program_args[] = {program_name, NULL};

static void write_error(char *out, size_t capacity, const char *message, size_t message_len) {
    if (out == NULL || capacity == 0) {
        return;
    }
    size_t copy_len = message_len < capacity - 1 ? message_len : capacity - 1;
    memcpy(out, message, copy_len);
    out[copy_len] = '\0';
}

static int32_t initialize_verifier(char *error_out, size_t error_capacity) {
    if (initialized) {
        return 0;
    }
    if (initialization_failed) {
        static const char message[] = "Lean VM verifier initialization previously failed";
        write_error(error_out, error_capacity, message, sizeof(message) - 1);
        return OPENVM_VM_INIT_ERROR;
    }

    lean_setup_args(1, program_args);
    lean_initialize_runtime_module();
    lean_set_panic_messages(false);
    lean_object *result = initialize_VmVerifier_Spec_Runtime(1 /* builtin */);
    lean_set_panic_messages(true);
    lean_io_mark_end_initialization();

    if (lean_io_result_is_error(result)) {
        initialization_failed = true;
        lean_dec(result);
        static const char message[] = "Lean VM verifier initialization failed";
        write_error(error_out, error_capacity, message, sizeof(message) - 1);
        return OPENVM_VM_INIT_ERROR;
    }

    lean_dec(result);
    lean_init_task_manager();
    initialized = true;
    return 0;
}

static lean_object *byte_array(const uint8_t *bytes, size_t len) {
    lean_object *array = lean_alloc_sarray(1, len, len);
    if (len != 0) {
        memcpy(lean_sarray_cptr(array), bytes, len);
    }
    return array;
}

int32_t openvm_vm_verify(const uint8_t *vk, size_t vk_len, const uint8_t *baseline,
                         size_t baseline_len, const uint8_t *proof, size_t proof_len,
                         const uint8_t *public_values, size_t public_values_len,
                         const uint8_t *user_public_values, size_t user_public_values_len,
                         char *error_out, size_t error_capacity) {
    if ((vk == NULL && vk_len != 0) || (baseline == NULL && baseline_len != 0) ||
        (proof == NULL && proof_len != 0) ||
        (public_values == NULL && public_values_len != 0) ||
        (user_public_values == NULL && user_public_values_len != 0) ||
        (error_out == NULL && error_capacity != 0)) {
        return OPENVM_VM_INVALID_ARGUMENT;
    }
    if (error_capacity != 0) {
        error_out[0] = '\0';
    }

    int32_t init_code = initialize_verifier(error_out, error_capacity);
    if (init_code != 0) {
        return init_code;
    }

    lean_object *result = l_VmVerifier_verifyVmStarkProof(
        byte_array(vk, vk_len), byte_array(baseline, baseline_len),
        byte_array(proof, proof_len), byte_array(public_values, public_values_len),
        byte_array(user_public_values, user_public_values_len));
    unsigned tag = lean_obj_tag(result);
    if (tag == 1) {
        lean_dec(result);
        return 0;
    }
    if (tag != 0) {
        lean_dec(result);
        static const char message[] = "Lean VM verifier returned an invalid Except tag";
        write_error(error_out, error_capacity, message, sizeof(message) - 1);
        return OPENVM_VM_INVALID_RESULT;
    }

    lean_object *error = lean_ctor_get(result, 0);
    uint32_t exit_code = lean_unbox_uint32(error);
    lean_dec(result);
    static const char message[] = "Lean VM verifier rejected the proof";
    write_error(error_out, error_capacity, message, sizeof(message) - 1);
    return (int32_t)exit_code;
}
