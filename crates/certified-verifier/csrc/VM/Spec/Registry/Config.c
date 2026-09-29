// Lean compiler output
// Module: VM.Spec.Registry.Config
// Imports: public import Init public meta import Init public import VM.Spec.Execution.Field public import VM.Spec.Machine.Config
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lp_mathlib_ZMod_val(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
static const lean_ctor_object lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(536870912) << 1) | 1))}};
static const lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___closed__0_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___boxed(lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__0_value;
static const lean_ctor_object lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*12 + 0, .m_other = 12, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(30) << 1) | 1)),((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(7) << 1) | 1)),((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)(((size_t)(29) << 1) | 1)),((lean_object*)(((size_t)(29) << 1) | 1)),((lean_object*)(((size_t)(32) << 1) | 1)),((lean_object*)(((size_t)(10) << 1) | 1)),((lean_object*)&lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__0_value)}};
static const lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__1_value;
LEAN_EXPORT const lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Registry_openVmConfig___closed__1_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec(lean_object* v_addressSpace_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_4_ = lean_unsigned_to_nat(1u);
v___x_5_ = lean_unsigned_to_nat(2013265921u);
v___x_6_ = lp_mathlib_ZMod_val(v___x_5_, v_addressSpace_3_);
v___x_7_ = lean_nat_dec_le(v___x_4_, v___x_6_);
if (v___x_7_ == 0)
{
lean_object* v___x_8_; 
lean_dec(v___x_6_);
v___x_8_ = lean_box(0);
return v___x_8_;
}
else
{
lean_object* v___x_9_; uint8_t v___x_10_; 
v___x_9_ = lean_unsigned_to_nat(9u);
v___x_10_ = lean_nat_dec_lt(v___x_6_, v___x_9_);
lean_dec(v___x_6_);
if (v___x_10_ == 0)
{
lean_object* v___x_11_; 
v___x_11_ = lean_box(0);
return v___x_11_;
}
else
{
lean_object* v___x_12_; 
v___x_12_ = ((lean_object*)(lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___closed__0));
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec___boxed(lean_object* v_addressSpace_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_openvm_x2dfv_VM_Spec_Registry_openVmAddressSpaceSpec(v_addressSpace_13_);
lean_dec(v_addressSpace_13_);
return v_res_14_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Execution_Field(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Config(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Registry_Config(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Execution_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
