// Lean compiler output
// Module: VM.Spec.Machine.Memory
// Imports: public import Init public meta import Init public import VM.Spec.Machine.Field public import VM.Spec.Machine.Config public import VM.Spec.Machine.Constants
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_readOperand___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_readOperand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_readOperand___redArg(lean_object* v_inst_1_, lean_object* v_mem_2_, lean_object* v_d_3_, lean_object* v_p_4_){
_start:
{
lean_object* v_asNat_5_; lean_object* v___x_6_; lean_object* v___x_7_; uint8_t v___x_8_; 
v_asNat_5_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_asNat_5_);
lean_dec_ref(v_inst_1_);
lean_inc(v_d_3_);
v___x_6_ = lean_apply_1(v_asNat_5_, v_d_3_);
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = lean_nat_dec_eq(v___x_6_, v___x_7_);
lean_dec(v___x_6_);
if (v___x_8_ == 0)
{
lean_object* v___x_9_; 
v___x_9_ = lean_apply_2(v_mem_2_, v_d_3_, v_p_4_);
return v___x_9_;
}
else
{
lean_dec(v_d_3_);
lean_dec(v_mem_2_);
return v_p_4_;
}
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_readOperand(lean_object* v_F_10_, lean_object* v_inst_11_, lean_object* v_mem_12_, lean_object* v_d_13_, lean_object* v_p_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_openvm_x2dfv_VM_Spec_Machine_readOperand___redArg(v_inst_11_, v_mem_12_, v_d_13_, v_p_14_);
return v___x_15_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Field(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Config(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Constants(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Memory(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_openvm_x2dfv_VM_Spec_Machine_Constants(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
