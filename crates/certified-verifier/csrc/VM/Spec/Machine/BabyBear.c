// Lean compiler output
// Module: VM.Spec.Machine.BabyBear
// Imports: public import Init public meta import Init public import VM.Spec.Machine.Field public import Fundamentals.Spec.BabyBear.Field
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
lean_object* lp_mathlib_ZMod_instField___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__0 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__0_value;
static const lean_closure_object lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1))} };
static const lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__1 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__1_value;
static const lean_ctor_object lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2013265921) << 1) | 1)),((lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__0_value),((lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__1_value)}};
static const lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__2 = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__2_value;
LEAN_EXPORT const lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField = (const lean_object*)&lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___closed__2_value;
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0(lean_object* v___x_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_ZMod_val(v___x_1_, v_x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0___boxed(lean_object* v___x_4_, lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__0(v___x_4_, v_x_5_);
lean_dec(v_x_5_);
lean_dec(v___x_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_openvm_x2dfv_VM_Spec_Machine_BabyBear_instCanonicalFieldField___lam__1(lean_object* v___x_7_, lean_object* v_n_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v_toRing_11_; lean_object* v___x_12_; lean_object* v_toAddMonoidWithOne_13_; lean_object* v_toNatCast_14_; lean_object* v___x_15_; 
v___x_9_ = lp_mathlib_ZMod_instField___redArg(v___x_7_);
v___x_10_ = lp_mathlib_Field_toDivisionRing___redArg(v___x_9_);
v_toRing_11_ = lean_ctor_get(v___x_10_, 0);
lean_inc_ref(v_toRing_11_);
lean_dec_ref(v___x_10_);
v___x_12_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_toRing_11_);
v_toAddMonoidWithOne_13_ = lean_ctor_get(v___x_12_, 1);
lean_inc_ref(v_toAddMonoidWithOne_13_);
lean_dec_ref(v___x_12_);
v_toNatCast_14_ = lean_ctor_get(v_toAddMonoidWithOne_13_, 0);
lean_inc(v_toNatCast_14_);
lean_dec_ref(v_toAddMonoidWithOne_13_);
v___x_15_ = lean_apply_1(v_toNatCast_14_, v_n_8_);
return v___x_15_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_Field(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Field(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_openvm_x2dfv_VM_Spec_Machine_BabyBear(uint8_t builtin) {
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
res = initialize_openvm_x2dfv_VM_Spec_Machine_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_BabyBear_Field(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
