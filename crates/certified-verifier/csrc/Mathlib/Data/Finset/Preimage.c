// Lean compiler output
// Module: Mathlib.Data.Finset.Preimage
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Pi public import Mathlib.Data.Finset.Sigma public import Mathlib.Data.Set.Finite.Basic
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__0(lean_object* v_e_1_, lean_object* v_a_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_e_1_, 0);
lean_inc(v_toFun_3_);
lean_dec_ref(v_e_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v_a_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__1(lean_object* v_e_5_, lean_object* v_b_6_){
_start:
{
lean_object* v___x_7_; lean_object* v_toFun_8_; lean_object* v___x_9_; 
v___x_7_ = lp_mathlib_Equiv_symm___redArg(v_e_5_);
v_toFun_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_toFun_8_);
lean_dec_ref(v___x_7_);
v___x_9_ = lean_apply_1(v_toFun_8_, v_b_6_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___redArg(lean_object* v_e_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___f_12_; lean_object* v___x_13_; 
lean_inc_ref(v_e_10_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__0), 2, 1);
lean_closure_set(v___f_11_, 0, v_e_10_);
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_restrictPreimageFinset___redArg___lam__1), 2, 1);
lean_closure_set(v___f_12_, 0, v_e_10_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___f_11_);
lean_ctor_set(v___x_13_, 1, v___f_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset(lean_object* v_00_u03b1_14_, lean_object* v_00_u03b2_15_, lean_object* v_e_16_, lean_object* v_s_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Equiv_restrictPreimageFinset___redArg(v_e_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_restrictPreimageFinset___boxed(lean_object* v_00_u03b1_19_, lean_object* v_00_u03b2_20_, lean_object* v_e_21_, lean_object* v_s_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Equiv_restrictPreimageFinset(v_00_u03b1_19_, v_00_u03b2_20_, v_e_21_, v_s_22_);
lean_dec(v_s_22_);
return v_res_23_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
}
#ifdef __cplusplus
}
#endif
