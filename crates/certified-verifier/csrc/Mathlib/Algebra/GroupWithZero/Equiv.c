// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Equiv
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.GroupWithZero.Hom
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
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_f_1_, 0);
lean_inc(v_toFun_3_);
lean_dec_ref(v_f_1_);
v___x_4_ = lean_apply_1(v_toFun_3_, v___y_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg(lean_object* v_f_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_f_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom(lean_object* v_G_7_, lean_object* v_H_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_f_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toMonoidWithZeroHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_12_, 0, v_f_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidWithZeroHom___boxed(lean_object* v_G_13_, lean_object* v_H_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_MulEquiv_toMonoidWithZeroHom(v_G_13_, v_H_14_, v_inst_15_, v_inst_16_, v_f_17_);
lean_dec_ref(v_inst_16_);
lean_dec_ref(v_inst_15_);
return v_res_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Equiv(builtin);
}
#ifdef __cplusplus
}
#endif
