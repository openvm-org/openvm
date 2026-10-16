// Lean compiler output
// Module: Mathlib.SetTheory.Ordinal.Univ
// Imports: public import Init public meta import Init public import Mathlib.SetTheory.Ordinal.Basic
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
lean_object* lp_mathlib_Ordinal_liftInitialSeg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_univ;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_univ;
static const lean_closure_object lp_mathlib_Ordinal_liftPrincipalSeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ordinal_liftInitialSeg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ordinal_liftPrincipalSeg___closed__0 = (const lean_object*)&lp_mathlib_Ordinal_liftPrincipalSeg___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordinal_liftPrincipalSeg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ordinal_liftPrincipalSeg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordinal_liftPrincipalSeg___closed__1 = (const lean_object*)&lp_mathlib_Ordinal_liftPrincipalSeg___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordinal_liftPrincipalSeg = (const lean_object*)&lp_mathlib_Ordinal_liftPrincipalSeg___closed__1_value;
static lean_object* _init_lp_mathlib_Ordinal_univ(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_univ(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Ordinal_univ = _init_lp_mathlib_Ordinal_univ();
lean_mark_persistent(lp_mathlib_Ordinal_univ);
lp_mathlib_Cardinal_univ = _init_lp_mathlib_Cardinal_univ();
lean_mark_persistent(lp_mathlib_Cardinal_univ);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Ordinal_Univ(builtin);
}
#ifdef __cplusplus
}
#endif
