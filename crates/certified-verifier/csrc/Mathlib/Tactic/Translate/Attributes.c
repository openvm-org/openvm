// Lean compiler output
// Module: Mathlib.Tactic.Translate.Attributes
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_generatingAttrs;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_registerGeneratingAttr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_registerGeneratingAttr___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_2_ = lean_box(1);
v___x_3_ = lean_st_mk_ref(v___x_2_);
v___x_4_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4_, 0, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2____boxed(lean_object* v_a_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2_();
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_registerGeneratingAttr(lean_object* v_attr_7_, lean_object* v_impl_8_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_10_ = lp_mathlib_Mathlib_Tactic_generatingAttrs;
v___x_11_ = lean_st_ref_take(v___x_10_);
v___x_12_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_attr_7_, v_impl_8_, v___x_11_);
v___x_13_ = lean_st_ref_set(v___x_10_, v___x_12_);
v___x_14_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_registerGeneratingAttr___boxed(lean_object* v_attr_15_, lean_object* v_impl_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Mathlib_Tactic_registerGeneratingAttr(v_attr_15_, v_impl_16_);
return v_res_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Translate_Attributes_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Translate_Attributes_23622836____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_generatingAttrs = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_generatingAttrs);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_Attributes(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_Attributes(builtin);
}
#ifdef __cplusplus
}
#endif
