// Lean compiler output
// Module: Mathlib.Data.Finset.Union
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Fold public import Mathlib.Data.Multiset.Bind public import Mathlib.Order.SetNotation
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
lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion___redArg___lam__0(lean_object* v_t_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_t_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion___redArg(lean_object* v_s_4_, lean_object* v_t_5_){
_start:
{
lean_object* v___f_6_; lean_object* v___x_7_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finset_disjiUnion___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_t_5_);
v___x_7_ = lp_mathlib_Multiset_bind___redArg(v_s_4_, v___f_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_disjiUnion(lean_object* v_00_u03b1_8_, lean_object* v_00_u03b2_9_, lean_object* v_s_10_, lean_object* v_t_11_, lean_object* v_hf_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Finset_disjiUnion___redArg(v_s_10_, v_t_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion___redArg___lam__0(lean_object* v_t_14_, lean_object* v_a_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_1(v_t_14_, v_a_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion___redArg(lean_object* v_inst_17_, lean_object* v_s_18_, lean_object* v_t_19_){
_start:
{
lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Finset_biUnion___redArg___lam__0), 2, 1);
lean_closure_set(v___f_20_, 0, v_t_19_);
v___x_21_ = lp_mathlib_Multiset_bind___redArg(v_s_18_, v___f_20_);
v___x_22_ = lp_mathlib_List_dedup___redArg(v_inst_17_, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_biUnion(lean_object* v_00_u03b1_23_, lean_object* v_00_u03b2_24_, lean_object* v_inst_25_, lean_object* v_s_26_, lean_object* v_t_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Finset_biUnion___redArg(v_inst_25_, v_s_26_, v_t_27_);
return v___x_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Union(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Union(builtin);
}
#ifdef __cplusplus
}
#endif
