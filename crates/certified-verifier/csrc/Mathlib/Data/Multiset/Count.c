// Lean compiler output
// Module: Mathlib.Data.Multiset.Count
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Nodup public import Mathlib.Data.Multiset.ZeroCons
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
lean_object* l_List_countP_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_countP___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_countP___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_b_2_);
v___x_4_ = lean_unbox(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___redArg___lam__0___boxed(lean_object* v_inst_5_, lean_object* v_b_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_mathlib_Multiset_countP___redArg___lam__0(v_inst_5_, v_b_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP___redArg(lean_object* v_inst_9_, lean_object* v_s_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_countP___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_11_, 0, v_inst_9_);
v___x_12_ = lean_unsigned_to_nat(0u);
v___x_13_ = l_List_countP_go___redArg(v___f_11_, v_s_10_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_countP(lean_object* v_00_u03b1_14_, lean_object* v_p_15_, lean_object* v_inst_16_, lean_object* v_s_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Multiset_countP___redArg(v_inst_16_, v_s_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count___redArg(lean_object* v_inst_19_, lean_object* v_a_20_, lean_object* v_s_21_){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = lean_apply_1(v_inst_19_, v_a_20_);
v___x_23_ = lp_mathlib_Multiset_countP___redArg(v___x_22_, v_s_21_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_count(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_a_26_, lean_object* v_s_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Multiset_count___redArg(v_inst_25_, v_a_26_, v_s_27_);
return v___x_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Nodup(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Count(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Nodup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_ZeroCons(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Count(builtin);
}
#ifdef __cplusplus
}
#endif
