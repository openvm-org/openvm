// Lean compiler output
// Module: Mathlib.Data.Multiset.Find
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Find public import Mathlib.Data.Multiset.AddSub public import Mathlib.Data.Multiset.Basic public import Mathlib.Data.Set.Subsingleton
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
lean_object* l_List_find_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_find_x3f___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_find_x3f___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_b_2_);
v___x_4_ = lean_unbox(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f___redArg___lam__0___boxed(lean_object* v_inst_5_, lean_object* v_b_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_mathlib_Multiset_find_x3f___redArg___lam__0(v_inst_5_, v_b_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f___redArg(lean_object* v_inst_9_, lean_object* v_s_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_find_x3f___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_11_, 0, v_inst_9_);
v___x_12_ = l_List_find_x3f___redArg(v___f_11_, v_s_10_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_find_x3f(lean_object* v_00_u03b1_13_, lean_object* v_p_14_, lean_object* v_inst_15_, lean_object* v_s_16_, lean_object* v_a_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Multiset_find_x3f___redArg(v_inst_15_, v_s_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___redArg(uint8_t v_x_19_, lean_object* v_h__1_20_, lean_object* v_h__2_21_){
_start:
{
if (v_x_19_ == 0)
{
lean_object* v___x_22_; lean_object* v___x_23_; 
lean_dec(v_h__1_20_);
v___x_22_ = lean_box(0);
v___x_23_ = lean_apply_1(v_h__2_21_, v___x_22_);
return v___x_23_;
}
else
{
lean_object* v___x_24_; lean_object* v___x_25_; 
lean_dec(v_h__2_21_);
v___x_24_ = lean_box(0);
v___x_25_ = lean_apply_1(v_h__1_20_, v___x_24_);
return v___x_25_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___redArg___boxed(lean_object* v_x_26_, lean_object* v_h__1_27_, lean_object* v_h__2_28_){
_start:
{
uint8_t v_x_24__boxed_29_; lean_object* v_res_30_; 
v_x_24__boxed_29_ = lean_unbox(v_x_26_);
v_res_30_ = lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___redArg(v_x_24__boxed_29_, v_h__1_27_, v_h__2_28_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter(lean_object* v_motive_31_, uint8_t v_x_32_, lean_object* v_h__1_33_, lean_object* v_h__2_34_){
_start:
{
if (v_x_32_ == 0)
{
lean_object* v___x_35_; lean_object* v___x_36_; 
lean_dec(v_h__1_33_);
v___x_35_ = lean_box(0);
v___x_36_ = lean_apply_1(v_h__2_34_, v___x_35_);
return v___x_36_;
}
else
{
lean_object* v___x_37_; lean_object* v___x_38_; 
lean_dec(v_h__2_34_);
v___x_37_ = lean_box(0);
v___x_38_ = lean_apply_1(v_h__1_33_, v___x_37_);
return v___x_38_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter___boxed(lean_object* v_motive_39_, lean_object* v_x_40_, lean_object* v_h__1_41_, lean_object* v_h__2_42_){
_start:
{
uint8_t v_x_35__boxed_43_; lean_object* v_res_44_; 
v_x_35__boxed_43_ = lean_unbox(v_x_40_);
v_res_44_ = lp_mathlib___private_Mathlib_Data_Multiset_Find_0__List_filter_match__1_splitter(v_motive_39_, v_x_35__boxed_43_, v_h__1_41_, v_h__2_42_);
return v_res_44_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Find(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Find(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_AddSub(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Find(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_AddSub(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Find(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_AddSub(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Find(builtin);
}
#ifdef __cplusplus
}
#endif
