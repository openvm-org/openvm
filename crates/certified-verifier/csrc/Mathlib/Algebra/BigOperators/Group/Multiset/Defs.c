// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Group.Multiset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.List.Defs public import Mathlib.Algebra.Group.Basic public import Mathlib.Data.Multiset.Basic public import Mathlib.Data.Multiset.Filter
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toMul_1_, v_x1_2_, v_x2_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg(lean_object* v_inst_5_, lean_object* v_s_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_toOne_9_; lean_object* v_toMul_10_; lean_object* v___f_11_; lean_object* v___x_12_; 
v___x_7_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_5_);
v___x_8_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_7_);
v_toOne_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_toOne_9_);
v_toMul_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_toMul_10_);
lean_dec_ref(v___x_8_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_prod___redArg___lam__0), 3, 1);
lean_closure_set(v___f_11_, 0, v_toMul_10_);
v___x_12_ = l_List_foldrTR___redArg(v___f_11_, v_toOne_9_, v_s_6_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___redArg___boxed(lean_object* v_inst_13_, lean_object* v_s_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_Multiset_prod___redArg(v_inst_13_, v_s_14_);
lean_dec_ref(v_inst_13_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod(lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Multiset_prod___redArg(v_inst_17_, v_s_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_prod___boxed(lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_s_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Multiset_prod(v_M_20_, v_inst_21_, v_s_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg___lam__0(lean_object* v_toAdd_24_, lean_object* v_x1_25_, lean_object* v_x2_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_apply_2(v_toAdd_24_, v_x1_25_, v_x2_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg(lean_object* v_inst_28_, lean_object* v_s_29_){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v_toZero_32_; lean_object* v_toAdd_33_; lean_object* v___f_34_; lean_object* v___x_35_; 
v___x_30_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_28_);
v___x_31_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_30_);
v_toZero_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc(v_toZero_32_);
v_toAdd_33_ = lean_ctor_get(v___x_31_, 1);
lean_inc(v_toAdd_33_);
lean_dec_ref(v___x_31_);
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sum___redArg___lam__0), 3, 1);
lean_closure_set(v___f_34_, 0, v_toAdd_33_);
v___x_35_ = l_List_foldrTR___redArg(v___f_34_, v_toZero_32_, v_s_29_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___redArg___boxed(lean_object* v_inst_36_, lean_object* v_s_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Multiset_sum___redArg(v_inst_36_, v_s_37_);
lean_dec_ref(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum(lean_object* v_M_39_, lean_object* v_inst_40_, lean_object* v_s_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_Multiset_sum___redArg(v_inst_40_, v_s_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___boxed(lean_object* v_M_43_, lean_object* v_inst_44_, lean_object* v_s_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Multiset_sum(v_M_43_, v_inst_44_, v_s_45_);
lean_dec_ref(v_inst_44_);
return v_res_46_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
