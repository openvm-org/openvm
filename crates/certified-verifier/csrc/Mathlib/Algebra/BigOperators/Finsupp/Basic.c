// Lean compiler output
// Module: Mathlib.Algebra.BigOperators.Finsupp.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Sigma public import Mathlib.Algebra.BigOperators.Pi public import Mathlib.Algebra.BigOperators.Ring.Finset public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Data.Finsupp.Ext public import Mathlib.Data.Finsupp.Indicator
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
lean_object* lp_mathlib_Finset_sum___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg___lam__0(lean_object* v_toFun_1_, lean_object* v_g_2_, lean_object* v_a_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
lean_inc(v_a_3_);
v___x_4_ = lean_apply_1(v_toFun_1_, v_a_3_);
v___x_5_ = lean_apply_2(v_g_2_, v_a_3_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg(lean_object* v_inst_6_, lean_object* v_f_7_, lean_object* v_g_8_){
_start:
{
lean_object* v_support_9_; lean_object* v_toFun_10_; lean_object* v___f_11_; lean_object* v___x_12_; 
v_support_9_ = lean_ctor_get(v_f_7_, 0);
lean_inc(v_support_9_);
v_toFun_10_ = lean_ctor_get(v_f_7_, 1);
lean_inc(v_toFun_10_);
lean_dec_ref(v_f_7_);
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_11_, 0, v_toFun_10_);
lean_closure_set(v___f_11_, 1, v_g_8_);
v___x_12_ = lp_mathlib_Finset_prod___redArg(v_inst_6_, v_support_9_, v___f_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___redArg___boxed(lean_object* v_inst_13_, lean_object* v_f_14_, lean_object* v_g_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Finsupp_prod___redArg(v_inst_13_, v_f_14_, v_g_15_);
lean_dec_ref(v_inst_13_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod(lean_object* v_00_u03b1_17_, lean_object* v_M_18_, lean_object* v_N_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_f_22_, lean_object* v_g_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Finsupp_prod___redArg(v_inst_21_, v_f_22_, v_g_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_prod___boxed(lean_object* v_00_u03b1_25_, lean_object* v_M_26_, lean_object* v_N_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_f_30_, lean_object* v_g_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Finsupp_prod(v_00_u03b1_25_, v_M_26_, v_N_27_, v_inst_28_, v_inst_29_, v_f_30_, v_g_31_);
lean_dec_ref(v_inst_29_);
lean_dec(v_inst_28_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___redArg(lean_object* v_inst_33_, lean_object* v_f_34_, lean_object* v_g_35_){
_start:
{
lean_object* v_support_36_; lean_object* v_toFun_37_; lean_object* v___f_38_; lean_object* v___x_39_; 
v_support_36_ = lean_ctor_get(v_f_34_, 0);
lean_inc(v_support_36_);
v_toFun_37_ = lean_ctor_get(v_f_34_, 1);
lean_inc(v_toFun_37_);
lean_dec_ref(v_f_34_);
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_38_, 0, v_toFun_37_);
lean_closure_set(v___f_38_, 1, v_g_35_);
v___x_39_ = lp_mathlib_Finset_sum___redArg(v_inst_33_, v_support_36_, v___f_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___redArg___boxed(lean_object* v_inst_40_, lean_object* v_f_41_, lean_object* v_g_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Finsupp_sum___redArg(v_inst_40_, v_f_41_, v_g_42_);
lean_dec_ref(v_inst_40_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum(lean_object* v_00_u03b1_44_, lean_object* v_M_45_, lean_object* v_N_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_f_49_, lean_object* v_g_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Finsupp_sum___redArg(v_inst_48_, v_f_49_, v_g_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___boxed(lean_object* v_00_u03b1_52_, lean_object* v_M_53_, lean_object* v_N_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_f_57_, lean_object* v_g_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Finsupp_sum(v_00_u03b1_52_, v_M_53_, v_N_54_, v_inst_55_, v_inst_56_, v_f_57_, v_g_58_);
lean_dec_ref(v_inst_56_);
lean_dec(v_inst_55_);
return v_res_59_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Ext(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Indicator(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Ext(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Indicator(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
