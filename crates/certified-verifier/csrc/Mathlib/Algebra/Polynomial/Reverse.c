// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Reverse
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Polynomial.Degree.TrailingDegree public import Mathlib.Algebra.Polynomial.EraseLead
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
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun(lean_object* v_N_1_, lean_object* v_i_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_le(v_i_2_, v_N_1_);
if (v___x_3_ == 0)
{
lean_inc(v_i_2_);
return v_i_2_;
}
else
{
lean_object* v___x_4_; 
v___x_4_ = lean_nat_sub(v_N_1_, v_i_2_);
return v___x_4_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAtFun___boxed(lean_object* v_N_5_, lean_object* v_i_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Polynomial_revAtFun(v_N_5_, v_i_6_);
lean_dec(v_i_6_);
lean_dec(v_N_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0(lean_object* v_N_8_, lean_object* v_i_9_){
_start:
{
uint8_t v___x_10_; 
v___x_10_ = lean_nat_dec_le(v_i_9_, v_N_8_);
if (v___x_10_ == 0)
{
lean_inc(v_i_9_);
return v_i_9_;
}
else
{
lean_object* v___x_11_; 
v___x_11_ = lean_nat_sub(v_N_8_, v_i_9_);
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt___lam__0___boxed(lean_object* v_N_12_, lean_object* v_i_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Polynomial_revAt___lam__0(v_N_12_, v_i_13_);
lean_dec(v_i_13_);
lean_dec(v_N_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_revAt(lean_object* v_N_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_revAt___lam__0___boxed), 2, 1);
lean_closure_set(v___f_16_, 0, v_N_15_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___redArg(lean_object* v_x_17_, lean_object* v_h__1_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_apply_1(v_h__1_18_, v_x_17_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_motive_22_, lean_object* v_x_23_, lean_object* v_h__1_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_apply_1(v_h__1_24_, v_x_23_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter___boxed(lean_object* v_R_26_, lean_object* v_inst_27_, lean_object* v_motive_28_, lean_object* v_x_29_, lean_object* v_h__1_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib___private_Mathlib_Algebra_Polynomial_Reverse_0__Polynomial_reflect_match__1_splitter(v_R_26_, v_inst_27_, v_motive_28_, v_x_29_, v_h__1_30_);
lean_dec_ref(v_inst_27_);
return v_res_31_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_TrailingDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Reverse(builtin);
}
#ifdef __cplusplus
}
#endif
