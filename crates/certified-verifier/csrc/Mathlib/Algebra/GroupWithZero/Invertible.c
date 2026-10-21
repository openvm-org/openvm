// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Invertible
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Invertible.Basic public import Mathlib.Algebra.GroupWithZero.Units.Basic
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
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfNonzero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfNonzero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfNonzero___redArg(lean_object* v_inst_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v_toInv_5_; lean_object* v___x_6_; 
v___x_3_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_1_);
v___x_4_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_3_);
lean_dec_ref(v___x_3_);
v_toInv_5_ = lean_ctor_get(v___x_4_, 1);
lean_inc(v_toInv_5_);
lean_dec_ref(v___x_4_);
v___x_6_ = lean_apply_1(v_toInv_5_, v_a_2_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfNonzero(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_a_9_, lean_object* v_h_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_invertibleOfNonzero___redArg(v_inst_8_, v_a_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___redArg(lean_object* v_a_12_){
_start:
{
lean_inc(v_a_12_);
return v_a_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___redArg___boxed(lean_object* v_a_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_invertibleInv___redArg(v_a_13_);
lean_dec(v_a_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_a_17_, lean_object* v_inst_18_){
_start:
{
lean_inc(v_a_17_);
return v_a_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleInv___boxed(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_a_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_invertibleInv(v_00_u03b1_19_, v_inst_20_, v_a_21_, v_inst_22_);
lean_dec(v_inst_22_);
lean_dec(v_a_21_);
lean_dec_ref(v_inst_20_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv___redArg(lean_object* v_inst_24_, lean_object* v_a_25_, lean_object* v_b_26_){
_start:
{
lean_object* v___x_27_; lean_object* v_toDiv_28_; lean_object* v___x_29_; 
v___x_27_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_24_);
v_toDiv_28_ = lean_ctor_get(v___x_27_, 2);
lean_inc(v_toDiv_28_);
lean_dec_ref(v___x_27_);
v___x_29_ = lean_apply_2(v_toDiv_28_, v_b_26_, v_a_25_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv(lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_a_32_, lean_object* v_b_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_invertibleDiv___redArg(v_inst_31_, v_a_32_, v_b_33_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleDiv___boxed(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_a_39_, lean_object* v_b_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_invertibleDiv(v_00_u03b1_37_, v_inst_38_, v_a_39_, v_b_40_, v_inst_41_, v_inst_42_);
lean_dec(v_inst_42_);
lean_dec(v_inst_41_);
return v_res_43_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
}
#ifdef __cplusplus
}
#endif
