// Lean compiler output
// Module: Mathlib.Algebra.Ring.Invertible
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Invertible public import Mathlib.Algebra.Ring.Defs
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulLeft___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_y_3_){
_start:
{
lean_object* v___x_4_; lean_object* v_toMul_5_; lean_object* v_val_6_; lean_object* v_neg_7_; lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_16_; 
v___x_4_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_1_);
v_toMul_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc(v_toMul_5_);
lean_dec_ref(v___x_4_);
v_val_6_ = lean_ctor_get(v_x_2_, 0);
v_neg_7_ = lean_ctor_get(v_x_2_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_x_2_);
if (v_isSharedCheck_16_ == 0)
{
v___x_9_ = v_x_2_;
v_isShared_10_ = v_isSharedCheck_16_;
goto v_resetjp_8_;
}
else
{
lean_inc(v_neg_7_);
lean_inc(v_val_6_);
lean_dec(v_x_2_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_16_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_14_; 
lean_inc(v_toMul_5_);
lean_inc(v_y_3_);
v___x_11_ = lean_apply_2(v_toMul_5_, v_y_3_, v_val_6_);
v___x_12_ = lean_apply_2(v_toMul_5_, v_y_3_, v_neg_7_);
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 1, v___x_12_);
lean_ctor_set(v___x_9_, 0, v___x_11_);
v___x_14_ = v___x_9_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_11_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v___x_12_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulLeft(lean_object* v_R_17_, lean_object* v_inst_18_, lean_object* v_x_19_, lean_object* v_y_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_AddUnits_mulLeft___redArg(v_inst_18_, v_x_19_, v_y_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulRight___redArg(lean_object* v_inst_22_, lean_object* v_x_23_, lean_object* v_y_24_){
_start:
{
lean_object* v___x_25_; lean_object* v_toMul_26_; lean_object* v_val_27_; lean_object* v_neg_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_37_; 
v___x_25_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_22_);
v_toMul_26_ = lean_ctor_get(v___x_25_, 0);
lean_inc(v_toMul_26_);
lean_dec_ref(v___x_25_);
v_val_27_ = lean_ctor_get(v_x_23_, 0);
v_neg_28_ = lean_ctor_get(v_x_23_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v_x_23_);
if (v_isSharedCheck_37_ == 0)
{
v___x_30_ = v_x_23_;
v_isShared_31_ = v_isSharedCheck_37_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_neg_28_);
lean_inc(v_val_27_);
lean_dec(v_x_23_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_37_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_35_; 
lean_inc(v_toMul_26_);
lean_inc(v_y_24_);
v___x_32_ = lean_apply_2(v_toMul_26_, v_val_27_, v_y_24_);
v___x_33_ = lean_apply_2(v_toMul_26_, v_neg_28_, v_y_24_);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 1, v___x_33_);
lean_ctor_set(v___x_30_, 0, v___x_32_);
v___x_35_ = v___x_30_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v___x_32_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v___x_33_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_mulRight(lean_object* v_R_38_, lean_object* v_inst_39_, lean_object* v_x_40_, lean_object* v_y_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_AddUnits_mulRight___redArg(v_inst_39_, v_x_40_, v_y_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_apply_1(v_inst_43_, v_inst_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg(lean_object* v_R_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_a_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_apply_1(v_inst_49_, v_inst_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleNeg___boxed(lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_a_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_invertibleNeg(v_R_53_, v_inst_54_, v_inst_55_, v_inst_56_, v_a_57_, v_inst_58_);
lean_dec(v_a_57_);
lean_dec(v_inst_55_);
lean_dec(v_inst_54_);
return v_res_59_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Invertible(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Invertible(builtin);
}
#ifdef __cplusplus
}
#endif
