// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Finset
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Fold public import Mathlib.Algebra.GCDMonoid.Multiset public import Mathlib.Algebra.GCDMonoid.Nat
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
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Finset_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_lcm___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_lcm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_gcd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_gcd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_lcm___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_s_3_, lean_object* v_f_4_){
_start:
{
lean_object* v_toGCDMonoid_5_; lean_object* v_lcm_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_toMulOneClass_9_; lean_object* v___x_10_; lean_object* v_toOne_11_; lean_object* v___x_12_; 
v_toGCDMonoid_5_ = lean_ctor_get(v_inst_2_, 1);
lean_inc_ref(v_toGCDMonoid_5_);
lean_dec_ref(v_inst_2_);
v_lcm_6_ = lean_ctor_get(v_toGCDMonoid_5_, 1);
lean_inc(v_lcm_6_);
lean_dec_ref(v_toGCDMonoid_5_);
v___x_7_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_1_);
v___x_8_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_7_);
v_toMulOneClass_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc_ref(v_toMulOneClass_9_);
lean_dec_ref(v___x_8_);
v___x_10_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_9_);
v_toOne_11_ = lean_ctor_get(v___x_10_, 0);
lean_inc(v_toOne_11_);
lean_dec_ref(v___x_10_);
v___x_12_ = lp_mathlib_Finset_fold___redArg(v_lcm_6_, v_toOne_11_, v_f_4_, v_s_3_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_lcm(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_s_17_, lean_object* v_f_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Finset_lcm___redArg(v_inst_15_, v_inst_16_, v_s_17_, v_f_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_gcd___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_s_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_toGCDMonoid_24_; lean_object* v_gcd_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v_toZero_29_; lean_object* v___x_30_; 
v_toGCDMonoid_24_ = lean_ctor_get(v_inst_21_, 1);
lean_inc_ref(v_toGCDMonoid_24_);
lean_dec_ref(v_inst_21_);
v_gcd_25_ = lean_ctor_get(v_toGCDMonoid_24_, 0);
lean_inc(v_gcd_25_);
lean_dec_ref(v_toGCDMonoid_24_);
v___x_26_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_20_);
v___x_27_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_26_);
v___x_28_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_27_);
v_toZero_29_ = lean_ctor_get(v___x_28_, 1);
lean_inc(v_toZero_29_);
lean_dec_ref(v___x_28_);
v___x_30_ = lp_mathlib_Finset_fold___redArg(v_gcd_25_, v_toZero_29_, v_f_23_, v_s_22_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_gcd(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_s_35_, lean_object* v_f_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Finset_gcd___redArg(v_inst_33_, v_inst_34_, v_s_35_, v_f_36_);
return v___x_37_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
}
#ifdef __cplusplus
}
#endif
