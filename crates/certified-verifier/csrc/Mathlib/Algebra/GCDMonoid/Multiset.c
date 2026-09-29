// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Multiset
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GCDMonoid.Basic public import Mathlib.Algebra.Order.Group.Multiset public import Mathlib.Data.Multiset.FinsetOps public import Mathlib.Data.Multiset.Fold
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
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_lcm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_lcm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_gcd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_gcd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_lcm___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_s_3_){
_start:
{
lean_object* v_toGCDMonoid_4_; lean_object* v_lcm_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v_toMulOneClass_8_; lean_object* v___x_9_; lean_object* v_toOne_10_; lean_object* v___x_11_; 
v_toGCDMonoid_4_ = lean_ctor_get(v_inst_2_, 1);
lean_inc_ref(v_toGCDMonoid_4_);
lean_dec_ref(v_inst_2_);
v_lcm_5_ = lean_ctor_get(v_toGCDMonoid_4_, 1);
lean_inc(v_lcm_5_);
lean_dec_ref(v_toGCDMonoid_4_);
v___x_6_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_1_);
v___x_7_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_6_);
v_toMulOneClass_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_toMulOneClass_8_);
lean_dec_ref(v___x_7_);
v___x_9_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_8_);
v_toOne_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc(v_toOne_10_);
lean_dec_ref(v___x_9_);
v___x_11_ = l_List_foldrTR___redArg(v_lcm_5_, v_toOne_10_, v_s_3_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_lcm(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_s_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Multiset_lcm___redArg(v_inst_13_, v_inst_14_, v_s_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_gcd___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_s_19_){
_start:
{
lean_object* v_toGCDMonoid_20_; lean_object* v_gcd_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v_toZero_25_; lean_object* v___x_26_; 
v_toGCDMonoid_20_ = lean_ctor_get(v_inst_18_, 1);
lean_inc_ref(v_toGCDMonoid_20_);
lean_dec_ref(v_inst_18_);
v_gcd_21_ = lean_ctor_get(v_toGCDMonoid_20_, 0);
lean_inc(v_gcd_21_);
lean_dec_ref(v_toGCDMonoid_20_);
v___x_22_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_17_);
v___x_23_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_22_);
v___x_24_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_23_);
v_toZero_25_ = lean_ctor_get(v___x_24_, 1);
lean_inc(v_toZero_25_);
lean_dec_ref(v___x_24_);
v___x_26_ = l_List_foldrTR___redArg(v_gcd_21_, v_toZero_25_, v_s_19_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_gcd(lean_object* v_00_u03b1_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_s_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Multiset_gcd___redArg(v_inst_28_, v_inst_29_, v_s_30_);
return v___x_31_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Fold(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_FinsetOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GCDMonoid_Multiset(builtin);
}
#ifdef __cplusplus
}
#endif
