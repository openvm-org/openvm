// Lean compiler output
// Module: Mathlib.Algebra.Group.Commutator
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Data.Bracket
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement___redArg___lam__0(lean_object* v_toMul_1_, lean_object* v_toInv_2_, lean_object* v_g_u2081_3_, lean_object* v_g_u2082_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
lean_inc_n(v_toMul_1_, 2);
lean_inc(v_g_u2082_4_);
lean_inc(v_g_u2081_3_);
v___x_5_ = lean_apply_2(v_toMul_1_, v_g_u2081_3_, v_g_u2082_4_);
lean_inc(v_toInv_2_);
v___x_6_ = lean_apply_1(v_toInv_2_, v_g_u2081_3_);
v___x_7_ = lean_apply_2(v_toMul_1_, v___x_5_, v___x_6_);
v___x_8_ = lean_apply_1(v_toInv_2_, v_g_u2082_4_);
v___x_9_ = lean_apply_2(v_toMul_1_, v___x_7_, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v_toMonoid_11_; lean_object* v_toInv_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v_toMul_15_; lean_object* v___f_16_; 
v_toMonoid_11_ = lean_ctor_get(v_inst_10_, 0);
lean_inc_ref(v_toMonoid_11_);
v_toInv_12_ = lean_ctor_get(v_inst_10_, 1);
lean_inc(v_toInv_12_);
lean_dec_ref(v_inst_10_);
v___x_13_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_11_);
lean_dec_ref(v_toMonoid_11_);
v___x_14_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_13_);
v_toMul_15_ = lean_ctor_get(v___x_14_, 1);
lean_inc(v_toMul_15_);
lean_dec_ref(v___x_14_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_commutatorElement___redArg___lam__0), 4, 2);
lean_closure_set(v___f_16_, 0, v_toMul_15_);
lean_closure_set(v___f_16_, 1, v_toInv_12_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commutatorElement(lean_object* v_G_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_commutatorElement___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement___redArg___lam__0(lean_object* v_toAdd_20_, lean_object* v_toNeg_21_, lean_object* v_g_u2081_22_, lean_object* v_g_u2082_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
lean_inc_n(v_toAdd_20_, 2);
lean_inc(v_g_u2082_23_);
lean_inc(v_g_u2081_22_);
v___x_24_ = lean_apply_2(v_toAdd_20_, v_g_u2081_22_, v_g_u2082_23_);
lean_inc(v_toNeg_21_);
v___x_25_ = lean_apply_1(v_toNeg_21_, v_g_u2081_22_);
v___x_26_ = lean_apply_2(v_toAdd_20_, v___x_24_, v___x_25_);
v___x_27_ = lean_apply_1(v_toNeg_21_, v_g_u2082_23_);
v___x_28_ = lean_apply_2(v_toAdd_20_, v___x_26_, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v_toAddMonoid_30_; lean_object* v_toNeg_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v_toAdd_34_; lean_object* v___f_35_; 
v_toAddMonoid_30_ = lean_ctor_get(v_inst_29_, 0);
lean_inc_ref(v_toAddMonoid_30_);
v_toNeg_31_ = lean_ctor_get(v_inst_29_, 1);
lean_inc(v_toNeg_31_);
lean_dec_ref(v_inst_29_);
v___x_32_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_30_);
lean_dec_ref(v_toAddMonoid_30_);
v___x_33_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_32_);
v_toAdd_34_ = lean_ctor_get(v___x_33_, 1);
lean_inc(v_toAdd_34_);
lean_dec_ref(v___x_33_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_addCommutatorElement___redArg___lam__0), 4, 2);
lean_closure_set(v___f_35_, 0, v_toAdd_34_);
lean_closure_set(v___f_35_, 1, v_toNeg_31_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addCommutatorElement(lean_object* v_G_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_addCommutatorElement___redArg(v_inst_37_);
return v___x_38_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Bracket(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Bracket(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Bracket(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Bracket(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
}
#ifdef __cplusplus
}
#endif
