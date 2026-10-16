// Lean compiler output
// Module: Mathlib.Algebra.Field.ZMod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Basic public import Mathlib.Data.ZMod.Basic
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
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
lean_object* lp_mathlib_ZMod_inv___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__0(lean_object* v_toNatCast_1_, lean_object* v_toIntCast_2_, lean_object* v___x_3_, lean_object* v_toMul_4_, lean_object* v_x_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lp_mathlib_Rat_castRec___redArg(v_toNatCast_1_, v_toIntCast_2_, v___x_3_, v_x_5_);
v___x_8_ = lean_apply_2(v_toMul_4_, v___x_7_, v___y_6_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg___lam__1(lean_object* v_toNatCast_9_, lean_object* v___x_10_, lean_object* v_toMul_11_, lean_object* v_x_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lp_mathlib_NNRat_castRec___redArg(v_toNatCast_9_, v___x_10_, v_x_12_);
v___x_15_ = lean_apply_2(v_toMul_11_, v___x_14_, v___y_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField___redArg(lean_object* v_p_16_){
_start:
{
lean_object* v___x_17_; lean_object* v_toSemiring_18_; lean_object* v_toMonoid_19_; lean_object* v_toIntCast_20_; lean_object* v_toNatCast_21_; lean_object* v_toOne_22_; lean_object* v_toMul_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___f_26_; lean_object* v___f_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
lean_inc(v_p_16_);
v___x_17_ = lp_mathlib_ZMod_commRing(v_p_16_);
v_toSemiring_18_ = lean_ctor_get(v___x_17_, 0);
lean_inc_ref(v_toSemiring_18_);
v_toMonoid_19_ = lean_ctor_get(v_toSemiring_18_, 1);
lean_inc_ref(v_toMonoid_19_);
v_toIntCast_20_ = lean_ctor_get(v___x_17_, 4);
lean_inc_n(v_toIntCast_20_, 2);
v_toNatCast_21_ = lean_ctor_get(v_toSemiring_18_, 2);
lean_inc_n(v_toNatCast_21_, 4);
lean_dec_ref(v_toSemiring_18_);
v_toOne_22_ = lean_ctor_get(v_toMonoid_19_, 0);
lean_inc_n(v_toOne_22_, 2);
v_toMul_23_ = lean_ctor_get(v_toMonoid_19_, 1);
lean_inc_n(v_toMul_23_, 4);
v___x_24_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_inv___boxed), 2, 1);
lean_closure_set(v___x_24_, 0, v_p_16_);
lean_inc_ref_n(v___x_24_, 2);
v___x_25_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_25_, 0, lean_box(0));
lean_closure_set(v___x_25_, 1, v_toMonoid_19_);
lean_closure_set(v___x_25_, 2, v___x_24_);
lean_inc_ref_n(v___x_25_, 4);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_instField___redArg___lam__0), 6, 4);
lean_closure_set(v___f_26_, 0, v_toNatCast_21_);
lean_closure_set(v___f_26_, 1, v_toIntCast_20_);
lean_closure_set(v___f_26_, 2, v___x_25_);
lean_closure_set(v___f_26_, 3, v_toMul_23_);
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_ZMod_instField___redArg___lam__1), 5, 3);
lean_closure_set(v___f_27_, 0, v_toNatCast_21_);
lean_closure_set(v___f_27_, 1, v___x_25_);
lean_closure_set(v___f_27_, 2, v_toMul_23_);
v___x_28_ = lean_alloc_closure((void*)(l_npowRec___boxed), 5, 3);
lean_closure_set(v___x_28_, 0, lean_box(0));
lean_closure_set(v___x_28_, 1, v_toOne_22_);
lean_closure_set(v___x_28_, 2, v_toMul_23_);
v___x_29_ = lean_alloc_closure((void*)(lp_mathlib_zpowRec___boxed), 7, 5);
lean_closure_set(v___x_29_, 0, lean_box(0));
lean_closure_set(v___x_29_, 1, v_toOne_22_);
lean_closure_set(v___x_29_, 2, v_toMul_23_);
lean_closure_set(v___x_29_, 3, v___x_24_);
lean_closure_set(v___x_29_, 4, v___x_28_);
v___x_30_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_castRec), 4, 3);
lean_closure_set(v___x_30_, 0, lean_box(0));
lean_closure_set(v___x_30_, 1, v_toNatCast_21_);
lean_closure_set(v___x_30_, 2, v___x_25_);
v___x_31_ = lean_alloc_closure((void*)(lp_mathlib_Rat_castRec), 5, 4);
lean_closure_set(v___x_31_, 0, lean_box(0));
lean_closure_set(v___x_31_, 1, v_toNatCast_21_);
lean_closure_set(v___x_31_, 2, v_toIntCast_20_);
lean_closure_set(v___x_31_, 3, v___x_25_);
v___x_32_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_32_, 0, v___x_17_);
lean_ctor_set(v___x_32_, 1, v___x_24_);
lean_ctor_set(v___x_32_, 2, v___x_25_);
lean_ctor_set(v___x_32_, 3, v___x_29_);
lean_ctor_set(v___x_32_, 4, v___x_30_);
lean_ctor_set(v___x_32_, 5, v___x_31_);
lean_ctor_set(v___x_32_, 6, v___f_27_);
lean_ctor_set(v___x_32_, 7, v___f_26_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZMod_instField(lean_object* v_p_33_, lean_object* v_hp_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_ZMod_instField___redArg(v_p_33_);
return v___x_35_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_ZMod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_ZMod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_ZMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_ZMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_ZMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_ZMod(builtin);
}
#ifdef __cplusplus
}
#endif
