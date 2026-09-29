// Lean compiler output
// Module: Mathlib.GroupTheory.MonoidLocalization.MonoidWithZero
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.GroupWithZero.NonZeroDivisors public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.GroupTheory.MonoidLocalization.Maps public import Mathlib.RingTheory.OreLocalization.Basic
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
lean_object* lp_mathlib_OreLocalization_oreSetComm(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero___redArg(lean_object* v_inst_1_, lean_object* v_S_2_){
_start:
{
lean_object* v_toCommMonoid_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_toZero_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_17_; 
v_toCommMonoid_3_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref_n(v_toCommMonoid_3_, 2);
v___x_4_ = lp_mathlib_OreLocalization_oreSetComm(lean_box(0), v_toCommMonoid_3_, v_S_2_);
v___x_5_ = lp_mathlib_OreLocalization_instMonoid___redArg(v_toCommMonoid_3_, v_S_2_, v___x_4_);
v___x_6_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_1_);
v___x_7_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_6_);
v___x_8_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_7_);
v_toZero_9_ = lean_ctor_get(v___x_8_, 1);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_17_ == 0)
{
lean_object* v_unused_18_; 
v_unused_18_ = lean_ctor_get(v___x_8_, 0);
lean_dec(v_unused_18_);
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_17_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_toZero_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_17_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_13_; lean_object* v___x_15_; 
v___x_13_ = lp_mathlib_OreLocalization_zero___redArg(v_toCommMonoid_3_, v_toZero_9_);
lean_dec_ref(v_toCommMonoid_3_);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 1, v___x_13_);
lean_ctor_set(v___x_11_, 0, v___x_5_);
v___x_15_ = v___x_11_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_5_);
lean_ctor_set(v_reuseFailAlloc_16_, 1, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero(lean_object* v_M_19_, lean_object* v_inst_20_, lean_object* v_S_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Localization_instCommMonoidWithZero___redArg(v_inst_20_, v_S_21_);
return v___x_22_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Maps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Maps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
}
#ifdef __cplusplus
}
#endif
