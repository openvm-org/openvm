// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.Roots
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Polynomial.BigOperators public import Mathlib.Algebra.Polynomial.RingDivision public import Mathlib.Data.Set.Card public import Mathlib.Data.Set.Finite.Lemmas public import Mathlib.RingTheory.Coprime.Lemmas public import Mathlib.RingTheory.Localization.FractionRing public import Mathlib.SetTheory.Cardinal.Order public import Mathlib.Order.Filter.TendstoCofinite
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
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_g_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_g_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass(lean_object* v_R_7_, lean_object* v_S_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_G_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_15_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass___boxed(lean_object* v_R_19_, lean_object* v_S_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_G_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_f_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Polynomial_instMulActionElemRootSetOfSMulCommClass(v_R_19_, v_S_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_G_25_, v_inst_26_, v_inst_27_, v_inst_28_, v_f_29_);
lean_dec_ref(v_f_29_);
lean_dec_ref(v_inst_26_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec_ref(v_inst_21_);
return v_res_30_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_TendstoCofinite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_TendstoCofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_TendstoCofinite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_RingDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_TendstoCofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
}
#ifdef __cplusplus
}
#endif
