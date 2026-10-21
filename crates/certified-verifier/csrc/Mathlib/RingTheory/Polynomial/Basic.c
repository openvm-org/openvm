// Lean compiler output
// Module: Mathlib.RingTheory.Polynomial.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.MvPolynomial.Equiv public import Mathlib.Algebra.Polynomial.BigOperators public import Mathlib.Algebra.Ring.GeomSum public import Mathlib.RingTheory.Noetherian.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Ideal_ofPolynomial(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_ofPolynomial___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_ofPolynomial(lean_object* v_R_1_, lean_object* v_inst_2_, lean_object* v_I_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_ofPolynomial___boxed(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_I_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Ideal_ofPolynomial(v_R_5_, v_inst_6_, v_I_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_GeomSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Noetherian_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Polynomial_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
