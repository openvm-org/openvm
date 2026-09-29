// Lean compiler output
// Module: Mathlib.RingTheory.MvPolynomial.WeightedHomogeneous
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Finprod public import Mathlib.Algebra.DirectSum.Decomposition public import Mathlib.Algebra.GradedMonoid public import Mathlib.Algebra.MvPolynomial.Basic public import Mathlib.Algebra.Order.Monoid.Canonical.Defs public import Mathlib.Data.Finsupp.Weight public import Mathlib.RingTheory.GradedAlgebra.Homogeneous.Ideal public import Mathlib.RingTheory.MvPolynomial.Basic public import Mathlib.Tactic.Order
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
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_weightedHomogeneousSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_weightedHomogeneousSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_weightedHomogeneousSubmodule(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_00_u03c3_4_, lean_object* v_inst_5_, lean_object* v_w_6_, lean_object* v_m_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MvPolynomial_weightedHomogeneousSubmodule___boxed(lean_object* v_R_9_, lean_object* v_M_10_, lean_object* v_inst_11_, lean_object* v_00_u03c3_12_, lean_object* v_inst_13_, lean_object* v_w_14_, lean_object* v_m_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_MvPolynomial_weightedHomogeneousSubmodule(v_R_9_, v_M_10_, v_inst_11_, v_00_u03c3_12_, v_inst_13_, v_w_14_, v_m_15_);
lean_dec(v_m_15_);
lean_dec(v_w_14_);
lean_dec_ref(v_inst_13_);
lean_dec_ref(v_inst_11_);
return v_res_16_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Weight(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Weight(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Weight(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Finprod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_Canonical_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Weight(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_MvPolynomial_WeightedHomogeneous(builtin);
}
#ifdef __cplusplus
}
#endif
