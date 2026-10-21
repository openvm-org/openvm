// Lean compiler output
// Module: Mathlib.Order.Filter.Cofinite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.FiniteSupport.Defs public import Mathlib.Basic.Finite.Prod public import Mathlib.Data.Set.Finite.Lemmas public import Mathlib.Order.Filter.AtTopBot.Basic public import Mathlib.Order.Filter.CountablyGenerated public import Mathlib.Order.Filter.Ker public import Mathlib.Order.Filter.Pi public import Mathlib.Order.Filter.Prod public import Mathlib.Order.Heyting.Boundary
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
LEAN_EXPORT lean_object* lp_mathlib_Filter_cofinite(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_cofinite(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_CountablyGenerated(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Ker(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Boundary(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_CountablyGenerated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Boundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_CountablyGenerated(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Ker(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Heyting_Boundary(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Cofinite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FiniteSupport_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_AtTopBot_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_CountablyGenerated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Heyting_Boundary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Filter_Cofinite(builtin);
}
#ifdef __cplusplus
}
#endif
