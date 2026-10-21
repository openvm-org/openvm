// Lean compiler output
// Module: Mathlib.Tactic.NormNum
// Imports: public import Init public meta import Init public import Mathlib.Data.Rat.Cast.Order public import Mathlib.Tactic.NormNum.Abs public import Mathlib.Tactic.NormNum.Basic public import Mathlib.Tactic.NormNum.DivMod public import Mathlib.Tactic.NormNum.Eq public import Mathlib.Tactic.NormNum.Ineq public import Mathlib.Tactic.NormNum.Inv public import Mathlib.Tactic.NormNum.OfScientific public import Mathlib.Tactic.NormNum.Pow
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
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_NormNum(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Abs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Eq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Inv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Pow(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_NormNum(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Abs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_DivMod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Eq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Ineq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Inv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_OfScientific(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Pow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_NormNum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_NormNum(builtin);
}
#ifdef __cplusplus
}
#endif
