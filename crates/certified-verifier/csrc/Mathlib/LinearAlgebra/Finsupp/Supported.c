// Lean compiler output
// Module: Mathlib.LinearAlgebra.Finsupp.Supported
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Range public import Mathlib.LinearAlgebra.Finsupp.LSum public import Mathlib.LinearAlgebra.Span.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_supported(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_supported___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_supported(lean_object* v_00_u03b1_1_, lean_object* v_M_2_, lean_object* v_R_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_s_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_supported___boxed(lean_object* v_00_u03b1_9_, lean_object* v_M_10_, lean_object* v_R_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_s_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Finsupp_supported(v_00_u03b1_9_, v_M_10_, v_R_11_, v_inst_12_, v_inst_13_, v_inst_14_, v_s_15_);
lean_dec(v_inst_14_);
lean_dec_ref(v_inst_13_);
lean_dec_ref(v_inst_12_);
return v_res_16_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LSum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_Supported(builtin);
}
#ifdef __cplusplus
}
#endif
