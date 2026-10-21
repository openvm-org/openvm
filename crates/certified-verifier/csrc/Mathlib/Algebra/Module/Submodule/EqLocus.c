// Lean compiler output
// Module: Mathlib.Algebra.Module.Submodule.EqLocus
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Ker
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
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_eqLocus(lean_object* v_R_1_, lean_object* v_R_u2082_2_, lean_object* v_M_3_, lean_object* v_M_u2082_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_00_u03c4_u2081_u2082_11_, lean_object* v_f_12_, lean_object* v_g_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_box(0);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_eqLocus___boxed(lean_object* v_R_15_, lean_object* v_R_u2082_16_, lean_object* v_M_17_, lean_object* v_M_u2082_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_00_u03c4_u2081_u2082_25_, lean_object* v_f_26_, lean_object* v_g_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_LinearMap_eqLocus(v_R_15_, v_R_u2082_16_, v_M_17_, v_M_u2082_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_00_u03c4_u2081_u2082_25_, v_f_26_, v_g_27_);
lean_dec(v_g_27_);
lean_dec(v_f_26_);
lean_dec(v_00_u03c4_u2081_u2082_25_);
lean_dec(v_inst_24_);
lean_dec(v_inst_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_19_);
return v_res_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Submodule_EqLocus(builtin);
}
#ifdef __cplusplus
}
#endif
