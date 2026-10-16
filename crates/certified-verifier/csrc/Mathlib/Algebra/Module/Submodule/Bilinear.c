// Lean compiler output
// Module: Mathlib.Algebra.Module.Submodule.Bilinear
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Span.Basic public import Mathlib.LinearAlgebra.BilinearMap
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
lean_object* lp_mathlib_Submodule_completeLattice___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v_toConditionallyCompletePartialOrderSup_7_; lean_object* v_toSupSet_8_; lean_object* v___x_9_; 
v___x_4_ = lp_mathlib_Submodule_completeLattice___redArg(v_inst_1_, v_inst_2_, v_inst_3_);
v___x_5_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_4_);
v___x_6_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_5_);
v_toConditionallyCompletePartialOrderSup_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_7_);
lean_dec_ref(v___x_6_);
v_toSupSet_8_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_7_, 1);
lean_inc(v_toSupSet_8_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_7_);
v___x_9_ = lean_apply_1(v_toSupSet_8_, lean_box(0));
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___redArg___boxed(lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Submodule_map_u2082___redArg(v_inst_10_, v_inst_11_, v_inst_12_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082(lean_object* v_R_14_, lean_object* v_M_15_, lean_object* v_N_16_, lean_object* v_P_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_f_25_, lean_object* v_p_26_, lean_object* v_q_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Submodule_map_u2082___redArg(v_inst_18_, v_inst_21_, v_inst_24_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_map_u2082___boxed(lean_object* v_R_29_, lean_object* v_M_30_, lean_object* v_N_31_, lean_object* v_P_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_f_40_, lean_object* v_p_41_, lean_object* v_q_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Submodule_map_u2082(v_R_29_, v_M_30_, v_N_31_, v_P_32_, v_inst_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_inst_39_, v_f_40_, v_p_41_, v_q_42_);
lean_dec(v_f_40_);
lean_dec(v_inst_39_);
lean_dec(v_inst_38_);
lean_dec(v_inst_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_35_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
return v_res_43_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_BilinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Submodule_Bilinear(builtin);
}
#ifdef __cplusplus
}
#endif
