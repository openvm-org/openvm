// Lean compiler output
// Module: Mathlib.RingTheory.Ideal.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Defs public import Mathlib.Tactic.Abel
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
LEAN_EXPORT lean_object* lp_mathlib_Module_eqIdeal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_eqIdeal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_inertia(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_inertia___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_eqIdeal(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_m_6_, lean_object* v_m_x27_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_eqIdeal___boxed(lean_object* v_R_9_, lean_object* v_M_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_m_14_, lean_object* v_m_x27_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Module_eqIdeal(v_R_9_, v_M_10_, v_inst_11_, v_inst_12_, v_inst_13_, v_m_14_, v_m_x27_15_);
lean_dec(v_m_x27_15_);
lean_dec(v_m_14_);
lean_dec(v_inst_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_inertia(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_G_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_I_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_box(0);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_inertia___boxed(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_G_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_I_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Ideal_inertia(v_00_u03b1_24_, v_inst_25_, v_G_26_, v_inst_27_, v_inst_28_, v_I_29_);
lean_dec(v_inst_28_);
lean_dec_ref(v_inst_27_);
lean_dec_ref(v_inst_25_);
return v_res_30_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Ideal_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
