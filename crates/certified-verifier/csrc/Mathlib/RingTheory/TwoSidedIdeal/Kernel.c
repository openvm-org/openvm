// Lean compiler output
// Module: Mathlib.RingTheory.TwoSidedIdeal.Kernel
// Imports: public import Init public meta import Init public import Mathlib.RingTheory.TwoSidedIdeal.Basic public import Mathlib.RingTheory.TwoSidedIdeal.Lattice
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
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_ker(lean_object* v_R_1_, lean_object* v_S_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_F_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_f_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_ker___boxed(lean_object* v_R_10_, lean_object* v_S_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_F_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_TwoSidedIdeal_ker(v_R_10_, v_S_11_, v_inst_12_, v_inst_13_, v_F_14_, v_inst_15_, v_inst_16_, v_f_17_);
lean_dec(v_f_17_);
lean_dec(v_inst_15_);
lean_dec_ref(v_inst_13_);
lean_dec_ref(v_inst_12_);
return v_res_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Lattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Lattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Kernel(builtin);
}
#ifdef __cplusplus
}
#endif
