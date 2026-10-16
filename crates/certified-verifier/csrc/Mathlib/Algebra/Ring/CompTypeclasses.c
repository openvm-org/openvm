// Lean compiler output
// Module: Mathlib.Algebra.Ring.CompTypeclasses
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Equiv
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
lean_object* lp_mathlib_RingEquiv_ofRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv___redArg(lean_object* v_00_u03c3_1_, lean_object* v_00_u03c3_x27_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_RingEquiv_ofRingHom___redArg(v_00_u03c3_1_, v_00_u03c3_x27_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv(lean_object* v_R_u2081_4_, lean_object* v_R_u2082_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_00_u03c3_8_, lean_object* v_00_u03c3_x27_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_RingEquiv_ofRingHom___redArg(v_00_u03c3_8_, v_00_u03c3_x27_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHomInvPair_toRingEquiv___boxed(lean_object* v_R_u2081_12_, lean_object* v_R_u2082_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_00_u03c3_16_, lean_object* v_00_u03c3_x27_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_RingHomInvPair_toRingEquiv(v_R_u2081_12_, v_R_u2082_13_, v_inst_14_, v_inst_15_, v_00_u03c3_16_, v_00_u03c3_x27_17_, v_inst_18_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_19_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_CompTypeclasses(builtin);
}
#ifdef __cplusplus
}
#endif
