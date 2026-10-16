// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Idempotent
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Idempotent public import Mathlib.Algebra.GroupWithZero.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toZero_2_; 
v_toZero_2_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_toZero_2_);
return v_toZero_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___redArg___boxed(lean_object* v_inst_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_IsIdempotentElem_instZeroSubtype___redArg(v_inst_3_);
lean_dec_ref(v_inst_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype(lean_object* v_M_u2080_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v_toZero_7_; 
v_toZero_7_ = lean_ctor_get(v_inst_6_, 1);
lean_inc(v_toZero_7_);
return v_toZero_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsIdempotentElem_instZeroSubtype___boxed(lean_object* v_M_u2080_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_IsIdempotentElem_instZeroSubtype(v_M_u2080_8_, v_inst_9_);
lean_dec_ref(v_inst_9_);
return v_res_10_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Idempotent(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Idempotent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Idempotent(builtin);
}
#ifdef __cplusplus
}
#endif
