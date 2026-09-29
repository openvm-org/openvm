// Lean compiler output
// Module: Mathlib.LinearAlgebra.Basis.Cardinality
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Basis.Defs public import Mathlib.LinearAlgebra.LinearIndependent.Defs public import Mathlib.LinearAlgebra.Span.Basic public import Mathlib.SetTheory.Cardinal.Pigeonhole
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_Basis_Cardinality_0__Option_elim_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_Basis_Cardinality_0__Option_elim_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_Basis_Cardinality_0__Option_elim_match__1_splitter___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_x_3_, lean_object* v_h__1_4_, lean_object* v_h__2_5_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_6_; 
lean_dec(v_h__1_4_);
v___x_6_ = lean_apply_2(v_h__2_5_, v_x_2_, v_x_3_);
return v___x_6_;
}
else
{
lean_object* v_val_7_; lean_object* v___x_8_; 
lean_dec(v_h__2_5_);
v_val_7_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_val_7_);
lean_dec_ref_known(v_x_1_, 1);
v___x_8_ = lean_apply_3(v_h__1_4_, v_val_7_, v_x_2_, v_x_3_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_LinearAlgebra_Basis_Cardinality_0__Option_elim_match__1_splitter(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_motive_11_, lean_object* v_x_12_, lean_object* v_x_13_, lean_object* v_x_14_, lean_object* v_h__1_15_, lean_object* v_h__2_16_){
_start:
{
if (lean_obj_tag(v_x_12_) == 0)
{
lean_object* v___x_17_; 
lean_dec(v_h__1_15_);
v___x_17_ = lean_apply_2(v_h__2_16_, v_x_13_, v_x_14_);
return v___x_17_;
}
else
{
lean_object* v_val_18_; lean_object* v___x_19_; 
lean_dec(v_h__2_16_);
v_val_18_ = lean_ctor_get(v_x_12_, 0);
lean_inc(v_val_18_);
lean_dec_ref_known(v_x_12_, 1);
v___x_19_ = lean_apply_3(v_h__1_15_, v_val_18_, v_x_13_, v_x_14_);
return v___x_19_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Pigeonhole(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Pigeonhole(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Basis_Cardinality(builtin);
}
#ifdef __cplusplus
}
#endif
