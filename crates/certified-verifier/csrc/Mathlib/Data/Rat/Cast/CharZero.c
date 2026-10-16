// Lean compiler output
// Module: Mathlib.Data.Rat.Cast.CharZero
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Units.Lemmas public import Mathlib.Data.Rat.Cast.Defs
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
lean_object* lp_batteries_Rat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_castHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_castHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_castHom___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toRatCast_2_; lean_object* v___x_3_; 
v_toRatCast_2_ = lean_ctor_get(v_inst_1_, 5);
lean_inc(v_toRatCast_2_);
lean_dec_ref(v_inst_1_);
v___x_3_ = lean_alloc_closure((void*)(lp_batteries_Rat_cast), 3, 2);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v_toRatCast_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_castHom(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Rat_castHom___redArg(v_inst_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castHom___redArg(lean_object* v_inst_8_){
_start:
{
lean_object* v_toNNRatCast_9_; lean_object* v___x_10_; 
v_toNNRatCast_9_ = lean_ctor_get(v_inst_8_, 4);
lean_inc(v_toNNRatCast_9_);
lean_dec_ref(v_inst_8_);
v___x_10_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_cast), 3, 2);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, v_toNNRatCast_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castHom(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_NNRat_castHom___redArg(v_inst_12_);
return v___x_14_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Rat_Cast_CharZero(builtin);
}
#ifdef __cplusplus
}
#endif
