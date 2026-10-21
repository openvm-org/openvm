// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Algebra.GroupWithZero.NeZero public import Mathlib.Basic.Unique public import Mathlib.Tactic.Conv public import Batteries.Tactic.SeqFocus
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
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfZeroEqOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfZeroEqOne(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfZeroEqOne___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_toZero_3_; 
v___x_2_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_1_);
v_toZero_3_ = lean_ctor_get(v___x_2_, 1);
lean_inc(v_toZero_3_);
lean_dec_ref(v___x_2_);
return v_toZero_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfZeroEqOne(lean_object* v_M_u2080_4_, lean_object* v_inst_5_, lean_object* v_h_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_uniqueOfZeroEqOne___redArg(v_inst_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid___redArg(lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GroupWithZero_toDivisionMonoid(lean_object* v_G_u2080_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_11_);
return v___x_12_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
