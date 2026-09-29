// Lean compiler output
// Module: Mathlib.Algebra.Group.Semigroup
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Regular.Defs public import Mathlib.Tactic.MkIffOfInductiveProp public import Mathlib.Tactic.Simps
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
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_lower__cancel__priority;
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_CommSemigroup_toCommMagma___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma(lean_object* v_G_4_, lean_object* v_self_5_){
_start:
{
lean_inc(v_self_5_);
return v_self_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemigroup_toCommMagma___boxed(lean_object* v_G_6_, lean_object* v_self_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_CommSemigroup_toCommMagma(v_G_6_, v_self_7_);
lean_dec(v_self_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(lean_object* v_self_9_){
_start:
{
lean_inc(v_self_9_);
return v_self_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg___boxed(lean_object* v_self_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_AddCommSemigroup_toAddCommMagma___redArg(v_self_10_);
lean_dec(v_self_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma(lean_object* v_G_12_, lean_object* v_self_13_){
_start:
{
lean_inc(v_self_13_);
return v_self_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommSemigroup_toAddCommMagma___boxed(lean_object* v_G_14_, lean_object* v_self_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddCommSemigroup_toAddCommMagma(v_G_14_, v_self_15_);
lean_dec(v_self_15_);
return v_res_16_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_lower__cancel__priority(void){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_box(0);
return v___x_17_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Regular_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Regular_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_lower__cancel__priority = _init_lp_mathlib_LibraryNote_lower__cancel__priority();
lean_mark_persistent(lp_mathlib_LibraryNote_lower__cancel__priority);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Regular_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Semigroup(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Regular_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Semigroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Semigroup(builtin);
}
#ifdef __cplusplus
}
#endif
