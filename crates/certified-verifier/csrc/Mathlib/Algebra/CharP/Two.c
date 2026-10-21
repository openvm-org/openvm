// Lean compiler output
// Module: Mathlib.Algebra.CharP.Two
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Finset.Defs public import Mathlib.Algebra.CharP.Defs public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Algebra.Ring.Parity
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg___lam__0(lean_object* v_toNPow_1_, lean_object* v_x_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_unsigned_to_nat(2u);
v___x_4_ = lean_apply_2(v_toNPow_1_, v___x_3_, v_x_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v_toMonoid_6_; lean_object* v_toNPow_7_; lean_object* v___f_8_; 
v_toMonoid_6_ = lean_ctor_get(v_inst_5_, 1);
lean_inc_ref(v_toMonoid_6_);
lean_dec_ref(v_inst_5_);
v_toNPow_7_ = lean_ctor_get(v_toMonoid_6_, 2);
lean_inc(v_toNPow_7_);
lean_dec_ref(v_toMonoid_6_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_toNPow_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom(lean_object* v_R_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib___private_Mathlib_Algebra_CharP_Two_0__CharTwo_sqAddMonoidHom___redArg(v_inst_10_);
return v___x_12_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_CharP_Two(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharP_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_CharP_Two(builtin);
}
#ifdef __cplusplus
}
#endif
