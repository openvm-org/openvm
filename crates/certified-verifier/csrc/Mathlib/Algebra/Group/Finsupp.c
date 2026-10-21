// Lean compiler output
// Module: Mathlib.Algebra.Group.Finsupp
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.Group.Pi.Torsion public import Mathlib.Data.Finset.Max public import Mathlib.Data.Finsupp.Single
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
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___redArg___lam__0(lean_object* v_a_1_, lean_object* v_g_2_){
_start:
{
lean_object* v_toFun_3_; lean_object* v___x_4_; 
v_toFun_3_ = lean_ctor_get(v_g_2_, 1);
lean_inc(v_toFun_3_);
lean_dec_ref(v_g_2_);
v___x_4_ = lean_apply_1(v_toFun_3_, v_a_1_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___redArg(lean_object* v_a_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_applyAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_a_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom(lean_object* v_00_u03b9_7_, lean_object* v_M_8_, lean_object* v_inst_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___f_11_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_applyAddHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_11_, 0, v_a_10_);
return v___f_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_applyAddHom___boxed(lean_object* v_00_u03b9_12_, lean_object* v_M_13_, lean_object* v_inst_14_, lean_object* v_a_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Finsupp_applyAddHom(v_00_u03b9_12_, v_M_13_, v_inst_14_, v_a_15_);
lean_dec_ref(v_inst_14_);
return v_res_16_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Torsion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Single(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Finsupp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Single(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Finsupp(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Single(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Finsupp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Single(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Finsupp(builtin);
}
#ifdef __cplusplus
}
#endif
