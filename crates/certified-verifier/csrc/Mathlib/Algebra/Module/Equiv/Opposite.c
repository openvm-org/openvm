// Lean compiler output
// Module: Mathlib.Algebra.Module.Equiv.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.Opposite
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
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
static lean_once_cell_t lp_mathlib_MulOpposite_opLinearEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulOpposite_opLinearEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opLinearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_MulOpposite_opLinearEquiv___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opLinearEquiv(lean_object* v_R_2_, lean_object* v_M_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; lean_object* v_toFun_8_; lean_object* v_invFun_9_; lean_object* v___x_10_; 
v___x_7_ = lean_obj_once(&lp_mathlib_MulOpposite_opLinearEquiv___closed__0, &lp_mathlib_MulOpposite_opLinearEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opLinearEquiv___closed__0);
v_toFun_8_ = lean_ctor_get(v___x_7_, 0);
v_invFun_9_ = lean_ctor_get(v___x_7_, 1);
lean_inc(v_invFun_9_);
lean_inc(v_toFun_8_);
v___x_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_10_, 0, v_toFun_8_);
lean_ctor_set(v___x_10_, 1, v_invFun_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opLinearEquiv___boxed(lean_object* v_R_11_, lean_object* v_M_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_MulOpposite_opLinearEquiv(v_R_11_, v_M_12_, v_inst_13_, v_inst_14_, v_inst_15_);
lean_dec(v_inst_15_);
lean_dec_ref(v_inst_14_);
lean_dec_ref(v_inst_13_);
return v_res_16_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Equiv_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
