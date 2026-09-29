// Lean compiler output
// Module: Mathlib.Algebra.Ring.Action.Group
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.Basic public import Mathlib.Algebra.Ring.Action.Basic public import Mathlib.Algebra.Ring.Aut public import Mathlib.Algebra.Ring.Equiv
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
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instMulSemiringActionRingEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instMulSemiringActionRingEquiv___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instMulSemiringActionRingEquiv___closed__0 = (const lean_object*)&lp_mathlib_instMulSemiringActionRingEquiv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_DistribMulAction_toAddEquiv___redArg(v_inst_1_, v_inst_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0___boxed(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_x_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0(v_inst_5_, v_inst_6_, v_x_7_);
lean_dec_ref(v_inst_5_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___redArg(lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_11_, 0, v_inst_9_);
lean_closure_set(v___f_11_, 1, v_inst_10_);
return v___f_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv(lean_object* v_G_12_, lean_object* v_inst_13_, lean_object* v_R_14_, lean_object* v_inst_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_toRingEquiv___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_17_, 0, v_inst_13_);
lean_closure_set(v___f_17_, 1, v_inst_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingEquiv___boxed(lean_object* v_G_18_, lean_object* v_inst_19_, lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_MulSemiringAction_toRingEquiv(v_G_18_, v_inst_19_, v_R_20_, v_inst_21_, v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv___lam__0(lean_object* v_x1_24_, lean_object* v_x2_25_){
_start:
{
lean_object* v_toFun_26_; lean_object* v___x_27_; 
v_toFun_26_ = lean_ctor_get(v_x1_24_, 0);
lean_inc(v_toFun_26_);
lean_dec_ref(v_x1_24_);
v___x_27_ = lean_apply_1(v_toFun_26_, v_x2_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv(lean_object* v_R_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___f_31_; 
v___f_31_ = ((lean_object*)(lp_mathlib_instMulSemiringActionRingEquiv___closed__0));
return v___f_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulSemiringActionRingEquiv___boxed(lean_object* v_R_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_instMulSemiringActionRingEquiv(v_R_32_, v_inst_33_);
lean_dec_ref(v_inst_33_);
return v_res_34_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Aut(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Aut(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Aut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Action_Group(builtin);
}
#ifdef __cplusplus
}
#endif
