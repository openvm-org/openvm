// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Pointwise
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.End public import Mathlib.Algebra.Group.Subgroup.MulOppositeLemmas public import Mathlib.Algebra.Group.Subgroup.ZPowers.Basic public import Mathlib.Algebra.Group.Submonoid.Pointwise public import Mathlib.GroupTheory.GroupAction.ConjAct public import Mathlib.Algebra.Group.Pointwise.Set.Lattice
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
lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_pointwiseMulAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_pointwiseMulAction___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_pointwiseMulAction___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_pointwiseMulAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___lam__0(lean_object* v_a_1_, lean_object* v_S_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___lam__0___boxed(lean_object* v_a_4_, lean_object* v_S_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Subgroup_pointwiseMulAction___lam__0(v_a_4_, v_S_5_);
lean_dec(v_a_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction(lean_object* v_00_u03b1_8_, lean_object* v_G_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; 
v___f_13_ = ((lean_object*)(lp_mathlib_Subgroup_pointwiseMulAction___closed__0));
return v___f_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_pointwiseMulAction___boxed(lean_object* v_00_u03b1_14_, lean_object* v_G_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Subgroup_pointwiseMulAction(v_00_u03b1_14_, v_G_15_, v_inst_16_, v_inst_17_, v_inst_18_);
lean_dec(v_inst_18_);
lean_dec_ref(v_inst_17_);
lean_dec_ref(v_inst_16_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = lp_mathlib_MulDistribMulAction_toMulEquiv___redArg(v_inst_20_, v_inst_21_, v_a_22_);
v___x_24_ = lp_mathlib_MulEquiv_submonoidMap___redArg(v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___redArg___boxed(lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_a_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Subgroup_equivSMul___redArg(v_inst_25_, v_inst_26_, v_a_27_);
lean_dec_ref(v_inst_25_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul(lean_object* v_00_u03b1_29_, lean_object* v_G_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_a_34_, lean_object* v_H_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Subgroup_equivSMul___redArg(v_inst_32_, v_inst_33_, v_a_34_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_equivSMul___boxed(lean_object* v_00_u03b1_37_, lean_object* v_G_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_a_42_, lean_object* v_H_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Subgroup_equivSMul(v_00_u03b1_37_, v_G_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_a_42_, v_H_43_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
return v_res_44_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Lattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Lattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_MulOppositeLemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
}
#ifdef __cplusplus
}
#endif
