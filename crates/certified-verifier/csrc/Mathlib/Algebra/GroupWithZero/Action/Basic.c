// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.End public import Mathlib.Algebra.GroupWithZero.Action.Defs public import Mathlib.Algebra.Group.Action.Prod public import Mathlib.Algebra.GroupWithZero.Prod
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
lean_object* lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulAction_toPerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_smulMulHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddAut___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddAut(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___f_4_; lean_object* v___x_5_; lean_object* v_invFun_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_13_; 
lean_inc(v_x_3_);
lean_inc(v_inst_2_);
v___f_4_ = lean_alloc_closure((void*)(lp_mathlib_SMulZeroClass_toZeroHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_4_, 0, v_inst_2_);
lean_closure_set(v___f_4_, 1, v_x_3_);
v___x_5_ = lp_mathlib_MulAction_toPerm___redArg(v_inst_1_, v_inst_2_, v_x_3_);
v_invFun_6_ = lean_ctor_get(v___x_5_, 1);
v_isSharedCheck_13_ = !lean_is_exclusive(v___x_5_);
if (v_isSharedCheck_13_ == 0)
{
lean_object* v_unused_14_; 
v_unused_14_ = lean_ctor_get(v___x_5_, 0);
lean_dec(v_unused_14_);
v___x_8_ = v___x_5_;
v_isShared_9_ = v_isSharedCheck_13_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_invFun_6_);
lean_dec(v___x_5_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_13_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_11_; 
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 0, v___f_4_);
v___x_11_ = v___x_8_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_12_; 
v_reuseFailAlloc_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_12_, 0, v___f_4_);
lean_ctor_set(v_reuseFailAlloc_12_, 1, v_invFun_6_);
v___x_11_ = v_reuseFailAlloc_12_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg___boxed(lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_x_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_DistribMulAction_toAddEquiv___redArg(v_inst_15_, v_inst_16_, v_x_17_);
lean_dec_ref(v_inst_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv(lean_object* v_G_19_, lean_object* v_A_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_x_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_DistribMulAction_toAddEquiv___redArg(v_inst_21_, v_inst_23_, v_x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddEquiv___boxed(lean_object* v_G_26_, lean_object* v_A_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_DistribMulAction_toAddEquiv(v_G_26_, v_A_27_, v_inst_28_, v_inst_29_, v_inst_30_, v_x_31_);
lean_dec_ref(v_inst_29_);
lean_dec_ref(v_inst_28_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddAut___redArg(lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toAddEquiv___boxed), 6, 5);
lean_closure_set(v___x_36_, 0, lean_box(0));
lean_closure_set(v___x_36_, 1, lean_box(0));
lean_closure_set(v___x_36_, 2, v_inst_33_);
lean_closure_set(v___x_36_, 3, v_inst_34_);
lean_closure_set(v___x_36_, 4, v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_toAddAut(lean_object* v_G_37_, lean_object* v_A_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toAddEquiv___boxed), 6, 5);
lean_closure_set(v___x_42_, 0, lean_box(0));
lean_closure_set(v___x_42_, 1, lean_box(0));
lean_closure_set(v___x_42_, 2, v_inst_39_);
lean_closure_set(v___x_42_, 3, v_inst_40_);
lean_closure_set(v___x_42_, 4, v_inst_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_44_, 0, v_inst_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom(lean_object* v_M_u2080_45_, lean_object* v_N_u2080_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_smulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_52_, 0, v_inst_49_);
return v___f_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_smulMonoidWithZeroHom___boxed(lean_object* v_M_u2080_53_, lean_object* v_N_u2080_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_smulMonoidWithZeroHom(v_M_u2080_53_, v_N_u2080_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_inst_59_);
lean_dec_ref(v_inst_56_);
lean_dec_ref(v_inst_55_);
return v_res_60_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
