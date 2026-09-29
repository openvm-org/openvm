// Lean compiler output
// Module: Mathlib.Algebra.Ring.AddAut
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.Basic public import Mathlib.Algebra.GroupWithZero.Action.Units public import Mathlib.Algebra.Group.Units.Opposite public import Mathlib.Algebra.Module.Opposite
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
lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Units_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Units_opEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toOppositeModule___redArg(lean_object*);
lean_object* lp_mathlib_Units_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_DistribMulAction_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMonoid_2_; lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v_toNonUnitalNonAssocSemiring_5_; lean_object* v_toAddCommMonoid_6_; lean_object* v___x_7_; lean_object* v___f_8_; lean_object* v___x_9_; 
v_toMonoid_2_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_toMonoid_2_);
v___x_3_ = lp_mathlib_Units_instDivInvMonoid___redArg(v_toMonoid_2_);
lean_inc_ref(v_inst_1_);
v___x_4_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_1_);
v_toNonUnitalNonAssocSemiring_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_5_);
lean_dec_ref(v___x_4_);
v_toAddCommMonoid_6_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_5_, 0);
lean_inc_ref(v_toAddCommMonoid_6_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_5_);
v___x_7_ = lp_mathlib_Semiring_toModule___redArg(v_inst_1_);
lean_dec_ref(v_inst_1_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_8_, 0, v___x_7_);
v___x_9_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_toAddEquiv___boxed), 6, 5);
lean_closure_set(v___x_9_, 0, lean_box(0));
lean_closure_set(v___x_9_, 1, lean_box(0));
lean_closure_set(v___x_9_, 2, v___x_3_);
lean_closure_set(v___x_9_, 3, v_toAddCommMonoid_6_);
lean_closure_set(v___x_9_, 4, v___f_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulLeft(lean_object* v_R_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_AddAut_mulLeft___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight___redArg(lean_object* v_inst_13_, lean_object* v_u_14_){
_start:
{
lean_object* v_toMonoid_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v_toFun_20_; lean_object* v___x_21_; lean_object* v___f_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v_toMonoid_15_ = lean_ctor_get(v_inst_13_, 1);
lean_inc_ref(v_toMonoid_15_);
v___x_16_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_toMonoid_15_);
v___x_17_ = lp_mathlib_Units_instDivInvMonoid___redArg(v___x_16_);
v___x_18_ = lp_mathlib_Units_opEquiv(lean_box(0), v_toMonoid_15_);
v___x_19_ = lp_mathlib_Equiv_symm___redArg(v___x_18_);
v_toFun_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc(v_toFun_20_);
lean_dec_ref(v___x_19_);
v___x_21_ = lp_mathlib_Semiring_toOppositeModule___redArg(v_inst_13_);
lean_dec_ref(v_inst_13_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Units_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v___x_21_);
v___x_23_ = lean_apply_1(v_toFun_20_, v_u_14_);
v___x_24_ = lp_mathlib_DistribMulAction_toAddEquiv___redArg(v___x_17_, v___f_22_, v___x_23_);
lean_dec_ref(v___x_17_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddAut_mulRight(lean_object* v_R_25_, lean_object* v_inst_26_, lean_object* v_u_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_AddAut_mulRight___redArg(v_inst_26_, v_u_27_);
return v___x_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_AddAut(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_AddAut(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_AddAut(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_AddAut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_AddAut(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_AddAut(builtin);
}
#ifdef __cplusplus
}
#endif
