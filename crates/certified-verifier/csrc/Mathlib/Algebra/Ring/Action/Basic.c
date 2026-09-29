// Lean compiler output
// Module: Mathlib.Algebra.Ring.Action.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Basic public import Mathlib.Algebra.GroupWithZero.Action.End public import Mathlib.Algebra.Ring.Hom.Defs
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
lean_object* lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SMul_comp_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingHom_applyMulSemiringAction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_applyMulSemiringAction___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingHom_applyMulSemiringAction___closed__0 = (const lean_object*)&lp_mathlib_RingHom_applyMulSemiringAction___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___redArg(lean_object* v_h_1_){
_start:
{
lean_inc(v_h_1_);
return v_h_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___redArg___boxed(lean_object* v_h_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_MulSemiringAction_toMulDistribMulAction___redArg(v_h_2_);
lean_dec(v_h_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction(lean_object* v_M_4_, lean_object* v_R_5_, lean_object* v_x_6_, lean_object* v_x_7_, lean_object* v_h_8_){
_start:
{
lean_inc(v_h_8_);
return v_h_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toMulDistribMulAction___boxed(lean_object* v_M_9_, lean_object* v_R_10_, lean_object* v_x_11_, lean_object* v_x_12_, lean_object* v_h_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_MulSemiringAction_toMulDistribMulAction(v_M_9_, v_R_10_, v_x_11_, v_x_12_, v_h_13_);
lean_dec(v_h_13_);
lean_dec_ref(v_x_12_);
lean_dec_ref(v_x_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom___redArg(lean_object* v_inst_15_, lean_object* v_x_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_17_, 0, v_inst_15_);
lean_closure_set(v___f_17_, 1, v_x_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom(lean_object* v_M_18_, lean_object* v_inst_19_, lean_object* v_R_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMonoidHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_24_, 0, v_inst_22_);
lean_closure_set(v___f_24_, 1, v_x_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_toRingHom___boxed(lean_object* v_M_25_, lean_object* v_inst_26_, lean_object* v_R_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_MulSemiringAction_toRingHom(v_M_25_, v_inst_26_, v_R_27_, v_inst_28_, v_inst_29_, v_x_30_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_26_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction___lam__0(lean_object* v_x1_32_, lean_object* v_x2_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lean_apply_1(v_x1_32_, v_x2_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction(lean_object* v_R_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = ((lean_object*)(lp_mathlib_RingHom_applyMulSemiringAction___closed__0));
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_applyMulSemiringAction___boxed(lean_object* v_R_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_RingHom_applyMulSemiringAction(v_R_39_, v_inst_40_);
lean_dec_ref(v_inst_40_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___redArg___lam__0(lean_object* v_f_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lean_apply_1(v_f_42_, v___y_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___redArg(lean_object* v_f_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; lean_object* v___x_48_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_compHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_47_, 0, v_f_45_);
v___x_48_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_48_, 0, lean_box(0));
lean_closure_set(v___x_48_, 1, lean_box(0));
lean_closure_set(v___x_48_, 2, lean_box(0));
lean_closure_set(v___x_48_, 3, v_inst_46_);
lean_closure_set(v___x_48_, 4, v___f_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom(lean_object* v_M_49_, lean_object* v_N_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_f_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___x_58_; 
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_MulSemiringAction_compHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_57_, 0, v_f_55_);
v___x_58_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_58_, 0, lean_box(0));
lean_closure_set(v___x_58_, 1, lean_box(0));
lean_closure_set(v___x_58_, 2, lean_box(0));
lean_closure_set(v___x_58_, 3, v_inst_56_);
lean_closure_set(v___x_58_, 4, v___f_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulSemiringAction_compHom___boxed(lean_object* v_M_59_, lean_object* v_N_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_R_63_, lean_object* v_inst_64_, lean_object* v_f_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_MulSemiringAction_compHom(v_M_59_, v_N_60_, v_inst_61_, v_inst_62_, v_R_63_, v_inst_64_, v_f_65_, v_inst_66_);
lean_dec_ref(v_inst_64_);
lean_dec_ref(v_inst_62_);
lean_dec_ref(v_inst_61_);
return v_res_67_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
