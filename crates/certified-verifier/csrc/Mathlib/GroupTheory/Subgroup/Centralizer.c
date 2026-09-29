// Lean compiler output
// Module: Mathlib.GroupTheory.Subgroup.Centralizer
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.End public import Mathlib.Algebra.Group.Commutator public import Mathlib.GroupTheory.Subgroup.Center public import Mathlib.GroupTheory.Submonoid.Centralizer
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
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer(lean_object* v_G_1_, lean_object* v_inst_2_, lean_object* v_s_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centralizer___boxed(lean_object* v_G_5_, lean_object* v_inst_6_, lean_object* v_s_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Subgroup_centralizer(v_G_5_, v_inst_6_, v_s_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer(lean_object* v_G_9_, lean_object* v_inst_10_, lean_object* v_s_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_centralizer___boxed(lean_object* v_G_13_, lean_object* v_inst_14_, lean_object* v_s_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddSubgroup_centralizer(v_G_13_, v_inst_14_, v_s_15_);
lean_dec_ref(v_inst_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm___redArg(lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closureCommGroupOfComm(lean_object* v_G_19_, lean_object* v_inst_20_, lean_object* v_k_21_, lean_object* v_hcomm_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_20_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm___redArg(lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closureAddCommGroupOfComm(lean_object* v_G_26_, lean_object* v_inst_27_, lean_object* v_k_28_, lean_object* v_hcomm_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_27_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0(lean_object* v_inst_31_, lean_object* v_toMul_32_, lean_object* v_g_33_, lean_object* v_h_34_){
_start:
{
lean_object* v___x_35_; lean_object* v_toInv_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_35_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_31_);
v_toInv_36_ = lean_ctor_get(v___x_35_, 1);
lean_inc(v_toInv_36_);
lean_dec_ref(v___x_35_);
lean_inc(v_toMul_32_);
lean_inc(v_g_33_);
v___x_37_ = lean_apply_2(v_toMul_32_, v_g_33_, v_h_34_);
v___x_38_ = lean_apply_1(v_toInv_36_, v_g_33_);
v___x_39_ = lean_apply_2(v_toMul_32_, v___x_37_, v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0___boxed(lean_object* v_inst_40_, lean_object* v_toMul_41_, lean_object* v_g_42_, lean_object* v_h_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0(v_inst_40_, v_toMul_41_, v_g_42_, v_h_43_);
lean_dec_ref(v_inst_40_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v_toMonoid_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v_toMul_49_; lean_object* v___f_50_; 
v_toMonoid_46_ = lean_ctor_get(v_inst_45_, 0);
v___x_47_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_46_);
v___x_48_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_47_);
v_toMul_49_ = lean_ctor_get(v___x_48_, 1);
lean_inc(v_toMul_49_);
lean_dec_ref(v___x_48_);
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_50_, 0, v_inst_45_);
lean_closure_set(v___f_50_, 1, v_toMul_49_);
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe(lean_object* v_G_51_, lean_object* v_inst_52_, lean_object* v_H_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg(v_inst_52_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom___redArg(lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; lean_object* v_toMonoid_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
lean_inc_ref(v_inst_55_);
v___x_56_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_55_);
v_toMonoid_57_ = lean_ctor_get(v_inst_55_, 0);
lean_inc_ref(v_toMonoid_57_);
v___x_58_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_toMonoid_57_);
v___x_59_ = lp_mathlib_Subgroup_instMulDistribMulActionSubtypeMemNormalizerCoe___redArg(v_inst_55_);
v___x_60_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(v___x_60_, 0, lean_box(0));
lean_closure_set(v___x_60_, 1, lean_box(0));
lean_closure_set(v___x_60_, 2, v___x_56_);
lean_closure_set(v___x_60_, 3, v___x_58_);
lean_closure_set(v___x_60_, 4, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizerMonoidHom(lean_object* v_G_61_, lean_object* v_inst_62_, lean_object* v_H_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Subgroup_normalizerMonoidHom___redArg(v_inst_62_);
return v___x_64_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Submonoid_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
}
#ifdef __cplusplus
}
#endif
