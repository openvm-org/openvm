// Lean compiler output
// Module: Aesop.Tree.RunMetaM
// Imports: public import Init public meta import Init public import Aesop.Tree.Data
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
lean_object* l_Lean_Meta_saveState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_runInMetaState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_parentRapp_x3f(lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_saveState___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "aesop: internal error: expected goal "};
static const lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1;
static const lean_string_object lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = " to be normalised (but not proven by normalisation)."};
static const lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0(lean_object* v_x_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_apply_1(v_x_1_, lean_box(0));
v___x_8_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0___boxed(lean_object* v_x_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0(v_x_9_, v___y_10_, v___y_11_, v___y_12_, v___y_13_);
lean_dec(v___y_13_);
lean_dec_ref(v___y_12_);
lean_dec(v___y_11_);
lean_dec_ref(v___y_10_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__1(lean_object* v_inst_16_, lean_object* v_00_u03b1_17_, lean_object* v_x_18_){
_start:
{
lean_object* v___f_19_; lean_object* v___x_20_; 
v___f_19_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_19_, 0, v_x_18_);
v___x_20_ = lean_apply_2(v_inst_16_, lean_box(0), v___f_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; 
v___f_22_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__1), 3, 1);
lean_closure_set(v___f_22_, 0, v_inst_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop(lean_object* v_m_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadLiftTSTRealWorld__aesop___redArg___lam__1), 3, 1);
lean_closure_set(v___f_25_, 0, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__0(lean_object* v_r_26_, lean_object* v_toPure_27_, lean_object* v_s_28_){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v_r_26_);
lean_ctor_set(v___x_29_, 1, v_s_28_);
v___x_30_ = lean_apply_2(v_toPure_27_, lean_box(0), v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1(lean_object* v_toPure_32_, lean_object* v_inst_33_, lean_object* v_toBind_34_, lean_object* v_r_35_){
_start:
{
lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___f_36_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__0), 3, 2);
lean_closure_set(v___f_36_, 0, v_r_35_);
lean_closure_set(v___f_36_, 1, v_toPure_32_);
v___x_37_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1___closed__0));
v___x_38_ = lean_apply_2(v_inst_33_, lean_box(0), v___x_37_);
v___x_39_ = lean_apply_4(v_toBind_34_, lean_box(0), lean_box(0), v___x_38_, v___f_36_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg(lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_x_42_){
_start:
{
lean_object* v_toApplicative_43_; lean_object* v_toBind_44_; lean_object* v_toPure_45_; lean_object* v___f_46_; lean_object* v___x_47_; 
v_toApplicative_43_ = lean_ctor_get(v_inst_40_, 0);
lean_inc_ref(v_toApplicative_43_);
v_toBind_44_ = lean_ctor_get(v_inst_40_, 1);
lean_inc_n(v_toBind_44_, 2);
lean_dec_ref(v_inst_40_);
v_toPure_45_ = lean_ctor_get(v_toApplicative_43_, 1);
lean_inc(v_toPure_45_);
lean_dec_ref(v_toApplicative_43_);
v___f_46_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1), 4, 3);
lean_closure_set(v___f_46_, 0, v_toPure_45_);
lean_closure_set(v___f_46_, 1, v_inst_41_);
lean_closure_set(v___f_46_, 2, v_toBind_44_);
v___x_47_ = lean_apply_4(v_toBind_44_, lean_box(0), lean_box(0), v_x_42_, v___f_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState(lean_object* v_m_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_00_u03b1_51_, lean_object* v_x_52_){
_start:
{
lean_object* v_toApplicative_53_; lean_object* v_toBind_54_; lean_object* v_toPure_55_; lean_object* v___f_56_; lean_object* v___x_57_; 
v_toApplicative_53_ = lean_ctor_get(v_inst_49_, 0);
lean_inc_ref(v_toApplicative_53_);
v_toBind_54_ = lean_ctor_get(v_inst_49_, 1);
lean_inc_n(v_toBind_54_, 2);
lean_dec_ref(v_inst_49_);
v_toPure_55_ = lean_ctor_get(v_toApplicative_53_, 1);
lean_inc(v_toPure_55_);
lean_dec_ref(v_toApplicative_53_);
v___f_56_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1), 4, 3);
lean_closure_set(v___f_56_, 0, v_toPure_55_);
lean_closure_set(v___f_56_, 1, v_inst_50_);
lean_closure_set(v___f_56_, 2, v_toBind_54_);
v___x_57_ = lean_apply_4(v_toBind_54_, lean_box(0), lean_box(0), v_x_52_, v___f_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27___redArg(lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_x_61_, lean_object* v_r_62_){
_start:
{
lean_object* v___x_63_; lean_object* v_elimRapp_64_; lean_object* v___x_65_; lean_object* v_metaState_66_; lean_object* v___x_67_; 
v___x_63_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_64_ = lean_ctor_get(v___x_63_, 3);
lean_inc_ref(v_elimRapp_64_);
v___x_65_ = lean_apply_1(v_elimRapp_64_, v_r_62_);
v_metaState_66_ = lean_ctor_get(v___x_65_, 6);
lean_inc_ref(v_metaState_66_);
lean_dec_ref(v___x_65_);
v___x_67_ = lp_aesop_Aesop_runInMetaState___redArg(v_inst_58_, v_inst_59_, v_inst_60_, v_metaState_66_, v_x_61_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM_x27(lean_object* v_m_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_00_u03b1_72_, lean_object* v_x_73_, lean_object* v_r_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_aesop_Aesop_Rapp_runMetaM_x27___redArg(v_inst_69_, v_inst_70_, v_inst_71_, v_x_73_, v_r_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM___redArg(lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_x_79_, lean_object* v_r_80_){
_start:
{
lean_object* v_toApplicative_81_; lean_object* v_toBind_82_; lean_object* v_toPure_83_; lean_object* v___f_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v_toApplicative_81_ = lean_ctor_get(v_inst_76_, 0);
v_toBind_82_ = lean_ctor_get(v_inst_76_, 1);
v_toPure_83_ = lean_ctor_get(v_toApplicative_81_, 1);
lean_inc_n(v_toBind_82_, 2);
lean_inc(v_inst_77_);
lean_inc(v_toPure_83_);
v___f_84_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1), 4, 3);
lean_closure_set(v___f_84_, 0, v_toPure_83_);
lean_closure_set(v___f_84_, 1, v_inst_77_);
lean_closure_set(v___f_84_, 2, v_toBind_82_);
v___x_85_ = lean_apply_4(v_toBind_82_, lean_box(0), lean_box(0), v_x_79_, v___f_84_);
v___x_86_ = lp_aesop_Aesop_Rapp_runMetaM_x27___redArg(v_inst_76_, v_inst_77_, v_inst_78_, v___x_85_, v_r_80_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaM(lean_object* v_m_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_00_u03b1_91_, lean_object* v_x_92_, lean_object* v_r_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_aesop_Aesop_Rapp_runMetaM___redArg(v_inst_88_, v_inst_89_, v_inst_90_, v_x_92_, v_r_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying___redArg___lam__0(lean_object* v_r_95_, lean_object* v_toPure_96_, lean_object* v_____x_97_){
_start:
{
lean_object* v_fst_98_; lean_object* v_snd_99_; lean_object* v___x_101_; uint8_t v_isShared_102_; uint8_t v_isSharedCheck_131_; 
v_fst_98_ = lean_ctor_get(v_____x_97_, 0);
v_snd_99_ = lean_ctor_get(v_____x_97_, 1);
v_isSharedCheck_131_ = !lean_is_exclusive(v_____x_97_);
if (v_isSharedCheck_131_ == 0)
{
v___x_101_ = v_____x_97_;
v_isShared_102_ = v_isSharedCheck_131_;
goto v_resetjp_100_;
}
else
{
lean_inc(v_snd_99_);
lean_inc(v_fst_98_);
lean_dec(v_____x_97_);
v___x_101_ = lean_box(0);
v_isShared_102_ = v_isSharedCheck_131_;
goto v_resetjp_100_;
}
v_resetjp_100_:
{
lean_object* v___x_103_; lean_object* v_introRapp_104_; lean_object* v_elimRapp_105_; lean_object* v___x_106_; lean_object* v_id_107_; lean_object* v_parent_108_; lean_object* v_children_109_; uint8_t v_state_110_; uint8_t v_isIrrelevant_111_; lean_object* v_appliedRule_112_; lean_object* v_scriptSteps_x3f_113_; lean_object* v_originalSubgoals_114_; double v_successProbability_115_; lean_object* v_introducedMVars_116_; lean_object* v_assignedMVars_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_129_; 
v___x_103_ = lp_aesop_Aesop_treeImpl;
v_introRapp_104_ = lean_ctor_get(v___x_103_, 2);
v_elimRapp_105_ = lean_ctor_get(v___x_103_, 3);
lean_inc_ref(v_elimRapp_105_);
v___x_106_ = lean_apply_1(v_elimRapp_105_, v_r_95_);
v_id_107_ = lean_ctor_get(v___x_106_, 0);
v_parent_108_ = lean_ctor_get(v___x_106_, 1);
v_children_109_ = lean_ctor_get(v___x_106_, 2);
v_state_110_ = lean_ctor_get_uint8(v___x_106_, sizeof(void*)*9 + 8);
v_isIrrelevant_111_ = lean_ctor_get_uint8(v___x_106_, sizeof(void*)*9 + 9);
v_appliedRule_112_ = lean_ctor_get(v___x_106_, 3);
v_scriptSteps_x3f_113_ = lean_ctor_get(v___x_106_, 4);
v_originalSubgoals_114_ = lean_ctor_get(v___x_106_, 5);
v_successProbability_115_ = lean_ctor_get_float(v___x_106_, sizeof(void*)*9);
v_introducedMVars_116_ = lean_ctor_get(v___x_106_, 7);
v_assignedMVars_117_ = lean_ctor_get(v___x_106_, 8);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v___x_106_, 6);
lean_dec(v_unused_130_);
v___x_119_ = v___x_106_;
v_isShared_120_ = v_isSharedCheck_129_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_assignedMVars_117_);
lean_inc(v_introducedMVars_116_);
lean_inc(v_originalSubgoals_114_);
lean_inc(v_scriptSteps_x3f_113_);
lean_inc(v_appliedRule_112_);
lean_inc(v_children_109_);
lean_inc(v_parent_108_);
lean_inc(v_id_107_);
lean_dec(v___x_106_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_129_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_122_; 
if (v_isShared_120_ == 0)
{
lean_ctor_set(v___x_119_, 6, v_snd_99_);
v___x_122_ = v___x_119_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v_id_107_);
lean_ctor_set(v_reuseFailAlloc_128_, 1, v_parent_108_);
lean_ctor_set(v_reuseFailAlloc_128_, 2, v_children_109_);
lean_ctor_set(v_reuseFailAlloc_128_, 3, v_appliedRule_112_);
lean_ctor_set(v_reuseFailAlloc_128_, 4, v_scriptSteps_x3f_113_);
lean_ctor_set(v_reuseFailAlloc_128_, 5, v_originalSubgoals_114_);
lean_ctor_set(v_reuseFailAlloc_128_, 6, v_snd_99_);
lean_ctor_set(v_reuseFailAlloc_128_, 7, v_introducedMVars_116_);
lean_ctor_set(v_reuseFailAlloc_128_, 8, v_assignedMVars_117_);
lean_ctor_set_uint8(v_reuseFailAlloc_128_, sizeof(void*)*9 + 8, v_state_110_);
lean_ctor_set_uint8(v_reuseFailAlloc_128_, sizeof(void*)*9 + 9, v_isIrrelevant_111_);
lean_ctor_set_float(v_reuseFailAlloc_128_, sizeof(void*)*9, v_successProbability_115_);
v___x_122_ = v_reuseFailAlloc_128_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
lean_object* v___x_123_; lean_object* v___x_125_; 
lean_inc(v_introRapp_104_);
v___x_123_ = lean_apply_1(v_introRapp_104_, v___x_122_);
if (v_isShared_102_ == 0)
{
lean_ctor_set(v___x_101_, 1, v___x_123_);
v___x_125_ = v___x_101_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v_fst_98_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v___x_123_);
v___x_125_ = v_reuseFailAlloc_127_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
lean_object* v___x_126_; 
v___x_126_ = lean_apply_2(v_toPure_96_, lean_box(0), v___x_125_);
return v___x_126_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying___redArg(lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_x_135_, lean_object* v_r_136_){
_start:
{
lean_object* v_toApplicative_137_; lean_object* v_toBind_138_; lean_object* v_toPure_139_; lean_object* v___x_140_; lean_object* v___f_141_; lean_object* v___x_142_; 
v_toApplicative_137_ = lean_ctor_get(v_inst_132_, 0);
v_toBind_138_ = lean_ctor_get(v_inst_132_, 1);
lean_inc(v_toBind_138_);
v_toPure_139_ = lean_ctor_get(v_toApplicative_137_, 1);
lean_inc(v_toPure_139_);
lean_inc(v_r_136_);
v___x_140_ = lp_aesop_Aesop_Rapp_runMetaM___redArg(v_inst_132_, v_inst_133_, v_inst_134_, v_x_135_, v_r_136_);
v___f_141_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMModifying___redArg___lam__0), 3, 2);
lean_closure_set(v___f_141_, 0, v_r_136_);
lean_closure_set(v___f_141_, 1, v_toPure_139_);
v___x_142_ = lean_apply_4(v_toBind_138_, lean_box(0), lean_box(0), v___x_140_, v___f_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifying(lean_object* v_m_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_00_u03b1_147_, lean_object* v_x_148_, lean_object* v_r_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_aesop_Aesop_Rapp_runMetaMModifying___redArg(v_inst_144_, v_inst_145_, v_inst_146_, v_x_148_, v_r_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0(lean_object* v_rref_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = lean_st_ref_get(v_rref_151_);
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0___boxed(lean_object* v_rref_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0(v_rref_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_);
lean_dec(v___y_163_);
lean_dec_ref(v___y_162_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
lean_dec(v_rref_159_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1(lean_object* v_rref_166_, lean_object* v_snd_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_173_ = lean_st_ref_set(v_rref_166_, v_snd_167_);
v___x_174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1___boxed(lean_object* v_rref_175_, lean_object* v_snd_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1(v_rref_175_, v_snd_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v_rref_175_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__2(lean_object* v_toPure_183_, lean_object* v_fst_184_, lean_object* v_____r_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_apply_2(v_toPure_183_, lean_box(0), v_fst_184_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__3(lean_object* v_rref_187_, lean_object* v_toPure_188_, lean_object* v_inst_189_, lean_object* v_toBind_190_, lean_object* v_____x_191_){
_start:
{
lean_object* v_fst_192_; lean_object* v_snd_193_; lean_object* v___f_194_; lean_object* v___f_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v_fst_192_ = lean_ctor_get(v_____x_191_, 0);
lean_inc(v_fst_192_);
v_snd_193_ = lean_ctor_get(v_____x_191_, 1);
lean_inc(v_snd_193_);
lean_dec_ref(v_____x_191_);
v___f_194_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__1___boxed), 7, 2);
lean_closure_set(v___f_194_, 0, v_rref_187_);
lean_closure_set(v___f_194_, 1, v_snd_193_);
v___f_195_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__2), 3, 2);
lean_closure_set(v___f_195_, 0, v_toPure_188_);
lean_closure_set(v___f_195_, 1, v_fst_192_);
v___x_196_ = lean_apply_2(v_inst_189_, lean_box(0), v___f_194_);
v___x_197_ = lean_apply_4(v_toBind_190_, lean_box(0), lean_box(0), v___x_196_, v___f_195_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__4(lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_x_201_, lean_object* v_toBind_202_, lean_object* v___f_203_, lean_object* v_____do__lift_204_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = lp_aesop_Aesop_Rapp_runMetaMModifying___redArg(v_inst_198_, v_inst_199_, v_inst_200_, v_x_201_, v_____do__lift_204_);
v___x_206_ = lean_apply_4(v_toBind_202_, lean_box(0), lean_box(0), v___x_205_, v___f_203_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying___redArg(lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_x_210_, lean_object* v_rref_211_){
_start:
{
lean_object* v_toApplicative_212_; lean_object* v_toBind_213_; lean_object* v_toPure_214_; lean_object* v___f_215_; lean_object* v___x_216_; lean_object* v___f_217_; lean_object* v___f_218_; lean_object* v___x_219_; 
v_toApplicative_212_ = lean_ctor_get(v_inst_207_, 0);
v_toBind_213_ = lean_ctor_get(v_inst_207_, 1);
lean_inc_n(v_toBind_213_, 3);
v_toPure_214_ = lean_ctor_get(v_toApplicative_212_, 1);
lean_inc(v_rref_211_);
v___f_215_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_215_, 0, v_rref_211_);
lean_inc_n(v_inst_208_, 2);
v___x_216_ = lean_apply_2(v_inst_208_, lean_box(0), v___f_215_);
lean_inc(v_toPure_214_);
v___f_217_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__3), 5, 4);
lean_closure_set(v___f_217_, 0, v_rref_211_);
lean_closure_set(v___f_217_, 1, v_toPure_214_);
lean_closure_set(v___f_217_, 2, v_inst_208_);
lean_closure_set(v___f_217_, 3, v_toBind_213_);
v___f_218_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RappRef_runMetaMModifying___redArg___lam__4), 7, 6);
lean_closure_set(v___f_218_, 0, v_inst_207_);
lean_closure_set(v___f_218_, 1, v_inst_208_);
lean_closure_set(v___f_218_, 2, v_inst_209_);
lean_closure_set(v___f_218_, 3, v_x_210_);
lean_closure_set(v___f_218_, 4, v_toBind_213_);
lean_closure_set(v___f_218_, 5, v___f_217_);
v___x_219_ = lean_apply_4(v_toBind_213_, lean_box(0), lean_box(0), v___x_216_, v___f_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_runMetaMModifying(lean_object* v_m_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_00_u03b1_224_, lean_object* v_x_225_, lean_object* v_rref_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_aesop_Aesop_RappRef_runMetaMModifying___redArg(v_inst_221_, v_inst_222_, v_inst_223_, v_x_225_, v_rref_226_);
return v___x_227_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = ((lean_object*)(lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__0));
v___x_230_ = l_Lean_stringToMessageData(v___x_229_);
return v___x_230_;
}
}
static lean_object* _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = ((lean_object*)(lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__2));
v___x_233_ = l_Lean_stringToMessageData(v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_x_238_, lean_object* v_g_239_){
_start:
{
lean_object* v___x_240_; lean_object* v_elimGoal_241_; lean_object* v___x_242_; lean_object* v_normalizationState_243_; 
v___x_240_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_241_ = lean_ctor_get(v___x_240_, 1);
lean_inc_ref(v_elimGoal_241_);
v___x_242_ = lean_apply_1(v_elimGoal_241_, v_g_239_);
v_normalizationState_243_ = lean_ctor_get(v___x_242_, 6);
lean_inc(v_normalizationState_243_);
if (lean_obj_tag(v_normalizationState_243_) == 1)
{
lean_object* v_postGoal_244_; lean_object* v_postState_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
lean_dec_ref(v___x_242_);
lean_dec_ref(v_inst_237_);
v_postGoal_244_ = lean_ctor_get(v_normalizationState_243_, 0);
lean_inc(v_postGoal_244_);
v_postState_245_ = lean_ctor_get(v_normalizationState_243_, 1);
lean_inc_ref(v_postState_245_);
lean_dec_ref_known(v_normalizationState_243_, 3);
v___x_246_ = lean_apply_1(v_x_238_, v_postGoal_244_);
v___x_247_ = lp_aesop_Aesop_runInMetaState___redArg(v_inst_234_, v_inst_235_, v_inst_236_, v_postState_245_, v___x_246_);
return v___x_247_;
}
else
{
lean_object* v_id_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
lean_dec(v_normalizationState_243_);
lean_dec(v_x_238_);
lean_dec(v_inst_236_);
lean_dec(v_inst_235_);
v_id_248_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_id_248_);
lean_dec_ref(v___x_242_);
v___x_249_ = lean_obj_once(&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1, &lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1_once, _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__1);
v___x_250_ = l_Nat_reprFast(v_id_248_);
v___x_251_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
v___x_252_ = l_Lean_MessageData_ofFormat(v___x_251_);
v___x_253_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_249_);
lean_ctor_set(v___x_253_, 1, v___x_252_);
v___x_254_ = lean_obj_once(&lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3, &lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3_once, _init_lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg___closed__3);
v___x_255_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_253_);
lean_ctor_set(v___x_255_, 1, v___x_254_);
v___x_256_ = l_Lean_throwError___redArg(v_inst_234_, v_inst_237_, v___x_255_);
return v___x_256_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27(lean_object* v_m_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_00_u03b1_261_, lean_object* v_inst_262_, lean_object* v_x_263_, lean_object* v_g_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v_inst_258_, v_inst_259_, v_inst_260_, v_inst_262_, v_x_263_, v_g_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg___lam__2(lean_object* v_inst_266_, lean_object* v_x_267_, lean_object* v_inst_268_, lean_object* v_g_269_){
_start:
{
lean_object* v_toApplicative_270_; lean_object* v_toBind_271_; lean_object* v_toPure_272_; lean_object* v___x_273_; lean_object* v___f_274_; lean_object* v___x_275_; 
v_toApplicative_270_ = lean_ctor_get(v_inst_266_, 0);
lean_inc_ref(v_toApplicative_270_);
v_toBind_271_ = lean_ctor_get(v_inst_266_, 1);
lean_inc_n(v_toBind_271_, 2);
lean_dec_ref(v_inst_266_);
v_toPure_272_ = lean_ctor_get(v_toApplicative_270_, 1);
lean_inc(v_toPure_272_);
lean_dec_ref(v_toApplicative_270_);
v___x_273_ = lean_apply_1(v_x_267_, v_g_269_);
v___f_274_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1), 4, 3);
lean_closure_set(v___f_274_, 0, v_toPure_272_);
lean_closure_set(v___f_274_, 1, v_inst_268_);
lean_closure_set(v___f_274_, 2, v_toBind_271_);
v___x_275_ = lean_apply_4(v_toBind_271_, lean_box(0), lean_box(0), v___x_273_, v___f_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg(lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_x_280_, lean_object* v_g_281_){
_start:
{
lean_object* v___f_282_; lean_object* v___x_283_; 
lean_inc(v_inst_277_);
lean_inc_ref(v_inst_276_);
v___f_282_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg___lam__2), 4, 3);
lean_closure_set(v___f_282_, 0, v_inst_276_);
lean_closure_set(v___f_282_, 1, v_x_280_);
lean_closure_set(v___f_282_, 2, v_inst_277_);
v___x_283_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v_inst_276_, v_inst_277_, v_inst_278_, v_inst_279_, v___f_282_, v_g_281_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState(lean_object* v_m_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_00_u03b1_288_, lean_object* v_inst_289_, lean_object* v_x_290_, lean_object* v_g_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState___redArg(v_inst_285_, v_inst_286_, v_inst_287_, v_inst_289_, v_x_290_, v_g_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0(lean_object* v_g_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_299_ = lp_aesop_Aesop_Goal_parentRapp_x3f(v_g_293_);
v___x_300_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0___boxed(lean_object* v_g_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0(v_g_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__1(lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_x_311_, lean_object* v_____do__lift_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_aesop_Aesop_Rapp_runMetaM_x27___redArg(v_inst_308_, v_inst_309_, v_inst_310_, v_x_311_, v_____do__lift_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2(lean_object* v_x_314_){
_start:
{
lean_object* v_fst_315_; 
v_fst_315_ = lean_ctor_get(v_x_314_, 0);
lean_inc(v_fst_315_);
return v_fst_315_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2___boxed(lean_object* v_x_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__2(v_x_316_);
lean_dec_ref(v_x_316_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3(lean_object* v___x_318_, lean_object* v_x_319_){
_start:
{
lean_inc(v___x_318_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3___boxed(lean_object* v___x_320_, lean_object* v_x_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3(v___x_320_, v_x_321_);
lean_dec(v_x_321_);
lean_dec(v___x_320_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__4(lean_object* v_toFunctor_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_x_326_, lean_object* v___f_327_, lean_object* v_initialState_328_){
_start:
{
lean_object* v_map_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___f_332_; lean_object* v_y_333_; lean_object* v___x_334_; 
v_map_329_ = lean_ctor_get(v_toFunctor_323_, 0);
lean_inc(v_map_329_);
lean_dec_ref(v_toFunctor_323_);
v___x_330_ = lean_alloc_closure((void*)(l_Lean_Meta_SavedState_restore___boxed), 6, 1);
lean_closure_set(v___x_330_, 0, v_initialState_328_);
v___x_331_ = lean_apply_2(v_inst_324_, lean_box(0), v___x_330_);
v___f_332_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_332_, 0, v___x_331_);
v_y_333_ = lean_apply_4(v_inst_325_, lean_box(0), lean_box(0), v_x_326_, v___f_332_);
v___x_334_ = lean_apply_4(v_map_329_, lean_box(0), lean_box(0), v___f_327_, v_y_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5(lean_object* v_val_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_341_ = lean_st_ref_get(v_val_335_);
v___x_342_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5___boxed(lean_object* v_val_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5(v_val_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec(v_val_343_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__6(lean_object* v_inst_350_, lean_object* v_toBind_351_, lean_object* v___f_352_, lean_object* v___f_353_, lean_object* v_____do__lift_354_){
_start:
{
if (lean_obj_tag(v_____do__lift_354_) == 0)
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
lean_dec(v___f_353_);
v___x_355_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1___closed__0));
v___x_356_ = lean_apply_2(v_inst_350_, lean_box(0), v___x_355_);
v___x_357_ = lean_apply_4(v_toBind_351_, lean_box(0), lean_box(0), v___x_356_, v___f_352_);
return v___x_357_;
}
else
{
lean_object* v_val_358_; lean_object* v___f_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
lean_dec(v___f_352_);
v_val_358_ = lean_ctor_get(v_____do__lift_354_, 0);
lean_inc(v_val_358_);
lean_dec_ref_known(v_____do__lift_354_, 1);
v___f_359_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__5___boxed), 6, 1);
lean_closure_set(v___f_359_, 0, v_val_358_);
v___x_360_ = lean_apply_2(v_inst_350_, lean_box(0), v___f_359_);
v___x_361_ = lean_apply_4(v_toBind_351_, lean_box(0), lean_box(0), v___x_360_, v___f_353_);
return v___x_361_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg(lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_x_366_, lean_object* v_g_367_){
_start:
{
lean_object* v_toApplicative_368_; lean_object* v_toBind_369_; lean_object* v_toFunctor_370_; lean_object* v_this_371_; lean_object* v___f_372_; lean_object* v___f_373_; lean_object* v___x_374_; lean_object* v___f_375_; lean_object* v___f_376_; lean_object* v___x_377_; 
v_toApplicative_368_ = lean_ctor_get(v_inst_363_, 0);
v_toBind_369_ = lean_ctor_get(v_inst_363_, 1);
lean_inc_n(v_toBind_369_, 2);
v_toFunctor_370_ = lean_ctor_get(v_toApplicative_368_, 0);
lean_inc_ref(v_toFunctor_370_);
v_this_371_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v_this_371_, 0, v_g_367_);
lean_inc(v_x_366_);
lean_inc(v_inst_365_);
lean_inc_n(v_inst_364_, 3);
v___f_372_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__1), 5, 4);
lean_closure_set(v___f_372_, 0, v_inst_363_);
lean_closure_set(v___f_372_, 1, v_inst_364_);
lean_closure_set(v___f_372_, 2, v_inst_365_);
lean_closure_set(v___f_372_, 3, v_x_366_);
v___f_373_ = ((lean_object*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___closed__0));
v___x_374_ = lean_apply_2(v_inst_364_, lean_box(0), v_this_371_);
v___f_375_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__4), 6, 5);
lean_closure_set(v___f_375_, 0, v_toFunctor_370_);
lean_closure_set(v___f_375_, 1, v_inst_364_);
lean_closure_set(v___f_375_, 2, v_inst_365_);
lean_closure_set(v___f_375_, 3, v_x_366_);
lean_closure_set(v___f_375_, 4, v___f_373_);
v___f_376_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__6), 5, 4);
lean_closure_set(v___f_376_, 0, v_inst_364_);
lean_closure_set(v___f_376_, 1, v_toBind_369_);
lean_closure_set(v___f_376_, 2, v___f_375_);
lean_closure_set(v___f_376_, 3, v___f_372_);
v___x_377_ = lean_apply_4(v_toBind_369_, lean_box(0), lean_box(0), v___x_374_, v___f_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState_x27(lean_object* v_m_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_00_u03b1_382_, lean_object* v_x_383_, lean_object* v_g_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg(v_inst_379_, v_inst_380_, v_inst_381_, v_x_383_, v_g_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_x_389_, lean_object* v_g_390_){
_start:
{
lean_object* v_toApplicative_391_; lean_object* v_toBind_392_; lean_object* v_toPure_393_; lean_object* v___f_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v_toApplicative_391_ = lean_ctor_get(v_inst_386_, 0);
v_toBind_392_ = lean_ctor_get(v_inst_386_, 1);
v_toPure_393_ = lean_ctor_get(v_toApplicative_391_, 1);
lean_inc_n(v_toBind_392_, 2);
lean_inc(v_inst_387_);
lean_inc(v_toPure_393_);
v___f_394_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_RunMetaM_0__Aesop_withSaveState___redArg___lam__1), 4, 3);
lean_closure_set(v___f_394_, 0, v_toPure_393_);
lean_closure_set(v___f_394_, 1, v_inst_387_);
lean_closure_set(v___f_394_, 2, v_toBind_392_);
v___x_395_ = lean_apply_4(v_toBind_392_, lean_box(0), lean_box(0), v_x_389_, v___f_394_);
v___x_396_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg(v_inst_386_, v_inst_387_, v_inst_388_, v___x_395_, v_g_390_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMInParentState(lean_object* v_m_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_00_u03b1_401_, lean_object* v_x_402_, lean_object* v_g_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(v_inst_398_, v_inst_399_, v_inst_400_, v_x_402_, v_g_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg___lam__1(lean_object* v_x_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_____do__lift_409_){
_start:
{
if (lean_obj_tag(v_____do__lift_409_) == 0)
{
lean_dec(v_inst_408_);
lean_dec(v_inst_407_);
lean_dec_ref(v_inst_406_);
return v_x_405_;
}
else
{
lean_object* v_val_410_; lean_object* v___x_411_; 
v_val_410_ = lean_ctor_get(v_____do__lift_409_, 0);
lean_inc(v_val_410_);
lean_dec_ref_known(v_____do__lift_409_, 1);
v___x_411_ = lp_aesop_Aesop_RappRef_runMetaMModifying___redArg(v_inst_406_, v_inst_407_, v_inst_408_, v_x_405_, v_val_410_);
return v___x_411_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg(lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_x_415_, lean_object* v_g_416_){
_start:
{
lean_object* v_toBind_417_; lean_object* v_this_418_; lean_object* v___f_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
v_toBind_417_ = lean_ctor_get(v_inst_412_, 1);
lean_inc(v_toBind_417_);
v_this_418_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v_this_418_, 0, v_g_416_);
lean_inc(v_inst_413_);
v___f_419_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg___lam__1), 5, 4);
lean_closure_set(v___f_419_, 0, v_x_415_);
lean_closure_set(v___f_419_, 1, v_inst_412_);
lean_closure_set(v___f_419_, 2, v_inst_413_);
lean_closure_set(v___f_419_, 3, v_inst_414_);
v___x_420_ = lean_apply_2(v_inst_413_, lean_box(0), v_this_418_);
v___x_421_ = lean_apply_4(v_toBind_417_, lean_box(0), lean_box(0), v___x_420_, v___f_419_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_runMetaMModifyingParentState(lean_object* v_m_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_00_u03b1_426_, lean_object* v_x_427_, lean_object* v_g_428_){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg(v_inst_423_, v_inst_424_, v_inst_425_, v_x_427_, v_g_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__0(lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_x_433_, lean_object* v_____do__lift_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_aesop_Aesop_Goal_runMetaMInParentState___redArg(v_inst_430_, v_inst_431_, v_inst_432_, v_x_433_, v_____do__lift_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1(lean_object* v_parent_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_442_ = lean_st_ref_get(v_parent_436_);
v___x_443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1___boxed(lean_object* v_parent_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1(v_parent_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_);
lean_dec(v___y_448_);
lean_dec_ref(v___y_447_);
lean_dec(v___y_446_);
lean_dec_ref(v___y_445_);
lean_dec(v_parent_444_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg(lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_x_454_, lean_object* v_r_455_){
_start:
{
lean_object* v_toBind_456_; lean_object* v___x_457_; lean_object* v_elimRapp_458_; lean_object* v___x_459_; lean_object* v_parent_460_; lean_object* v___f_461_; lean_object* v___f_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
v_toBind_456_ = lean_ctor_get(v_inst_451_, 1);
lean_inc(v_toBind_456_);
v___x_457_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_458_ = lean_ctor_get(v___x_457_, 3);
lean_inc_ref(v_elimRapp_458_);
v___x_459_ = lean_apply_1(v_elimRapp_458_, v_r_455_);
v_parent_460_ = lean_ctor_get(v___x_459_, 1);
lean_inc(v_parent_460_);
lean_dec_ref(v___x_459_);
lean_inc(v_inst_452_);
v___f_461_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__0), 5, 4);
lean_closure_set(v___f_461_, 0, v_inst_451_);
lean_closure_set(v___f_461_, 1, v_inst_452_);
lean_closure_set(v___f_461_, 2, v_inst_453_);
lean_closure_set(v___f_461_, 3, v_x_454_);
v___f_462_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1___boxed), 6, 1);
lean_closure_set(v___f_462_, 0, v_parent_460_);
v___x_463_ = lean_apply_2(v_inst_452_, lean_box(0), v___f_462_);
v___x_464_ = lean_apply_4(v_toBind_456_, lean_box(0), lean_box(0), v___x_463_, v___f_461_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState(lean_object* v_m_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_00_u03b1_469_, lean_object* v_x_470_, lean_object* v_r_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg(v_inst_466_, v_inst_467_, v_inst_468_, v_x_470_, v_r_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg___lam__0(lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_x_476_, lean_object* v_____do__lift_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_aesop_Aesop_Goal_runMetaMInParentState_x27___redArg(v_inst_473_, v_inst_474_, v_inst_475_, v_x_476_, v_____do__lift_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg(lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_x_482_, lean_object* v_r_483_){
_start:
{
lean_object* v_toBind_484_; lean_object* v___x_485_; lean_object* v_elimRapp_486_; lean_object* v___x_487_; lean_object* v_parent_488_; lean_object* v___f_489_; lean_object* v___f_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v_toBind_484_ = lean_ctor_get(v_inst_479_, 1);
lean_inc(v_toBind_484_);
v___x_485_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_486_ = lean_ctor_get(v___x_485_, 3);
lean_inc_ref(v_elimRapp_486_);
v___x_487_ = lean_apply_1(v_elimRapp_486_, v_r_483_);
v_parent_488_ = lean_ctor_get(v___x_487_, 1);
lean_inc(v_parent_488_);
lean_dec_ref(v___x_487_);
lean_inc(v_inst_480_);
v___f_489_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg___lam__0), 5, 4);
lean_closure_set(v___f_489_, 0, v_inst_479_);
lean_closure_set(v___f_489_, 1, v_inst_480_);
lean_closure_set(v___f_489_, 2, v_inst_481_);
lean_closure_set(v___f_489_, 3, v_x_482_);
v___f_490_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1___boxed), 6, 1);
lean_closure_set(v___f_490_, 0, v_parent_488_);
v___x_491_ = lean_apply_2(v_inst_480_, lean_box(0), v___f_490_);
v___x_492_ = lean_apply_4(v_toBind_484_, lean_box(0), lean_box(0), v___x_491_, v___f_489_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMInParentState_x27(lean_object* v_m_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_inst_496_, lean_object* v_00_u03b1_497_, lean_object* v_x_498_, lean_object* v_r_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_aesop_Aesop_Rapp_runMetaMInParentState_x27___redArg(v_inst_494_, v_inst_495_, v_inst_496_, v_x_498_, v_r_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg___lam__0(lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_x_504_, lean_object* v_____do__lift_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_aesop_Aesop_Goal_runMetaMModifyingParentState___redArg(v_inst_501_, v_inst_502_, v_inst_503_, v_x_504_, v_____do__lift_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg(lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_x_510_, lean_object* v_r_511_){
_start:
{
lean_object* v_toBind_512_; lean_object* v___x_513_; lean_object* v_elimRapp_514_; lean_object* v___x_515_; lean_object* v_parent_516_; lean_object* v___f_517_; lean_object* v___f_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
v_toBind_512_ = lean_ctor_get(v_inst_507_, 1);
lean_inc(v_toBind_512_);
v___x_513_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_514_ = lean_ctor_get(v___x_513_, 3);
lean_inc_ref(v_elimRapp_514_);
v___x_515_ = lean_apply_1(v_elimRapp_514_, v_r_511_);
v_parent_516_ = lean_ctor_get(v___x_515_, 1);
lean_inc(v_parent_516_);
lean_dec_ref(v___x_515_);
lean_inc(v_inst_508_);
v___f_517_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg___lam__0), 5, 4);
lean_closure_set(v___f_517_, 0, v_inst_507_);
lean_closure_set(v___f_517_, 1, v_inst_508_);
lean_closure_set(v___f_517_, 2, v_inst_509_);
lean_closure_set(v___f_517_, 3, v_x_510_);
v___f_518_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Rapp_runMetaMInParentState___redArg___lam__1___boxed), 6, 1);
lean_closure_set(v___f_518_, 0, v_parent_516_);
v___x_519_ = lean_apply_2(v_inst_508_, lean_box(0), v___f_518_);
v___x_520_ = lean_apply_4(v_toBind_512_, lean_box(0), lean_box(0), v___x_519_, v___f_517_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_runMetaMModifyingParentState(lean_object* v_m_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_00_u03b1_525_, lean_object* v_x_526_, lean_object* v_r_527_){
_start:
{
lean_object* v___x_528_; 
v___x_528_ = lp_aesop_Aesop_Rapp_runMetaMModifyingParentState___redArg(v_inst_522_, v_inst_523_, v_inst_524_, v_x_526_, v_r_527_);
return v___x_528_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_Data(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Data(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_RunMetaM(builtin);
}
#ifdef __cplusplus
}
#endif
