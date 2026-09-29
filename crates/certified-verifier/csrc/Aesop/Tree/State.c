// Lean compiler output
// Module: Aesop.Tree.State
// Imports: public import Init public meta import Init public import Aesop.Tree.Traversal
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
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
uint8_t lp_aesop_Aesop_Goal_isExhausted(lean_object*);
uint8_t lp_aesop_Aesop_NodeState_isUnprovable(uint8_t);
uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
uint8_t lp_aesop_Aesop_GoalState_isUnprovable(uint8_t);
uint8_t lp_aesop_Aesop_NormalizationState_isProvenByNormalization(lean_object*);
uint8_t lp_aesop_Aesop_NodeState_isIrrelevant(uint8_t);
uint8_t lp_aesop_Aesop_GoalState_isIrrelevant(uint8_t);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevantNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevantNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevantNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevantNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevantNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevantNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_markSubtreeIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_markSubtreeIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markSubtreeIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markSubtreeIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isProvenNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isProvenNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isProvenNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isProvenNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markProvenCore(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markProvenCore___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markProvenByNormalization(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markProvenByNormalization___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markProven(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markProven___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0(uint8_t, uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnprovableNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnprovableNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isUnprovableNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isUnprovableNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isUnprovableNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isUnprovableNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markUnprovableCore(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markUnprovableCore___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markUnprovable(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markUnprovable___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_checkAndMarkUnprovable(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_checkAndMarkUnprovable___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_stateNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stateNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_stateNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_stateNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_stateNoCache(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_stateNoCache___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isIrrelevantNoCache(lean_object* v_g_1_){
_start:
{
lean_object* v___x_3_; lean_object* v_elimGoal_4_; lean_object* v_elimMVarCluster_5_; lean_object* v___x_6_; lean_object* v_parent_7_; uint8_t v_state_8_; uint8_t v___x_9_; 
v___x_3_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_4_ = lean_ctor_get(v___x_3_, 1);
v_elimMVarCluster_5_ = lean_ctor_get(v___x_3_, 5);
lean_inc_ref(v_elimGoal_4_);
v___x_6_ = lean_apply_1(v_elimGoal_4_, v_g_1_);
v_parent_7_ = lean_ctor_get(v___x_6_, 1);
lean_inc(v_parent_7_);
v_state_8_ = lean_ctor_get_uint8(v___x_6_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_6_);
v___x_9_ = lp_aesop_Aesop_GoalState_isIrrelevant(v_state_8_);
if (v___x_9_ == 0)
{
lean_object* v___x_10_; lean_object* v___x_11_; uint8_t v_isIrrelevant_12_; 
v___x_10_ = lean_st_ref_get(v_parent_7_);
lean_dec(v_parent_7_);
lean_inc_ref(v_elimMVarCluster_5_);
v___x_11_ = lean_apply_1(v_elimMVarCluster_5_, v___x_10_);
v_isIrrelevant_12_ = lean_ctor_get_uint8(v___x_11_, sizeof(void*)*2);
lean_dec_ref(v___x_11_);
return v_isIrrelevant_12_;
}
else
{
lean_dec(v_parent_7_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isIrrelevantNoCache___boxed(lean_object* v_g_13_, lean_object* v_a_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_aesop_Aesop_Goal_isIrrelevantNoCache(v_g_13_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isIrrelevantNoCache(lean_object* v_r_17_){
_start:
{
lean_object* v___x_19_; lean_object* v_elimGoal_20_; lean_object* v_elimRapp_21_; lean_object* v___x_22_; lean_object* v_parent_23_; uint8_t v_state_24_; uint8_t v___x_25_; 
v___x_19_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_20_ = lean_ctor_get(v___x_19_, 1);
v_elimRapp_21_ = lean_ctor_get(v___x_19_, 3);
lean_inc_ref(v_elimRapp_21_);
v___x_22_ = lean_apply_1(v_elimRapp_21_, v_r_17_);
v_parent_23_ = lean_ctor_get(v___x_22_, 1);
lean_inc(v_parent_23_);
v_state_24_ = lean_ctor_get_uint8(v___x_22_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_22_);
v___x_25_ = lp_aesop_Aesop_NodeState_isIrrelevant(v_state_24_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; uint8_t v_isIrrelevant_28_; 
v___x_26_ = lean_st_ref_get(v_parent_23_);
lean_dec(v_parent_23_);
lean_inc_ref(v_elimGoal_20_);
v___x_27_ = lean_apply_1(v_elimGoal_20_, v___x_26_);
v_isIrrelevant_28_ = lean_ctor_get_uint8(v___x_27_, sizeof(void*)*14 + 9);
lean_dec_ref(v___x_27_);
return v_isIrrelevant_28_;
}
else
{
lean_dec(v_parent_23_);
return v___x_25_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isIrrelevantNoCache___boxed(lean_object* v_r_29_, lean_object* v_a_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_aesop_Aesop_Rapp_isIrrelevantNoCache(v_r_29_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isIrrelevantNoCache(lean_object* v_c_33_){
_start:
{
lean_object* v___x_35_; lean_object* v_elimRapp_36_; lean_object* v_elimMVarCluster_37_; lean_object* v___x_38_; lean_object* v_parent_x3f_39_; uint8_t v_state_40_; uint8_t v___x_41_; 
v___x_35_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_36_ = lean_ctor_get(v___x_35_, 3);
v_elimMVarCluster_37_ = lean_ctor_get(v___x_35_, 5);
lean_inc_ref(v_elimMVarCluster_37_);
v___x_38_ = lean_apply_1(v_elimMVarCluster_37_, v_c_33_);
v_parent_x3f_39_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_parent_x3f_39_);
v_state_40_ = lean_ctor_get_uint8(v___x_38_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_38_);
v___x_41_ = lp_aesop_Aesop_NodeState_isIrrelevant(v_state_40_);
if (v___x_41_ == 0)
{
if (lean_obj_tag(v_parent_x3f_39_) == 0)
{
return v___x_41_;
}
else
{
lean_object* v_val_42_; lean_object* v___x_43_; lean_object* v___x_44_; uint8_t v_isIrrelevant_45_; 
v_val_42_ = lean_ctor_get(v_parent_x3f_39_, 0);
lean_inc(v_val_42_);
lean_dec_ref_known(v_parent_x3f_39_, 1);
v___x_43_ = lean_st_ref_get(v_val_42_);
lean_dec(v_val_42_);
lean_inc_ref(v_elimRapp_36_);
v___x_44_ = lean_apply_1(v_elimRapp_36_, v___x_43_);
v_isIrrelevant_45_ = lean_ctor_get_uint8(v___x_44_, sizeof(void*)*9 + 9);
lean_dec_ref(v___x_44_);
return v_isIrrelevant_45_;
}
}
else
{
lean_dec(v_parent_x3f_39_);
return v___x_41_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isIrrelevantNoCache___boxed(lean_object* v_c_46_, lean_object* v_a_47_){
_start:
{
uint8_t v_res_48_; lean_object* v_r_49_; 
v_res_48_ = lp_aesop_Aesop_MVarCluster_isIrrelevantNoCache(v_c_46_);
v_r_49_ = lean_box(v_res_48_);
return v_r_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1(lean_object* v_as_50_, size_t v_i_51_, size_t v_stop_52_, lean_object* v_b_53_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = lean_usize_dec_eq(v_i_51_, v_stop_52_);
if (v___x_55_ == 0)
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; size_t v___x_59_; size_t v___x_60_; 
v___x_56_ = lean_array_uget_borrowed(v_as_50_, v_i_51_);
lean_inc(v___x_56_);
v___x_57_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
v___x_58_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_57_);
lean_dec_ref_known(v___x_57_, 1);
v___x_59_ = ((size_t)1ULL);
v___x_60_ = lean_usize_add(v_i_51_, v___x_59_);
v_i_51_ = v___x_60_;
v_b_53_ = v___x_58_;
goto _start;
}
else
{
return v_b_53_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2(lean_object* v_as_62_, size_t v_i_63_, size_t v_stop_64_, lean_object* v_b_65_){
_start:
{
uint8_t v___x_67_; 
v___x_67_ = lean_usize_dec_eq(v_i_63_, v_stop_64_);
if (v___x_67_ == 0)
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; size_t v___x_71_; size_t v___x_72_; 
v___x_68_ = lean_array_uget_borrowed(v_as_62_, v_i_63_);
lean_inc(v___x_68_);
v___x_69_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
v___x_70_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_69_);
lean_dec_ref_known(v___x_69_, 1);
v___x_71_ = ((size_t)1ULL);
v___x_72_ = lean_usize_add(v_i_63_, v___x_71_);
v_i_63_ = v___x_72_;
v_b_65_ = v___x_70_;
goto _start;
}
else
{
return v_b_65_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(lean_object* v_x_74_){
_start:
{
switch(lean_obj_tag(v_x_74_))
{
case 0:
{
lean_object* v_gref_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v_introGoal_85_; lean_object* v_elimGoal_86_; lean_object* v___x_87_; uint8_t v_isIrrelevant_88_; 
v_gref_82_ = lean_ctor_get(v_x_74_, 0);
v___x_83_ = lean_st_ref_get(v_gref_82_);
v___x_84_ = lp_aesop_Aesop_treeImpl;
v_introGoal_85_ = lean_ctor_get(v___x_84_, 0);
v_elimGoal_86_ = lean_ctor_get(v___x_84_, 1);
lean_inc_ref(v_elimGoal_86_);
v___x_87_ = lean_apply_1(v_elimGoal_86_, v___x_83_);
v_isIrrelevant_88_ = lean_ctor_get_uint8(v___x_87_, sizeof(void*)*14 + 9);
lean_dec_ref(v___x_87_);
if (v_isIrrelevant_88_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v_id_91_; lean_object* v_parent_92_; lean_object* v_children_93_; lean_object* v_origin_94_; lean_object* v_depth_95_; uint8_t v_state_96_; uint8_t v_isForcedUnprovable_97_; lean_object* v_preNormGoal_98_; lean_object* v_normalizationState_99_; lean_object* v_mvars_100_; lean_object* v_forwardState_101_; lean_object* v_forwardRuleMatches_102_; double v_successProbability_103_; lean_object* v_addedInIteration_104_; lean_object* v_lastExpandedInIteration_105_; uint8_t v_unsafeRulesSelected_106_; lean_object* v_unsafeQueue_107_; lean_object* v_failedRapps_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_132_; 
v___x_89_ = lean_st_ref_take(v_gref_82_);
lean_inc_ref(v_elimGoal_86_);
v___x_90_ = lean_apply_1(v_elimGoal_86_, v___x_89_);
v_id_91_ = lean_ctor_get(v___x_90_, 0);
v_parent_92_ = lean_ctor_get(v___x_90_, 1);
v_children_93_ = lean_ctor_get(v___x_90_, 2);
v_origin_94_ = lean_ctor_get(v___x_90_, 3);
v_depth_95_ = lean_ctor_get(v___x_90_, 4);
v_state_96_ = lean_ctor_get_uint8(v___x_90_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_97_ = lean_ctor_get_uint8(v___x_90_, sizeof(void*)*14 + 10);
v_preNormGoal_98_ = lean_ctor_get(v___x_90_, 5);
v_normalizationState_99_ = lean_ctor_get(v___x_90_, 6);
v_mvars_100_ = lean_ctor_get(v___x_90_, 7);
v_forwardState_101_ = lean_ctor_get(v___x_90_, 8);
v_forwardRuleMatches_102_ = lean_ctor_get(v___x_90_, 9);
v_successProbability_103_ = lean_ctor_get_float(v___x_90_, sizeof(void*)*14);
v_addedInIteration_104_ = lean_ctor_get(v___x_90_, 10);
v_lastExpandedInIteration_105_ = lean_ctor_get(v___x_90_, 11);
v_unsafeRulesSelected_106_ = lean_ctor_get_uint8(v___x_90_, sizeof(void*)*14 + 11);
v_unsafeQueue_107_ = lean_ctor_get(v___x_90_, 12);
v_failedRapps_108_ = lean_ctor_get(v___x_90_, 13);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_90_);
if (v_isSharedCheck_132_ == 0)
{
v___x_110_ = v___x_90_;
v_isShared_111_ = v_isSharedCheck_132_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_failedRapps_108_);
lean_inc(v_unsafeQueue_107_);
lean_inc(v_lastExpandedInIteration_105_);
lean_inc(v_addedInIteration_104_);
lean_inc(v_forwardRuleMatches_102_);
lean_inc(v_forwardState_101_);
lean_inc(v_mvars_100_);
lean_inc(v_normalizationState_99_);
lean_inc(v_preNormGoal_98_);
lean_inc(v_depth_95_);
lean_inc(v_origin_94_);
lean_inc(v_children_93_);
lean_inc(v_parent_92_);
lean_inc(v_id_91_);
lean_dec(v___x_90_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_132_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
uint8_t v___x_112_; lean_object* v___x_114_; 
v___x_112_ = 1;
if (v_isShared_111_ == 0)
{
v___x_114_ = v___x_110_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_id_91_);
lean_ctor_set(v_reuseFailAlloc_131_, 1, v_parent_92_);
lean_ctor_set(v_reuseFailAlloc_131_, 2, v_children_93_);
lean_ctor_set(v_reuseFailAlloc_131_, 3, v_origin_94_);
lean_ctor_set(v_reuseFailAlloc_131_, 4, v_depth_95_);
lean_ctor_set(v_reuseFailAlloc_131_, 5, v_preNormGoal_98_);
lean_ctor_set(v_reuseFailAlloc_131_, 6, v_normalizationState_99_);
lean_ctor_set(v_reuseFailAlloc_131_, 7, v_mvars_100_);
lean_ctor_set(v_reuseFailAlloc_131_, 8, v_forwardState_101_);
lean_ctor_set(v_reuseFailAlloc_131_, 9, v_forwardRuleMatches_102_);
lean_ctor_set(v_reuseFailAlloc_131_, 10, v_addedInIteration_104_);
lean_ctor_set(v_reuseFailAlloc_131_, 11, v_lastExpandedInIteration_105_);
lean_ctor_set(v_reuseFailAlloc_131_, 12, v_unsafeQueue_107_);
lean_ctor_set(v_reuseFailAlloc_131_, 13, v_failedRapps_108_);
lean_ctor_set_uint8(v_reuseFailAlloc_131_, sizeof(void*)*14 + 8, v_state_96_);
lean_ctor_set_uint8(v_reuseFailAlloc_131_, sizeof(void*)*14 + 10, v_isForcedUnprovable_97_);
lean_ctor_set_float(v_reuseFailAlloc_131_, sizeof(void*)*14, v_successProbability_103_);
lean_ctor_set_uint8(v_reuseFailAlloc_131_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_106_);
v___x_114_ = v_reuseFailAlloc_131_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v_children_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
lean_ctor_set_uint8(v___x_114_, sizeof(void*)*14 + 9, v___x_112_);
lean_inc(v_introGoal_85_);
v___x_115_ = lean_apply_1(v_introGoal_85_, v___x_114_);
v___x_116_ = lean_st_ref_set(v_gref_82_, v___x_115_);
v___x_117_ = lean_st_ref_get(v_gref_82_);
lean_inc_ref(v_elimGoal_86_);
v___x_118_ = lean_apply_1(v_elimGoal_86_, v___x_117_);
v_children_119_ = lean_ctor_get(v___x_118_, 2);
lean_inc_ref(v_children_119_);
lean_dec_ref(v___x_118_);
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = lean_array_get_size(v_children_119_);
v___x_122_ = lean_nat_dec_lt(v___x_120_, v___x_121_);
if (v___x_122_ == 0)
{
lean_dec_ref(v_children_119_);
goto v___jp_76_;
}
else
{
lean_object* v___x_123_; uint8_t v___x_124_; 
v___x_123_ = lean_box(0);
v___x_124_ = lean_nat_dec_le(v___x_121_, v___x_121_);
if (v___x_124_ == 0)
{
if (v___x_122_ == 0)
{
lean_dec_ref(v_children_119_);
goto v___jp_76_;
}
else
{
size_t v___x_125_; size_t v___x_126_; lean_object* v___x_127_; 
v___x_125_ = ((size_t)0ULL);
v___x_126_ = lean_usize_of_nat(v___x_121_);
v___x_127_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0(v_children_119_, v___x_125_, v___x_126_, v___x_123_);
lean_dec_ref(v_children_119_);
goto v___jp_76_;
}
}
else
{
size_t v___x_128_; size_t v___x_129_; lean_object* v___x_130_; 
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lean_usize_of_nat(v___x_121_);
v___x_130_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0(v_children_119_, v___x_128_, v___x_129_, v___x_123_);
lean_dec_ref(v_children_119_);
goto v___jp_76_;
}
}
}
}
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_box(0);
return v___x_133_;
}
}
case 1:
{
lean_object* v_rref_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_introRapp_137_; lean_object* v_elimRapp_138_; lean_object* v___x_139_; uint8_t v_isIrrelevant_140_; 
v_rref_134_ = lean_ctor_get(v_x_74_, 0);
v___x_135_ = lean_st_ref_get(v_rref_134_);
v___x_136_ = lp_aesop_Aesop_treeImpl;
v_introRapp_137_ = lean_ctor_get(v___x_136_, 2);
v_elimRapp_138_ = lean_ctor_get(v___x_136_, 3);
lean_inc_ref(v_elimRapp_138_);
v___x_139_ = lean_apply_1(v_elimRapp_138_, v___x_135_);
v_isIrrelevant_140_ = lean_ctor_get_uint8(v___x_139_, sizeof(void*)*9 + 9);
lean_dec_ref(v___x_139_);
if (v_isIrrelevant_140_ == 0)
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v_id_143_; lean_object* v_parent_144_; lean_object* v_children_145_; uint8_t v_state_146_; lean_object* v_appliedRule_147_; lean_object* v_scriptSteps_x3f_148_; lean_object* v_originalSubgoals_149_; double v_successProbability_150_; lean_object* v_metaState_151_; lean_object* v_introducedMVars_152_; lean_object* v_assignedMVars_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_177_; 
v___x_141_ = lean_st_ref_take(v_rref_134_);
lean_inc_ref(v_elimRapp_138_);
v___x_142_ = lean_apply_1(v_elimRapp_138_, v___x_141_);
v_id_143_ = lean_ctor_get(v___x_142_, 0);
v_parent_144_ = lean_ctor_get(v___x_142_, 1);
v_children_145_ = lean_ctor_get(v___x_142_, 2);
v_state_146_ = lean_ctor_get_uint8(v___x_142_, sizeof(void*)*9 + 8);
v_appliedRule_147_ = lean_ctor_get(v___x_142_, 3);
v_scriptSteps_x3f_148_ = lean_ctor_get(v___x_142_, 4);
v_originalSubgoals_149_ = lean_ctor_get(v___x_142_, 5);
v_successProbability_150_ = lean_ctor_get_float(v___x_142_, sizeof(void*)*9);
v_metaState_151_ = lean_ctor_get(v___x_142_, 6);
v_introducedMVars_152_ = lean_ctor_get(v___x_142_, 7);
v_assignedMVars_153_ = lean_ctor_get(v___x_142_, 8);
v_isSharedCheck_177_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_177_ == 0)
{
v___x_155_ = v___x_142_;
v_isShared_156_ = v_isSharedCheck_177_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_assignedMVars_153_);
lean_inc(v_introducedMVars_152_);
lean_inc(v_metaState_151_);
lean_inc(v_originalSubgoals_149_);
lean_inc(v_scriptSteps_x3f_148_);
lean_inc(v_appliedRule_147_);
lean_inc(v_children_145_);
lean_inc(v_parent_144_);
lean_inc(v_id_143_);
lean_dec(v___x_142_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_177_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
uint8_t v___x_157_; lean_object* v___x_159_; 
v___x_157_ = 1;
if (v_isShared_156_ == 0)
{
v___x_159_ = v___x_155_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v_id_143_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v_parent_144_);
lean_ctor_set(v_reuseFailAlloc_176_, 2, v_children_145_);
lean_ctor_set(v_reuseFailAlloc_176_, 3, v_appliedRule_147_);
lean_ctor_set(v_reuseFailAlloc_176_, 4, v_scriptSteps_x3f_148_);
lean_ctor_set(v_reuseFailAlloc_176_, 5, v_originalSubgoals_149_);
lean_ctor_set(v_reuseFailAlloc_176_, 6, v_metaState_151_);
lean_ctor_set(v_reuseFailAlloc_176_, 7, v_introducedMVars_152_);
lean_ctor_set(v_reuseFailAlloc_176_, 8, v_assignedMVars_153_);
lean_ctor_set_uint8(v_reuseFailAlloc_176_, sizeof(void*)*9 + 8, v_state_146_);
lean_ctor_set_float(v_reuseFailAlloc_176_, sizeof(void*)*9, v_successProbability_150_);
v___x_159_ = v_reuseFailAlloc_176_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v_children_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
lean_ctor_set_uint8(v___x_159_, sizeof(void*)*9 + 9, v___x_157_);
lean_inc(v_introRapp_137_);
v___x_160_ = lean_apply_1(v_introRapp_137_, v___x_159_);
v___x_161_ = lean_st_ref_set(v_rref_134_, v___x_160_);
v___x_162_ = lean_st_ref_get(v_rref_134_);
lean_inc_ref(v_elimRapp_138_);
v___x_163_ = lean_apply_1(v_elimRapp_138_, v___x_162_);
v_children_164_ = lean_ctor_get(v___x_163_, 2);
lean_inc_ref(v_children_164_);
lean_dec_ref(v___x_163_);
v___x_165_ = lean_unsigned_to_nat(0u);
v___x_166_ = lean_array_get_size(v_children_164_);
v___x_167_ = lean_nat_dec_lt(v___x_165_, v___x_166_);
if (v___x_167_ == 0)
{
lean_dec_ref(v_children_164_);
goto v___jp_78_;
}
else
{
lean_object* v___x_168_; uint8_t v___x_169_; 
v___x_168_ = lean_box(0);
v___x_169_ = lean_nat_dec_le(v___x_166_, v___x_166_);
if (v___x_169_ == 0)
{
if (v___x_167_ == 0)
{
lean_dec_ref(v_children_164_);
goto v___jp_78_;
}
else
{
size_t v___x_170_; size_t v___x_171_; lean_object* v___x_172_; 
v___x_170_ = ((size_t)0ULL);
v___x_171_ = lean_usize_of_nat(v___x_166_);
v___x_172_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1(v_children_164_, v___x_170_, v___x_171_, v___x_168_);
lean_dec_ref(v_children_164_);
goto v___jp_78_;
}
}
else
{
size_t v___x_173_; size_t v___x_174_; lean_object* v___x_175_; 
v___x_173_ = ((size_t)0ULL);
v___x_174_ = lean_usize_of_nat(v___x_166_);
v___x_175_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1(v_children_164_, v___x_173_, v___x_174_, v___x_168_);
lean_dec_ref(v_children_164_);
goto v___jp_78_;
}
}
}
}
}
else
{
lean_object* v___x_178_; 
v___x_178_ = lean_box(0);
return v___x_178_;
}
}
default: 
{
lean_object* v_cref_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v_introMVarCluster_182_; lean_object* v_elimMVarCluster_183_; lean_object* v___x_184_; uint8_t v_isIrrelevant_185_; 
v_cref_179_ = lean_ctor_get(v_x_74_, 0);
v___x_180_ = lean_st_ref_get(v_cref_179_);
v___x_181_ = lp_aesop_Aesop_treeImpl;
v_introMVarCluster_182_ = lean_ctor_get(v___x_181_, 4);
v_elimMVarCluster_183_ = lean_ctor_get(v___x_181_, 5);
lean_inc_ref(v_elimMVarCluster_183_);
v___x_184_ = lean_apply_1(v_elimMVarCluster_183_, v___x_180_);
v_isIrrelevant_185_ = lean_ctor_get_uint8(v___x_184_, sizeof(void*)*2);
lean_dec_ref(v___x_184_);
if (v_isIrrelevant_185_ == 0)
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v_parent_x3f_188_; lean_object* v_goals_189_; uint8_t v_state_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_214_; 
v___x_186_ = lean_st_ref_take(v_cref_179_);
lean_inc_ref(v_elimMVarCluster_183_);
v___x_187_ = lean_apply_1(v_elimMVarCluster_183_, v___x_186_);
v_parent_x3f_188_ = lean_ctor_get(v___x_187_, 0);
v_goals_189_ = lean_ctor_get(v___x_187_, 1);
v_state_190_ = lean_ctor_get_uint8(v___x_187_, sizeof(void*)*2 + 1);
v_isSharedCheck_214_ = !lean_is_exclusive(v___x_187_);
if (v_isSharedCheck_214_ == 0)
{
v___x_192_ = v___x_187_;
v_isShared_193_ = v_isSharedCheck_214_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_goals_189_);
lean_inc(v_parent_x3f_188_);
lean_dec(v___x_187_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_214_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
uint8_t v___x_194_; lean_object* v___x_196_; 
v___x_194_ = 1;
if (v_isShared_193_ == 0)
{
v___x_196_ = v___x_192_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_parent_x3f_188_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v_goals_189_);
lean_ctor_set_uint8(v_reuseFailAlloc_213_, sizeof(void*)*2 + 1, v_state_190_);
v___x_196_ = v_reuseFailAlloc_213_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v_goals_201_; lean_object* v___x_202_; lean_object* v___x_203_; uint8_t v___x_204_; 
lean_ctor_set_uint8(v___x_196_, sizeof(void*)*2, v___x_194_);
lean_inc(v_introMVarCluster_182_);
v___x_197_ = lean_apply_1(v_introMVarCluster_182_, v___x_196_);
v___x_198_ = lean_st_ref_set(v_cref_179_, v___x_197_);
v___x_199_ = lean_st_ref_get(v_cref_179_);
lean_inc_ref(v_elimMVarCluster_183_);
v___x_200_ = lean_apply_1(v_elimMVarCluster_183_, v___x_199_);
v_goals_201_ = lean_ctor_get(v___x_200_, 1);
lean_inc_ref(v_goals_201_);
lean_dec_ref(v___x_200_);
v___x_202_ = lean_unsigned_to_nat(0u);
v___x_203_ = lean_array_get_size(v_goals_201_);
v___x_204_ = lean_nat_dec_lt(v___x_202_, v___x_203_);
if (v___x_204_ == 0)
{
lean_dec_ref(v_goals_201_);
goto v___jp_80_;
}
else
{
lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_205_ = lean_box(0);
v___x_206_ = lean_nat_dec_le(v___x_203_, v___x_203_);
if (v___x_206_ == 0)
{
if (v___x_204_ == 0)
{
lean_dec_ref(v_goals_201_);
goto v___jp_80_;
}
else
{
size_t v___x_207_; size_t v___x_208_; lean_object* v___x_209_; 
v___x_207_ = ((size_t)0ULL);
v___x_208_ = lean_usize_of_nat(v___x_203_);
v___x_209_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2(v_goals_201_, v___x_207_, v___x_208_, v___x_205_);
lean_dec_ref(v_goals_201_);
goto v___jp_80_;
}
}
else
{
size_t v___x_210_; size_t v___x_211_; lean_object* v___x_212_; 
v___x_210_ = ((size_t)0ULL);
v___x_211_ = lean_usize_of_nat(v___x_203_);
v___x_212_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2(v_goals_201_, v___x_210_, v___x_211_, v___x_205_);
lean_dec_ref(v_goals_201_);
goto v___jp_80_;
}
}
}
}
}
else
{
lean_object* v___x_215_; 
v___x_215_ = lean_box(0);
return v___x_215_;
}
}
}
v___jp_76_:
{
lean_object* v___x_77_; 
v___x_77_ = lean_box(0);
return v___x_77_;
}
v___jp_78_:
{
lean_object* v___x_79_; 
v___x_79_ = lean_box(0);
return v___x_79_;
}
v___jp_80_:
{
lean_object* v___x_81_; 
v___x_81_ = lean_box(0);
return v___x_81_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0(lean_object* v_as_216_, size_t v_i_217_, size_t v_stop_218_, lean_object* v_b_219_){
_start:
{
uint8_t v___x_221_; 
v___x_221_ = lean_usize_dec_eq(v_i_217_, v_stop_218_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; size_t v___x_225_; size_t v___x_226_; 
v___x_222_ = lean_array_uget_borrowed(v_as_216_, v_i_217_);
lean_inc(v___x_222_);
v___x_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
v___x_224_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_223_);
lean_dec_ref_known(v___x_223_, 1);
v___x_225_ = ((size_t)1ULL);
v___x_226_ = lean_usize_add(v_i_217_, v___x_225_);
v_i_217_ = v___x_226_;
v_b_219_ = v___x_224_;
goto _start;
}
else
{
return v_b_219_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0___boxed(lean_object* v_as_228_, lean_object* v_i_229_, lean_object* v_stop_230_, lean_object* v_b_231_, lean_object* v___y_232_){
_start:
{
size_t v_i_boxed_233_; size_t v_stop_boxed_234_; lean_object* v_res_235_; 
v_i_boxed_233_ = lean_unbox_usize(v_i_229_);
lean_dec(v_i_229_);
v_stop_boxed_234_ = lean_unbox_usize(v_stop_230_);
lean_dec(v_stop_230_);
v_res_235_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__0(v_as_228_, v_i_boxed_233_, v_stop_boxed_234_, v_b_231_);
lean_dec_ref(v_as_228_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1___boxed(lean_object* v_as_236_, lean_object* v_i_237_, lean_object* v_stop_238_, lean_object* v_b_239_, lean_object* v___y_240_){
_start:
{
size_t v_i_boxed_241_; size_t v_stop_boxed_242_; lean_object* v_res_243_; 
v_i_boxed_241_ = lean_unbox_usize(v_i_237_);
lean_dec(v_i_237_);
v_stop_boxed_242_ = lean_unbox_usize(v_stop_238_);
lean_dec(v_stop_238_);
v_res_243_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__1(v_as_236_, v_i_boxed_241_, v_stop_boxed_242_, v_b_239_);
lean_dec_ref(v_as_236_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2___boxed(lean_object* v_as_244_, lean_object* v_i_245_, lean_object* v_stop_246_, lean_object* v_b_247_, lean_object* v___y_248_){
_start:
{
size_t v_i_boxed_249_; size_t v_stop_boxed_250_; lean_object* v_res_251_; 
v_i_boxed_249_ = lean_unbox_usize(v_i_245_);
lean_dec(v_i_245_);
v_stop_boxed_250_ = lean_unbox_usize(v_stop_246_);
lean_dec(v_stop_246_);
v_res_251_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0_spec__2(v_as_244_, v_i_boxed_249_, v_stop_boxed_250_, v_b_247_);
lean_dec_ref(v_as_244_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0___boxed(lean_object* v_x_252_, lean_object* v___y_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v_x_252_);
lean_dec_ref(v_x_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_markSubtreeIrrelevant(lean_object* v_a_255_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v_a_255_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TreeRef_markSubtreeIrrelevant___boxed(lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_aesop_Aesop_TreeRef_markSubtreeIrrelevant(v_a_258_);
lean_dec_ref(v_a_258_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant(lean_object* v_gref_261_){
_start:
{
lean_object* v___x_263_; lean_object* v___x_264_; 
v___x_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_263_, 0, v_gref_261_);
v___x_264_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_263_);
lean_dec_ref_known(v___x_263_, 1);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant___boxed(lean_object* v_gref_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant(v_gref_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markSubtreeIrrelevant(lean_object* v_rref_268_){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_270_, 0, v_rref_268_);
v___x_271_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_270_);
lean_dec_ref_known(v___x_270_, 1);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markSubtreeIrrelevant___boxed(lean_object* v_rref_272_, lean_object* v_a_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_aesop_Aesop_RappRef_markSubtreeIrrelevant(v_rref_272_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant(lean_object* v_cref_275_){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_277_, 0, v_cref_275_);
v___x_278_ = lp_aesop_Aesop_traverseDown___at___00Aesop_TreeRef_markSubtreeIrrelevant_spec__0(v___x_277_);
lean_dec_ref_known(v___x_277_, 1);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant___boxed(lean_object* v_cref_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant(v_cref_279_);
return v_res_281_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache(lean_object* v_g_282_){
_start:
{
lean_object* v___x_283_; lean_object* v_elimGoal_284_; lean_object* v___x_285_; lean_object* v_normalizationState_286_; uint8_t v___x_287_; 
v___x_283_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_284_ = lean_ctor_get(v___x_283_, 1);
lean_inc_ref(v_elimGoal_284_);
v___x_285_ = lean_apply_1(v_elimGoal_284_, v_g_282_);
v_normalizationState_286_ = lean_ctor_get(v___x_285_, 6);
lean_inc(v_normalizationState_286_);
lean_dec_ref(v___x_285_);
v___x_287_ = lp_aesop_Aesop_NormalizationState_isProvenByNormalization(v_normalizationState_286_);
lean_dec(v_normalizationState_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache___boxed(lean_object* v_g_288_){
_start:
{
uint8_t v_res_289_; lean_object* v_r_290_; 
v_res_289_ = lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache(v_g_288_);
v_r_290_ = lean_box(v_res_289_);
return v_r_290_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0(lean_object* v_as_291_, size_t v_i_292_, size_t v_stop_293_){
_start:
{
uint8_t v___x_295_; 
v___x_295_ = lean_usize_dec_eq(v_i_292_, v_stop_293_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v_elimRapp_299_; lean_object* v___x_300_; uint8_t v_state_301_; uint8_t v___x_302_; 
v___x_296_ = lean_array_uget_borrowed(v_as_291_, v_i_292_);
v___x_297_ = lean_st_ref_get(v___x_296_);
v___x_298_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_299_ = lean_ctor_get(v___x_298_, 3);
lean_inc_ref(v_elimRapp_299_);
v___x_300_ = lean_apply_1(v_elimRapp_299_, v___x_297_);
v_state_301_ = lean_ctor_get_uint8(v___x_300_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_300_);
v___x_302_ = lp_aesop_Aesop_NodeState_isProven(v_state_301_);
if (v___x_302_ == 0)
{
size_t v___x_303_; size_t v___x_304_; 
v___x_303_ = ((size_t)1ULL);
v___x_304_ = lean_usize_add(v_i_292_, v___x_303_);
v_i_292_ = v___x_304_;
goto _start;
}
else
{
return v___x_302_;
}
}
else
{
uint8_t v___x_306_; 
v___x_306_ = 0;
return v___x_306_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0___boxed(lean_object* v_as_307_, lean_object* v_i_308_, lean_object* v_stop_309_, lean_object* v___y_310_){
_start:
{
size_t v_i_boxed_311_; size_t v_stop_boxed_312_; uint8_t v_res_313_; lean_object* v_r_314_; 
v_i_boxed_311_ = lean_unbox_usize(v_i_308_);
lean_dec(v_i_308_);
v_stop_boxed_312_ = lean_unbox_usize(v_stop_309_);
lean_dec(v_stop_309_);
v_res_313_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0(v_as_307_, v_i_boxed_311_, v_stop_boxed_312_);
lean_dec_ref(v_as_307_);
v_r_314_ = lean_box(v_res_313_);
return v_r_314_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache(lean_object* v_g_315_){
_start:
{
lean_object* v___x_317_; lean_object* v_elimGoal_318_; lean_object* v___x_319_; lean_object* v_children_320_; lean_object* v___x_321_; lean_object* v___x_322_; uint8_t v___x_323_; 
v___x_317_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_318_ = lean_ctor_get(v___x_317_, 1);
lean_inc_ref(v_elimGoal_318_);
v___x_319_ = lean_apply_1(v_elimGoal_318_, v_g_315_);
v_children_320_ = lean_ctor_get(v___x_319_, 2);
lean_inc_ref(v_children_320_);
lean_dec_ref(v___x_319_);
v___x_321_ = lean_unsigned_to_nat(0u);
v___x_322_ = lean_array_get_size(v_children_320_);
v___x_323_ = lean_nat_dec_lt(v___x_321_, v___x_322_);
if (v___x_323_ == 0)
{
lean_dec_ref(v_children_320_);
return v___x_323_;
}
else
{
if (v___x_323_ == 0)
{
lean_dec_ref(v_children_320_);
return v___x_323_;
}
else
{
size_t v___x_324_; size_t v___x_325_; uint8_t v___x_326_; 
v___x_324_ = ((size_t)0ULL);
v___x_325_ = lean_usize_of_nat(v___x_322_);
v___x_326_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isProvenByRuleApplicationNoCache_spec__0(v_children_320_, v___x_324_, v___x_325_);
lean_dec_ref(v_children_320_);
return v___x_326_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache___boxed(lean_object* v_g_327_, lean_object* v_a_328_){
_start:
{
uint8_t v_res_329_; lean_object* v_r_330_; 
v_res_329_ = lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache(v_g_327_);
v_r_330_ = lean_box(v_res_329_);
return v_r_330_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isProvenNoCache(lean_object* v_g_331_){
_start:
{
uint8_t v___x_333_; 
lean_inc(v_g_331_);
v___x_333_ = lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache(v_g_331_);
if (v___x_333_ == 0)
{
uint8_t v___x_334_; 
v___x_334_ = lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache(v_g_331_);
return v___x_334_;
}
else
{
lean_dec(v_g_331_);
return v___x_333_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isProvenNoCache___boxed(lean_object* v_g_335_, lean_object* v_a_336_){
_start:
{
uint8_t v_res_337_; lean_object* v_r_338_; 
v_res_337_ = lp_aesop_Aesop_Goal_isProvenNoCache(v_g_335_);
v_r_338_ = lean_box(v_res_337_);
return v_r_338_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0(lean_object* v_as_339_, size_t v_i_340_, size_t v_stop_341_){
_start:
{
uint8_t v___x_343_; 
v___x_343_ = lean_usize_dec_eq(v_i_340_, v_stop_341_);
if (v___x_343_ == 0)
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v_elimMVarCluster_347_; lean_object* v___x_348_; uint8_t v_state_349_; uint8_t v___x_350_; uint8_t v___x_351_; 
v___x_344_ = lean_array_uget_borrowed(v_as_339_, v_i_340_);
v___x_345_ = lean_st_ref_get(v___x_344_);
v___x_346_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_347_ = lean_ctor_get(v___x_346_, 5);
lean_inc_ref(v_elimMVarCluster_347_);
v___x_348_ = lean_apply_1(v_elimMVarCluster_347_, v___x_345_);
v_state_349_ = lean_ctor_get_uint8(v___x_348_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_348_);
v___x_350_ = 1;
v___x_351_ = lp_aesop_Aesop_NodeState_isProven(v_state_349_);
if (v___x_351_ == 0)
{
return v___x_350_;
}
else
{
if (v___x_343_ == 0)
{
size_t v___x_352_; size_t v___x_353_; 
v___x_352_ = ((size_t)1ULL);
v___x_353_ = lean_usize_add(v_i_340_, v___x_352_);
v_i_340_ = v___x_353_;
goto _start;
}
else
{
return v___x_350_;
}
}
}
else
{
uint8_t v___x_355_; 
v___x_355_ = 0;
return v___x_355_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0___boxed(lean_object* v_as_356_, lean_object* v_i_357_, lean_object* v_stop_358_, lean_object* v___y_359_){
_start:
{
size_t v_i_boxed_360_; size_t v_stop_boxed_361_; uint8_t v_res_362_; lean_object* v_r_363_; 
v_i_boxed_360_ = lean_unbox_usize(v_i_357_);
lean_dec(v_i_357_);
v_stop_boxed_361_ = lean_unbox_usize(v_stop_358_);
lean_dec(v_stop_358_);
v_res_362_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0(v_as_356_, v_i_boxed_360_, v_stop_boxed_361_);
lean_dec_ref(v_as_356_);
v_r_363_ = lean_box(v_res_362_);
return v_r_363_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isProvenNoCache(lean_object* v_r_364_){
_start:
{
lean_object* v___x_368_; lean_object* v_elimRapp_369_; lean_object* v___x_370_; lean_object* v_children_371_; lean_object* v___x_372_; lean_object* v___x_373_; uint8_t v___x_374_; 
v___x_368_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_369_ = lean_ctor_get(v___x_368_, 3);
lean_inc_ref(v_elimRapp_369_);
v___x_370_ = lean_apply_1(v_elimRapp_369_, v_r_364_);
v_children_371_ = lean_ctor_get(v___x_370_, 2);
lean_inc_ref(v_children_371_);
lean_dec_ref(v___x_370_);
v___x_372_ = lean_unsigned_to_nat(0u);
v___x_373_ = lean_array_get_size(v_children_371_);
v___x_374_ = lean_nat_dec_lt(v___x_372_, v___x_373_);
if (v___x_374_ == 0)
{
lean_dec_ref(v_children_371_);
goto v___jp_366_;
}
else
{
if (v___x_374_ == 0)
{
lean_dec_ref(v_children_371_);
goto v___jp_366_;
}
else
{
size_t v___x_375_; size_t v___x_376_; uint8_t v___x_377_; 
v___x_375_ = ((size_t)0ULL);
v___x_376_ = lean_usize_of_nat(v___x_373_);
v___x_377_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isProvenNoCache_spec__0(v_children_371_, v___x_375_, v___x_376_);
lean_dec_ref(v_children_371_);
if (v___x_377_ == 0)
{
goto v___jp_366_;
}
else
{
uint8_t v___x_378_; 
v___x_378_ = 0;
return v___x_378_;
}
}
}
v___jp_366_:
{
uint8_t v___x_367_; 
v___x_367_ = 1;
return v___x_367_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isProvenNoCache___boxed(lean_object* v_r_379_, lean_object* v_a_380_){
_start:
{
uint8_t v_res_381_; lean_object* v_r_382_; 
v_res_381_ = lp_aesop_Aesop_Rapp_isProvenNoCache(v_r_379_);
v_r_382_ = lean_box(v_res_381_);
return v_r_382_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0(lean_object* v_as_383_, size_t v_i_384_, size_t v_stop_385_){
_start:
{
uint8_t v___x_387_; 
v___x_387_ = lean_usize_dec_eq(v_i_384_, v_stop_385_);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v_elimGoal_391_; lean_object* v___x_392_; uint8_t v_state_393_; uint8_t v___x_394_; 
v___x_388_ = lean_array_uget_borrowed(v_as_383_, v_i_384_);
v___x_389_ = lean_st_ref_get(v___x_388_);
v___x_390_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_391_ = lean_ctor_get(v___x_390_, 1);
lean_inc_ref(v_elimGoal_391_);
v___x_392_ = lean_apply_1(v_elimGoal_391_, v___x_389_);
v_state_393_ = lean_ctor_get_uint8(v___x_392_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_392_);
v___x_394_ = lp_aesop_Aesop_GoalState_isProven(v_state_393_);
if (v___x_394_ == 0)
{
size_t v___x_395_; size_t v___x_396_; 
v___x_395_ = ((size_t)1ULL);
v___x_396_ = lean_usize_add(v_i_384_, v___x_395_);
v_i_384_ = v___x_396_;
goto _start;
}
else
{
return v___x_394_;
}
}
else
{
uint8_t v___x_398_; 
v___x_398_ = 0;
return v___x_398_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0___boxed(lean_object* v_as_399_, lean_object* v_i_400_, lean_object* v_stop_401_, lean_object* v___y_402_){
_start:
{
size_t v_i_boxed_403_; size_t v_stop_boxed_404_; uint8_t v_res_405_; lean_object* v_r_406_; 
v_i_boxed_403_ = lean_unbox_usize(v_i_400_);
lean_dec(v_i_400_);
v_stop_boxed_404_ = lean_unbox_usize(v_stop_401_);
lean_dec(v_stop_401_);
v_res_405_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0(v_as_399_, v_i_boxed_403_, v_stop_boxed_404_);
lean_dec_ref(v_as_399_);
v_r_406_ = lean_box(v_res_405_);
return v_r_406_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isProvenNoCache(lean_object* v_c_407_){
_start:
{
lean_object* v___x_409_; lean_object* v_elimMVarCluster_410_; lean_object* v___x_411_; lean_object* v_goals_412_; lean_object* v___x_413_; lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_409_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_410_ = lean_ctor_get(v___x_409_, 5);
lean_inc_ref(v_elimMVarCluster_410_);
v___x_411_ = lean_apply_1(v_elimMVarCluster_410_, v_c_407_);
v_goals_412_ = lean_ctor_get(v___x_411_, 1);
lean_inc_ref(v_goals_412_);
lean_dec_ref(v___x_411_);
v___x_413_ = lean_unsigned_to_nat(0u);
v___x_414_ = lean_array_get_size(v_goals_412_);
v___x_415_ = lean_nat_dec_lt(v___x_413_, v___x_414_);
if (v___x_415_ == 0)
{
lean_dec_ref(v_goals_412_);
return v___x_415_;
}
else
{
if (v___x_415_ == 0)
{
lean_dec_ref(v_goals_412_);
return v___x_415_;
}
else
{
size_t v___x_416_; size_t v___x_417_; uint8_t v___x_418_; 
v___x_416_ = ((size_t)0ULL);
v___x_417_ = lean_usize_of_nat(v___x_414_);
v___x_418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isProvenNoCache_spec__0(v_goals_412_, v___x_416_, v___x_417_);
lean_dec_ref(v_goals_412_);
return v___x_418_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isProvenNoCache___boxed(lean_object* v_c_419_, lean_object* v_a_420_){
_start:
{
uint8_t v_res_421_; lean_object* v_r_422_; 
v_res_421_ = lp_aesop_Aesop_MVarCluster_isProvenNoCache(v_c_419_);
v_r_422_ = lean_box(v_res_421_);
return v_r_422_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0(lean_object* v_as_423_, size_t v_i_424_, size_t v_stop_425_, lean_object* v_b_426_){
_start:
{
uint8_t v___x_428_; 
v___x_428_ = lean_usize_dec_eq(v_i_424_, v_stop_425_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; size_t v___x_431_; size_t v___x_432_; 
v___x_429_ = lean_array_uget_borrowed(v_as_423_, v_i_424_);
lean_inc(v___x_429_);
v___x_430_ = lp_aesop_Aesop_GoalRef_markSubtreeIrrelevant(v___x_429_);
v___x_431_ = ((size_t)1ULL);
v___x_432_ = lean_usize_add(v_i_424_, v___x_431_);
v_i_424_ = v___x_432_;
v_b_426_ = v___x_430_;
goto _start;
}
else
{
return v_b_426_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0___boxed(lean_object* v_as_434_, lean_object* v_i_435_, lean_object* v_stop_436_, lean_object* v_b_437_, lean_object* v___y_438_){
_start:
{
size_t v_i_boxed_439_; size_t v_stop_boxed_440_; lean_object* v_res_441_; 
v_i_boxed_439_ = lean_unbox_usize(v_i_435_);
lean_dec(v_i_435_);
v_stop_boxed_440_ = lean_unbox_usize(v_stop_436_);
lean_dec(v_stop_436_);
v_res_441_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0(v_as_434_, v_i_boxed_439_, v_stop_boxed_440_, v_b_437_);
lean_dec_ref(v_as_434_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(lean_object* v_as_442_, size_t v_i_443_, size_t v_stop_444_, lean_object* v_b_445_){
_start:
{
uint8_t v___x_447_; 
v___x_447_ = lean_usize_dec_eq(v_i_443_, v_stop_444_);
if (v___x_447_ == 0)
{
lean_object* v___x_448_; lean_object* v___x_449_; size_t v___x_450_; size_t v___x_451_; 
v___x_448_ = lean_array_uget_borrowed(v_as_442_, v_i_443_);
lean_inc(v___x_448_);
v___x_449_ = lp_aesop_Aesop_RappRef_markSubtreeIrrelevant(v___x_448_);
v___x_450_ = ((size_t)1ULL);
v___x_451_ = lean_usize_add(v_i_443_, v___x_450_);
v_i_443_ = v___x_451_;
v_b_445_ = v___x_449_;
goto _start;
}
else
{
return v_b_445_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1___boxed(lean_object* v_as_453_, lean_object* v_i_454_, lean_object* v_stop_455_, lean_object* v_b_456_, lean_object* v___y_457_){
_start:
{
size_t v_i_boxed_458_; size_t v_stop_boxed_459_; lean_object* v_res_460_; 
v_i_boxed_458_ = lean_unbox_usize(v_i_454_);
lean_dec(v_i_454_);
v_stop_boxed_459_ = lean_unbox_usize(v_stop_455_);
lean_dec(v_stop_455_);
v_res_460_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_as_453_, v_i_boxed_458_, v_stop_boxed_459_, v_b_456_);
lean_dec_ref(v_as_453_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(lean_object* v_x_461_){
_start:
{
switch(lean_obj_tag(v_x_461_))
{
case 0:
{
lean_object* v_gref_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_554_; 
v_gref_465_ = lean_ctor_get(v_x_461_, 0);
v_isSharedCheck_554_ = !lean_is_exclusive(v_x_461_);
if (v_isSharedCheck_554_ == 0)
{
v___x_467_ = v_x_461_;
v_isShared_468_ = v_isSharedCheck_554_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_gref_465_);
lean_dec(v_x_461_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_554_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___y_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v_introGoal_484_; lean_object* v_elimGoal_485_; lean_object* v___x_486_; lean_object* v_id_487_; lean_object* v_parent_488_; lean_object* v_children_489_; lean_object* v_origin_490_; lean_object* v_depth_491_; uint8_t v_isIrrelevant_492_; uint8_t v_isForcedUnprovable_493_; lean_object* v_preNormGoal_494_; lean_object* v_normalizationState_495_; lean_object* v_mvars_496_; lean_object* v_forwardState_497_; lean_object* v_forwardRuleMatches_498_; double v_successProbability_499_; lean_object* v_addedInIteration_500_; lean_object* v_lastExpandedInIteration_501_; uint8_t v_unsafeRulesSelected_502_; lean_object* v_unsafeQueue_503_; lean_object* v_failedRapps_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_553_; 
v___x_482_ = lean_st_ref_get(v_gref_465_);
v___x_483_ = lp_aesop_Aesop_treeImpl;
v_introGoal_484_ = lean_ctor_get(v___x_483_, 0);
v_elimGoal_485_ = lean_ctor_get(v___x_483_, 1);
lean_inc_ref(v_elimGoal_485_);
v___x_486_ = lean_apply_1(v_elimGoal_485_, v___x_482_);
v_id_487_ = lean_ctor_get(v___x_486_, 0);
v_parent_488_ = lean_ctor_get(v___x_486_, 1);
v_children_489_ = lean_ctor_get(v___x_486_, 2);
v_origin_490_ = lean_ctor_get(v___x_486_, 3);
v_depth_491_ = lean_ctor_get(v___x_486_, 4);
v_isIrrelevant_492_ = lean_ctor_get_uint8(v___x_486_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_493_ = lean_ctor_get_uint8(v___x_486_, sizeof(void*)*14 + 10);
v_preNormGoal_494_ = lean_ctor_get(v___x_486_, 5);
v_normalizationState_495_ = lean_ctor_get(v___x_486_, 6);
v_mvars_496_ = lean_ctor_get(v___x_486_, 7);
v_forwardState_497_ = lean_ctor_get(v___x_486_, 8);
v_forwardRuleMatches_498_ = lean_ctor_get(v___x_486_, 9);
v_successProbability_499_ = lean_ctor_get_float(v___x_486_, sizeof(void*)*14);
v_addedInIteration_500_ = lean_ctor_get(v___x_486_, 10);
v_lastExpandedInIteration_501_ = lean_ctor_get(v___x_486_, 11);
v_unsafeRulesSelected_502_ = lean_ctor_get_uint8(v___x_486_, sizeof(void*)*14 + 11);
v_unsafeQueue_503_ = lean_ctor_get(v___x_486_, 12);
v_failedRapps_504_ = lean_ctor_get(v___x_486_, 13);
v_isSharedCheck_553_ = !lean_is_exclusive(v___x_486_);
if (v_isSharedCheck_553_ == 0)
{
v___x_506_ = v___x_486_;
v_isShared_507_ = v_isSharedCheck_553_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_failedRapps_504_);
lean_inc(v_unsafeQueue_503_);
lean_inc(v_lastExpandedInIteration_501_);
lean_inc(v_addedInIteration_500_);
lean_inc(v_forwardRuleMatches_498_);
lean_inc(v_forwardState_497_);
lean_inc(v_mvars_496_);
lean_inc(v_normalizationState_495_);
lean_inc(v_preNormGoal_494_);
lean_inc(v_depth_491_);
lean_inc(v_origin_490_);
lean_inc(v_children_489_);
lean_inc(v_parent_488_);
lean_inc(v_id_487_);
lean_dec(v___x_486_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_553_;
goto v_resetjp_505_;
}
v___jp_469_:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v_elimGoal_472_; lean_object* v___x_473_; lean_object* v_parent_474_; lean_object* v___x_476_; 
v___x_470_ = lean_st_ref_get(v_gref_465_);
lean_dec(v_gref_465_);
v___x_471_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_472_ = lean_ctor_get(v___x_471_, 1);
lean_inc_ref(v_elimGoal_472_);
v___x_473_ = lean_apply_1(v_elimGoal_472_, v___x_470_);
v_parent_474_ = lean_ctor_get(v___x_473_, 1);
lean_inc(v_parent_474_);
lean_dec_ref(v___x_473_);
if (v_isShared_468_ == 0)
{
lean_ctor_set_tag(v___x_467_, 2);
lean_ctor_set(v___x_467_, 0, v_parent_474_);
v___x_476_ = v___x_467_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_479_; 
v_reuseFailAlloc_479_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_479_, 0, v_parent_474_);
v___x_476_ = v_reuseFailAlloc_479_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_477_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v___x_476_);
v___x_478_ = lean_box(0);
return v___x_478_;
}
}
v___jp_480_:
{
goto v___jp_469_;
}
v_resetjp_505_:
{
uint8_t v___x_508_; lean_object* v___x_510_; 
v___x_508_ = 1;
lean_inc_ref(v_children_489_);
if (v_isShared_507_ == 0)
{
v___x_510_ = v___x_506_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_id_487_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v_parent_488_);
lean_ctor_set(v_reuseFailAlloc_552_, 2, v_children_489_);
lean_ctor_set(v_reuseFailAlloc_552_, 3, v_origin_490_);
lean_ctor_set(v_reuseFailAlloc_552_, 4, v_depth_491_);
lean_ctor_set(v_reuseFailAlloc_552_, 5, v_preNormGoal_494_);
lean_ctor_set(v_reuseFailAlloc_552_, 6, v_normalizationState_495_);
lean_ctor_set(v_reuseFailAlloc_552_, 7, v_mvars_496_);
lean_ctor_set(v_reuseFailAlloc_552_, 8, v_forwardState_497_);
lean_ctor_set(v_reuseFailAlloc_552_, 9, v_forwardRuleMatches_498_);
lean_ctor_set(v_reuseFailAlloc_552_, 10, v_addedInIteration_500_);
lean_ctor_set(v_reuseFailAlloc_552_, 11, v_lastExpandedInIteration_501_);
lean_ctor_set(v_reuseFailAlloc_552_, 12, v_unsafeQueue_503_);
lean_ctor_set(v_reuseFailAlloc_552_, 13, v_failedRapps_504_);
lean_ctor_set_uint8(v_reuseFailAlloc_552_, sizeof(void*)*14 + 9, v_isIrrelevant_492_);
lean_ctor_set_uint8(v_reuseFailAlloc_552_, sizeof(void*)*14 + 10, v_isForcedUnprovable_493_);
lean_ctor_set_float(v_reuseFailAlloc_552_, sizeof(void*)*14, v_successProbability_499_);
lean_ctor_set_uint8(v_reuseFailAlloc_552_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_502_);
v___x_510_ = v_reuseFailAlloc_552_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v_id_513_; lean_object* v_parent_514_; lean_object* v_children_515_; lean_object* v_origin_516_; lean_object* v_depth_517_; uint8_t v_state_518_; uint8_t v_isForcedUnprovable_519_; lean_object* v_preNormGoal_520_; lean_object* v_normalizationState_521_; lean_object* v_mvars_522_; lean_object* v_forwardState_523_; lean_object* v_forwardRuleMatches_524_; double v_successProbability_525_; lean_object* v_addedInIteration_526_; lean_object* v_lastExpandedInIteration_527_; uint8_t v_unsafeRulesSelected_528_; lean_object* v_unsafeQueue_529_; lean_object* v_failedRapps_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_551_; 
lean_ctor_set_uint8(v___x_510_, sizeof(void*)*14 + 8, v___x_508_);
lean_inc(v_introGoal_484_);
v___x_511_ = lean_apply_1(v_introGoal_484_, v___x_510_);
lean_inc_ref(v_elimGoal_485_);
v___x_512_ = lean_apply_1(v_elimGoal_485_, v___x_511_);
v_id_513_ = lean_ctor_get(v___x_512_, 0);
v_parent_514_ = lean_ctor_get(v___x_512_, 1);
v_children_515_ = lean_ctor_get(v___x_512_, 2);
v_origin_516_ = lean_ctor_get(v___x_512_, 3);
v_depth_517_ = lean_ctor_get(v___x_512_, 4);
v_state_518_ = lean_ctor_get_uint8(v___x_512_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_519_ = lean_ctor_get_uint8(v___x_512_, sizeof(void*)*14 + 10);
v_preNormGoal_520_ = lean_ctor_get(v___x_512_, 5);
v_normalizationState_521_ = lean_ctor_get(v___x_512_, 6);
v_mvars_522_ = lean_ctor_get(v___x_512_, 7);
v_forwardState_523_ = lean_ctor_get(v___x_512_, 8);
v_forwardRuleMatches_524_ = lean_ctor_get(v___x_512_, 9);
v_successProbability_525_ = lean_ctor_get_float(v___x_512_, sizeof(void*)*14);
v_addedInIteration_526_ = lean_ctor_get(v___x_512_, 10);
v_lastExpandedInIteration_527_ = lean_ctor_get(v___x_512_, 11);
v_unsafeRulesSelected_528_ = lean_ctor_get_uint8(v___x_512_, sizeof(void*)*14 + 11);
v_unsafeQueue_529_ = lean_ctor_get(v___x_512_, 12);
v_failedRapps_530_ = lean_ctor_get(v___x_512_, 13);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_551_ == 0)
{
v___x_532_ = v___x_512_;
v_isShared_533_ = v_isSharedCheck_551_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_failedRapps_530_);
lean_inc(v_unsafeQueue_529_);
lean_inc(v_lastExpandedInIteration_527_);
lean_inc(v_addedInIteration_526_);
lean_inc(v_forwardRuleMatches_524_);
lean_inc(v_forwardState_523_);
lean_inc(v_mvars_522_);
lean_inc(v_normalizationState_521_);
lean_inc(v_preNormGoal_520_);
lean_inc(v_depth_517_);
lean_inc(v_origin_516_);
lean_inc(v_children_515_);
lean_inc(v_parent_514_);
lean_inc(v_id_513_);
lean_dec(v___x_512_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_551_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
uint8_t v___x_534_; lean_object* v___x_536_; 
v___x_534_ = 1;
if (v_isShared_533_ == 0)
{
v___x_536_ = v___x_532_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v_id_513_);
lean_ctor_set(v_reuseFailAlloc_550_, 1, v_parent_514_);
lean_ctor_set(v_reuseFailAlloc_550_, 2, v_children_515_);
lean_ctor_set(v_reuseFailAlloc_550_, 3, v_origin_516_);
lean_ctor_set(v_reuseFailAlloc_550_, 4, v_depth_517_);
lean_ctor_set(v_reuseFailAlloc_550_, 5, v_preNormGoal_520_);
lean_ctor_set(v_reuseFailAlloc_550_, 6, v_normalizationState_521_);
lean_ctor_set(v_reuseFailAlloc_550_, 7, v_mvars_522_);
lean_ctor_set(v_reuseFailAlloc_550_, 8, v_forwardState_523_);
lean_ctor_set(v_reuseFailAlloc_550_, 9, v_forwardRuleMatches_524_);
lean_ctor_set(v_reuseFailAlloc_550_, 10, v_addedInIteration_526_);
lean_ctor_set(v_reuseFailAlloc_550_, 11, v_lastExpandedInIteration_527_);
lean_ctor_set(v_reuseFailAlloc_550_, 12, v_unsafeQueue_529_);
lean_ctor_set(v_reuseFailAlloc_550_, 13, v_failedRapps_530_);
lean_ctor_set_uint8(v_reuseFailAlloc_550_, sizeof(void*)*14 + 8, v_state_518_);
lean_ctor_set_uint8(v_reuseFailAlloc_550_, sizeof(void*)*14 + 10, v_isForcedUnprovable_519_);
lean_ctor_set_float(v_reuseFailAlloc_550_, sizeof(void*)*14, v_successProbability_525_);
lean_ctor_set_uint8(v_reuseFailAlloc_550_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_528_);
v___x_536_ = v_reuseFailAlloc_550_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; uint8_t v___x_541_; 
lean_ctor_set_uint8(v___x_536_, sizeof(void*)*14 + 9, v___x_534_);
lean_inc(v_introGoal_484_);
v___x_537_ = lean_apply_1(v_introGoal_484_, v___x_536_);
v___x_538_ = lean_st_ref_set(v_gref_465_, v___x_537_);
v___x_539_ = lean_unsigned_to_nat(0u);
v___x_540_ = lean_array_get_size(v_children_489_);
v___x_541_ = lean_nat_dec_lt(v___x_539_, v___x_540_);
if (v___x_541_ == 0)
{
lean_dec_ref(v_children_489_);
goto v___jp_469_;
}
else
{
lean_object* v___x_542_; uint8_t v___x_543_; 
v___x_542_ = lean_box(0);
v___x_543_ = lean_nat_dec_le(v___x_540_, v___x_540_);
if (v___x_543_ == 0)
{
if (v___x_541_ == 0)
{
lean_dec_ref(v_children_489_);
goto v___jp_469_;
}
else
{
size_t v___x_544_; size_t v___x_545_; lean_object* v___x_546_; 
v___x_544_ = ((size_t)0ULL);
v___x_545_ = lean_usize_of_nat(v___x_540_);
v___x_546_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_489_, v___x_544_, v___x_545_, v___x_542_);
lean_dec_ref(v_children_489_);
v___y_481_ = v___x_546_;
goto v___jp_480_;
}
}
else
{
size_t v___x_547_; size_t v___x_548_; lean_object* v___x_549_; 
v___x_547_ = ((size_t)0ULL);
v___x_548_ = lean_usize_of_nat(v___x_540_);
v___x_549_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_489_, v___x_547_, v___x_548_, v___x_542_);
lean_dec_ref(v_children_489_);
v___y_481_ = v___x_549_;
goto v___jp_480_;
}
}
}
}
}
}
}
}
case 1:
{
lean_object* v_rref_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_616_; 
v_rref_555_ = lean_ctor_get(v_x_461_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v_x_461_);
if (v_isSharedCheck_616_ == 0)
{
v___x_557_ = v_x_461_;
v_isShared_558_ = v_isSharedCheck_616_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_rref_555_);
lean_dec(v_x_461_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_616_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_559_; uint8_t v___x_560_; 
v___x_559_ = lean_st_ref_get(v_rref_555_);
lean_inc(v___x_559_);
v___x_560_ = lp_aesop_Aesop_Rapp_isProvenNoCache(v___x_559_);
if (v___x_560_ == 0)
{
lean_object* v___x_561_; 
lean_dec(v___x_559_);
lean_del_object(v___x_557_);
lean_dec(v_rref_555_);
v___x_561_ = lean_box(0);
return v___x_561_;
}
else
{
lean_object* v___x_562_; lean_object* v_introRapp_563_; lean_object* v_elimRapp_564_; lean_object* v___x_565_; lean_object* v_id_566_; lean_object* v_parent_567_; lean_object* v_children_568_; uint8_t v_isIrrelevant_569_; lean_object* v_appliedRule_570_; lean_object* v_scriptSteps_x3f_571_; lean_object* v_originalSubgoals_572_; double v_successProbability_573_; lean_object* v_metaState_574_; lean_object* v_introducedMVars_575_; lean_object* v_assignedMVars_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_615_; 
v___x_562_ = lp_aesop_Aesop_treeImpl;
v_introRapp_563_ = lean_ctor_get(v___x_562_, 2);
v_elimRapp_564_ = lean_ctor_get(v___x_562_, 3);
lean_inc_ref(v_elimRapp_564_);
v___x_565_ = lean_apply_1(v_elimRapp_564_, v___x_559_);
v_id_566_ = lean_ctor_get(v___x_565_, 0);
v_parent_567_ = lean_ctor_get(v___x_565_, 1);
v_children_568_ = lean_ctor_get(v___x_565_, 2);
v_isIrrelevant_569_ = lean_ctor_get_uint8(v___x_565_, sizeof(void*)*9 + 9);
v_appliedRule_570_ = lean_ctor_get(v___x_565_, 3);
v_scriptSteps_x3f_571_ = lean_ctor_get(v___x_565_, 4);
v_originalSubgoals_572_ = lean_ctor_get(v___x_565_, 5);
v_successProbability_573_ = lean_ctor_get_float(v___x_565_, sizeof(void*)*9);
v_metaState_574_ = lean_ctor_get(v___x_565_, 6);
v_introducedMVars_575_ = lean_ctor_get(v___x_565_, 7);
v_assignedMVars_576_ = lean_ctor_get(v___x_565_, 8);
v_isSharedCheck_615_ = !lean_is_exclusive(v___x_565_);
if (v_isSharedCheck_615_ == 0)
{
v___x_578_ = v___x_565_;
v_isShared_579_ = v_isSharedCheck_615_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_assignedMVars_576_);
lean_inc(v_introducedMVars_575_);
lean_inc(v_metaState_574_);
lean_inc(v_originalSubgoals_572_);
lean_inc(v_scriptSteps_x3f_571_);
lean_inc(v_appliedRule_570_);
lean_inc(v_children_568_);
lean_inc(v_parent_567_);
lean_inc(v_id_566_);
lean_dec(v___x_565_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_615_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
uint8_t v___x_580_; lean_object* v___x_582_; 
v___x_580_ = 1;
if (v_isShared_579_ == 0)
{
v___x_582_ = v___x_578_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v_id_566_);
lean_ctor_set(v_reuseFailAlloc_614_, 1, v_parent_567_);
lean_ctor_set(v_reuseFailAlloc_614_, 2, v_children_568_);
lean_ctor_set(v_reuseFailAlloc_614_, 3, v_appliedRule_570_);
lean_ctor_set(v_reuseFailAlloc_614_, 4, v_scriptSteps_x3f_571_);
lean_ctor_set(v_reuseFailAlloc_614_, 5, v_originalSubgoals_572_);
lean_ctor_set(v_reuseFailAlloc_614_, 6, v_metaState_574_);
lean_ctor_set(v_reuseFailAlloc_614_, 7, v_introducedMVars_575_);
lean_ctor_set(v_reuseFailAlloc_614_, 8, v_assignedMVars_576_);
lean_ctor_set_uint8(v_reuseFailAlloc_614_, sizeof(void*)*9 + 9, v_isIrrelevant_569_);
lean_ctor_set_float(v_reuseFailAlloc_614_, sizeof(void*)*9, v_successProbability_573_);
v___x_582_ = v_reuseFailAlloc_614_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v_id_585_; lean_object* v_parent_586_; lean_object* v_children_587_; uint8_t v_state_588_; lean_object* v_appliedRule_589_; lean_object* v_scriptSteps_x3f_590_; lean_object* v_originalSubgoals_591_; double v_successProbability_592_; lean_object* v_metaState_593_; lean_object* v_introducedMVars_594_; lean_object* v_assignedMVars_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_613_; 
lean_ctor_set_uint8(v___x_582_, sizeof(void*)*9 + 8, v___x_580_);
lean_inc(v_introRapp_563_);
v___x_583_ = lean_apply_1(v_introRapp_563_, v___x_582_);
lean_inc_ref(v_elimRapp_564_);
v___x_584_ = lean_apply_1(v_elimRapp_564_, v___x_583_);
v_id_585_ = lean_ctor_get(v___x_584_, 0);
v_parent_586_ = lean_ctor_get(v___x_584_, 1);
v_children_587_ = lean_ctor_get(v___x_584_, 2);
v_state_588_ = lean_ctor_get_uint8(v___x_584_, sizeof(void*)*9 + 8);
v_appliedRule_589_ = lean_ctor_get(v___x_584_, 3);
v_scriptSteps_x3f_590_ = lean_ctor_get(v___x_584_, 4);
v_originalSubgoals_591_ = lean_ctor_get(v___x_584_, 5);
v_successProbability_592_ = lean_ctor_get_float(v___x_584_, sizeof(void*)*9);
v_metaState_593_ = lean_ctor_get(v___x_584_, 6);
v_introducedMVars_594_ = lean_ctor_get(v___x_584_, 7);
v_assignedMVars_595_ = lean_ctor_get(v___x_584_, 8);
v_isSharedCheck_613_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_613_ == 0)
{
v___x_597_ = v___x_584_;
v_isShared_598_ = v_isSharedCheck_613_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_assignedMVars_595_);
lean_inc(v_introducedMVars_594_);
lean_inc(v_metaState_593_);
lean_inc(v_originalSubgoals_591_);
lean_inc(v_scriptSteps_x3f_590_);
lean_inc(v_appliedRule_589_);
lean_inc(v_children_587_);
lean_inc(v_parent_586_);
lean_inc(v_id_585_);
lean_dec(v___x_584_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_613_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_600_; 
if (v_isShared_598_ == 0)
{
v___x_600_ = v___x_597_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v_id_585_);
lean_ctor_set(v_reuseFailAlloc_612_, 1, v_parent_586_);
lean_ctor_set(v_reuseFailAlloc_612_, 2, v_children_587_);
lean_ctor_set(v_reuseFailAlloc_612_, 3, v_appliedRule_589_);
lean_ctor_set(v_reuseFailAlloc_612_, 4, v_scriptSteps_x3f_590_);
lean_ctor_set(v_reuseFailAlloc_612_, 5, v_originalSubgoals_591_);
lean_ctor_set(v_reuseFailAlloc_612_, 6, v_metaState_593_);
lean_ctor_set(v_reuseFailAlloc_612_, 7, v_introducedMVars_594_);
lean_ctor_set(v_reuseFailAlloc_612_, 8, v_assignedMVars_595_);
lean_ctor_set_uint8(v_reuseFailAlloc_612_, sizeof(void*)*9 + 8, v_state_588_);
lean_ctor_set_float(v_reuseFailAlloc_612_, sizeof(void*)*9, v_successProbability_592_);
v___x_600_ = v_reuseFailAlloc_612_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v_elimRapp_604_; lean_object* v___x_605_; lean_object* v_parent_606_; lean_object* v___x_608_; 
lean_ctor_set_uint8(v___x_600_, sizeof(void*)*9 + 9, v___x_560_);
lean_inc(v_introRapp_563_);
v___x_601_ = lean_apply_1(v_introRapp_563_, v___x_600_);
v___x_602_ = lean_st_ref_set(v_rref_555_, v___x_601_);
v___x_603_ = lean_st_ref_get(v_rref_555_);
lean_dec(v_rref_555_);
v_elimRapp_604_ = lean_ctor_get(v___x_562_, 3);
lean_inc_ref(v_elimRapp_604_);
v___x_605_ = lean_apply_1(v_elimRapp_604_, v___x_603_);
v_parent_606_ = lean_ctor_get(v___x_605_, 1);
lean_inc(v_parent_606_);
lean_dec_ref(v___x_605_);
if (v_isShared_558_ == 0)
{
lean_ctor_set_tag(v___x_557_, 0);
lean_ctor_set(v___x_557_, 0, v_parent_606_);
v___x_608_ = v___x_557_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_parent_606_);
v___x_608_ = v_reuseFailAlloc_611_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
lean_object* v___x_609_; lean_object* v___x_610_; 
v___x_609_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v___x_608_);
v___x_610_ = lean_box(0);
return v___x_610_;
}
}
}
}
}
}
}
}
default: 
{
lean_object* v_cref_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_676_; 
v_cref_617_ = lean_ctor_get(v_x_461_, 0);
v_isSharedCheck_676_ = !lean_is_exclusive(v_x_461_);
if (v_isSharedCheck_676_ == 0)
{
v___x_619_ = v_x_461_;
v_isShared_620_ = v_isSharedCheck_676_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_cref_617_);
lean_dec(v_x_461_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_676_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
lean_object* v___y_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v_introMVarCluster_636_; lean_object* v_elimMVarCluster_637_; lean_object* v___x_638_; lean_object* v_parent_x3f_639_; lean_object* v_goals_640_; uint8_t v_isIrrelevant_641_; lean_object* v___x_643_; uint8_t v_isShared_644_; uint8_t v_isSharedCheck_675_; 
v___x_634_ = lean_st_ref_get(v_cref_617_);
v___x_635_ = lp_aesop_Aesop_treeImpl;
v_introMVarCluster_636_ = lean_ctor_get(v___x_635_, 4);
v_elimMVarCluster_637_ = lean_ctor_get(v___x_635_, 5);
lean_inc_ref(v_elimMVarCluster_637_);
v___x_638_ = lean_apply_1(v_elimMVarCluster_637_, v___x_634_);
v_parent_x3f_639_ = lean_ctor_get(v___x_638_, 0);
v_goals_640_ = lean_ctor_get(v___x_638_, 1);
v_isIrrelevant_641_ = lean_ctor_get_uint8(v___x_638_, sizeof(void*)*2);
v_isSharedCheck_675_ = !lean_is_exclusive(v___x_638_);
if (v_isSharedCheck_675_ == 0)
{
v___x_643_ = v___x_638_;
v_isShared_644_ = v_isSharedCheck_675_;
goto v_resetjp_642_;
}
else
{
lean_inc(v_goals_640_);
lean_inc(v_parent_x3f_639_);
lean_dec(v___x_638_);
v___x_643_ = lean_box(0);
v_isShared_644_ = v_isSharedCheck_675_;
goto v_resetjp_642_;
}
v___jp_621_:
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v_elimMVarCluster_624_; lean_object* v___x_625_; lean_object* v_parent_x3f_626_; 
v___x_622_ = lean_st_ref_get(v_cref_617_);
lean_dec(v_cref_617_);
v___x_623_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_624_ = lean_ctor_get(v___x_623_, 5);
lean_inc_ref(v_elimMVarCluster_624_);
v___x_625_ = lean_apply_1(v_elimMVarCluster_624_, v___x_622_);
v_parent_x3f_626_ = lean_ctor_get(v___x_625_, 0);
lean_inc(v_parent_x3f_626_);
lean_dec_ref(v___x_625_);
if (lean_obj_tag(v_parent_x3f_626_) == 1)
{
lean_object* v_val_627_; lean_object* v___x_629_; 
v_val_627_ = lean_ctor_get(v_parent_x3f_626_, 0);
lean_inc(v_val_627_);
lean_dec_ref_known(v_parent_x3f_626_, 1);
if (v_isShared_620_ == 0)
{
lean_ctor_set_tag(v___x_619_, 1);
lean_ctor_set(v___x_619_, 0, v_val_627_);
v___x_629_ = v___x_619_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_val_627_);
v___x_629_ = v_reuseFailAlloc_631_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
lean_object* v___x_630_; 
v___x_630_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v___x_629_);
goto v___jp_463_;
}
}
else
{
lean_dec(v_parent_x3f_626_);
lean_del_object(v___x_619_);
goto v___jp_463_;
}
}
v___jp_632_:
{
goto v___jp_621_;
}
v_resetjp_642_:
{
uint8_t v___x_645_; lean_object* v___x_647_; 
v___x_645_ = 1;
lean_inc_ref(v_goals_640_);
if (v_isShared_644_ == 0)
{
v___x_647_ = v___x_643_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_674_; 
v_reuseFailAlloc_674_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_674_, 0, v_parent_x3f_639_);
lean_ctor_set(v_reuseFailAlloc_674_, 1, v_goals_640_);
lean_ctor_set_uint8(v_reuseFailAlloc_674_, sizeof(void*)*2, v_isIrrelevant_641_);
v___x_647_ = v_reuseFailAlloc_674_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v_parent_x3f_650_; lean_object* v_goals_651_; uint8_t v_state_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_673_; 
lean_ctor_set_uint8(v___x_647_, sizeof(void*)*2 + 1, v___x_645_);
lean_inc(v_introMVarCluster_636_);
v___x_648_ = lean_apply_1(v_introMVarCluster_636_, v___x_647_);
lean_inc_ref(v_elimMVarCluster_637_);
v___x_649_ = lean_apply_1(v_elimMVarCluster_637_, v___x_648_);
v_parent_x3f_650_ = lean_ctor_get(v___x_649_, 0);
v_goals_651_ = lean_ctor_get(v___x_649_, 1);
v_state_652_ = lean_ctor_get_uint8(v___x_649_, sizeof(void*)*2 + 1);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_673_ == 0)
{
v___x_654_ = v___x_649_;
v_isShared_655_ = v_isSharedCheck_673_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_goals_651_);
lean_inc(v_parent_x3f_650_);
lean_dec(v___x_649_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_673_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
uint8_t v___x_656_; lean_object* v___x_658_; 
v___x_656_ = 1;
if (v_isShared_655_ == 0)
{
v___x_658_ = v___x_654_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_parent_x3f_650_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_goals_651_);
lean_ctor_set_uint8(v_reuseFailAlloc_672_, sizeof(void*)*2 + 1, v_state_652_);
v___x_658_ = v_reuseFailAlloc_672_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; uint8_t v___x_663_; 
lean_ctor_set_uint8(v___x_658_, sizeof(void*)*2, v___x_656_);
lean_inc(v_introMVarCluster_636_);
v___x_659_ = lean_apply_1(v_introMVarCluster_636_, v___x_658_);
v___x_660_ = lean_st_ref_set(v_cref_617_, v___x_659_);
v___x_661_ = lean_unsigned_to_nat(0u);
v___x_662_ = lean_array_get_size(v_goals_640_);
v___x_663_ = lean_nat_dec_lt(v___x_661_, v___x_662_);
if (v___x_663_ == 0)
{
lean_dec_ref(v_goals_640_);
goto v___jp_621_;
}
else
{
lean_object* v___x_664_; uint8_t v___x_665_; 
v___x_664_ = lean_box(0);
v___x_665_ = lean_nat_dec_le(v___x_662_, v___x_662_);
if (v___x_665_ == 0)
{
if (v___x_663_ == 0)
{
lean_dec_ref(v_goals_640_);
goto v___jp_621_;
}
else
{
size_t v___x_666_; size_t v___x_667_; lean_object* v___x_668_; 
v___x_666_ = ((size_t)0ULL);
v___x_667_ = lean_usize_of_nat(v___x_662_);
v___x_668_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0(v_goals_640_, v___x_666_, v___x_667_, v___x_664_);
lean_dec_ref(v_goals_640_);
v___y_633_ = v___x_668_;
goto v___jp_632_;
}
}
else
{
size_t v___x_669_; size_t v___x_670_; lean_object* v___x_671_; 
v___x_669_ = ((size_t)0ULL);
v___x_670_ = lean_usize_of_nat(v___x_662_);
v___x_671_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__0(v_goals_640_, v___x_669_, v___x_670_, v___x_664_);
lean_dec_ref(v_goals_640_);
v___y_633_ = v___x_671_;
goto v___jp_632_;
}
}
}
}
}
}
}
}
}
v___jp_463_:
{
lean_object* v___x_464_; 
v___x_464_ = lean_box(0);
return v___x_464_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2___boxed(lean_object* v_x_677_, lean_object* v___y_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v_x_677_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markProvenCore(lean_object* v_root_680_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v_root_680_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markProvenCore___boxed(lean_object* v_root_683_, lean_object* v_a_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_aesop___private_Aesop_Tree_State_0__Aesop_markProvenCore(v_root_683_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markProvenByNormalization(lean_object* v_gref_686_){
_start:
{
lean_object* v___x_688_; lean_object* v___y_697_; lean_object* v___x_698_; lean_object* v_introGoal_699_; lean_object* v_elimGoal_700_; lean_object* v___x_701_; lean_object* v_id_702_; lean_object* v_parent_703_; lean_object* v_children_704_; lean_object* v_origin_705_; lean_object* v_depth_706_; uint8_t v_isIrrelevant_707_; uint8_t v_isForcedUnprovable_708_; lean_object* v_preNormGoal_709_; lean_object* v_normalizationState_710_; lean_object* v_mvars_711_; lean_object* v_forwardState_712_; lean_object* v_forwardRuleMatches_713_; double v_successProbability_714_; lean_object* v_addedInIteration_715_; lean_object* v_lastExpandedInIteration_716_; uint8_t v_unsafeRulesSelected_717_; lean_object* v_unsafeQueue_718_; lean_object* v_failedRapps_719_; lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_768_; 
v___x_688_ = lean_st_ref_get(v_gref_686_);
v___x_698_ = lp_aesop_Aesop_treeImpl;
v_introGoal_699_ = lean_ctor_get(v___x_698_, 0);
v_elimGoal_700_ = lean_ctor_get(v___x_698_, 1);
lean_inc_ref(v_elimGoal_700_);
lean_inc(v___x_688_);
v___x_701_ = lean_apply_1(v_elimGoal_700_, v___x_688_);
v_id_702_ = lean_ctor_get(v___x_701_, 0);
v_parent_703_ = lean_ctor_get(v___x_701_, 1);
v_children_704_ = lean_ctor_get(v___x_701_, 2);
v_origin_705_ = lean_ctor_get(v___x_701_, 3);
v_depth_706_ = lean_ctor_get(v___x_701_, 4);
v_isIrrelevant_707_ = lean_ctor_get_uint8(v___x_701_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_708_ = lean_ctor_get_uint8(v___x_701_, sizeof(void*)*14 + 10);
v_preNormGoal_709_ = lean_ctor_get(v___x_701_, 5);
v_normalizationState_710_ = lean_ctor_get(v___x_701_, 6);
v_mvars_711_ = lean_ctor_get(v___x_701_, 7);
v_forwardState_712_ = lean_ctor_get(v___x_701_, 8);
v_forwardRuleMatches_713_ = lean_ctor_get(v___x_701_, 9);
v_successProbability_714_ = lean_ctor_get_float(v___x_701_, sizeof(void*)*14);
v_addedInIteration_715_ = lean_ctor_get(v___x_701_, 10);
v_lastExpandedInIteration_716_ = lean_ctor_get(v___x_701_, 11);
v_unsafeRulesSelected_717_ = lean_ctor_get_uint8(v___x_701_, sizeof(void*)*14 + 11);
v_unsafeQueue_718_ = lean_ctor_get(v___x_701_, 12);
v_failedRapps_719_ = lean_ctor_get(v___x_701_, 13);
v_isSharedCheck_768_ = !lean_is_exclusive(v___x_701_);
if (v_isSharedCheck_768_ == 0)
{
v___x_721_ = v___x_701_;
v_isShared_722_ = v_isSharedCheck_768_;
goto v_resetjp_720_;
}
else
{
lean_inc(v_failedRapps_719_);
lean_inc(v_unsafeQueue_718_);
lean_inc(v_lastExpandedInIteration_716_);
lean_inc(v_addedInIteration_715_);
lean_inc(v_forwardRuleMatches_713_);
lean_inc(v_forwardState_712_);
lean_inc(v_mvars_711_);
lean_inc(v_normalizationState_710_);
lean_inc(v_preNormGoal_709_);
lean_inc(v_depth_706_);
lean_inc(v_origin_705_);
lean_inc(v_children_704_);
lean_inc(v_parent_703_);
lean_inc(v_id_702_);
lean_dec(v___x_701_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_768_;
goto v_resetjp_720_;
}
v___jp_689_:
{
lean_object* v___x_690_; lean_object* v_elimGoal_691_; lean_object* v___x_692_; lean_object* v_parent_693_; lean_object* v___x_694_; lean_object* v___x_695_; 
v___x_690_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_691_ = lean_ctor_get(v___x_690_, 1);
lean_inc_ref(v_elimGoal_691_);
v___x_692_ = lean_apply_1(v_elimGoal_691_, v___x_688_);
v_parent_693_ = lean_ctor_get(v___x_692_, 1);
lean_inc(v_parent_693_);
lean_dec_ref(v___x_692_);
v___x_694_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_694_, 0, v_parent_693_);
v___x_695_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v___x_694_);
return v___x_695_;
}
v___jp_696_:
{
goto v___jp_689_;
}
v_resetjp_720_:
{
uint8_t v___x_723_; lean_object* v___x_725_; 
v___x_723_ = 2;
lean_inc_ref(v_children_704_);
if (v_isShared_722_ == 0)
{
v___x_725_ = v___x_721_;
goto v_reusejp_724_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_id_702_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v_parent_703_);
lean_ctor_set(v_reuseFailAlloc_767_, 2, v_children_704_);
lean_ctor_set(v_reuseFailAlloc_767_, 3, v_origin_705_);
lean_ctor_set(v_reuseFailAlloc_767_, 4, v_depth_706_);
lean_ctor_set(v_reuseFailAlloc_767_, 5, v_preNormGoal_709_);
lean_ctor_set(v_reuseFailAlloc_767_, 6, v_normalizationState_710_);
lean_ctor_set(v_reuseFailAlloc_767_, 7, v_mvars_711_);
lean_ctor_set(v_reuseFailAlloc_767_, 8, v_forwardState_712_);
lean_ctor_set(v_reuseFailAlloc_767_, 9, v_forwardRuleMatches_713_);
lean_ctor_set(v_reuseFailAlloc_767_, 10, v_addedInIteration_715_);
lean_ctor_set(v_reuseFailAlloc_767_, 11, v_lastExpandedInIteration_716_);
lean_ctor_set(v_reuseFailAlloc_767_, 12, v_unsafeQueue_718_);
lean_ctor_set(v_reuseFailAlloc_767_, 13, v_failedRapps_719_);
lean_ctor_set_uint8(v_reuseFailAlloc_767_, sizeof(void*)*14 + 9, v_isIrrelevant_707_);
lean_ctor_set_uint8(v_reuseFailAlloc_767_, sizeof(void*)*14 + 10, v_isForcedUnprovable_708_);
lean_ctor_set_float(v_reuseFailAlloc_767_, sizeof(void*)*14, v_successProbability_714_);
lean_ctor_set_uint8(v_reuseFailAlloc_767_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_717_);
v___x_725_ = v_reuseFailAlloc_767_;
goto v_reusejp_724_;
}
v_reusejp_724_:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v_id_728_; lean_object* v_parent_729_; lean_object* v_children_730_; lean_object* v_origin_731_; lean_object* v_depth_732_; uint8_t v_state_733_; uint8_t v_isForcedUnprovable_734_; lean_object* v_preNormGoal_735_; lean_object* v_normalizationState_736_; lean_object* v_mvars_737_; lean_object* v_forwardState_738_; lean_object* v_forwardRuleMatches_739_; double v_successProbability_740_; lean_object* v_addedInIteration_741_; lean_object* v_lastExpandedInIteration_742_; uint8_t v_unsafeRulesSelected_743_; lean_object* v_unsafeQueue_744_; lean_object* v_failedRapps_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_766_; 
lean_ctor_set_uint8(v___x_725_, sizeof(void*)*14 + 8, v___x_723_);
lean_inc(v_introGoal_699_);
v___x_726_ = lean_apply_1(v_introGoal_699_, v___x_725_);
lean_inc_ref(v_elimGoal_700_);
v___x_727_ = lean_apply_1(v_elimGoal_700_, v___x_726_);
v_id_728_ = lean_ctor_get(v___x_727_, 0);
v_parent_729_ = lean_ctor_get(v___x_727_, 1);
v_children_730_ = lean_ctor_get(v___x_727_, 2);
v_origin_731_ = lean_ctor_get(v___x_727_, 3);
v_depth_732_ = lean_ctor_get(v___x_727_, 4);
v_state_733_ = lean_ctor_get_uint8(v___x_727_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_734_ = lean_ctor_get_uint8(v___x_727_, sizeof(void*)*14 + 10);
v_preNormGoal_735_ = lean_ctor_get(v___x_727_, 5);
v_normalizationState_736_ = lean_ctor_get(v___x_727_, 6);
v_mvars_737_ = lean_ctor_get(v___x_727_, 7);
v_forwardState_738_ = lean_ctor_get(v___x_727_, 8);
v_forwardRuleMatches_739_ = lean_ctor_get(v___x_727_, 9);
v_successProbability_740_ = lean_ctor_get_float(v___x_727_, sizeof(void*)*14);
v_addedInIteration_741_ = lean_ctor_get(v___x_727_, 10);
v_lastExpandedInIteration_742_ = lean_ctor_get(v___x_727_, 11);
v_unsafeRulesSelected_743_ = lean_ctor_get_uint8(v___x_727_, sizeof(void*)*14 + 11);
v_unsafeQueue_744_ = lean_ctor_get(v___x_727_, 12);
v_failedRapps_745_ = lean_ctor_get(v___x_727_, 13);
v_isSharedCheck_766_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_766_ == 0)
{
v___x_747_ = v___x_727_;
v_isShared_748_ = v_isSharedCheck_766_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_failedRapps_745_);
lean_inc(v_unsafeQueue_744_);
lean_inc(v_lastExpandedInIteration_742_);
lean_inc(v_addedInIteration_741_);
lean_inc(v_forwardRuleMatches_739_);
lean_inc(v_forwardState_738_);
lean_inc(v_mvars_737_);
lean_inc(v_normalizationState_736_);
lean_inc(v_preNormGoal_735_);
lean_inc(v_depth_732_);
lean_inc(v_origin_731_);
lean_inc(v_children_730_);
lean_inc(v_parent_729_);
lean_inc(v_id_728_);
lean_dec(v___x_727_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_766_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
uint8_t v___x_749_; lean_object* v___x_751_; 
v___x_749_ = 1;
if (v_isShared_748_ == 0)
{
v___x_751_ = v___x_747_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_id_728_);
lean_ctor_set(v_reuseFailAlloc_765_, 1, v_parent_729_);
lean_ctor_set(v_reuseFailAlloc_765_, 2, v_children_730_);
lean_ctor_set(v_reuseFailAlloc_765_, 3, v_origin_731_);
lean_ctor_set(v_reuseFailAlloc_765_, 4, v_depth_732_);
lean_ctor_set(v_reuseFailAlloc_765_, 5, v_preNormGoal_735_);
lean_ctor_set(v_reuseFailAlloc_765_, 6, v_normalizationState_736_);
lean_ctor_set(v_reuseFailAlloc_765_, 7, v_mvars_737_);
lean_ctor_set(v_reuseFailAlloc_765_, 8, v_forwardState_738_);
lean_ctor_set(v_reuseFailAlloc_765_, 9, v_forwardRuleMatches_739_);
lean_ctor_set(v_reuseFailAlloc_765_, 10, v_addedInIteration_741_);
lean_ctor_set(v_reuseFailAlloc_765_, 11, v_lastExpandedInIteration_742_);
lean_ctor_set(v_reuseFailAlloc_765_, 12, v_unsafeQueue_744_);
lean_ctor_set(v_reuseFailAlloc_765_, 13, v_failedRapps_745_);
lean_ctor_set_uint8(v_reuseFailAlloc_765_, sizeof(void*)*14 + 8, v_state_733_);
lean_ctor_set_uint8(v_reuseFailAlloc_765_, sizeof(void*)*14 + 10, v_isForcedUnprovable_734_);
lean_ctor_set_float(v_reuseFailAlloc_765_, sizeof(void*)*14, v_successProbability_740_);
lean_ctor_set_uint8(v_reuseFailAlloc_765_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_743_);
v___x_751_ = v_reuseFailAlloc_765_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; uint8_t v___x_756_; 
lean_ctor_set_uint8(v___x_751_, sizeof(void*)*14 + 9, v___x_749_);
lean_inc(v_introGoal_699_);
v___x_752_ = lean_apply_1(v_introGoal_699_, v___x_751_);
v___x_753_ = lean_st_ref_set(v_gref_686_, v___x_752_);
v___x_754_ = lean_unsigned_to_nat(0u);
v___x_755_ = lean_array_get_size(v_children_704_);
v___x_756_ = lean_nat_dec_lt(v___x_754_, v___x_755_);
if (v___x_756_ == 0)
{
lean_dec_ref(v_children_704_);
goto v___jp_689_;
}
else
{
lean_object* v___x_757_; uint8_t v___x_758_; 
v___x_757_ = lean_box(0);
v___x_758_ = lean_nat_dec_le(v___x_755_, v___x_755_);
if (v___x_758_ == 0)
{
if (v___x_756_ == 0)
{
lean_dec_ref(v_children_704_);
goto v___jp_689_;
}
else
{
size_t v___x_759_; size_t v___x_760_; lean_object* v___x_761_; 
v___x_759_ = ((size_t)0ULL);
v___x_760_ = lean_usize_of_nat(v___x_755_);
v___x_761_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_704_, v___x_759_, v___x_760_, v___x_757_);
lean_dec_ref(v_children_704_);
v___y_697_ = v___x_761_;
goto v___jp_696_;
}
}
else
{
size_t v___x_762_; size_t v___x_763_; lean_object* v___x_764_; 
v___x_762_ = ((size_t)0ULL);
v___x_763_ = lean_usize_of_nat(v___x_755_);
v___x_764_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_704_, v___x_762_, v___x_763_, v___x_757_);
lean_dec_ref(v_children_704_);
v___y_697_ = v___x_764_;
goto v___jp_696_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markProvenByNormalization___boxed(lean_object* v_gref_769_, lean_object* v_a_770_){
_start:
{
lean_object* v_res_771_; 
v_res_771_ = lp_aesop_Aesop_GoalRef_markProvenByNormalization(v_gref_769_);
lean_dec(v_gref_769_);
return v_res_771_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markProven(lean_object* v_rref_772_){
_start:
{
lean_object* v___x_774_; lean_object* v___x_775_; 
v___x_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_774_, 0, v_rref_772_);
v___x_775_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__2(v___x_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RappRef_markProven___boxed(lean_object* v_rref_776_, lean_object* v_a_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_aesop_Aesop_RappRef_markProven(v_rref_776_);
return v_res_778_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0(uint8_t v_val_779_, uint8_t v___x_780_, lean_object* v_as_781_, size_t v_i_782_, size_t v_stop_783_){
_start:
{
uint8_t v___x_785_; 
v___x_785_ = lean_usize_dec_eq(v_i_782_, v_stop_783_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v_elimRapp_789_; lean_object* v___x_790_; uint8_t v_state_791_; uint8_t v___x_792_; uint8_t v_val_794_; uint8_t v___x_798_; 
v___x_786_ = lean_array_uget_borrowed(v_as_781_, v_i_782_);
v___x_787_ = lean_st_ref_get(v___x_786_);
v___x_788_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_789_ = lean_ctor_get(v___x_788_, 3);
lean_inc_ref(v_elimRapp_789_);
v___x_790_ = lean_apply_1(v_elimRapp_789_, v___x_787_);
v_state_791_ = lean_ctor_get_uint8(v___x_790_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_790_);
v___x_792_ = 1;
v___x_798_ = lp_aesop_Aesop_NodeState_isUnprovable(v_state_791_);
if (v___x_798_ == 0)
{
v_val_794_ = v_val_779_;
goto v___jp_793_;
}
else
{
v_val_794_ = v___x_780_;
goto v___jp_793_;
}
v___jp_793_:
{
if (v_val_794_ == 0)
{
size_t v___x_795_; size_t v___x_796_; 
v___x_795_ = ((size_t)1ULL);
v___x_796_ = lean_usize_add(v_i_782_, v___x_795_);
v_i_782_ = v___x_796_;
goto _start;
}
else
{
return v___x_792_;
}
}
}
else
{
uint8_t v___x_799_; 
v___x_799_ = 0;
return v___x_799_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0___boxed(lean_object* v_val_800_, lean_object* v___x_801_, lean_object* v_as_802_, lean_object* v_i_803_, lean_object* v_stop_804_, lean_object* v___y_805_){
_start:
{
uint8_t v_val_819__boxed_806_; uint8_t v___x_820__boxed_807_; size_t v_i_boxed_808_; size_t v_stop_boxed_809_; uint8_t v_res_810_; lean_object* v_r_811_; 
v_val_819__boxed_806_ = lean_unbox(v_val_800_);
v___x_820__boxed_807_ = lean_unbox(v___x_801_);
v_i_boxed_808_ = lean_unbox_usize(v_i_803_);
lean_dec(v_i_803_);
v_stop_boxed_809_ = lean_unbox_usize(v_stop_804_);
lean_dec(v_stop_804_);
v_res_810_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0(v_val_819__boxed_806_, v___x_820__boxed_807_, v_as_802_, v_i_boxed_808_, v_stop_boxed_809_);
lean_dec_ref(v_as_802_);
v_r_811_ = lean_box(v_res_810_);
return v_r_811_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_isUnprovableNoCache(lean_object* v_g_812_){
_start:
{
lean_object* v___x_814_; lean_object* v_elimGoal_815_; lean_object* v___x_816_; uint8_t v_isForcedUnprovable_817_; 
v___x_814_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_815_ = lean_ctor_get(v___x_814_, 1);
lean_inc_ref(v_elimGoal_815_);
lean_inc(v_g_812_);
v___x_816_ = lean_apply_1(v_elimGoal_815_, v_g_812_);
v_isForcedUnprovable_817_ = lean_ctor_get_uint8(v___x_816_, sizeof(void*)*14 + 10);
if (v_isForcedUnprovable_817_ == 0)
{
lean_object* v_children_818_; uint8_t v___x_819_; 
v_children_818_ = lean_ctor_get(v___x_816_, 2);
lean_inc_ref(v_children_818_);
lean_dec_ref(v___x_816_);
v___x_819_ = lp_aesop_Aesop_Goal_isExhausted(v_g_812_);
if (v___x_819_ == 0)
{
lean_dec_ref(v_children_818_);
return v___x_819_;
}
else
{
lean_object* v___x_820_; lean_object* v___x_821_; uint8_t v___x_822_; 
v___x_820_ = lean_unsigned_to_nat(0u);
v___x_821_ = lean_array_get_size(v_children_818_);
v___x_822_ = lean_nat_dec_lt(v___x_820_, v___x_821_);
if (v___x_822_ == 0)
{
lean_dec_ref(v_children_818_);
return v___x_819_;
}
else
{
if (v___x_822_ == 0)
{
lean_dec_ref(v_children_818_);
return v___x_819_;
}
else
{
size_t v___x_823_; size_t v___x_824_; uint8_t v___x_825_; 
v___x_823_ = ((size_t)0ULL);
v___x_824_ = lean_usize_of_nat(v___x_821_);
v___x_825_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Goal_isUnprovableNoCache_spec__0(v___x_819_, v_isForcedUnprovable_817_, v_children_818_, v___x_823_, v___x_824_);
lean_dec_ref(v_children_818_);
if (v___x_825_ == 0)
{
return v___x_819_;
}
else
{
return v_isForcedUnprovable_817_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_816_);
lean_dec(v_g_812_);
return v_isForcedUnprovable_817_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_isUnprovableNoCache___boxed(lean_object* v_g_826_, lean_object* v_a_827_){
_start:
{
uint8_t v_res_828_; lean_object* v_r_829_; 
v_res_828_ = lp_aesop_Aesop_Goal_isUnprovableNoCache(v_g_826_);
v_r_829_ = lean_box(v_res_828_);
return v_r_829_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0(lean_object* v_as_830_, size_t v_i_831_, size_t v_stop_832_){
_start:
{
uint8_t v___x_834_; 
v___x_834_ = lean_usize_dec_eq(v_i_831_, v_stop_832_);
if (v___x_834_ == 0)
{
lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v_elimMVarCluster_838_; lean_object* v___x_839_; uint8_t v_state_840_; uint8_t v___x_841_; 
v___x_835_ = lean_array_uget_borrowed(v_as_830_, v_i_831_);
v___x_836_ = lean_st_ref_get(v___x_835_);
v___x_837_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_838_ = lean_ctor_get(v___x_837_, 5);
lean_inc_ref(v_elimMVarCluster_838_);
v___x_839_ = lean_apply_1(v_elimMVarCluster_838_, v___x_836_);
v_state_840_ = lean_ctor_get_uint8(v___x_839_, sizeof(void*)*2 + 1);
lean_dec_ref(v___x_839_);
v___x_841_ = lp_aesop_Aesop_NodeState_isUnprovable(v_state_840_);
if (v___x_841_ == 0)
{
size_t v___x_842_; size_t v___x_843_; 
v___x_842_ = ((size_t)1ULL);
v___x_843_ = lean_usize_add(v_i_831_, v___x_842_);
v_i_831_ = v___x_843_;
goto _start;
}
else
{
return v___x_841_;
}
}
else
{
uint8_t v___x_845_; 
v___x_845_ = 0;
return v___x_845_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0___boxed(lean_object* v_as_846_, lean_object* v_i_847_, lean_object* v_stop_848_, lean_object* v___y_849_){
_start:
{
size_t v_i_boxed_850_; size_t v_stop_boxed_851_; uint8_t v_res_852_; lean_object* v_r_853_; 
v_i_boxed_850_ = lean_unbox_usize(v_i_847_);
lean_dec(v_i_847_);
v_stop_boxed_851_ = lean_unbox_usize(v_stop_848_);
lean_dec(v_stop_848_);
v_res_852_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0(v_as_846_, v_i_boxed_850_, v_stop_boxed_851_);
lean_dec_ref(v_as_846_);
v_r_853_ = lean_box(v_res_852_);
return v_r_853_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_isUnprovableNoCache(lean_object* v_r_854_){
_start:
{
lean_object* v___x_856_; lean_object* v_elimRapp_857_; lean_object* v___x_858_; lean_object* v_children_859_; lean_object* v___x_860_; lean_object* v___x_861_; uint8_t v___x_862_; 
v___x_856_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_857_ = lean_ctor_get(v___x_856_, 3);
lean_inc_ref(v_elimRapp_857_);
v___x_858_ = lean_apply_1(v_elimRapp_857_, v_r_854_);
v_children_859_ = lean_ctor_get(v___x_858_, 2);
lean_inc_ref(v_children_859_);
lean_dec_ref(v___x_858_);
v___x_860_ = lean_unsigned_to_nat(0u);
v___x_861_ = lean_array_get_size(v_children_859_);
v___x_862_ = lean_nat_dec_lt(v___x_860_, v___x_861_);
if (v___x_862_ == 0)
{
lean_dec_ref(v_children_859_);
return v___x_862_;
}
else
{
if (v___x_862_ == 0)
{
lean_dec_ref(v_children_859_);
return v___x_862_;
}
else
{
size_t v___x_863_; size_t v___x_864_; uint8_t v___x_865_; 
v___x_863_ = ((size_t)0ULL);
v___x_864_ = lean_usize_of_nat(v___x_861_);
v___x_865_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Rapp_isUnprovableNoCache_spec__0(v_children_859_, v___x_863_, v___x_864_);
lean_dec_ref(v_children_859_);
return v___x_865_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_isUnprovableNoCache___boxed(lean_object* v_r_866_, lean_object* v_a_867_){
_start:
{
uint8_t v_res_868_; lean_object* v_r_869_; 
v_res_868_ = lp_aesop_Aesop_Rapp_isUnprovableNoCache(v_r_866_);
v_r_869_ = lean_box(v_res_868_);
return v_r_869_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0(lean_object* v_as_870_, size_t v_i_871_, size_t v_stop_872_){
_start:
{
uint8_t v___x_874_; 
v___x_874_ = lean_usize_dec_eq(v_i_871_, v_stop_872_);
if (v___x_874_ == 0)
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v_elimGoal_878_; lean_object* v___x_879_; uint8_t v_state_880_; uint8_t v___x_881_; uint8_t v___x_882_; 
v___x_875_ = lean_array_uget_borrowed(v_as_870_, v_i_871_);
v___x_876_ = lean_st_ref_get(v___x_875_);
v___x_877_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_878_ = lean_ctor_get(v___x_877_, 1);
lean_inc_ref(v_elimGoal_878_);
v___x_879_ = lean_apply_1(v_elimGoal_878_, v___x_876_);
v_state_880_ = lean_ctor_get_uint8(v___x_879_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_879_);
v___x_881_ = 1;
v___x_882_ = lp_aesop_Aesop_GoalState_isUnprovable(v_state_880_);
if (v___x_882_ == 0)
{
return v___x_881_;
}
else
{
if (v___x_874_ == 0)
{
size_t v___x_883_; size_t v___x_884_; 
v___x_883_ = ((size_t)1ULL);
v___x_884_ = lean_usize_add(v_i_871_, v___x_883_);
v_i_871_ = v___x_884_;
goto _start;
}
else
{
return v___x_881_;
}
}
}
else
{
uint8_t v___x_886_; 
v___x_886_ = 0;
return v___x_886_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0___boxed(lean_object* v_as_887_, lean_object* v_i_888_, lean_object* v_stop_889_, lean_object* v___y_890_){
_start:
{
size_t v_i_boxed_891_; size_t v_stop_boxed_892_; uint8_t v_res_893_; lean_object* v_r_894_; 
v_i_boxed_891_ = lean_unbox_usize(v_i_888_);
lean_dec(v_i_888_);
v_stop_boxed_892_ = lean_unbox_usize(v_stop_889_);
lean_dec(v_stop_889_);
v_res_893_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0(v_as_887_, v_i_boxed_891_, v_stop_boxed_892_);
lean_dec_ref(v_as_887_);
v_r_894_ = lean_box(v_res_893_);
return v_r_894_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_isUnprovableNoCache(lean_object* v_c_895_){
_start:
{
lean_object* v___x_899_; lean_object* v_elimMVarCluster_900_; lean_object* v___x_901_; lean_object* v_goals_902_; lean_object* v___x_903_; lean_object* v___x_904_; uint8_t v___x_905_; 
v___x_899_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_900_ = lean_ctor_get(v___x_899_, 5);
lean_inc_ref(v_elimMVarCluster_900_);
v___x_901_ = lean_apply_1(v_elimMVarCluster_900_, v_c_895_);
v_goals_902_ = lean_ctor_get(v___x_901_, 1);
lean_inc_ref(v_goals_902_);
lean_dec_ref(v___x_901_);
v___x_903_ = lean_unsigned_to_nat(0u);
v___x_904_ = lean_array_get_size(v_goals_902_);
v___x_905_ = lean_nat_dec_lt(v___x_903_, v___x_904_);
if (v___x_905_ == 0)
{
lean_dec_ref(v_goals_902_);
goto v___jp_897_;
}
else
{
if (v___x_905_ == 0)
{
lean_dec_ref(v_goals_902_);
goto v___jp_897_;
}
else
{
size_t v___x_906_; size_t v___x_907_; uint8_t v___x_908_; 
v___x_906_ = ((size_t)0ULL);
v___x_907_ = lean_usize_of_nat(v___x_904_);
v___x_908_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_MVarCluster_isUnprovableNoCache_spec__0(v_goals_902_, v___x_906_, v___x_907_);
lean_dec_ref(v_goals_902_);
if (v___x_908_ == 0)
{
goto v___jp_897_;
}
else
{
uint8_t v___x_909_; 
v___x_909_ = 0;
return v___x_909_;
}
}
}
v___jp_897_:
{
uint8_t v___x_898_; 
v___x_898_ = 1;
return v___x_898_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_isUnprovableNoCache___boxed(lean_object* v_c_910_, lean_object* v_a_911_){
_start:
{
uint8_t v_res_912_; lean_object* v_r_913_; 
v_res_912_ = lp_aesop_Aesop_MVarCluster_isUnprovableNoCache(v_c_910_);
v_r_913_ = lean_box(v_res_912_);
return v_r_913_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0(lean_object* v_as_914_, size_t v_i_915_, size_t v_stop_916_, lean_object* v_b_917_){
_start:
{
uint8_t v___x_919_; 
v___x_919_ = lean_usize_dec_eq(v_i_915_, v_stop_916_);
if (v___x_919_ == 0)
{
lean_object* v___x_920_; lean_object* v___x_921_; size_t v___x_922_; size_t v___x_923_; 
v___x_920_ = lean_array_uget_borrowed(v_as_914_, v_i_915_);
lean_inc(v___x_920_);
v___x_921_ = lp_aesop_Aesop_MVarClusterRef_markSubtreeIrrelevant(v___x_920_);
v___x_922_ = ((size_t)1ULL);
v___x_923_ = lean_usize_add(v_i_915_, v___x_922_);
v_i_915_ = v___x_923_;
v_b_917_ = v___x_921_;
goto _start;
}
else
{
return v_b_917_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0___boxed(lean_object* v_as_925_, lean_object* v_i_926_, lean_object* v_stop_927_, lean_object* v_b_928_, lean_object* v___y_929_){
_start:
{
size_t v_i_boxed_930_; size_t v_stop_boxed_931_; lean_object* v_res_932_; 
v_i_boxed_930_ = lean_unbox_usize(v_i_926_);
lean_dec(v_i_926_);
v_stop_boxed_931_ = lean_unbox_usize(v_stop_927_);
lean_dec(v_stop_927_);
v_res_932_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0(v_as_925_, v_i_boxed_930_, v_stop_boxed_931_, v_b_928_);
lean_dec_ref(v_as_925_);
return v_res_932_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(lean_object* v_x_933_){
_start:
{
switch(lean_obj_tag(v_x_933_))
{
case 0:
{
lean_object* v_gref_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_1012_; 
v_gref_937_ = lean_ctor_get(v_x_933_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v_x_933_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_939_ = v_x_933_;
v_isShared_940_ = v_isSharedCheck_1012_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_gref_937_);
lean_dec(v_x_933_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_1012_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_941_; uint8_t v___x_942_; 
v___x_941_ = lean_st_ref_get(v_gref_937_);
lean_inc(v___x_941_);
v___x_942_ = lp_aesop_Aesop_Goal_isUnprovableNoCache(v___x_941_);
if (v___x_942_ == 0)
{
lean_object* v___x_943_; 
lean_dec(v___x_941_);
lean_del_object(v___x_939_);
lean_dec(v_gref_937_);
v___x_943_ = lean_box(0);
return v___x_943_;
}
else
{
lean_object* v___x_944_; lean_object* v_introGoal_945_; lean_object* v_elimGoal_946_; lean_object* v___x_947_; lean_object* v_id_948_; lean_object* v_parent_949_; lean_object* v_children_950_; lean_object* v_origin_951_; lean_object* v_depth_952_; uint8_t v_isIrrelevant_953_; uint8_t v_isForcedUnprovable_954_; lean_object* v_preNormGoal_955_; lean_object* v_normalizationState_956_; lean_object* v_mvars_957_; lean_object* v_forwardState_958_; lean_object* v_forwardRuleMatches_959_; double v_successProbability_960_; lean_object* v_addedInIteration_961_; lean_object* v_lastExpandedInIteration_962_; uint8_t v_unsafeRulesSelected_963_; lean_object* v_unsafeQueue_964_; lean_object* v_failedRapps_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_1011_; 
v___x_944_ = lp_aesop_Aesop_treeImpl;
v_introGoal_945_ = lean_ctor_get(v___x_944_, 0);
v_elimGoal_946_ = lean_ctor_get(v___x_944_, 1);
lean_inc_ref(v_elimGoal_946_);
v___x_947_ = lean_apply_1(v_elimGoal_946_, v___x_941_);
v_id_948_ = lean_ctor_get(v___x_947_, 0);
v_parent_949_ = lean_ctor_get(v___x_947_, 1);
v_children_950_ = lean_ctor_get(v___x_947_, 2);
v_origin_951_ = lean_ctor_get(v___x_947_, 3);
v_depth_952_ = lean_ctor_get(v___x_947_, 4);
v_isIrrelevant_953_ = lean_ctor_get_uint8(v___x_947_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_954_ = lean_ctor_get_uint8(v___x_947_, sizeof(void*)*14 + 10);
v_preNormGoal_955_ = lean_ctor_get(v___x_947_, 5);
v_normalizationState_956_ = lean_ctor_get(v___x_947_, 6);
v_mvars_957_ = lean_ctor_get(v___x_947_, 7);
v_forwardState_958_ = lean_ctor_get(v___x_947_, 8);
v_forwardRuleMatches_959_ = lean_ctor_get(v___x_947_, 9);
v_successProbability_960_ = lean_ctor_get_float(v___x_947_, sizeof(void*)*14);
v_addedInIteration_961_ = lean_ctor_get(v___x_947_, 10);
v_lastExpandedInIteration_962_ = lean_ctor_get(v___x_947_, 11);
v_unsafeRulesSelected_963_ = lean_ctor_get_uint8(v___x_947_, sizeof(void*)*14 + 11);
v_unsafeQueue_964_ = lean_ctor_get(v___x_947_, 12);
v_failedRapps_965_ = lean_ctor_get(v___x_947_, 13);
v_isSharedCheck_1011_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_1011_ == 0)
{
v___x_967_ = v___x_947_;
v_isShared_968_ = v_isSharedCheck_1011_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_failedRapps_965_);
lean_inc(v_unsafeQueue_964_);
lean_inc(v_lastExpandedInIteration_962_);
lean_inc(v_addedInIteration_961_);
lean_inc(v_forwardRuleMatches_959_);
lean_inc(v_forwardState_958_);
lean_inc(v_mvars_957_);
lean_inc(v_normalizationState_956_);
lean_inc(v_preNormGoal_955_);
lean_inc(v_depth_952_);
lean_inc(v_origin_951_);
lean_inc(v_children_950_);
lean_inc(v_parent_949_);
lean_inc(v_id_948_);
lean_dec(v___x_947_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_1011_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
uint8_t v___x_969_; lean_object* v___x_971_; 
v___x_969_ = 3;
if (v_isShared_968_ == 0)
{
v___x_971_ = v___x_967_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_1010_; 
v_reuseFailAlloc_1010_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1010_, 0, v_id_948_);
lean_ctor_set(v_reuseFailAlloc_1010_, 1, v_parent_949_);
lean_ctor_set(v_reuseFailAlloc_1010_, 2, v_children_950_);
lean_ctor_set(v_reuseFailAlloc_1010_, 3, v_origin_951_);
lean_ctor_set(v_reuseFailAlloc_1010_, 4, v_depth_952_);
lean_ctor_set(v_reuseFailAlloc_1010_, 5, v_preNormGoal_955_);
lean_ctor_set(v_reuseFailAlloc_1010_, 6, v_normalizationState_956_);
lean_ctor_set(v_reuseFailAlloc_1010_, 7, v_mvars_957_);
lean_ctor_set(v_reuseFailAlloc_1010_, 8, v_forwardState_958_);
lean_ctor_set(v_reuseFailAlloc_1010_, 9, v_forwardRuleMatches_959_);
lean_ctor_set(v_reuseFailAlloc_1010_, 10, v_addedInIteration_961_);
lean_ctor_set(v_reuseFailAlloc_1010_, 11, v_lastExpandedInIteration_962_);
lean_ctor_set(v_reuseFailAlloc_1010_, 12, v_unsafeQueue_964_);
lean_ctor_set(v_reuseFailAlloc_1010_, 13, v_failedRapps_965_);
lean_ctor_set_uint8(v_reuseFailAlloc_1010_, sizeof(void*)*14 + 9, v_isIrrelevant_953_);
lean_ctor_set_uint8(v_reuseFailAlloc_1010_, sizeof(void*)*14 + 10, v_isForcedUnprovable_954_);
lean_ctor_set_float(v_reuseFailAlloc_1010_, sizeof(void*)*14, v_successProbability_960_);
lean_ctor_set_uint8(v_reuseFailAlloc_1010_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_963_);
v___x_971_ = v_reuseFailAlloc_1010_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v_id_974_; lean_object* v_parent_975_; lean_object* v_children_976_; lean_object* v_origin_977_; lean_object* v_depth_978_; uint8_t v_state_979_; uint8_t v_isForcedUnprovable_980_; lean_object* v_preNormGoal_981_; lean_object* v_normalizationState_982_; lean_object* v_mvars_983_; lean_object* v_forwardState_984_; lean_object* v_forwardRuleMatches_985_; double v_successProbability_986_; lean_object* v_addedInIteration_987_; lean_object* v_lastExpandedInIteration_988_; uint8_t v_unsafeRulesSelected_989_; lean_object* v_unsafeQueue_990_; lean_object* v_failedRapps_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_1009_; 
lean_ctor_set_uint8(v___x_971_, sizeof(void*)*14 + 8, v___x_969_);
lean_inc(v_introGoal_945_);
v___x_972_ = lean_apply_1(v_introGoal_945_, v___x_971_);
lean_inc_ref(v_elimGoal_946_);
v___x_973_ = lean_apply_1(v_elimGoal_946_, v___x_972_);
v_id_974_ = lean_ctor_get(v___x_973_, 0);
v_parent_975_ = lean_ctor_get(v___x_973_, 1);
v_children_976_ = lean_ctor_get(v___x_973_, 2);
v_origin_977_ = lean_ctor_get(v___x_973_, 3);
v_depth_978_ = lean_ctor_get(v___x_973_, 4);
v_state_979_ = lean_ctor_get_uint8(v___x_973_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_980_ = lean_ctor_get_uint8(v___x_973_, sizeof(void*)*14 + 10);
v_preNormGoal_981_ = lean_ctor_get(v___x_973_, 5);
v_normalizationState_982_ = lean_ctor_get(v___x_973_, 6);
v_mvars_983_ = lean_ctor_get(v___x_973_, 7);
v_forwardState_984_ = lean_ctor_get(v___x_973_, 8);
v_forwardRuleMatches_985_ = lean_ctor_get(v___x_973_, 9);
v_successProbability_986_ = lean_ctor_get_float(v___x_973_, sizeof(void*)*14);
v_addedInIteration_987_ = lean_ctor_get(v___x_973_, 10);
v_lastExpandedInIteration_988_ = lean_ctor_get(v___x_973_, 11);
v_unsafeRulesSelected_989_ = lean_ctor_get_uint8(v___x_973_, sizeof(void*)*14 + 11);
v_unsafeQueue_990_ = lean_ctor_get(v___x_973_, 12);
v_failedRapps_991_ = lean_ctor_get(v___x_973_, 13);
v_isSharedCheck_1009_ = !lean_is_exclusive(v___x_973_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_993_ = v___x_973_;
v_isShared_994_ = v_isSharedCheck_1009_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_failedRapps_991_);
lean_inc(v_unsafeQueue_990_);
lean_inc(v_lastExpandedInIteration_988_);
lean_inc(v_addedInIteration_987_);
lean_inc(v_forwardRuleMatches_985_);
lean_inc(v_forwardState_984_);
lean_inc(v_mvars_983_);
lean_inc(v_normalizationState_982_);
lean_inc(v_preNormGoal_981_);
lean_inc(v_depth_978_);
lean_inc(v_origin_977_);
lean_inc(v_children_976_);
lean_inc(v_parent_975_);
lean_inc(v_id_974_);
lean_dec(v___x_973_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_1009_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_996_; 
if (v_isShared_994_ == 0)
{
v___x_996_ = v___x_993_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v_id_974_);
lean_ctor_set(v_reuseFailAlloc_1008_, 1, v_parent_975_);
lean_ctor_set(v_reuseFailAlloc_1008_, 2, v_children_976_);
lean_ctor_set(v_reuseFailAlloc_1008_, 3, v_origin_977_);
lean_ctor_set(v_reuseFailAlloc_1008_, 4, v_depth_978_);
lean_ctor_set(v_reuseFailAlloc_1008_, 5, v_preNormGoal_981_);
lean_ctor_set(v_reuseFailAlloc_1008_, 6, v_normalizationState_982_);
lean_ctor_set(v_reuseFailAlloc_1008_, 7, v_mvars_983_);
lean_ctor_set(v_reuseFailAlloc_1008_, 8, v_forwardState_984_);
lean_ctor_set(v_reuseFailAlloc_1008_, 9, v_forwardRuleMatches_985_);
lean_ctor_set(v_reuseFailAlloc_1008_, 10, v_addedInIteration_987_);
lean_ctor_set(v_reuseFailAlloc_1008_, 11, v_lastExpandedInIteration_988_);
lean_ctor_set(v_reuseFailAlloc_1008_, 12, v_unsafeQueue_990_);
lean_ctor_set(v_reuseFailAlloc_1008_, 13, v_failedRapps_991_);
lean_ctor_set_uint8(v_reuseFailAlloc_1008_, sizeof(void*)*14 + 8, v_state_979_);
lean_ctor_set_uint8(v_reuseFailAlloc_1008_, sizeof(void*)*14 + 10, v_isForcedUnprovable_980_);
lean_ctor_set_float(v_reuseFailAlloc_1008_, sizeof(void*)*14, v_successProbability_986_);
lean_ctor_set_uint8(v_reuseFailAlloc_1008_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_989_);
v___x_996_ = v_reuseFailAlloc_1008_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v_elimGoal_1000_; lean_object* v___x_1001_; lean_object* v_parent_1002_; lean_object* v___x_1004_; 
lean_ctor_set_uint8(v___x_996_, sizeof(void*)*14 + 9, v___x_942_);
lean_inc(v_introGoal_945_);
v___x_997_ = lean_apply_1(v_introGoal_945_, v___x_996_);
v___x_998_ = lean_st_ref_set(v_gref_937_, v___x_997_);
v___x_999_ = lean_st_ref_get(v_gref_937_);
lean_dec(v_gref_937_);
v_elimGoal_1000_ = lean_ctor_get(v___x_944_, 1);
lean_inc_ref(v_elimGoal_1000_);
v___x_1001_ = lean_apply_1(v_elimGoal_1000_, v___x_999_);
v_parent_1002_ = lean_ctor_get(v___x_1001_, 1);
lean_inc(v_parent_1002_);
lean_dec_ref(v___x_1001_);
if (v_isShared_940_ == 0)
{
lean_ctor_set_tag(v___x_939_, 2);
lean_ctor_set(v___x_939_, 0, v_parent_1002_);
v___x_1004_ = v___x_939_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v_parent_1002_);
v___x_1004_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
lean_object* v___x_1005_; lean_object* v___x_1006_; 
v___x_1005_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v___x_1004_);
v___x_1006_ = lean_box(0);
return v___x_1006_;
}
}
}
}
}
}
}
}
case 1:
{
lean_object* v_rref_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1088_; 
v_rref_1013_ = lean_ctor_get(v_x_933_, 0);
v_isSharedCheck_1088_ = !lean_is_exclusive(v_x_933_);
if (v_isSharedCheck_1088_ == 0)
{
v___x_1015_ = v_x_933_;
v_isShared_1016_ = v_isSharedCheck_1088_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_rref_1013_);
lean_dec(v_x_933_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1088_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
lean_object* v___y_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v_introRapp_1032_; lean_object* v_elimRapp_1033_; lean_object* v___x_1034_; lean_object* v_id_1035_; lean_object* v_parent_1036_; lean_object* v_children_1037_; uint8_t v_isIrrelevant_1038_; lean_object* v_appliedRule_1039_; lean_object* v_scriptSteps_x3f_1040_; lean_object* v_originalSubgoals_1041_; double v_successProbability_1042_; lean_object* v_metaState_1043_; lean_object* v_introducedMVars_1044_; lean_object* v_assignedMVars_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1087_; 
v___x_1030_ = lean_st_ref_get(v_rref_1013_);
v___x_1031_ = lp_aesop_Aesop_treeImpl;
v_introRapp_1032_ = lean_ctor_get(v___x_1031_, 2);
v_elimRapp_1033_ = lean_ctor_get(v___x_1031_, 3);
lean_inc_ref(v_elimRapp_1033_);
v___x_1034_ = lean_apply_1(v_elimRapp_1033_, v___x_1030_);
v_id_1035_ = lean_ctor_get(v___x_1034_, 0);
v_parent_1036_ = lean_ctor_get(v___x_1034_, 1);
v_children_1037_ = lean_ctor_get(v___x_1034_, 2);
v_isIrrelevant_1038_ = lean_ctor_get_uint8(v___x_1034_, sizeof(void*)*9 + 9);
v_appliedRule_1039_ = lean_ctor_get(v___x_1034_, 3);
v_scriptSteps_x3f_1040_ = lean_ctor_get(v___x_1034_, 4);
v_originalSubgoals_1041_ = lean_ctor_get(v___x_1034_, 5);
v_successProbability_1042_ = lean_ctor_get_float(v___x_1034_, sizeof(void*)*9);
v_metaState_1043_ = lean_ctor_get(v___x_1034_, 6);
v_introducedMVars_1044_ = lean_ctor_get(v___x_1034_, 7);
v_assignedMVars_1045_ = lean_ctor_get(v___x_1034_, 8);
v_isSharedCheck_1087_ = !lean_is_exclusive(v___x_1034_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1047_ = v___x_1034_;
v_isShared_1048_ = v_isSharedCheck_1087_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_assignedMVars_1045_);
lean_inc(v_introducedMVars_1044_);
lean_inc(v_metaState_1043_);
lean_inc(v_originalSubgoals_1041_);
lean_inc(v_scriptSteps_x3f_1040_);
lean_inc(v_appliedRule_1039_);
lean_inc(v_children_1037_);
lean_inc(v_parent_1036_);
lean_inc(v_id_1035_);
lean_dec(v___x_1034_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1087_;
goto v_resetjp_1046_;
}
v___jp_1017_:
{
lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v_elimRapp_1020_; lean_object* v___x_1021_; lean_object* v_parent_1022_; lean_object* v___x_1024_; 
v___x_1018_ = lean_st_ref_get(v_rref_1013_);
lean_dec(v_rref_1013_);
v___x_1019_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_1020_ = lean_ctor_get(v___x_1019_, 3);
lean_inc_ref(v_elimRapp_1020_);
v___x_1021_ = lean_apply_1(v_elimRapp_1020_, v___x_1018_);
v_parent_1022_ = lean_ctor_get(v___x_1021_, 1);
lean_inc(v_parent_1022_);
lean_dec_ref(v___x_1021_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set_tag(v___x_1015_, 0);
lean_ctor_set(v___x_1015_, 0, v_parent_1022_);
v___x_1024_ = v___x_1015_;
goto v_reusejp_1023_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1027_, 0, v_parent_1022_);
v___x_1024_ = v_reuseFailAlloc_1027_;
goto v_reusejp_1023_;
}
v_reusejp_1023_:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1025_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v___x_1024_);
v___x_1026_ = lean_box(0);
return v___x_1026_;
}
}
v___jp_1028_:
{
goto v___jp_1017_;
}
v_resetjp_1046_:
{
uint8_t v___x_1049_; lean_object* v___x_1051_; 
v___x_1049_ = 2;
lean_inc_ref(v_children_1037_);
if (v_isShared_1048_ == 0)
{
v___x_1051_ = v___x_1047_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v_id_1035_);
lean_ctor_set(v_reuseFailAlloc_1086_, 1, v_parent_1036_);
lean_ctor_set(v_reuseFailAlloc_1086_, 2, v_children_1037_);
lean_ctor_set(v_reuseFailAlloc_1086_, 3, v_appliedRule_1039_);
lean_ctor_set(v_reuseFailAlloc_1086_, 4, v_scriptSteps_x3f_1040_);
lean_ctor_set(v_reuseFailAlloc_1086_, 5, v_originalSubgoals_1041_);
lean_ctor_set(v_reuseFailAlloc_1086_, 6, v_metaState_1043_);
lean_ctor_set(v_reuseFailAlloc_1086_, 7, v_introducedMVars_1044_);
lean_ctor_set(v_reuseFailAlloc_1086_, 8, v_assignedMVars_1045_);
lean_ctor_set_uint8(v_reuseFailAlloc_1086_, sizeof(void*)*9 + 9, v_isIrrelevant_1038_);
lean_ctor_set_float(v_reuseFailAlloc_1086_, sizeof(void*)*9, v_successProbability_1042_);
v___x_1051_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v_id_1054_; lean_object* v_parent_1055_; lean_object* v_children_1056_; uint8_t v_state_1057_; lean_object* v_appliedRule_1058_; lean_object* v_scriptSteps_x3f_1059_; lean_object* v_originalSubgoals_1060_; double v_successProbability_1061_; lean_object* v_metaState_1062_; lean_object* v_introducedMVars_1063_; lean_object* v_assignedMVars_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1085_; 
lean_ctor_set_uint8(v___x_1051_, sizeof(void*)*9 + 8, v___x_1049_);
lean_inc(v_introRapp_1032_);
v___x_1052_ = lean_apply_1(v_introRapp_1032_, v___x_1051_);
lean_inc_ref(v_elimRapp_1033_);
v___x_1053_ = lean_apply_1(v_elimRapp_1033_, v___x_1052_);
v_id_1054_ = lean_ctor_get(v___x_1053_, 0);
v_parent_1055_ = lean_ctor_get(v___x_1053_, 1);
v_children_1056_ = lean_ctor_get(v___x_1053_, 2);
v_state_1057_ = lean_ctor_get_uint8(v___x_1053_, sizeof(void*)*9 + 8);
v_appliedRule_1058_ = lean_ctor_get(v___x_1053_, 3);
v_scriptSteps_x3f_1059_ = lean_ctor_get(v___x_1053_, 4);
v_originalSubgoals_1060_ = lean_ctor_get(v___x_1053_, 5);
v_successProbability_1061_ = lean_ctor_get_float(v___x_1053_, sizeof(void*)*9);
v_metaState_1062_ = lean_ctor_get(v___x_1053_, 6);
v_introducedMVars_1063_ = lean_ctor_get(v___x_1053_, 7);
v_assignedMVars_1064_ = lean_ctor_get(v___x_1053_, 8);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1053_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1066_ = v___x_1053_;
v_isShared_1067_ = v_isSharedCheck_1085_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_assignedMVars_1064_);
lean_inc(v_introducedMVars_1063_);
lean_inc(v_metaState_1062_);
lean_inc(v_originalSubgoals_1060_);
lean_inc(v_scriptSteps_x3f_1059_);
lean_inc(v_appliedRule_1058_);
lean_inc(v_children_1056_);
lean_inc(v_parent_1055_);
lean_inc(v_id_1054_);
lean_dec(v___x_1053_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1085_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
uint8_t v___x_1068_; lean_object* v___x_1070_; 
v___x_1068_ = 1;
if (v_isShared_1067_ == 0)
{
v___x_1070_ = v___x_1066_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(0, 9, 10);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_id_1054_);
lean_ctor_set(v_reuseFailAlloc_1084_, 1, v_parent_1055_);
lean_ctor_set(v_reuseFailAlloc_1084_, 2, v_children_1056_);
lean_ctor_set(v_reuseFailAlloc_1084_, 3, v_appliedRule_1058_);
lean_ctor_set(v_reuseFailAlloc_1084_, 4, v_scriptSteps_x3f_1059_);
lean_ctor_set(v_reuseFailAlloc_1084_, 5, v_originalSubgoals_1060_);
lean_ctor_set(v_reuseFailAlloc_1084_, 6, v_metaState_1062_);
lean_ctor_set(v_reuseFailAlloc_1084_, 7, v_introducedMVars_1063_);
lean_ctor_set(v_reuseFailAlloc_1084_, 8, v_assignedMVars_1064_);
lean_ctor_set_uint8(v_reuseFailAlloc_1084_, sizeof(void*)*9 + 8, v_state_1057_);
lean_ctor_set_float(v_reuseFailAlloc_1084_, sizeof(void*)*9, v_successProbability_1061_);
v___x_1070_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; uint8_t v___x_1075_; 
lean_ctor_set_uint8(v___x_1070_, sizeof(void*)*9 + 9, v___x_1068_);
lean_inc(v_introRapp_1032_);
v___x_1071_ = lean_apply_1(v_introRapp_1032_, v___x_1070_);
v___x_1072_ = lean_st_ref_set(v_rref_1013_, v___x_1071_);
v___x_1073_ = lean_unsigned_to_nat(0u);
v___x_1074_ = lean_array_get_size(v_children_1037_);
v___x_1075_ = lean_nat_dec_lt(v___x_1073_, v___x_1074_);
if (v___x_1075_ == 0)
{
lean_dec_ref(v_children_1037_);
goto v___jp_1017_;
}
else
{
lean_object* v___x_1076_; uint8_t v___x_1077_; 
v___x_1076_ = lean_box(0);
v___x_1077_ = lean_nat_dec_le(v___x_1074_, v___x_1074_);
if (v___x_1077_ == 0)
{
if (v___x_1075_ == 0)
{
lean_dec_ref(v_children_1037_);
goto v___jp_1017_;
}
else
{
size_t v___x_1078_; size_t v___x_1079_; lean_object* v___x_1080_; 
v___x_1078_ = ((size_t)0ULL);
v___x_1079_ = lean_usize_of_nat(v___x_1074_);
v___x_1080_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0(v_children_1037_, v___x_1078_, v___x_1079_, v___x_1076_);
lean_dec_ref(v_children_1037_);
v___y_1029_ = v___x_1080_;
goto v___jp_1028_;
}
}
else
{
size_t v___x_1081_; size_t v___x_1082_; lean_object* v___x_1083_; 
v___x_1081_ = ((size_t)0ULL);
v___x_1082_ = lean_usize_of_nat(v___x_1074_);
v___x_1083_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__0(v_children_1037_, v___x_1081_, v___x_1082_, v___x_1076_);
lean_dec_ref(v_children_1037_);
v___y_1029_ = v___x_1083_;
goto v___jp_1028_;
}
}
}
}
}
}
}
}
default: 
{
lean_object* v_cref_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1134_; 
v_cref_1089_ = lean_ctor_get(v_x_933_, 0);
v_isSharedCheck_1134_ = !lean_is_exclusive(v_x_933_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1091_ = v_x_933_;
v_isShared_1092_ = v_isSharedCheck_1134_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_cref_1089_);
lean_dec(v_x_933_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1134_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1093_; uint8_t v___x_1094_; 
v___x_1093_ = lean_st_ref_get(v_cref_1089_);
lean_inc(v___x_1093_);
v___x_1094_ = lp_aesop_Aesop_MVarCluster_isUnprovableNoCache(v___x_1093_);
if (v___x_1094_ == 0)
{
lean_object* v___x_1095_; 
lean_dec(v___x_1093_);
lean_del_object(v___x_1091_);
lean_dec(v_cref_1089_);
v___x_1095_ = lean_box(0);
return v___x_1095_;
}
else
{
lean_object* v___x_1096_; lean_object* v_introMVarCluster_1097_; lean_object* v_elimMVarCluster_1098_; lean_object* v___x_1099_; lean_object* v_parent_x3f_1100_; lean_object* v_goals_1101_; uint8_t v_isIrrelevant_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1133_; 
v___x_1096_ = lp_aesop_Aesop_treeImpl;
v_introMVarCluster_1097_ = lean_ctor_get(v___x_1096_, 4);
v_elimMVarCluster_1098_ = lean_ctor_get(v___x_1096_, 5);
lean_inc_ref(v_elimMVarCluster_1098_);
v___x_1099_ = lean_apply_1(v_elimMVarCluster_1098_, v___x_1093_);
v_parent_x3f_1100_ = lean_ctor_get(v___x_1099_, 0);
v_goals_1101_ = lean_ctor_get(v___x_1099_, 1);
v_isIrrelevant_1102_ = lean_ctor_get_uint8(v___x_1099_, sizeof(void*)*2);
v_isSharedCheck_1133_ = !lean_is_exclusive(v___x_1099_);
if (v_isSharedCheck_1133_ == 0)
{
v___x_1104_ = v___x_1099_;
v_isShared_1105_ = v_isSharedCheck_1133_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_goals_1101_);
lean_inc(v_parent_x3f_1100_);
lean_dec(v___x_1099_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1133_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
uint8_t v___x_1106_; lean_object* v___x_1108_; 
v___x_1106_ = 2;
if (v_isShared_1105_ == 0)
{
v___x_1108_ = v___x_1104_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v_parent_x3f_1100_);
lean_ctor_set(v_reuseFailAlloc_1132_, 1, v_goals_1101_);
lean_ctor_set_uint8(v_reuseFailAlloc_1132_, sizeof(void*)*2, v_isIrrelevant_1102_);
v___x_1108_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v_parent_x3f_1111_; lean_object* v_goals_1112_; uint8_t v_state_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1131_; 
lean_ctor_set_uint8(v___x_1108_, sizeof(void*)*2 + 1, v___x_1106_);
lean_inc(v_introMVarCluster_1097_);
v___x_1109_ = lean_apply_1(v_introMVarCluster_1097_, v___x_1108_);
lean_inc_ref(v_elimMVarCluster_1098_);
v___x_1110_ = lean_apply_1(v_elimMVarCluster_1098_, v___x_1109_);
v_parent_x3f_1111_ = lean_ctor_get(v___x_1110_, 0);
v_goals_1112_ = lean_ctor_get(v___x_1110_, 1);
v_state_1113_ = lean_ctor_get_uint8(v___x_1110_, sizeof(void*)*2 + 1);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1110_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1115_ = v___x_1110_;
v_isShared_1116_ = v_isSharedCheck_1131_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_goals_1112_);
lean_inc(v_parent_x3f_1111_);
lean_dec(v___x_1110_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1131_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1118_; 
if (v_isShared_1116_ == 0)
{
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(0, 2, 2);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_parent_x3f_1111_);
lean_ctor_set(v_reuseFailAlloc_1130_, 1, v_goals_1112_);
lean_ctor_set_uint8(v_reuseFailAlloc_1130_, sizeof(void*)*2 + 1, v_state_1113_);
v___x_1118_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v_elimMVarCluster_1122_; lean_object* v___x_1123_; lean_object* v_parent_x3f_1124_; 
lean_ctor_set_uint8(v___x_1118_, sizeof(void*)*2, v___x_1094_);
lean_inc(v_introMVarCluster_1097_);
v___x_1119_ = lean_apply_1(v_introMVarCluster_1097_, v___x_1118_);
v___x_1120_ = lean_st_ref_set(v_cref_1089_, v___x_1119_);
v___x_1121_ = lean_st_ref_get(v_cref_1089_);
lean_dec(v_cref_1089_);
v_elimMVarCluster_1122_ = lean_ctor_get(v___x_1096_, 5);
lean_inc_ref(v_elimMVarCluster_1122_);
v___x_1123_ = lean_apply_1(v_elimMVarCluster_1122_, v___x_1121_);
v_parent_x3f_1124_ = lean_ctor_get(v___x_1123_, 0);
lean_inc(v_parent_x3f_1124_);
lean_dec_ref(v___x_1123_);
if (lean_obj_tag(v_parent_x3f_1124_) == 1)
{
lean_object* v_val_1125_; lean_object* v___x_1127_; 
v_val_1125_ = lean_ctor_get(v_parent_x3f_1124_, 0);
lean_inc(v_val_1125_);
lean_dec_ref_known(v_parent_x3f_1124_, 1);
if (v_isShared_1092_ == 0)
{
lean_ctor_set_tag(v___x_1091_, 1);
lean_ctor_set(v___x_1091_, 0, v_val_1125_);
v___x_1127_ = v___x_1091_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_val_1125_);
v___x_1127_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
lean_object* v___x_1128_; 
v___x_1128_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v___x_1127_);
goto v___jp_935_;
}
}
else
{
lean_dec(v_parent_x3f_1124_);
lean_del_object(v___x_1091_);
goto v___jp_935_;
}
}
}
}
}
}
}
}
}
v___jp_935_:
{
lean_object* v___x_936_; 
v___x_936_ = lean_box(0);
return v___x_936_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1___boxed(lean_object* v_x_1135_, lean_object* v___y_1136_){
_start:
{
lean_object* v_res_1137_; 
v_res_1137_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v_x_1135_);
return v_res_1137_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markUnprovableCore(lean_object* v_a_1138_){
_start:
{
lean_object* v___x_1140_; 
v___x_1140_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v_a_1138_);
return v___x_1140_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_State_0__Aesop_markUnprovableCore___boxed(lean_object* v_a_1141_, lean_object* v_a_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_aesop___private_Aesop_Tree_State_0__Aesop_markUnprovableCore(v_a_1141_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markUnprovable(lean_object* v_gref_1144_){
_start:
{
lean_object* v___x_1146_; lean_object* v___y_1155_; lean_object* v___x_1156_; lean_object* v_introGoal_1157_; lean_object* v_elimGoal_1158_; lean_object* v___x_1159_; lean_object* v_id_1160_; lean_object* v_parent_1161_; lean_object* v_children_1162_; lean_object* v_origin_1163_; lean_object* v_depth_1164_; uint8_t v_isIrrelevant_1165_; uint8_t v_isForcedUnprovable_1166_; lean_object* v_preNormGoal_1167_; lean_object* v_normalizationState_1168_; lean_object* v_mvars_1169_; lean_object* v_forwardState_1170_; lean_object* v_forwardRuleMatches_1171_; double v_successProbability_1172_; lean_object* v_addedInIteration_1173_; lean_object* v_lastExpandedInIteration_1174_; uint8_t v_unsafeRulesSelected_1175_; lean_object* v_unsafeQueue_1176_; lean_object* v_failedRapps_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1226_; 
v___x_1146_ = lean_st_ref_get(v_gref_1144_);
v___x_1156_ = lp_aesop_Aesop_treeImpl;
v_introGoal_1157_ = lean_ctor_get(v___x_1156_, 0);
v_elimGoal_1158_ = lean_ctor_get(v___x_1156_, 1);
lean_inc_ref(v_elimGoal_1158_);
lean_inc(v___x_1146_);
v___x_1159_ = lean_apply_1(v_elimGoal_1158_, v___x_1146_);
v_id_1160_ = lean_ctor_get(v___x_1159_, 0);
v_parent_1161_ = lean_ctor_get(v___x_1159_, 1);
v_children_1162_ = lean_ctor_get(v___x_1159_, 2);
v_origin_1163_ = lean_ctor_get(v___x_1159_, 3);
v_depth_1164_ = lean_ctor_get(v___x_1159_, 4);
v_isIrrelevant_1165_ = lean_ctor_get_uint8(v___x_1159_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_1166_ = lean_ctor_get_uint8(v___x_1159_, sizeof(void*)*14 + 10);
v_preNormGoal_1167_ = lean_ctor_get(v___x_1159_, 5);
v_normalizationState_1168_ = lean_ctor_get(v___x_1159_, 6);
v_mvars_1169_ = lean_ctor_get(v___x_1159_, 7);
v_forwardState_1170_ = lean_ctor_get(v___x_1159_, 8);
v_forwardRuleMatches_1171_ = lean_ctor_get(v___x_1159_, 9);
v_successProbability_1172_ = lean_ctor_get_float(v___x_1159_, sizeof(void*)*14);
v_addedInIteration_1173_ = lean_ctor_get(v___x_1159_, 10);
v_lastExpandedInIteration_1174_ = lean_ctor_get(v___x_1159_, 11);
v_unsafeRulesSelected_1175_ = lean_ctor_get_uint8(v___x_1159_, sizeof(void*)*14 + 11);
v_unsafeQueue_1176_ = lean_ctor_get(v___x_1159_, 12);
v_failedRapps_1177_ = lean_ctor_get(v___x_1159_, 13);
v_isSharedCheck_1226_ = !lean_is_exclusive(v___x_1159_);
if (v_isSharedCheck_1226_ == 0)
{
v___x_1179_ = v___x_1159_;
v_isShared_1180_ = v_isSharedCheck_1226_;
goto v_resetjp_1178_;
}
else
{
lean_inc(v_failedRapps_1177_);
lean_inc(v_unsafeQueue_1176_);
lean_inc(v_lastExpandedInIteration_1174_);
lean_inc(v_addedInIteration_1173_);
lean_inc(v_forwardRuleMatches_1171_);
lean_inc(v_forwardState_1170_);
lean_inc(v_mvars_1169_);
lean_inc(v_normalizationState_1168_);
lean_inc(v_preNormGoal_1167_);
lean_inc(v_depth_1164_);
lean_inc(v_origin_1163_);
lean_inc(v_children_1162_);
lean_inc(v_parent_1161_);
lean_inc(v_id_1160_);
lean_dec(v___x_1159_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1226_;
goto v_resetjp_1178_;
}
v___jp_1147_:
{
lean_object* v___x_1148_; lean_object* v_elimGoal_1149_; lean_object* v___x_1150_; lean_object* v_parent_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___x_1148_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_1149_ = lean_ctor_get(v___x_1148_, 1);
lean_inc_ref(v_elimGoal_1149_);
v___x_1150_ = lean_apply_1(v_elimGoal_1149_, v___x_1146_);
v_parent_1151_ = lean_ctor_get(v___x_1150_, 1);
lean_inc(v_parent_1151_);
lean_dec_ref(v___x_1150_);
v___x_1152_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1152_, 0, v_parent_1151_);
v___x_1153_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v___x_1152_);
return v___x_1153_;
}
v___jp_1154_:
{
goto v___jp_1147_;
}
v_resetjp_1178_:
{
uint8_t v___x_1181_; lean_object* v___x_1183_; 
v___x_1181_ = 3;
lean_inc_ref(v_children_1162_);
if (v_isShared_1180_ == 0)
{
v___x_1183_ = v___x_1179_;
goto v_reusejp_1182_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v_id_1160_);
lean_ctor_set(v_reuseFailAlloc_1225_, 1, v_parent_1161_);
lean_ctor_set(v_reuseFailAlloc_1225_, 2, v_children_1162_);
lean_ctor_set(v_reuseFailAlloc_1225_, 3, v_origin_1163_);
lean_ctor_set(v_reuseFailAlloc_1225_, 4, v_depth_1164_);
lean_ctor_set(v_reuseFailAlloc_1225_, 5, v_preNormGoal_1167_);
lean_ctor_set(v_reuseFailAlloc_1225_, 6, v_normalizationState_1168_);
lean_ctor_set(v_reuseFailAlloc_1225_, 7, v_mvars_1169_);
lean_ctor_set(v_reuseFailAlloc_1225_, 8, v_forwardState_1170_);
lean_ctor_set(v_reuseFailAlloc_1225_, 9, v_forwardRuleMatches_1171_);
lean_ctor_set(v_reuseFailAlloc_1225_, 10, v_addedInIteration_1173_);
lean_ctor_set(v_reuseFailAlloc_1225_, 11, v_lastExpandedInIteration_1174_);
lean_ctor_set(v_reuseFailAlloc_1225_, 12, v_unsafeQueue_1176_);
lean_ctor_set(v_reuseFailAlloc_1225_, 13, v_failedRapps_1177_);
lean_ctor_set_uint8(v_reuseFailAlloc_1225_, sizeof(void*)*14 + 9, v_isIrrelevant_1165_);
lean_ctor_set_uint8(v_reuseFailAlloc_1225_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1166_);
lean_ctor_set_float(v_reuseFailAlloc_1225_, sizeof(void*)*14, v_successProbability_1172_);
lean_ctor_set_uint8(v_reuseFailAlloc_1225_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1175_);
v___x_1183_ = v_reuseFailAlloc_1225_;
goto v_reusejp_1182_;
}
v_reusejp_1182_:
{
lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v_id_1186_; lean_object* v_parent_1187_; lean_object* v_children_1188_; lean_object* v_origin_1189_; lean_object* v_depth_1190_; uint8_t v_state_1191_; uint8_t v_isForcedUnprovable_1192_; lean_object* v_preNormGoal_1193_; lean_object* v_normalizationState_1194_; lean_object* v_mvars_1195_; lean_object* v_forwardState_1196_; lean_object* v_forwardRuleMatches_1197_; double v_successProbability_1198_; lean_object* v_addedInIteration_1199_; lean_object* v_lastExpandedInIteration_1200_; uint8_t v_unsafeRulesSelected_1201_; lean_object* v_unsafeQueue_1202_; lean_object* v_failedRapps_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1224_; 
lean_ctor_set_uint8(v___x_1183_, sizeof(void*)*14 + 8, v___x_1181_);
lean_inc(v_introGoal_1157_);
v___x_1184_ = lean_apply_1(v_introGoal_1157_, v___x_1183_);
lean_inc_ref(v_elimGoal_1158_);
v___x_1185_ = lean_apply_1(v_elimGoal_1158_, v___x_1184_);
v_id_1186_ = lean_ctor_get(v___x_1185_, 0);
v_parent_1187_ = lean_ctor_get(v___x_1185_, 1);
v_children_1188_ = lean_ctor_get(v___x_1185_, 2);
v_origin_1189_ = lean_ctor_get(v___x_1185_, 3);
v_depth_1190_ = lean_ctor_get(v___x_1185_, 4);
v_state_1191_ = lean_ctor_get_uint8(v___x_1185_, sizeof(void*)*14 + 8);
v_isForcedUnprovable_1192_ = lean_ctor_get_uint8(v___x_1185_, sizeof(void*)*14 + 10);
v_preNormGoal_1193_ = lean_ctor_get(v___x_1185_, 5);
v_normalizationState_1194_ = lean_ctor_get(v___x_1185_, 6);
v_mvars_1195_ = lean_ctor_get(v___x_1185_, 7);
v_forwardState_1196_ = lean_ctor_get(v___x_1185_, 8);
v_forwardRuleMatches_1197_ = lean_ctor_get(v___x_1185_, 9);
v_successProbability_1198_ = lean_ctor_get_float(v___x_1185_, sizeof(void*)*14);
v_addedInIteration_1199_ = lean_ctor_get(v___x_1185_, 10);
v_lastExpandedInIteration_1200_ = lean_ctor_get(v___x_1185_, 11);
v_unsafeRulesSelected_1201_ = lean_ctor_get_uint8(v___x_1185_, sizeof(void*)*14 + 11);
v_unsafeQueue_1202_ = lean_ctor_get(v___x_1185_, 12);
v_failedRapps_1203_ = lean_ctor_get(v___x_1185_, 13);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1205_ = v___x_1185_;
v_isShared_1206_ = v_isSharedCheck_1224_;
goto v_resetjp_1204_;
}
else
{
lean_inc(v_failedRapps_1203_);
lean_inc(v_unsafeQueue_1202_);
lean_inc(v_lastExpandedInIteration_1200_);
lean_inc(v_addedInIteration_1199_);
lean_inc(v_forwardRuleMatches_1197_);
lean_inc(v_forwardState_1196_);
lean_inc(v_mvars_1195_);
lean_inc(v_normalizationState_1194_);
lean_inc(v_preNormGoal_1193_);
lean_inc(v_depth_1190_);
lean_inc(v_origin_1189_);
lean_inc(v_children_1188_);
lean_inc(v_parent_1187_);
lean_inc(v_id_1186_);
lean_dec(v___x_1185_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1224_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
uint8_t v___x_1207_; lean_object* v___x_1209_; 
v___x_1207_ = 1;
if (v_isShared_1206_ == 0)
{
v___x_1209_ = v___x_1205_;
goto v_reusejp_1208_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_id_1186_);
lean_ctor_set(v_reuseFailAlloc_1223_, 1, v_parent_1187_);
lean_ctor_set(v_reuseFailAlloc_1223_, 2, v_children_1188_);
lean_ctor_set(v_reuseFailAlloc_1223_, 3, v_origin_1189_);
lean_ctor_set(v_reuseFailAlloc_1223_, 4, v_depth_1190_);
lean_ctor_set(v_reuseFailAlloc_1223_, 5, v_preNormGoal_1193_);
lean_ctor_set(v_reuseFailAlloc_1223_, 6, v_normalizationState_1194_);
lean_ctor_set(v_reuseFailAlloc_1223_, 7, v_mvars_1195_);
lean_ctor_set(v_reuseFailAlloc_1223_, 8, v_forwardState_1196_);
lean_ctor_set(v_reuseFailAlloc_1223_, 9, v_forwardRuleMatches_1197_);
lean_ctor_set(v_reuseFailAlloc_1223_, 10, v_addedInIteration_1199_);
lean_ctor_set(v_reuseFailAlloc_1223_, 11, v_lastExpandedInIteration_1200_);
lean_ctor_set(v_reuseFailAlloc_1223_, 12, v_unsafeQueue_1202_);
lean_ctor_set(v_reuseFailAlloc_1223_, 13, v_failedRapps_1203_);
lean_ctor_set_uint8(v_reuseFailAlloc_1223_, sizeof(void*)*14 + 8, v_state_1191_);
lean_ctor_set_uint8(v_reuseFailAlloc_1223_, sizeof(void*)*14 + 10, v_isForcedUnprovable_1192_);
lean_ctor_set_float(v_reuseFailAlloc_1223_, sizeof(void*)*14, v_successProbability_1198_);
lean_ctor_set_uint8(v_reuseFailAlloc_1223_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1201_);
v___x_1209_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1208_;
}
v_reusejp_1208_:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; uint8_t v___x_1214_; 
lean_ctor_set_uint8(v___x_1209_, sizeof(void*)*14 + 9, v___x_1207_);
lean_inc(v_introGoal_1157_);
v___x_1210_ = lean_apply_1(v_introGoal_1157_, v___x_1209_);
v___x_1211_ = lean_st_ref_set(v_gref_1144_, v___x_1210_);
v___x_1212_ = lean_unsigned_to_nat(0u);
v___x_1213_ = lean_array_get_size(v_children_1162_);
v___x_1214_ = lean_nat_dec_lt(v___x_1212_, v___x_1213_);
if (v___x_1214_ == 0)
{
lean_dec_ref(v_children_1162_);
goto v___jp_1147_;
}
else
{
lean_object* v___x_1215_; uint8_t v___x_1216_; 
v___x_1215_ = lean_box(0);
v___x_1216_ = lean_nat_dec_le(v___x_1213_, v___x_1213_);
if (v___x_1216_ == 0)
{
if (v___x_1214_ == 0)
{
lean_dec_ref(v_children_1162_);
goto v___jp_1147_;
}
else
{
size_t v___x_1217_; size_t v___x_1218_; lean_object* v___x_1219_; 
v___x_1217_ = ((size_t)0ULL);
v___x_1218_ = lean_usize_of_nat(v___x_1213_);
v___x_1219_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_1162_, v___x_1217_, v___x_1218_, v___x_1215_);
lean_dec_ref(v_children_1162_);
v___y_1155_ = v___x_1219_;
goto v___jp_1154_;
}
}
else
{
size_t v___x_1220_; size_t v___x_1221_; lean_object* v___x_1222_; 
v___x_1220_ = ((size_t)0ULL);
v___x_1221_ = lean_usize_of_nat(v___x_1213_);
v___x_1222_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_State_0__Aesop_markProvenCore_spec__1(v_children_1162_, v___x_1220_, v___x_1221_, v___x_1215_);
lean_dec_ref(v_children_1162_);
v___y_1155_ = v___x_1222_;
goto v___jp_1154_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markUnprovable___boxed(lean_object* v_gref_1227_, lean_object* v_a_1228_){
_start:
{
lean_object* v_res_1229_; 
v_res_1229_ = lp_aesop_Aesop_GoalRef_markUnprovable(v_gref_1227_);
lean_dec(v_gref_1227_);
return v_res_1229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable(lean_object* v_gref_1230_){
_start:
{
lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v_introGoal_1234_; lean_object* v_elimGoal_1235_; lean_object* v___x_1236_; lean_object* v_id_1237_; lean_object* v_parent_1238_; lean_object* v_children_1239_; lean_object* v_origin_1240_; lean_object* v_depth_1241_; uint8_t v_state_1242_; uint8_t v_isIrrelevant_1243_; lean_object* v_preNormGoal_1244_; lean_object* v_normalizationState_1245_; lean_object* v_mvars_1246_; lean_object* v_forwardState_1247_; lean_object* v_forwardRuleMatches_1248_; double v_successProbability_1249_; lean_object* v_addedInIteration_1250_; lean_object* v_lastExpandedInIteration_1251_; uint8_t v_unsafeRulesSelected_1252_; lean_object* v_unsafeQueue_1253_; lean_object* v_failedRapps_1254_; lean_object* v___x_1256_; uint8_t v_isShared_1257_; uint8_t v_isSharedCheck_1265_; 
v___x_1232_ = lean_st_ref_take(v_gref_1230_);
v___x_1233_ = lp_aesop_Aesop_treeImpl;
v_introGoal_1234_ = lean_ctor_get(v___x_1233_, 0);
v_elimGoal_1235_ = lean_ctor_get(v___x_1233_, 1);
lean_inc_ref(v_elimGoal_1235_);
v___x_1236_ = lean_apply_1(v_elimGoal_1235_, v___x_1232_);
v_id_1237_ = lean_ctor_get(v___x_1236_, 0);
v_parent_1238_ = lean_ctor_get(v___x_1236_, 1);
v_children_1239_ = lean_ctor_get(v___x_1236_, 2);
v_origin_1240_ = lean_ctor_get(v___x_1236_, 3);
v_depth_1241_ = lean_ctor_get(v___x_1236_, 4);
v_state_1242_ = lean_ctor_get_uint8(v___x_1236_, sizeof(void*)*14 + 8);
v_isIrrelevant_1243_ = lean_ctor_get_uint8(v___x_1236_, sizeof(void*)*14 + 9);
v_preNormGoal_1244_ = lean_ctor_get(v___x_1236_, 5);
v_normalizationState_1245_ = lean_ctor_get(v___x_1236_, 6);
v_mvars_1246_ = lean_ctor_get(v___x_1236_, 7);
v_forwardState_1247_ = lean_ctor_get(v___x_1236_, 8);
v_forwardRuleMatches_1248_ = lean_ctor_get(v___x_1236_, 9);
v_successProbability_1249_ = lean_ctor_get_float(v___x_1236_, sizeof(void*)*14);
v_addedInIteration_1250_ = lean_ctor_get(v___x_1236_, 10);
v_lastExpandedInIteration_1251_ = lean_ctor_get(v___x_1236_, 11);
v_unsafeRulesSelected_1252_ = lean_ctor_get_uint8(v___x_1236_, sizeof(void*)*14 + 11);
v_unsafeQueue_1253_ = lean_ctor_get(v___x_1236_, 12);
v_failedRapps_1254_ = lean_ctor_get(v___x_1236_, 13);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1236_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1256_ = v___x_1236_;
v_isShared_1257_ = v_isSharedCheck_1265_;
goto v_resetjp_1255_;
}
else
{
lean_inc(v_failedRapps_1254_);
lean_inc(v_unsafeQueue_1253_);
lean_inc(v_lastExpandedInIteration_1251_);
lean_inc(v_addedInIteration_1250_);
lean_inc(v_forwardRuleMatches_1248_);
lean_inc(v_forwardState_1247_);
lean_inc(v_mvars_1246_);
lean_inc(v_normalizationState_1245_);
lean_inc(v_preNormGoal_1244_);
lean_inc(v_depth_1241_);
lean_inc(v_origin_1240_);
lean_inc(v_children_1239_);
lean_inc(v_parent_1238_);
lean_inc(v_id_1237_);
lean_dec(v___x_1236_);
v___x_1256_ = lean_box(0);
v_isShared_1257_ = v_isSharedCheck_1265_;
goto v_resetjp_1255_;
}
v_resetjp_1255_:
{
uint8_t v___x_1258_; lean_object* v___x_1260_; 
v___x_1258_ = 1;
if (v_isShared_1257_ == 0)
{
v___x_1260_ = v___x_1256_;
goto v_reusejp_1259_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v_id_1237_);
lean_ctor_set(v_reuseFailAlloc_1264_, 1, v_parent_1238_);
lean_ctor_set(v_reuseFailAlloc_1264_, 2, v_children_1239_);
lean_ctor_set(v_reuseFailAlloc_1264_, 3, v_origin_1240_);
lean_ctor_set(v_reuseFailAlloc_1264_, 4, v_depth_1241_);
lean_ctor_set(v_reuseFailAlloc_1264_, 5, v_preNormGoal_1244_);
lean_ctor_set(v_reuseFailAlloc_1264_, 6, v_normalizationState_1245_);
lean_ctor_set(v_reuseFailAlloc_1264_, 7, v_mvars_1246_);
lean_ctor_set(v_reuseFailAlloc_1264_, 8, v_forwardState_1247_);
lean_ctor_set(v_reuseFailAlloc_1264_, 9, v_forwardRuleMatches_1248_);
lean_ctor_set(v_reuseFailAlloc_1264_, 10, v_addedInIteration_1250_);
lean_ctor_set(v_reuseFailAlloc_1264_, 11, v_lastExpandedInIteration_1251_);
lean_ctor_set(v_reuseFailAlloc_1264_, 12, v_unsafeQueue_1253_);
lean_ctor_set(v_reuseFailAlloc_1264_, 13, v_failedRapps_1254_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*14 + 8, v_state_1242_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*14 + 9, v_isIrrelevant_1243_);
lean_ctor_set_float(v_reuseFailAlloc_1264_, sizeof(void*)*14, v_successProbability_1249_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_1252_);
v___x_1260_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1259_;
}
v_reusejp_1259_:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
lean_ctor_set_uint8(v___x_1260_, sizeof(void*)*14 + 10, v___x_1258_);
lean_inc(v_introGoal_1234_);
v___x_1261_ = lean_apply_1(v_introGoal_1234_, v___x_1260_);
v___x_1262_ = lean_st_ref_set(v_gref_1230_, v___x_1261_);
v___x_1263_ = lp_aesop_Aesop_GoalRef_markUnprovable(v_gref_1230_);
return v___x_1263_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_markForcedUnprovable___boxed(lean_object* v_gref_1266_, lean_object* v_a_1267_){
_start:
{
lean_object* v_res_1268_; 
v_res_1268_ = lp_aesop_Aesop_GoalRef_markForcedUnprovable(v_gref_1266_);
lean_dec(v_gref_1266_);
return v_res_1268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_checkAndMarkUnprovable(lean_object* v_gref_1269_){
_start:
{
lean_object* v___x_1271_; lean_object* v___x_1272_; 
v___x_1271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1271_, 0, v_gref_1269_);
v___x_1272_ = lp_aesop_Aesop_traverseUp___at___00__private_Aesop_Tree_State_0__Aesop_markUnprovableCore_spec__1(v___x_1271_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalRef_checkAndMarkUnprovable___boxed(lean_object* v_gref_1273_, lean_object* v_a_1274_){
_start:
{
lean_object* v_res_1275_; 
v_res_1275_ = lp_aesop_Aesop_GoalRef_checkAndMarkUnprovable(v_gref_1273_);
return v_res_1275_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Goal_stateNoCache(lean_object* v_g_1276_){
_start:
{
uint8_t v___x_1278_; 
lean_inc(v_g_1276_);
v___x_1278_ = lp_aesop_Aesop_Goal_isProvenByNormalizationNoCache(v_g_1276_);
if (v___x_1278_ == 0)
{
uint8_t v___x_1279_; 
lean_inc(v_g_1276_);
v___x_1279_ = lp_aesop_Aesop_Goal_isProvenByRuleApplicationNoCache(v_g_1276_);
if (v___x_1279_ == 0)
{
uint8_t v___x_1280_; 
v___x_1280_ = lp_aesop_Aesop_Goal_isUnprovableNoCache(v_g_1276_);
if (v___x_1280_ == 0)
{
uint8_t v___x_1281_; 
v___x_1281_ = 0;
return v___x_1281_;
}
else
{
uint8_t v___x_1282_; 
v___x_1282_ = 3;
return v___x_1282_;
}
}
else
{
uint8_t v___x_1283_; 
lean_dec(v_g_1276_);
v___x_1283_ = 1;
return v___x_1283_;
}
}
else
{
uint8_t v___x_1284_; 
lean_dec(v_g_1276_);
v___x_1284_ = 2;
return v___x_1284_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_stateNoCache___boxed(lean_object* v_g_1285_, lean_object* v_a_1286_){
_start:
{
uint8_t v_res_1287_; lean_object* v_r_1288_; 
v_res_1287_ = lp_aesop_Aesop_Goal_stateNoCache(v_g_1285_);
v_r_1288_ = lean_box(v_res_1287_);
return v_r_1288_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Rapp_stateNoCache(lean_object* v_r_1289_){
_start:
{
uint8_t v___x_1291_; 
lean_inc(v_r_1289_);
v___x_1291_ = lp_aesop_Aesop_Rapp_isProvenNoCache(v_r_1289_);
if (v___x_1291_ == 0)
{
uint8_t v___x_1292_; 
v___x_1292_ = lp_aesop_Aesop_Rapp_isUnprovableNoCache(v_r_1289_);
if (v___x_1292_ == 0)
{
uint8_t v___x_1293_; 
v___x_1293_ = 0;
return v___x_1293_;
}
else
{
uint8_t v___x_1294_; 
v___x_1294_ = 2;
return v___x_1294_;
}
}
else
{
uint8_t v___x_1295_; 
lean_dec(v_r_1289_);
v___x_1295_ = 1;
return v___x_1295_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Rapp_stateNoCache___boxed(lean_object* v_r_1296_, lean_object* v_a_1297_){
_start:
{
uint8_t v_res_1298_; lean_object* v_r_1299_; 
v_res_1298_ = lp_aesop_Aesop_Rapp_stateNoCache(v_r_1296_);
v_r_1299_ = lean_box(v_res_1298_);
return v_r_1299_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_MVarCluster_stateNoCache(lean_object* v_c_1300_){
_start:
{
uint8_t v___x_1302_; 
lean_inc(v_c_1300_);
v___x_1302_ = lp_aesop_Aesop_MVarCluster_isProvenNoCache(v_c_1300_);
if (v___x_1302_ == 0)
{
uint8_t v___x_1303_; 
v___x_1303_ = lp_aesop_Aesop_MVarCluster_isUnprovableNoCache(v_c_1300_);
if (v___x_1303_ == 0)
{
uint8_t v___x_1304_; 
v___x_1304_ = 0;
return v___x_1304_;
}
else
{
uint8_t v___x_1305_; 
v___x_1305_ = 2;
return v___x_1305_;
}
}
else
{
uint8_t v___x_1306_; 
lean_dec(v_c_1300_);
v___x_1306_ = 1;
return v___x_1306_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_MVarCluster_stateNoCache___boxed(lean_object* v_c_1307_, lean_object* v_a_1308_){
_start:
{
uint8_t v_res_1309_; lean_object* v_r_1310_; 
v_res_1309_ = lp_aesop_Aesop_MVarCluster_stateNoCache(v_c_1307_);
v_r_1310_ = lean_box(v_res_1309_);
return v_r_1310_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_State(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_State(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_Traversal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_State(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_Traversal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_State(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_State(builtin);
}
#ifdef __cplusplus
}
#endif
