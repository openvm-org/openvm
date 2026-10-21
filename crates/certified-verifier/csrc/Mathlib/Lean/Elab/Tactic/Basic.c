// Lean compiler output
// Module: Mathlib.Lean.Elab.Tactic.Basic
// Imports: public import Init public meta import Init public import Mathlib.Lean.Meta
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Elab_InfoTree_substitute(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_instInhabitedFileMap_default;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg(lean_object* v_a_1_, lean_object* v_a_2_, lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1_, v_a_2_, v_a_3_, v_a_4_, v_a_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_9_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_a_8_);
lean_dec_ref_known(v___x_7_, 1);
v___x_9_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_a_8_, v_a_2_, v_a_3_, v_a_4_, v_a_5_);
return v___x_9_;
}
else
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_17_; 
v_a_10_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_17_ == 0)
{
v___x_12_ = v___x_7_;
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_7_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_15_; 
if (v_isShared_13_ == 0)
{
v___x_15_ = v___x_12_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v_a_10_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg___boxed(lean_object* v_a_18_, lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v_a_21_, lean_object* v_a_22_, lean_object* v_a_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg(v_a_18_, v_a_19_, v_a_20_, v_a_21_, v_a_22_);
lean_dec(v_a_22_);
lean_dec_ref(v_a_21_);
lean_dec(v_a_20_);
lean_dec_ref(v_a_19_);
lean_dec(v_a_18_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27(lean_object* v_a_25_, lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_, lean_object* v_a_31_, lean_object* v_a_32_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___redArg(v_a_26_, v_a_29_, v_a_30_, v_a_31_, v_a_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27___boxed(lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_, lean_object* v_a_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Lean_Elab_Tactic_getMainTarget_x27_x27(v_a_35_, v_a_36_, v_a_37_, v_a_38_, v_a_39_, v_a_40_, v_a_41_, v_a_42_);
lean_dec(v_a_42_);
lean_dec_ref(v_a_41_);
lean_dec(v_a_40_);
lean_dec_ref(v_a_39_);
lean_dec(v_a_38_);
lean_dec_ref(v_a_37_);
lean_dec(v_a_36_);
lean_dec_ref(v_a_35_);
return v_res_44_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = lean_unsigned_to_nat(32u);
v___x_46_ = lean_mk_empty_array_with_capacity(v___x_45_);
v___x_47_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1(void){
_start:
{
size_t v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_48_ = ((size_t)5ULL);
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_unsigned_to_nat(32u);
v___x_51_ = lean_mk_empty_array_with_capacity(v___x_50_);
v___x_52_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__0);
v___x_53_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_53_, 0, v___x_52_);
lean_ctor_set(v___x_53_, 1, v___x_51_);
lean_ctor_set(v___x_53_, 2, v___x_49_);
lean_ctor_set(v___x_53_, 3, v___x_49_);
lean_ctor_set_usize(v___x_53_, 4, v___x_48_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg(lean_object* v___y_54_){
_start:
{
lean_object* v___x_56_; lean_object* v_infoState_57_; lean_object* v_trees_58_; lean_object* v___x_59_; lean_object* v_infoState_60_; lean_object* v_env_61_; lean_object* v_nextMacroScope_62_; lean_object* v_ngen_63_; lean_object* v_auxDeclNGen_64_; lean_object* v_traceState_65_; lean_object* v_cache_66_; lean_object* v_messages_67_; lean_object* v_snapshotTasks_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_89_; 
v___x_56_ = lean_st_ref_get(v___y_54_);
v_infoState_57_ = lean_ctor_get(v___x_56_, 7);
lean_inc_ref(v_infoState_57_);
lean_dec(v___x_56_);
v_trees_58_ = lean_ctor_get(v_infoState_57_, 2);
lean_inc_ref(v_trees_58_);
lean_dec_ref(v_infoState_57_);
v___x_59_ = lean_st_ref_take(v___y_54_);
v_infoState_60_ = lean_ctor_get(v___x_59_, 7);
v_env_61_ = lean_ctor_get(v___x_59_, 0);
v_nextMacroScope_62_ = lean_ctor_get(v___x_59_, 1);
v_ngen_63_ = lean_ctor_get(v___x_59_, 2);
v_auxDeclNGen_64_ = lean_ctor_get(v___x_59_, 3);
v_traceState_65_ = lean_ctor_get(v___x_59_, 4);
v_cache_66_ = lean_ctor_get(v___x_59_, 5);
v_messages_67_ = lean_ctor_get(v___x_59_, 6);
v_snapshotTasks_68_ = lean_ctor_get(v___x_59_, 8);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_59_);
if (v_isSharedCheck_89_ == 0)
{
v___x_70_ = v___x_59_;
v_isShared_71_ = v_isSharedCheck_89_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_snapshotTasks_68_);
lean_inc(v_infoState_60_);
lean_inc(v_messages_67_);
lean_inc(v_cache_66_);
lean_inc(v_traceState_65_);
lean_inc(v_auxDeclNGen_64_);
lean_inc(v_ngen_63_);
lean_inc(v_nextMacroScope_62_);
lean_inc(v_env_61_);
lean_dec(v___x_59_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_89_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
uint8_t v_enabled_72_; lean_object* v_assignment_73_; lean_object* v_lazyAssignment_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_87_; 
v_enabled_72_ = lean_ctor_get_uint8(v_infoState_60_, sizeof(void*)*3);
v_assignment_73_ = lean_ctor_get(v_infoState_60_, 0);
v_lazyAssignment_74_ = lean_ctor_get(v_infoState_60_, 1);
v_isSharedCheck_87_ = !lean_is_exclusive(v_infoState_60_);
if (v_isSharedCheck_87_ == 0)
{
lean_object* v_unused_88_; 
v_unused_88_ = lean_ctor_get(v_infoState_60_, 2);
lean_dec(v_unused_88_);
v___x_76_ = v_infoState_60_;
v_isShared_77_ = v_isSharedCheck_87_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_lazyAssignment_74_);
lean_inc(v_assignment_73_);
lean_dec(v_infoState_60_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_87_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_78_; lean_object* v___x_80_; 
v___x_78_ = lean_obj_once(&lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___closed__1);
if (v_isShared_77_ == 0)
{
lean_ctor_set(v___x_76_, 2, v___x_78_);
v___x_80_ = v___x_76_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v_assignment_73_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v_lazyAssignment_74_);
lean_ctor_set(v_reuseFailAlloc_86_, 2, v___x_78_);
lean_ctor_set_uint8(v_reuseFailAlloc_86_, sizeof(void*)*3, v_enabled_72_);
v___x_80_ = v_reuseFailAlloc_86_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
lean_object* v___x_82_; 
if (v_isShared_71_ == 0)
{
lean_ctor_set(v___x_70_, 7, v___x_80_);
v___x_82_ = v___x_70_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_env_61_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v_nextMacroScope_62_);
lean_ctor_set(v_reuseFailAlloc_85_, 2, v_ngen_63_);
lean_ctor_set(v_reuseFailAlloc_85_, 3, v_auxDeclNGen_64_);
lean_ctor_set(v_reuseFailAlloc_85_, 4, v_traceState_65_);
lean_ctor_set(v_reuseFailAlloc_85_, 5, v_cache_66_);
lean_ctor_set(v_reuseFailAlloc_85_, 6, v_messages_67_);
lean_ctor_set(v_reuseFailAlloc_85_, 7, v___x_80_);
lean_ctor_set(v_reuseFailAlloc_85_, 8, v_snapshotTasks_68_);
v___x_82_ = v_reuseFailAlloc_85_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_83_ = lean_st_ref_set(v___y_54_, v___x_82_);
v___x_84_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_84_, 0, v_trees_58_);
return v___x_84_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg(v___y_90_);
lean_dec(v___y_90_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6(lean_object* v___x_93_, lean_object* v_ctx_x3f_94_, size_t v_sz_95_, size_t v_i_96_, lean_object* v_bs_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
uint8_t v___x_107_; 
v___x_107_ = lean_usize_dec_lt(v_i_96_, v_sz_95_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; 
lean_dec_ref(v_ctx_x3f_94_);
v___x_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_108_, 0, v_bs_97_);
return v___x_108_;
}
else
{
lean_object* v_assignment_109_; lean_object* v___x_110_; 
v_assignment_109_ = lean_ctor_get(v___x_93_, 0);
lean_inc_ref(v_ctx_x3f_94_);
lean_inc(v___y_105_);
lean_inc_ref(v___y_104_);
lean_inc(v___y_103_);
lean_inc_ref(v___y_102_);
lean_inc(v___y_101_);
lean_inc_ref(v___y_100_);
lean_inc(v___y_99_);
lean_inc_ref(v___y_98_);
v___x_110_ = lean_apply_9(v_ctx_x3f_94_, v___y_98_, v___y_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_, v___y_105_, lean_box(0));
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v_a_111_; lean_object* v_v_112_; lean_object* v___x_113_; lean_object* v_bs_x27_114_; lean_object* v_a_116_; lean_object* v_tree_121_; 
v_a_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_a_111_);
lean_dec_ref_known(v___x_110_, 1);
v_v_112_ = lean_array_uget(v_bs_97_, v_i_96_);
v___x_113_ = lean_unsigned_to_nat(0u);
v_bs_x27_114_ = lean_array_uset(v_bs_97_, v_i_96_, v___x_113_);
v_tree_121_ = l_Lean_Elab_InfoTree_substitute(v_v_112_, v_assignment_109_);
if (lean_obj_tag(v_a_111_) == 0)
{
v_a_116_ = v_tree_121_;
goto v___jp_115_;
}
else
{
lean_object* v_val_122_; lean_object* v___x_123_; 
v_val_122_ = lean_ctor_get(v_a_111_, 0);
lean_inc(v_val_122_);
lean_dec_ref_known(v_a_111_, 1);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v_val_122_);
lean_ctor_set(v___x_123_, 1, v_tree_121_);
v_a_116_ = v___x_123_;
goto v___jp_115_;
}
v___jp_115_:
{
size_t v___x_117_; size_t v___x_118_; lean_object* v___x_119_; 
v___x_117_ = ((size_t)1ULL);
v___x_118_ = lean_usize_add(v_i_96_, v___x_117_);
v___x_119_ = lean_array_uset(v_bs_x27_114_, v_i_96_, v_a_116_);
v_i_96_ = v___x_118_;
v_bs_97_ = v___x_119_;
goto _start;
}
}
else
{
lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_131_; 
lean_dec_ref(v_bs_97_);
lean_dec_ref(v_ctx_x3f_94_);
v_a_124_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_131_ == 0)
{
v___x_126_ = v___x_110_;
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_110_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_131_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_129_; 
if (v_isShared_127_ == 0)
{
v___x_129_ = v___x_126_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v_a_124_);
v___x_129_ = v_reuseFailAlloc_130_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
return v___x_129_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6___boxed(lean_object* v___x_132_, lean_object* v_ctx_x3f_133_, lean_object* v_sz_134_, lean_object* v_i_135_, lean_object* v_bs_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
size_t v_sz_boxed_146_; size_t v_i_boxed_147_; lean_object* v_res_148_; 
v_sz_boxed_146_ = lean_unbox_usize(v_sz_134_);
lean_dec(v_sz_134_);
v_i_boxed_147_ = lean_unbox_usize(v_i_135_);
lean_dec(v_i_135_);
v_res_148_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6(v___x_132_, v_ctx_x3f_133_, v_sz_boxed_146_, v_i_boxed_147_, v_bs_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
lean_dec(v___y_140_);
lean_dec_ref(v___y_139_);
lean_dec(v___y_138_);
lean_dec_ref(v___y_137_);
lean_dec_ref(v___x_132_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5(lean_object* v___x_149_, lean_object* v_ctx_x3f_150_, lean_object* v_x_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_){
_start:
{
if (lean_obj_tag(v_x_151_) == 0)
{
lean_object* v_cs_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_187_; 
v_cs_161_ = lean_ctor_get(v_x_151_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v_x_151_);
if (v_isSharedCheck_187_ == 0)
{
v___x_163_ = v_x_151_;
v_isShared_164_ = v_isSharedCheck_187_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_cs_161_);
lean_dec(v_x_151_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_187_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
size_t v_sz_165_; size_t v___x_166_; lean_object* v___x_167_; 
v_sz_165_ = lean_array_size(v_cs_161_);
v___x_166_ = ((size_t)0ULL);
v___x_167_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6(v___x_149_, v_ctx_x3f_150_, v_sz_165_, v___x_166_, v_cs_161_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_);
if (lean_obj_tag(v___x_167_) == 0)
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_178_; 
v_a_168_ = lean_ctor_get(v___x_167_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_178_ == 0)
{
v___x_170_ = v___x_167_;
v_isShared_171_ = v_isSharedCheck_178_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_167_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_178_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_173_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v_a_168_);
v___x_173_ = v___x_163_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v_a_168_);
v___x_173_ = v_reuseFailAlloc_177_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
lean_object* v___x_175_; 
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 0, v___x_173_);
v___x_175_ = v___x_170_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_173_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
else
{
lean_object* v_a_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_186_; 
lean_del_object(v___x_163_);
v_a_179_ = lean_ctor_get(v___x_167_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_186_ == 0)
{
v___x_181_ = v___x_167_;
v_isShared_182_ = v_isSharedCheck_186_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_a_179_);
lean_dec(v___x_167_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_186_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_184_; 
if (v_isShared_182_ == 0)
{
v___x_184_ = v___x_181_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v_a_179_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
}
else
{
lean_object* v_vs_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_214_; 
v_vs_188_ = lean_ctor_get(v_x_151_, 0);
v_isSharedCheck_214_ = !lean_is_exclusive(v_x_151_);
if (v_isSharedCheck_214_ == 0)
{
v___x_190_ = v_x_151_;
v_isShared_191_ = v_isSharedCheck_214_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_vs_188_);
lean_dec(v_x_151_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_214_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
size_t v_sz_192_; size_t v___x_193_; lean_object* v___x_194_; 
v_sz_192_ = lean_array_size(v_vs_188_);
v___x_193_ = ((size_t)0ULL);
v___x_194_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6(v___x_149_, v_ctx_x3f_150_, v_sz_192_, v___x_193_, v_vs_188_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_);
if (lean_obj_tag(v___x_194_) == 0)
{
lean_object* v_a_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_205_; 
v_a_195_ = lean_ctor_get(v___x_194_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_194_);
if (v_isSharedCheck_205_ == 0)
{
v___x_197_ = v___x_194_;
v_isShared_198_ = v_isSharedCheck_205_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_a_195_);
lean_dec(v___x_194_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_205_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v___x_200_; 
if (v_isShared_191_ == 0)
{
lean_ctor_set(v___x_190_, 0, v_a_195_);
v___x_200_ = v___x_190_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_195_);
v___x_200_ = v_reuseFailAlloc_204_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
lean_object* v___x_202_; 
if (v_isShared_198_ == 0)
{
lean_ctor_set(v___x_197_, 0, v___x_200_);
v___x_202_ = v___x_197_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v___x_200_);
v___x_202_ = v_reuseFailAlloc_203_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
return v___x_202_;
}
}
}
}
else
{
lean_object* v_a_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_213_; 
lean_del_object(v___x_190_);
v_a_206_ = lean_ctor_get(v___x_194_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_194_);
if (v_isSharedCheck_213_ == 0)
{
v___x_208_ = v___x_194_;
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_a_206_);
lean_dec(v___x_194_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_213_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
if (v_isShared_209_ == 0)
{
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v_a_206_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6(lean_object* v___x_215_, lean_object* v_ctx_x3f_216_, size_t v_sz_217_, size_t v_i_218_, lean_object* v_bs_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
uint8_t v___x_229_; 
v___x_229_ = lean_usize_dec_lt(v_i_218_, v_sz_217_);
if (v___x_229_ == 0)
{
lean_object* v___x_230_; 
lean_dec_ref(v_ctx_x3f_216_);
v___x_230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_230_, 0, v_bs_219_);
return v___x_230_;
}
else
{
lean_object* v_v_231_; lean_object* v___x_232_; 
v_v_231_ = lean_array_uget_borrowed(v_bs_219_, v_i_218_);
lean_inc(v_v_231_);
lean_inc_ref(v_ctx_x3f_216_);
v___x_232_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5(v___x_215_, v_ctx_x3f_216_, v_v_231_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_, v___y_227_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v_a_233_; lean_object* v___x_234_; lean_object* v_bs_x27_235_; size_t v___x_236_; size_t v___x_237_; lean_object* v___x_238_; 
v_a_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_a_233_);
lean_dec_ref_known(v___x_232_, 1);
v___x_234_ = lean_unsigned_to_nat(0u);
v_bs_x27_235_ = lean_array_uset(v_bs_219_, v_i_218_, v___x_234_);
v___x_236_ = ((size_t)1ULL);
v___x_237_ = lean_usize_add(v_i_218_, v___x_236_);
v___x_238_ = lean_array_uset(v_bs_x27_235_, v_i_218_, v_a_233_);
v_i_218_ = v___x_237_;
v_bs_219_ = v___x_238_;
goto _start;
}
else
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
lean_dec_ref(v_bs_219_);
lean_dec_ref(v_ctx_x3f_216_);
v_a_240_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___x_232_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___x_232_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6___boxed(lean_object* v___x_248_, lean_object* v_ctx_x3f_249_, lean_object* v_sz_250_, lean_object* v_i_251_, lean_object* v_bs_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
size_t v_sz_boxed_262_; size_t v_i_boxed_263_; lean_object* v_res_264_; 
v_sz_boxed_262_ = lean_unbox_usize(v_sz_250_);
lean_dec(v_sz_250_);
v_i_boxed_263_ = lean_unbox_usize(v_i_251_);
lean_dec(v_i_251_);
v_res_264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5_spec__6(v___x_248_, v_ctx_x3f_249_, v_sz_boxed_262_, v_i_boxed_263_, v_bs_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
lean_dec(v___y_254_);
lean_dec_ref(v___y_253_);
lean_dec_ref(v___x_248_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5___boxed(lean_object* v___x_265_, lean_object* v_ctx_x3f_266_, lean_object* v_x_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5(v___x_265_, v_ctx_x3f_266_, v_x_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
lean_dec(v___y_275_);
lean_dec_ref(v___y_274_);
lean_dec(v___y_273_);
lean_dec_ref(v___y_272_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec_ref(v___x_265_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4(lean_object* v___x_278_, lean_object* v_ctx_x3f_279_, lean_object* v_t_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v_root_290_; lean_object* v_tail_291_; lean_object* v_size_292_; size_t v_shift_293_; lean_object* v_tailOff_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_330_; 
v_root_290_ = lean_ctor_get(v_t_280_, 0);
v_tail_291_ = lean_ctor_get(v_t_280_, 1);
v_size_292_ = lean_ctor_get(v_t_280_, 2);
v_shift_293_ = lean_ctor_get_usize(v_t_280_, 4);
v_tailOff_294_ = lean_ctor_get(v_t_280_, 3);
v_isSharedCheck_330_ = !lean_is_exclusive(v_t_280_);
if (v_isSharedCheck_330_ == 0)
{
v___x_296_ = v_t_280_;
v_isShared_297_ = v_isSharedCheck_330_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_tailOff_294_);
lean_inc(v_size_292_);
lean_inc(v_tail_291_);
lean_inc(v_root_290_);
lean_dec(v_t_280_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_330_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_298_; 
lean_inc_ref(v_ctx_x3f_279_);
v___x_298_ = lp_mathlib_Lean_PersistentArray_mapMAux___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__5(v___x_278_, v_ctx_x3f_279_, v_root_290_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
if (lean_obj_tag(v___x_298_) == 0)
{
lean_object* v_a_299_; size_t v_sz_300_; size_t v___x_301_; lean_object* v___x_302_; 
v_a_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc(v_a_299_);
lean_dec_ref_known(v___x_298_, 1);
v_sz_300_ = lean_array_size(v_tail_291_);
v___x_301_ = ((size_t)0ULL);
v___x_302_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4_spec__6(v___x_278_, v_ctx_x3f_279_, v_sz_300_, v___x_301_, v_tail_291_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_313_; 
v_a_303_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_313_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_313_ == 0)
{
v___x_305_ = v___x_302_;
v_isShared_306_ = v_isSharedCheck_313_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_302_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_313_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_297_ == 0)
{
lean_ctor_set(v___x_296_, 1, v_a_303_);
lean_ctor_set(v___x_296_, 0, v_a_299_);
v___x_308_ = v___x_296_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v_a_299_);
lean_ctor_set(v_reuseFailAlloc_312_, 1, v_a_303_);
lean_ctor_set(v_reuseFailAlloc_312_, 2, v_size_292_);
lean_ctor_set(v_reuseFailAlloc_312_, 3, v_tailOff_294_);
lean_ctor_set_usize(v_reuseFailAlloc_312_, 4, v_shift_293_);
v___x_308_ = v_reuseFailAlloc_312_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
lean_object* v___x_310_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 0, v___x_308_);
v___x_310_ = v___x_305_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v___x_308_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
else
{
lean_object* v_a_314_; lean_object* v___x_316_; uint8_t v_isShared_317_; uint8_t v_isSharedCheck_321_; 
lean_dec(v_a_299_);
lean_del_object(v___x_296_);
lean_dec(v_tailOff_294_);
lean_dec(v_size_292_);
v_a_314_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_321_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_321_ == 0)
{
v___x_316_ = v___x_302_;
v_isShared_317_ = v_isSharedCheck_321_;
goto v_resetjp_315_;
}
else
{
lean_inc(v_a_314_);
lean_dec(v___x_302_);
v___x_316_ = lean_box(0);
v_isShared_317_ = v_isSharedCheck_321_;
goto v_resetjp_315_;
}
v_resetjp_315_:
{
lean_object* v___x_319_; 
if (v_isShared_317_ == 0)
{
v___x_319_ = v___x_316_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v_a_314_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
else
{
lean_object* v_a_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_329_; 
lean_del_object(v___x_296_);
lean_dec(v_tailOff_294_);
lean_dec(v_size_292_);
lean_dec_ref(v_tail_291_);
lean_dec_ref(v_ctx_x3f_279_);
v_a_322_ = lean_ctor_get(v___x_298_, 0);
v_isSharedCheck_329_ = !lean_is_exclusive(v___x_298_);
if (v_isSharedCheck_329_ == 0)
{
v___x_324_ = v___x_298_;
v_isShared_325_ = v_isSharedCheck_329_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_a_322_);
lean_dec(v___x_298_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_329_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___x_327_; 
if (v_isShared_325_ == 0)
{
v___x_327_ = v___x_324_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_328_; 
v_reuseFailAlloc_328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_328_, 0, v_a_322_);
v___x_327_ = v_reuseFailAlloc_328_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
return v___x_327_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4___boxed(lean_object* v___x_331_, lean_object* v_ctx_x3f_332_, lean_object* v_t_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4(v___x_331_, v_ctx_x3f_332_, v_t_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_, v___y_338_, v___y_339_, v___y_340_, v___y_341_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
lean_dec(v___y_339_);
lean_dec_ref(v___y_338_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec_ref(v___x_331_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0(lean_object* v___y_344_, lean_object* v_ctx_x3f_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v_a_353_, lean_object* v_a_x3f_354_){
_start:
{
lean_object* v___x_356_; lean_object* v_infoState_357_; lean_object* v_trees_358_; lean_object* v___x_359_; 
v___x_356_ = lean_st_ref_get(v___y_344_);
v_infoState_357_ = lean_ctor_get(v___x_356_, 7);
lean_inc_ref(v_infoState_357_);
lean_dec(v___x_356_);
v_trees_358_ = lean_ctor_get(v_infoState_357_, 2);
lean_inc_ref(v_trees_358_);
v___x_359_ = lp_mathlib_Lean_PersistentArray_mapM___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__4(v_infoState_357_, v_ctx_x3f_345_, v_trees_358_, v___y_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_, v___y_351_, v___y_352_, v___y_344_);
lean_dec_ref(v_infoState_357_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_398_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_398_ == 0)
{
v___x_362_ = v___x_359_;
v_isShared_363_ = v_isSharedCheck_398_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_359_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_398_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_364_; lean_object* v_infoState_365_; lean_object* v_env_366_; lean_object* v_nextMacroScope_367_; lean_object* v_ngen_368_; lean_object* v_auxDeclNGen_369_; lean_object* v_traceState_370_; lean_object* v_cache_371_; lean_object* v_messages_372_; lean_object* v_snapshotTasks_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_397_; 
v___x_364_ = lean_st_ref_take(v___y_344_);
v_infoState_365_ = lean_ctor_get(v___x_364_, 7);
v_env_366_ = lean_ctor_get(v___x_364_, 0);
v_nextMacroScope_367_ = lean_ctor_get(v___x_364_, 1);
v_ngen_368_ = lean_ctor_get(v___x_364_, 2);
v_auxDeclNGen_369_ = lean_ctor_get(v___x_364_, 3);
v_traceState_370_ = lean_ctor_get(v___x_364_, 4);
v_cache_371_ = lean_ctor_get(v___x_364_, 5);
v_messages_372_ = lean_ctor_get(v___x_364_, 6);
v_snapshotTasks_373_ = lean_ctor_get(v___x_364_, 8);
v_isSharedCheck_397_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_397_ == 0)
{
v___x_375_ = v___x_364_;
v_isShared_376_ = v_isSharedCheck_397_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_snapshotTasks_373_);
lean_inc(v_infoState_365_);
lean_inc(v_messages_372_);
lean_inc(v_cache_371_);
lean_inc(v_traceState_370_);
lean_inc(v_auxDeclNGen_369_);
lean_inc(v_ngen_368_);
lean_inc(v_nextMacroScope_367_);
lean_inc(v_env_366_);
lean_dec(v___x_364_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_397_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
uint8_t v_enabled_377_; lean_object* v_assignment_378_; lean_object* v_lazyAssignment_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_395_; 
v_enabled_377_ = lean_ctor_get_uint8(v_infoState_365_, sizeof(void*)*3);
v_assignment_378_ = lean_ctor_get(v_infoState_365_, 0);
v_lazyAssignment_379_ = lean_ctor_get(v_infoState_365_, 1);
v_isSharedCheck_395_ = !lean_is_exclusive(v_infoState_365_);
if (v_isSharedCheck_395_ == 0)
{
lean_object* v_unused_396_; 
v_unused_396_ = lean_ctor_get(v_infoState_365_, 2);
lean_dec(v_unused_396_);
v___x_381_ = v_infoState_365_;
v_isShared_382_ = v_isSharedCheck_395_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_lazyAssignment_379_);
lean_inc(v_assignment_378_);
lean_dec(v_infoState_365_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_395_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_383_; lean_object* v___x_385_; 
v___x_383_ = l_Lean_PersistentArray_append___redArg(v_a_353_, v_a_360_);
lean_dec(v_a_360_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 2, v___x_383_);
v___x_385_ = v___x_381_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_assignment_378_);
lean_ctor_set(v_reuseFailAlloc_394_, 1, v_lazyAssignment_379_);
lean_ctor_set(v_reuseFailAlloc_394_, 2, v___x_383_);
lean_ctor_set_uint8(v_reuseFailAlloc_394_, sizeof(void*)*3, v_enabled_377_);
v___x_385_ = v_reuseFailAlloc_394_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
lean_object* v___x_387_; 
if (v_isShared_376_ == 0)
{
lean_ctor_set(v___x_375_, 7, v___x_385_);
v___x_387_ = v___x_375_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_env_366_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_nextMacroScope_367_);
lean_ctor_set(v_reuseFailAlloc_393_, 2, v_ngen_368_);
lean_ctor_set(v_reuseFailAlloc_393_, 3, v_auxDeclNGen_369_);
lean_ctor_set(v_reuseFailAlloc_393_, 4, v_traceState_370_);
lean_ctor_set(v_reuseFailAlloc_393_, 5, v_cache_371_);
lean_ctor_set(v_reuseFailAlloc_393_, 6, v_messages_372_);
lean_ctor_set(v_reuseFailAlloc_393_, 7, v___x_385_);
lean_ctor_set(v_reuseFailAlloc_393_, 8, v_snapshotTasks_373_);
v___x_387_ = v_reuseFailAlloc_393_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_391_; 
v___x_388_ = lean_st_ref_set(v___y_344_, v___x_387_);
v___x_389_ = lean_box(0);
if (v_isShared_363_ == 0)
{
lean_ctor_set(v___x_362_, 0, v___x_389_);
v___x_391_ = v___x_362_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v___x_389_);
v___x_391_ = v_reuseFailAlloc_392_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
return v___x_391_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_dec_ref(v_a_353_);
v_a_399_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_359_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_359_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0___boxed(lean_object* v___y_407_, lean_object* v_ctx_x3f_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v_a_416_, lean_object* v_a_x3f_417_, lean_object* v___y_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0(v___y_407_, v_ctx_x3f_408_, v___y_409_, v___y_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_, v_a_416_, v_a_x3f_417_);
lean_dec(v_a_x3f_417_);
lean_dec_ref(v___y_415_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec(v___y_407_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg(lean_object* v_x_420_, lean_object* v_ctx_x3f_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v___x_431_; lean_object* v_infoState_432_; uint8_t v_enabled_433_; 
v___x_431_ = lean_st_ref_get(v___y_429_);
v_infoState_432_ = lean_ctor_get(v___x_431_, 7);
lean_inc_ref(v_infoState_432_);
lean_dec(v___x_431_);
v_enabled_433_ = lean_ctor_get_uint8(v_infoState_432_, sizeof(void*)*3);
lean_dec_ref(v_infoState_432_);
if (v_enabled_433_ == 0)
{
lean_object* v___x_434_; 
lean_dec_ref(v_ctx_x3f_421_);
lean_inc(v___y_429_);
lean_inc_ref(v___y_428_);
lean_inc(v___y_427_);
lean_inc_ref(v___y_426_);
lean_inc(v___y_425_);
lean_inc_ref(v___y_424_);
lean_inc(v___y_423_);
lean_inc_ref(v___y_422_);
v___x_434_ = lean_apply_9(v_x_420_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_, lean_box(0));
return v___x_434_;
}
else
{
lean_object* v___x_435_; lean_object* v_a_436_; lean_object* v_r_437_; 
v___x_435_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg(v___y_429_);
v_a_436_ = lean_ctor_get(v___x_435_, 0);
lean_inc(v_a_436_);
lean_dec_ref(v___x_435_);
lean_inc(v___y_429_);
lean_inc_ref(v___y_428_);
lean_inc(v___y_427_);
lean_inc_ref(v___y_426_);
lean_inc(v___y_425_);
lean_inc_ref(v___y_424_);
lean_inc(v___y_423_);
lean_inc_ref(v___y_422_);
v_r_437_ = lean_apply_9(v_x_420_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_, lean_box(0));
if (lean_obj_tag(v_r_437_) == 0)
{
lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_462_; 
v_a_438_ = lean_ctor_get(v_r_437_, 0);
v_isSharedCheck_462_ = !lean_is_exclusive(v_r_437_);
if (v_isSharedCheck_462_ == 0)
{
v___x_440_ = v_r_437_;
v_isShared_441_ = v_isSharedCheck_462_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_dec(v_r_437_);
v___x_440_ = lean_box(0);
v_isShared_441_ = v_isSharedCheck_462_;
goto v_resetjp_439_;
}
v_resetjp_439_:
{
lean_object* v___x_443_; 
lean_inc(v_a_438_);
if (v_isShared_441_ == 0)
{
lean_ctor_set_tag(v___x_440_, 1);
v___x_443_ = v___x_440_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v_a_438_);
v___x_443_ = v_reuseFailAlloc_461_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
lean_object* v___x_444_; 
v___x_444_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0(v___y_429_, v_ctx_x3f_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v_a_436_, v___x_443_);
lean_dec_ref(v___x_443_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_451_; 
v_isSharedCheck_451_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_451_ == 0)
{
lean_object* v_unused_452_; 
v_unused_452_ = lean_ctor_get(v___x_444_, 0);
lean_dec(v_unused_452_);
v___x_446_ = v___x_444_;
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
else
{
lean_dec(v___x_444_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_451_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_449_; 
if (v_isShared_447_ == 0)
{
lean_ctor_set(v___x_446_, 0, v_a_438_);
v___x_449_ = v___x_446_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v_a_438_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
else
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
lean_dec(v_a_438_);
v_a_453_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_444_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_444_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_a_453_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
}
}
else
{
lean_object* v_a_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v_a_463_ = lean_ctor_get(v_r_437_, 0);
lean_inc(v_a_463_);
lean_dec_ref_known(v_r_437_, 1);
v___x_464_ = lean_box(0);
v___x_465_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___lam__0(v___y_429_, v_ctx_x3f_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v_a_436_, v___x_464_);
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_472_; 
v_isSharedCheck_472_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_472_ == 0)
{
lean_object* v_unused_473_; 
v_unused_473_ = lean_ctor_get(v___x_465_, 0);
lean_dec(v_unused_473_);
v___x_467_ = v___x_465_;
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
else
{
lean_dec(v___x_465_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___x_470_; 
if (v_isShared_468_ == 0)
{
lean_ctor_set_tag(v___x_467_, 1);
lean_ctor_set(v___x_467_, 0, v_a_463_);
v___x_470_ = v___x_467_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_a_463_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
}
else
{
lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_481_; 
lean_dec(v_a_463_);
v_a_474_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_481_ == 0)
{
v___x_476_ = v___x_465_;
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v___x_465_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_477_ == 0)
{
v___x_479_ = v___x_476_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_a_474_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg___boxed(lean_object* v_x_482_, lean_object* v_ctx_x3f_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg(v_x_482_, v_ctx_x3f_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
lean_dec(v___y_489_);
lean_dec_ref(v___y_488_);
lean_dec(v___y_487_);
lean_dec_ref(v___y_486_);
lean_dec(v___y_485_);
lean_dec_ref(v___y_484_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg(lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v___x_498_; lean_object* v_env_499_; lean_object* v___x_500_; lean_object* v_mctx_501_; lean_object* v_options_502_; lean_object* v_currNamespace_503_; lean_object* v_openDecls_504_; lean_object* v___x_505_; lean_object* v_ngen_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_498_ = lean_st_ref_get(v___y_496_);
v_env_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc_ref(v_env_499_);
lean_dec(v___x_498_);
v___x_500_ = lean_st_ref_get(v___y_494_);
v_mctx_501_ = lean_ctor_get(v___x_500_, 0);
lean_inc_ref(v_mctx_501_);
lean_dec(v___x_500_);
v_options_502_ = lean_ctor_get(v___y_495_, 2);
v_currNamespace_503_ = lean_ctor_get(v___y_495_, 6);
v_openDecls_504_ = lean_ctor_get(v___y_495_, 7);
v___x_505_ = lean_st_ref_get(v___y_496_);
v_ngen_506_ = lean_ctor_get(v___x_505_, 2);
lean_inc_ref(v_ngen_506_);
lean_dec(v___x_505_);
v___x_507_ = lean_box(0);
v___x_508_ = l_Lean_instInhabitedFileMap_default;
lean_inc(v_openDecls_504_);
lean_inc(v_currNamespace_503_);
lean_inc_ref(v_options_502_);
v___x_509_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_509_, 0, v_env_499_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
lean_ctor_set(v___x_509_, 2, v___x_508_);
lean_ctor_set(v___x_509_, 3, v_mctx_501_);
lean_ctor_set(v___x_509_, 4, v_options_502_);
lean_ctor_set(v___x_509_, 5, v_currNamespace_503_);
lean_ctor_set(v___x_509_, 6, v_openDecls_504_);
lean_ctor_set(v___x_509_, 7, v_ngen_506_);
v___x_510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg(v___y_511_, v___y_512_, v___y_513_);
lean_dec(v___y_513_);
lean_dec_ref(v___y_512_);
lean_dec(v___y_511_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0(lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v___x_525_; lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_550_; 
v___x_525_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg(v___y_521_, v___y_522_, v___y_523_);
v_a_526_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_550_ == 0)
{
v___x_528_ = v___x_525_;
v_isShared_529_ = v_isSharedCheck_550_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_525_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_550_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v_fileMap_530_; lean_object* v_env_531_; lean_object* v_mctx_532_; lean_object* v_options_533_; lean_object* v_currNamespace_534_; lean_object* v_openDecls_535_; lean_object* v_ngen_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_547_; 
v_fileMap_530_ = lean_ctor_get(v___y_522_, 1);
v_env_531_ = lean_ctor_get(v_a_526_, 0);
v_mctx_532_ = lean_ctor_get(v_a_526_, 3);
v_options_533_ = lean_ctor_get(v_a_526_, 4);
v_currNamespace_534_ = lean_ctor_get(v_a_526_, 5);
v_openDecls_535_ = lean_ctor_get(v_a_526_, 6);
v_ngen_536_ = lean_ctor_get(v_a_526_, 7);
v_isSharedCheck_547_ = !lean_is_exclusive(v_a_526_);
if (v_isSharedCheck_547_ == 0)
{
lean_object* v_unused_548_; lean_object* v_unused_549_; 
v_unused_548_ = lean_ctor_get(v_a_526_, 2);
lean_dec(v_unused_548_);
v_unused_549_ = lean_ctor_get(v_a_526_, 1);
lean_dec(v_unused_549_);
v___x_538_ = v_a_526_;
v_isShared_539_ = v_isSharedCheck_547_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_ngen_536_);
lean_inc(v_openDecls_535_);
lean_inc(v_currNamespace_534_);
lean_inc(v_options_533_);
lean_inc(v_mctx_532_);
lean_inc(v_env_531_);
lean_dec(v_a_526_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_547_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_540_; lean_object* v___x_542_; 
v___x_540_ = lean_box(0);
lean_inc_ref(v_fileMap_530_);
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 2, v_fileMap_530_);
lean_ctor_set(v___x_538_, 1, v___x_540_);
v___x_542_ = v___x_538_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v_env_531_);
lean_ctor_set(v_reuseFailAlloc_546_, 1, v___x_540_);
lean_ctor_set(v_reuseFailAlloc_546_, 2, v_fileMap_530_);
lean_ctor_set(v_reuseFailAlloc_546_, 3, v_mctx_532_);
lean_ctor_set(v_reuseFailAlloc_546_, 4, v_options_533_);
lean_ctor_set(v_reuseFailAlloc_546_, 5, v_currNamespace_534_);
lean_ctor_set(v_reuseFailAlloc_546_, 6, v_openDecls_535_);
lean_ctor_set(v_reuseFailAlloc_546_, 7, v_ngen_536_);
v___x_542_ = v_reuseFailAlloc_546_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
lean_object* v___x_544_; 
if (v_isShared_529_ == 0)
{
lean_ctor_set(v___x_528_, 0, v___x_542_);
v___x_544_ = v___x_528_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v___x_542_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0___boxed(lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0(v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_);
lean_dec(v___y_558_);
lean_dec_ref(v___y_557_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
lean_dec(v___y_552_);
lean_dec_ref(v___y_551_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0(lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_){
_start:
{
lean_object* v___x_570_; lean_object* v_a_571_; lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_580_; 
v___x_570_ = lp_mathlib_Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0(v___y_561_, v___y_562_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
v_a_571_ = lean_ctor_get(v___x_570_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_580_ == 0)
{
v___x_573_ = v___x_570_;
v_isShared_574_ = v_isSharedCheck_580_;
goto v_resetjp_572_;
}
else
{
lean_inc(v_a_571_);
lean_dec(v___x_570_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_580_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_578_; 
v___x_575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_575_, 0, v_a_571_);
v___x_576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_576_, 0, v___x_575_);
if (v_isShared_574_ == 0)
{
lean_ctor_set(v___x_573_, 0, v___x_576_);
v___x_578_ = v___x_573_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v___x_576_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0___boxed(lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___lam__0(v___y_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg(lean_object* v_x_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_){
_start:
{
lean_object* v___f_602_; lean_object* v___x_603_; 
v___f_602_ = ((lean_object*)(lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___closed__0));
v___x_603_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg(v_x_592_, v___f_602_, v___y_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_, v___y_600_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg___boxed(lean_object* v_x_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg(v_x_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_, v___y_612_);
lean_dec(v___y_612_);
lean_dec_ref(v___y_611_);
lean_dec(v___y_610_);
lean_dec_ref(v___y_609_);
lean_dec(v___y_608_);
lean_dec_ref(v___y_607_);
lean_dec(v___y_606_);
lean_dec_ref(v___y_605_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg(lean_object* v_x_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_){
_start:
{
lean_object* v___x_625_; 
v___x_625_ = l_Lean_Elab_Tactic_saveState___redArg(v_a_617_, v_a_619_, v_a_621_, v_a_623_);
if (lean_obj_tag(v___x_625_) == 0)
{
lean_object* v_a_626_; lean_object* v___x_627_; 
v_a_626_ = lean_ctor_get(v___x_625_, 0);
lean_inc(v_a_626_);
lean_dec_ref_known(v___x_625_, 1);
v___x_627_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg(v_x_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_, v_a_623_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_dec(v_a_626_);
return v___x_627_;
}
else
{
lean_object* v_a_628_; uint8_t v___y_630_; uint8_t v___x_714_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_a_628_);
v___x_714_ = l_Lean_Exception_isInterrupt(v_a_628_);
if (v___x_714_ == 0)
{
uint8_t v___x_715_; 
lean_inc(v_a_628_);
v___x_715_ = l_Lean_Exception_isRuntime(v_a_628_);
v___y_630_ = v___x_715_;
goto v___jp_629_;
}
else
{
v___y_630_ = v___x_714_;
goto v___jp_629_;
}
v___jp_629_:
{
if (v___y_630_ == 0)
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v_term_633_; lean_object* v_meta_634_; lean_object* v_core_635_; lean_object* v_toState_636_; lean_object* v_infoState_637_; lean_object* v_tactic_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_712_; 
lean_dec_ref_known(v___x_627_, 1);
v___x_631_ = lean_st_ref_get(v_a_623_);
v___x_632_ = lean_st_ref_get(v_a_623_);
v_term_633_ = lean_ctor_get(v_a_626_, 0);
lean_inc_ref(v_term_633_);
v_meta_634_ = lean_ctor_get(v_term_633_, 0);
lean_inc_ref(v_meta_634_);
v_core_635_ = lean_ctor_get(v_meta_634_, 0);
lean_inc_ref(v_core_635_);
v_toState_636_ = lean_ctor_get(v_core_635_, 0);
lean_inc_ref(v_toState_636_);
v_infoState_637_ = lean_ctor_get(v___x_631_, 7);
lean_inc_ref(v_infoState_637_);
lean_dec(v___x_631_);
v_tactic_638_ = lean_ctor_get(v_a_626_, 1);
v_isSharedCheck_712_ = !lean_is_exclusive(v_a_626_);
if (v_isSharedCheck_712_ == 0)
{
lean_object* v_unused_713_; 
v_unused_713_ = lean_ctor_get(v_a_626_, 0);
lean_dec(v_unused_713_);
v___x_640_ = v_a_626_;
v_isShared_641_ = v_isSharedCheck_712_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_tactic_638_);
lean_dec(v_a_626_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_712_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
lean_object* v_elab_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_710_; 
v_elab_642_ = lean_ctor_get(v_term_633_, 1);
v_isSharedCheck_710_ = !lean_is_exclusive(v_term_633_);
if (v_isSharedCheck_710_ == 0)
{
lean_object* v_unused_711_; 
v_unused_711_ = lean_ctor_get(v_term_633_, 0);
lean_dec(v_unused_711_);
v___x_644_ = v_term_633_;
v_isShared_645_ = v_isSharedCheck_710_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_elab_642_);
lean_dec(v_term_633_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_710_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v_meta_646_; lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_708_; 
v_meta_646_ = lean_ctor_get(v_meta_634_, 1);
v_isSharedCheck_708_ = !lean_is_exclusive(v_meta_634_);
if (v_isSharedCheck_708_ == 0)
{
lean_object* v_unused_709_; 
v_unused_709_ = lean_ctor_get(v_meta_634_, 0);
lean_dec(v_unused_709_);
v___x_648_ = v_meta_634_;
v_isShared_649_ = v_isSharedCheck_708_;
goto v_resetjp_647_;
}
else
{
lean_inc(v_meta_646_);
lean_dec(v_meta_634_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_708_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v_passedHeartbeats_650_; lean_object* v___x_652_; uint8_t v_isShared_653_; uint8_t v_isSharedCheck_706_; 
v_passedHeartbeats_650_ = lean_ctor_get(v_core_635_, 1);
v_isSharedCheck_706_ = !lean_is_exclusive(v_core_635_);
if (v_isSharedCheck_706_ == 0)
{
lean_object* v_unused_707_; 
v_unused_707_ = lean_ctor_get(v_core_635_, 0);
lean_dec(v_unused_707_);
v___x_652_ = v_core_635_;
v_isShared_653_ = v_isSharedCheck_706_;
goto v_resetjp_651_;
}
else
{
lean_inc(v_passedHeartbeats_650_);
lean_dec(v_core_635_);
v___x_652_ = lean_box(0);
v_isShared_653_ = v_isSharedCheck_706_;
goto v_resetjp_651_;
}
v_resetjp_651_:
{
lean_object* v_env_654_; lean_object* v_nextMacroScope_655_; lean_object* v_ngen_656_; lean_object* v_auxDeclNGen_657_; lean_object* v_traceState_658_; lean_object* v_cache_659_; lean_object* v_snapshotTasks_660_; lean_object* v_messages_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_697_; 
v_env_654_ = lean_ctor_get(v_toState_636_, 0);
lean_inc_ref(v_env_654_);
v_nextMacroScope_655_ = lean_ctor_get(v_toState_636_, 1);
lean_inc(v_nextMacroScope_655_);
v_ngen_656_ = lean_ctor_get(v_toState_636_, 2);
lean_inc_ref(v_ngen_656_);
v_auxDeclNGen_657_ = lean_ctor_get(v_toState_636_, 3);
lean_inc_ref(v_auxDeclNGen_657_);
v_traceState_658_ = lean_ctor_get(v_toState_636_, 4);
lean_inc_ref(v_traceState_658_);
v_cache_659_ = lean_ctor_get(v_toState_636_, 5);
lean_inc_ref(v_cache_659_);
v_snapshotTasks_660_ = lean_ctor_get(v_toState_636_, 8);
lean_inc_ref(v_snapshotTasks_660_);
lean_dec_ref(v_toState_636_);
v_messages_661_ = lean_ctor_get(v___x_632_, 6);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_697_ == 0)
{
lean_object* v_unused_698_; lean_object* v_unused_699_; lean_object* v_unused_700_; lean_object* v_unused_701_; lean_object* v_unused_702_; lean_object* v_unused_703_; lean_object* v_unused_704_; lean_object* v_unused_705_; 
v_unused_698_ = lean_ctor_get(v___x_632_, 8);
lean_dec(v_unused_698_);
v_unused_699_ = lean_ctor_get(v___x_632_, 7);
lean_dec(v_unused_699_);
v_unused_700_ = lean_ctor_get(v___x_632_, 5);
lean_dec(v_unused_700_);
v_unused_701_ = lean_ctor_get(v___x_632_, 4);
lean_dec(v_unused_701_);
v_unused_702_ = lean_ctor_get(v___x_632_, 3);
lean_dec(v_unused_702_);
v_unused_703_ = lean_ctor_get(v___x_632_, 2);
lean_dec(v_unused_703_);
v_unused_704_ = lean_ctor_get(v___x_632_, 1);
lean_dec(v_unused_704_);
v_unused_705_ = lean_ctor_get(v___x_632_, 0);
lean_dec(v_unused_705_);
v___x_663_ = v___x_632_;
v_isShared_664_ = v_isSharedCheck_697_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_messages_661_);
lean_dec(v___x_632_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_697_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___x_666_; 
if (v_isShared_664_ == 0)
{
lean_ctor_set(v___x_663_, 8, v_snapshotTasks_660_);
lean_ctor_set(v___x_663_, 7, v_infoState_637_);
lean_ctor_set(v___x_663_, 5, v_cache_659_);
lean_ctor_set(v___x_663_, 4, v_traceState_658_);
lean_ctor_set(v___x_663_, 3, v_auxDeclNGen_657_);
lean_ctor_set(v___x_663_, 2, v_ngen_656_);
lean_ctor_set(v___x_663_, 1, v_nextMacroScope_655_);
lean_ctor_set(v___x_663_, 0, v_env_654_);
v___x_666_ = v___x_663_;
goto v_reusejp_665_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_env_654_);
lean_ctor_set(v_reuseFailAlloc_696_, 1, v_nextMacroScope_655_);
lean_ctor_set(v_reuseFailAlloc_696_, 2, v_ngen_656_);
lean_ctor_set(v_reuseFailAlloc_696_, 3, v_auxDeclNGen_657_);
lean_ctor_set(v_reuseFailAlloc_696_, 4, v_traceState_658_);
lean_ctor_set(v_reuseFailAlloc_696_, 5, v_cache_659_);
lean_ctor_set(v_reuseFailAlloc_696_, 6, v_messages_661_);
lean_ctor_set(v_reuseFailAlloc_696_, 7, v_infoState_637_);
lean_ctor_set(v_reuseFailAlloc_696_, 8, v_snapshotTasks_660_);
v___x_666_ = v_reuseFailAlloc_696_;
goto v_reusejp_665_;
}
v_reusejp_665_:
{
lean_object* v___x_668_; 
if (v_isShared_653_ == 0)
{
lean_ctor_set(v___x_652_, 0, v___x_666_);
v___x_668_ = v___x_652_;
goto v_reusejp_667_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_666_);
lean_ctor_set(v_reuseFailAlloc_695_, 1, v_passedHeartbeats_650_);
v___x_668_ = v_reuseFailAlloc_695_;
goto v_reusejp_667_;
}
v_reusejp_667_:
{
lean_object* v___x_670_; 
if (v_isShared_649_ == 0)
{
lean_ctor_set(v___x_648_, 0, v___x_668_);
v___x_670_ = v___x_648_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v___x_668_);
lean_ctor_set(v_reuseFailAlloc_694_, 1, v_meta_646_);
v___x_670_ = v_reuseFailAlloc_694_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
lean_object* v___x_672_; 
if (v_isShared_645_ == 0)
{
lean_ctor_set(v___x_644_, 0, v___x_670_);
v___x_672_ = v___x_644_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v___x_670_);
lean_ctor_set(v_reuseFailAlloc_693_, 1, v_elab_642_);
v___x_672_ = v_reuseFailAlloc_693_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
lean_object* v___x_674_; 
if (v_isShared_641_ == 0)
{
lean_ctor_set(v___x_640_, 0, v___x_672_);
v___x_674_ = v___x_640_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v___x_672_);
lean_ctor_set(v_reuseFailAlloc_692_, 1, v_tactic_638_);
v___x_674_ = v_reuseFailAlloc_692_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
lean_object* v___x_675_; 
v___x_675_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___x_674_, v___y_630_, v_a_617_, v_a_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_, v_a_623_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_682_; 
v_isSharedCheck_682_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_682_ == 0)
{
lean_object* v_unused_683_; 
v_unused_683_ = lean_ctor_get(v___x_675_, 0);
lean_dec(v_unused_683_);
v___x_677_ = v___x_675_;
v_isShared_678_ = v_isSharedCheck_682_;
goto v_resetjp_676_;
}
else
{
lean_dec(v___x_675_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_682_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v___x_680_; 
if (v_isShared_678_ == 0)
{
lean_ctor_set_tag(v___x_677_, 1);
lean_ctor_set(v___x_677_, 0, v_a_628_);
v___x_680_ = v___x_677_;
goto v_reusejp_679_;
}
else
{
lean_object* v_reuseFailAlloc_681_; 
v_reuseFailAlloc_681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_681_, 0, v_a_628_);
v___x_680_ = v_reuseFailAlloc_681_;
goto v_reusejp_679_;
}
v_reusejp_679_:
{
return v___x_680_;
}
}
}
else
{
lean_object* v_a_684_; lean_object* v___x_686_; uint8_t v_isShared_687_; uint8_t v_isSharedCheck_691_; 
lean_dec(v_a_628_);
v_a_684_ = lean_ctor_get(v___x_675_, 0);
v_isSharedCheck_691_ = !lean_is_exclusive(v___x_675_);
if (v_isSharedCheck_691_ == 0)
{
v___x_686_ = v___x_675_;
v_isShared_687_ = v_isSharedCheck_691_;
goto v_resetjp_685_;
}
else
{
lean_inc(v_a_684_);
lean_dec(v___x_675_);
v___x_686_ = lean_box(0);
v_isShared_687_ = v_isSharedCheck_691_;
goto v_resetjp_685_;
}
v_resetjp_685_:
{
lean_object* v___x_689_; 
if (v_isShared_687_ == 0)
{
v___x_689_ = v___x_686_;
goto v_reusejp_688_;
}
else
{
lean_object* v_reuseFailAlloc_690_; 
v_reuseFailAlloc_690_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_690_, 0, v_a_684_);
v___x_689_ = v_reuseFailAlloc_690_;
goto v_reusejp_688_;
}
v_reusejp_688_:
{
return v___x_689_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
else
{
lean_dec(v_a_628_);
lean_dec(v_a_626_);
return v___x_627_;
}
}
}
}
else
{
lean_object* v_a_716_; lean_object* v___x_718_; uint8_t v_isShared_719_; uint8_t v_isSharedCheck_723_; 
lean_dec_ref(v_x_615_);
v_a_716_ = lean_ctor_get(v___x_625_, 0);
v_isSharedCheck_723_ = !lean_is_exclusive(v___x_625_);
if (v_isSharedCheck_723_ == 0)
{
v___x_718_ = v___x_625_;
v_isShared_719_ = v_isSharedCheck_723_;
goto v_resetjp_717_;
}
else
{
lean_inc(v_a_716_);
lean_dec(v___x_625_);
v___x_718_ = lean_box(0);
v_isShared_719_ = v_isSharedCheck_723_;
goto v_resetjp_717_;
}
v_resetjp_717_:
{
lean_object* v___x_721_; 
if (v_isShared_719_ == 0)
{
v___x_721_ = v___x_718_;
goto v_reusejp_720_;
}
else
{
lean_object* v_reuseFailAlloc_722_; 
v_reuseFailAlloc_722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_722_, 0, v_a_716_);
v___x_721_ = v_reuseFailAlloc_722_;
goto v_reusejp_720_;
}
v_reusejp_720_:
{
return v___x_721_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg___boxed(lean_object* v_x_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_){
_start:
{
lean_object* v_res_734_; 
v_res_734_ = lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg(v_x_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_, v_a_729_, v_a_730_, v_a_731_, v_a_732_);
lean_dec(v_a_732_);
lean_dec_ref(v_a_731_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
lean_dec(v_a_728_);
lean_dec_ref(v_a_727_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
return v_res_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages(lean_object* v_00_u03b1_735_, lean_object* v_x_736_, lean_object* v_a_737_, lean_object* v_a_738_, lean_object* v_a_739_, lean_object* v_a_740_, lean_object* v_a_741_, lean_object* v_a_742_, lean_object* v_a_743_, lean_object* v_a_744_){
_start:
{
lean_object* v___x_746_; 
v___x_746_ = lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___redArg(v_x_736_, v_a_737_, v_a_738_, v_a_739_, v_a_740_, v_a_741_, v_a_742_, v_a_743_, v_a_744_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___boxed(lean_object* v_00_u03b1_747_, lean_object* v_x_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages(v_00_u03b1_747_, v_x_748_, v_a_749_, v_a_750_, v_a_751_, v_a_752_, v_a_753_, v_a_754_, v_a_755_, v_a_756_);
lean_dec(v_a_756_);
lean_dec_ref(v_a_755_);
lean_dec(v_a_754_);
lean_dec_ref(v_a_753_);
lean_dec(v_a_752_);
lean_dec_ref(v_a_751_);
lean_dec(v_a_750_);
lean_dec_ref(v_a_749_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0(lean_object* v_00_u03b1_759_, lean_object* v_x_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___redArg(v_x_760_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0___boxed(lean_object* v_00_u03b1_771_, lean_object* v_x_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_mathlib_Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0(v_00_u03b1_771_, v_x_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1(lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___redArg(v___y_788_, v___y_789_, v___y_790_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1___boxed(lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_){
_start:
{
lean_object* v_res_802_; 
v_res_802_ = lp_mathlib_Lean_Elab_CommandContextInfo_saveNoFileMap___at___00Lean_Elab_CommandContextInfo_save___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__0_spec__1(v___y_793_, v___y_794_, v___y_795_, v___y_796_, v___y_797_, v___y_798_, v___y_799_, v___y_800_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
lean_dec(v___y_798_);
lean_dec_ref(v___y_797_);
lean_dec(v___y_796_);
lean_dec_ref(v___y_795_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
return v_res_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3(lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_){
_start:
{
lean_object* v___x_812_; 
v___x_812_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___redArg(v___y_810_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3___boxed(lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_Lean_Elab_getResetInfoTrees___at___00__private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1_spec__3(v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec(v___y_814_);
lean_dec_ref(v___y_813_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1(lean_object* v_00_u03b1_823_, lean_object* v_x_824_, lean_object* v_ctx_x3f_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_){
_start:
{
lean_object* v___x_835_; 
v___x_835_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___redArg(v_x_824_, v_ctx_x3f_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1___boxed(lean_object* v_00_u03b1_836_, lean_object* v_x_837_, lean_object* v_ctx_x3f_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_){
_start:
{
lean_object* v_res_848_; 
v_res_848_ = lp_mathlib___private_Lean_Elab_InfoTree_Main_0__Lean_Elab_withSavedPartialInfoContext___at___00Lean_Elab_withSaveInfoContext___at___00Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages_spec__0_spec__1(v_00_u03b1_836_, v_x_837_, v_ctx_x3f_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
lean_dec(v___y_842_);
lean_dec_ref(v___y_841_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
return v_res_848_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
