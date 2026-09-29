// Lean compiler output
// Module: Batteries.Lean.Meta.SavedState
// Imports: public import Init public meta import Init public import Batteries.Lean.Meta.Basic public import Batteries.Lean.MonadBacktrack
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
lean_object* lean_st_ref_get(lean_object*);
uint8_t lp_batteries_Lean_MetavarContext_isExprMVarDeclared(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MetavarContext_unassignedExprMVars(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Meta_getIntroducedExprMVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_getIntroducedExprMVars___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0(lean_object* v_s_1_, lean_object* v_x_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l_Lean_Meta_SavedState_restore___redArg(v_s_1_, v___y_4_, v___y_6_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v___x_9_; 
lean_dec_ref_known(v___x_8_, 1);
v___x_9_ = lean_apply_5(v_x_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_, lean_box(0));
return v___x_9_;
}
else
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_17_; 
lean_dec(v___y_6_);
lean_dec_ref(v___y_5_);
lean_dec(v___y_4_);
lean_dec_ref(v___y_3_);
lean_dec_ref(v_x_2_);
v_a_10_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_17_ == 0)
{
v___x_12_ = v___x_8_;
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_8_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0___boxed(lean_object* v_s_18_, lean_object* v_x_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0(v_s_18_, v_x_19_, v___y_20_, v___y_21_, v___y_22_, v___y_23_);
lean_dec_ref(v_s_18_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0(lean_object* v_x_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_){
_start:
{
lean_object* v___x_32_; 
lean_inc(v___y_30_);
lean_inc(v___y_28_);
v___x_32_ = lean_apply_5(v_x_26_, v___y_27_, v___y_28_, v___y_29_, v___y_30_, lean_box(0));
if (lean_obj_tag(v___x_32_) == 0)
{
lean_object* v_a_33_; lean_object* v___x_34_; 
v_a_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_a_33_);
lean_dec_ref_known(v___x_32_, 1);
v___x_34_ = l_Lean_Meta_saveState___redArg(v___y_28_, v___y_30_);
lean_dec(v___y_30_);
lean_dec(v___y_28_);
if (lean_obj_tag(v___x_34_) == 0)
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_43_; 
v_a_35_ = lean_ctor_get(v___x_34_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_34_);
if (v_isSharedCheck_43_ == 0)
{
v___x_37_ = v___x_34_;
v_isShared_38_ = v_isSharedCheck_43_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v___x_34_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_43_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_39_; lean_object* v___x_41_; 
v___x_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_39_, 0, v_a_33_);
lean_ctor_set(v___x_39_, 1, v_a_35_);
if (v_isShared_38_ == 0)
{
lean_ctor_set(v___x_37_, 0, v___x_39_);
v___x_41_ = v___x_37_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v___x_39_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
else
{
lean_object* v_a_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_51_; 
lean_dec(v_a_33_);
v_a_44_ = lean_ctor_get(v___x_34_, 0);
v_isSharedCheck_51_ = !lean_is_exclusive(v___x_34_);
if (v_isSharedCheck_51_ == 0)
{
v___x_46_ = v___x_34_;
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_a_44_);
lean_dec(v___x_34_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_51_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_49_; 
if (v_isShared_47_ == 0)
{
v___x_49_ = v___x_46_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v_a_44_);
v___x_49_ = v_reuseFailAlloc_50_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
return v___x_49_;
}
}
}
}
else
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
lean_dec(v___y_30_);
lean_dec(v___y_28_);
v_a_52_ = lean_ctor_get(v___x_32_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_32_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_32_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_32_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0___boxed(lean_object* v_x_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0(v_x_60_, v___y_61_, v___y_62_, v___y_63_, v___y_64_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(lean_object* v_x_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = l_Lean_Meta_saveState___redArg(v___y_69_, v___y_71_);
if (lean_obj_tag(v___x_73_) == 0)
{
lean_object* v_a_74_; lean_object* v_r_75_; 
v_a_74_ = lean_ctor_get(v___x_73_, 0);
lean_inc(v_a_74_);
lean_dec_ref_known(v___x_73_, 1);
lean_inc(v___y_71_);
lean_inc_ref(v___y_70_);
lean_inc(v___y_69_);
lean_inc_ref(v___y_68_);
v_r_75_ = lean_apply_5(v_x_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_, lean_box(0));
if (lean_obj_tag(v_r_75_) == 0)
{
lean_object* v_a_76_; lean_object* v___x_77_; 
v_a_76_ = lean_ctor_get(v_r_75_, 0);
lean_inc(v_a_76_);
lean_dec_ref_known(v_r_75_, 1);
v___x_77_ = l_Lean_Meta_SavedState_restore___redArg(v_a_74_, v___y_69_, v___y_71_);
lean_dec(v_a_74_);
if (lean_obj_tag(v___x_77_) == 0)
{
lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_84_; 
v_isSharedCheck_84_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_84_ == 0)
{
lean_object* v_unused_85_; 
v_unused_85_ = lean_ctor_get(v___x_77_, 0);
lean_dec(v_unused_85_);
v___x_79_ = v___x_77_;
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
else
{
lean_dec(v___x_77_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_82_; 
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 0, v_a_76_);
v___x_82_ = v___x_79_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_a_76_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
else
{
lean_object* v_a_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_93_; 
lean_dec(v_a_76_);
v_a_86_ = lean_ctor_get(v___x_77_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_93_ == 0)
{
v___x_88_ = v___x_77_;
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_a_86_);
lean_dec(v___x_77_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_93_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___x_91_; 
if (v_isShared_89_ == 0)
{
v___x_91_ = v___x_88_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v_a_86_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
}
else
{
lean_object* v_a_94_; lean_object* v___x_95_; 
v_a_94_ = lean_ctor_get(v_r_75_, 0);
lean_inc(v_a_94_);
lean_dec_ref_known(v_r_75_, 1);
v___x_95_ = l_Lean_Meta_SavedState_restore___redArg(v_a_74_, v___y_69_, v___y_71_);
lean_dec(v_a_74_);
if (lean_obj_tag(v___x_95_) == 0)
{
lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_102_; 
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_95_);
if (v_isSharedCheck_102_ == 0)
{
lean_object* v_unused_103_; 
v_unused_103_ = lean_ctor_get(v___x_95_, 0);
lean_dec(v_unused_103_);
v___x_97_ = v___x_95_;
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
else
{
lean_dec(v___x_95_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_100_; 
if (v_isShared_98_ == 0)
{
lean_ctor_set_tag(v___x_97_, 1);
lean_ctor_set(v___x_97_, 0, v_a_94_);
v___x_100_ = v___x_97_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_a_94_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
}
else
{
lean_object* v_a_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_111_; 
lean_dec(v_a_94_);
v_a_104_ = lean_ctor_get(v___x_95_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_95_);
if (v_isSharedCheck_111_ == 0)
{
v___x_106_ = v___x_95_;
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_a_104_);
lean_dec(v___x_95_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_111_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_109_; 
if (v_isShared_107_ == 0)
{
v___x_109_ = v___x_106_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v_a_104_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
}
else
{
lean_object* v_a_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_119_; 
lean_dec_ref(v_x_67_);
v_a_112_ = lean_ctor_get(v___x_73_, 0);
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_73_);
if (v_isSharedCheck_119_ == 0)
{
v___x_114_ = v___x_73_;
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_a_112_);
lean_dec(v___x_73_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_117_; 
if (v_isShared_115_ == 0)
{
v___x_117_ = v___x_114_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_a_112_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg___boxed(lean_object* v_x_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(v_x_120_, v___y_121_, v___y_122_, v___y_123_, v___y_124_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
lean_dec(v___y_122_);
lean_dec_ref(v___y_121_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg(lean_object* v_x_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v___f_133_; lean_object* v___x_134_; 
v___f_133_ = lean_alloc_closure((void*)(lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_133_, 0, v_x_127_);
v___x_134_ = lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(v___f_133_, v___y_128_, v___y_129_, v___y_130_, v___y_131_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg___boxed(lean_object* v_x_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg(v_x_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(lean_object* v_s_142_, lean_object* v_x_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_){
_start:
{
lean_object* v___f_149_; lean_object* v___x_150_; 
v___f_149_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_149_, 0, v_s_142_);
lean_closure_set(v___f_149_, 1, v_x_143_);
v___x_150_ = lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg(v___f_149_, v_a_144_, v_a_145_, v_a_146_, v_a_147_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___boxed(lean_object* v_s_151_, lean_object* v_x_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(v_s_151_, v_x_152_, v_a_153_, v_a_154_, v_a_155_, v_a_156_);
lean_dec(v_a_156_);
lean_dec_ref(v_a_155_);
lean_dec(v_a_154_);
lean_dec_ref(v_a_153_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM(lean_object* v_00_u03b1_159_, lean_object* v_s_160_, lean_object* v_x_161_, lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_batteries_Lean_Meta_SavedState_runMetaM___redArg(v_s_160_, v_x_161_, v_a_162_, v_a_163_, v_a_164_, v_a_165_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM___boxed(lean_object* v_00_u03b1_168_, lean_object* v_s_169_, lean_object* v_x_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_batteries_Lean_Meta_SavedState_runMetaM(v_00_u03b1_168_, v_s_169_, v_x_170_, v_a_171_, v_a_172_, v_a_173_, v_a_174_);
lean_dec(v_a_174_);
lean_dec_ref(v_a_173_);
lean_dec(v_a_172_);
lean_dec_ref(v_a_171_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0(lean_object* v_00_u03b1_177_, lean_object* v_x_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(v_x_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___boxed(lean_object* v_00_u03b1_185_, lean_object* v_x_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0(v_00_u03b1_185_, v_x_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0(lean_object* v_00_u03b1_193_, lean_object* v_x_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___redArg(v_x_194_, v___y_195_, v___y_196_, v___y_197_, v___y_198_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0___boxed(lean_object* v_00_u03b1_201_, lean_object* v_x_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_batteries_Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0(v_00_u03b1_201_, v_x_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object* v_s_209_, lean_object* v_x_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v___f_216_; lean_object* v___x_217_; 
v___f_216_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_SavedState_runMetaM___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_216_, 0, v_s_209_);
lean_closure_set(v___f_216_, 1, v_x_210_);
v___x_217_ = lp_batteries_Lean_withoutModifyingState___at___00Lean_withoutModifyingState_x27___at___00Lean_Meta_SavedState_runMetaM_spec__0_spec__0___redArg(v___f_216_, v_a_211_, v_a_212_, v_a_213_, v_a_214_);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg___boxed(lean_object* v_s_218_, lean_object* v_x_219_, lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_s_218_, v_x_219_, v_a_220_, v_a_221_, v_a_222_, v_a_223_);
lean_dec(v_a_223_);
lean_dec_ref(v_a_222_);
lean_dec(v_a_221_);
lean_dec_ref(v_a_220_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27(lean_object* v_00_u03b1_226_, lean_object* v_s_227_, lean_object* v_x_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_s_227_, v_x_228_, v_a_229_, v_a_230_, v_a_231_, v_a_232_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___boxed(lean_object* v_00_u03b1_235_, lean_object* v_s_236_, lean_object* v_x_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27(v_00_u03b1_235_, v_s_236_, v_x_237_, v_a_238_, v_a_239_, v_a_240_, v_a_241_);
lean_dec(v_a_241_);
lean_dec_ref(v_a_240_);
lean_dec(v_a_239_);
lean_dec_ref(v_a_238_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg(lean_object* v_mvarId_244_, lean_object* v___y_245_){
_start:
{
lean_object* v___x_247_; lean_object* v_mctx_248_; uint8_t v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_247_ = lean_st_ref_get(v___y_245_);
v_mctx_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc_ref(v_mctx_248_);
lean_dec(v___x_247_);
v___x_249_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_mctx_248_, v_mvarId_244_);
lean_dec_ref(v_mctx_248_);
v___x_250_ = lean_box(v___x_249_);
v___x_251_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg___boxed(lean_object* v_mvarId_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg(v_mvarId_252_, v___y_253_);
lean_dec(v___y_253_);
lean_dec(v_mvarId_252_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0(lean_object* v_mvarId_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg(v_mvarId_256_, v___y_258_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___boxed(lean_object* v_mvarId_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0(v_mvarId_263_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v_mvarId_263_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg(uint8_t v_includeDelayed_270_, lean_object* v___y_271_){
_start:
{
lean_object* v___x_273_; lean_object* v_mctx_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_273_ = lean_st_ref_get(v___y_271_);
v_mctx_274_ = lean_ctor_get(v___x_273_, 0);
lean_inc_ref(v_mctx_274_);
lean_dec(v___x_273_);
v___x_275_ = lp_batteries_Lean_MetavarContext_unassignedExprMVars(v_mctx_274_, v_includeDelayed_270_);
v___x_276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg___boxed(lean_object* v_includeDelayed_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
uint8_t v_includeDelayed_boxed_280_; lean_object* v_res_281_; 
v_includeDelayed_boxed_280_ = lean_unbox(v_includeDelayed_277_);
v_res_281_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg(v_includeDelayed_boxed_280_, v___y_278_);
lean_dec(v___y_278_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1(uint8_t v_includeDelayed_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___redArg(v_includeDelayed_282_, v___y_284_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1___boxed(lean_object* v_includeDelayed_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
uint8_t v_includeDelayed_boxed_295_; lean_object* v_res_296_; 
v_includeDelayed_boxed_295_ = lean_unbox(v_includeDelayed_289_);
v_res_296_ = lp_batteries_Lean_Meta_getUnassignedExprMVars___at___00Lean_Meta_getIntroducedExprMVars_spec__1(v_includeDelayed_boxed_295_, v___y_290_, v___y_291_, v___y_292_, v___y_293_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2(lean_object* v_as_297_, size_t v_i_298_, size_t v_stop_299_, lean_object* v_b_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_a_307_; uint8_t v___x_311_; 
v___x_311_ = lean_usize_dec_eq(v_i_298_, v_stop_299_);
if (v___x_311_ == 0)
{
lean_object* v___x_312_; lean_object* v___x_315_; 
v___x_312_ = lean_array_uget_borrowed(v_as_297_, v_i_298_);
v___x_315_ = lp_batteries_Lean_MVarId_isDeclared___at___00Lean_Meta_getIntroducedExprMVars_spec__0___redArg(v___x_312_, v___y_302_);
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v_a_316_; uint8_t v___x_317_; 
v_a_316_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_a_316_);
lean_dec_ref_known(v___x_315_, 1);
v___x_317_ = lean_unbox(v_a_316_);
lean_dec(v_a_316_);
if (v___x_317_ == 0)
{
goto v___jp_313_;
}
else
{
v_a_307_ = v_b_300_;
goto v___jp_306_;
}
}
else
{
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v_a_318_; uint8_t v___x_319_; 
v_a_318_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_a_318_);
lean_dec_ref_known(v___x_315_, 1);
v___x_319_ = lean_unbox(v_a_318_);
lean_dec(v_a_318_);
if (v___x_319_ == 0)
{
v_a_307_ = v_b_300_;
goto v___jp_306_;
}
else
{
goto v___jp_313_;
}
}
else
{
lean_object* v_a_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_327_; 
lean_dec_ref(v_b_300_);
v_a_320_ = lean_ctor_get(v___x_315_, 0);
v_isSharedCheck_327_ = !lean_is_exclusive(v___x_315_);
if (v_isSharedCheck_327_ == 0)
{
v___x_322_ = v___x_315_;
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_a_320_);
lean_dec(v___x_315_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_325_; 
if (v_isShared_323_ == 0)
{
v___x_325_ = v___x_322_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_a_320_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
}
v___jp_313_:
{
lean_object* v___x_314_; 
lean_inc(v___x_312_);
v___x_314_ = lean_array_push(v_b_300_, v___x_312_);
v_a_307_ = v___x_314_;
goto v___jp_306_;
}
}
else
{
lean_object* v___x_328_; 
v___x_328_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_328_, 0, v_b_300_);
return v___x_328_;
}
v___jp_306_:
{
size_t v___x_308_; size_t v___x_309_; 
v___x_308_ = ((size_t)1ULL);
v___x_309_ = lean_usize_add(v_i_298_, v___x_308_);
v_i_298_ = v___x_309_;
v_b_300_ = v_a_307_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2___boxed(lean_object* v_as_329_, lean_object* v_i_330_, lean_object* v_stop_331_, lean_object* v_b_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_){
_start:
{
size_t v_i_boxed_338_; size_t v_stop_boxed_339_; lean_object* v_res_340_; 
v_i_boxed_338_ = lean_unbox_usize(v_i_330_);
lean_dec(v_i_330_);
v_stop_boxed_339_ = lean_unbox_usize(v_stop_331_);
lean_dec(v_stop_331_);
v_res_340_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2(v_as_329_, v_i_boxed_338_, v_stop_boxed_339_, v_b_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
lean_dec_ref(v_as_329_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0(lean_object* v___x_341_, lean_object* v___x_342_, lean_object* v_a_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v___x_349_; uint8_t v___x_350_; 
v___x_349_ = lean_mk_empty_array_with_capacity(v___x_341_);
v___x_350_ = lean_nat_dec_lt(v___x_341_, v___x_342_);
if (v___x_350_ == 0)
{
lean_object* v___x_351_; 
v___x_351_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_351_, 0, v___x_349_);
return v___x_351_;
}
else
{
uint8_t v___x_352_; 
v___x_352_ = lean_nat_dec_le(v___x_342_, v___x_342_);
if (v___x_352_ == 0)
{
if (v___x_350_ == 0)
{
lean_object* v___x_353_; 
v___x_353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_353_, 0, v___x_349_);
return v___x_353_;
}
else
{
size_t v___x_354_; size_t v___x_355_; lean_object* v___x_356_; 
v___x_354_ = ((size_t)0ULL);
v___x_355_ = lean_usize_of_nat(v___x_342_);
v___x_356_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2(v_a_343_, v___x_354_, v___x_355_, v___x_349_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
return v___x_356_;
}
}
else
{
size_t v___x_357_; size_t v___x_358_; lean_object* v___x_359_; 
v___x_357_ = ((size_t)0ULL);
v___x_358_ = lean_usize_of_nat(v___x_342_);
v___x_359_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getIntroducedExprMVars_spec__2(v_a_343_, v___x_357_, v___x_358_, v___x_349_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
return v___x_359_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0___boxed(lean_object* v___x_360_, lean_object* v___x_361_, lean_object* v_a_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0(v___x_360_, v___x_361_, v_a_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v___y_364_);
lean_dec_ref(v___y_363_);
lean_dec_ref(v_a_362_);
lean_dec(v___x_361_);
lean_dec(v___x_360_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars(lean_object* v_preState_372_, lean_object* v_postState_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = ((lean_object*)(lp_batteries_Lean_Meta_getIntroducedExprMVars___closed__0));
v___x_380_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_373_, v___x_379_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_380_) == 0)
{
lean_object* v_a_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___f_384_; lean_object* v___x_385_; 
v_a_381_ = lean_ctor_get(v___x_380_, 0);
lean_inc(v_a_381_);
lean_dec_ref_known(v___x_380_, 1);
v___x_382_ = lean_unsigned_to_nat(0u);
v___x_383_ = lean_array_get_size(v_a_381_);
v___f_384_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_getIntroducedExprMVars___lam__0___boxed), 8, 3);
lean_closure_set(v___f_384_, 0, v___x_382_);
lean_closure_set(v___f_384_, 1, v___x_383_);
lean_closure_set(v___f_384_, 2, v_a_381_);
v___x_385_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_preState_372_, v___f_384_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
return v___x_385_;
}
else
{
lean_dec_ref(v_preState_372_);
return v___x_380_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getIntroducedExprMVars___boxed(lean_object* v_preState_386_, lean_object* v_postState_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_batteries_Lean_Meta_getIntroducedExprMVars(v_preState_386_, v_postState_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_);
lean_dec(v_a_391_);
lean_dec_ref(v_a_390_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
return v_res_393_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_keys_394_, lean_object* v_i_395_, lean_object* v_k_396_){
_start:
{
lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_397_ = lean_array_get_size(v_keys_394_);
v___x_398_ = lean_nat_dec_lt(v_i_395_, v___x_397_);
if (v___x_398_ == 0)
{
lean_dec(v_i_395_);
return v___x_398_;
}
else
{
lean_object* v_k_x27_399_; uint8_t v___x_400_; 
v_k_x27_399_ = lean_array_fget_borrowed(v_keys_394_, v_i_395_);
v___x_400_ = l_Lean_instBEqMVarId_beq(v_k_396_, v_k_x27_399_);
if (v___x_400_ == 0)
{
lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_401_ = lean_unsigned_to_nat(1u);
v___x_402_ = lean_nat_add(v_i_395_, v___x_401_);
lean_dec(v_i_395_);
v_i_395_ = v___x_402_;
goto _start;
}
else
{
lean_dec(v_i_395_);
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_keys_404_, lean_object* v_i_405_, lean_object* v_k_406_){
_start:
{
uint8_t v_res_407_; lean_object* v_r_408_; 
v_res_407_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_keys_404_, v_i_405_, v_k_406_);
lean_dec(v_k_406_);
lean_dec_ref(v_keys_404_);
v_r_408_ = lean_box(v_res_407_);
return v_r_408_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg(lean_object* v_x_409_, size_t v_x_410_, lean_object* v_x_411_){
_start:
{
if (lean_obj_tag(v_x_409_) == 0)
{
lean_object* v_es_412_; lean_object* v___x_413_; size_t v___x_414_; size_t v___x_415_; lean_object* v_j_416_; lean_object* v___x_417_; 
v_es_412_ = lean_ctor_get(v_x_409_, 0);
v___x_413_ = lean_box(2);
v___x_414_ = ((size_t)31ULL);
v___x_415_ = lean_usize_land(v_x_410_, v___x_414_);
v_j_416_ = lean_usize_to_nat(v___x_415_);
v___x_417_ = lean_array_get_borrowed(v___x_413_, v_es_412_, v_j_416_);
lean_dec(v_j_416_);
switch(lean_obj_tag(v___x_417_))
{
case 0:
{
lean_object* v_key_418_; uint8_t v___x_419_; 
v_key_418_ = lean_ctor_get(v___x_417_, 0);
v___x_419_ = l_Lean_instBEqMVarId_beq(v_x_411_, v_key_418_);
return v___x_419_;
}
case 1:
{
lean_object* v_node_420_; size_t v___x_421_; size_t v___x_422_; 
v_node_420_ = lean_ctor_get(v___x_417_, 0);
v___x_421_ = ((size_t)5ULL);
v___x_422_ = lean_usize_shift_right(v_x_410_, v___x_421_);
v_x_409_ = v_node_420_;
v_x_410_ = v___x_422_;
goto _start;
}
default: 
{
uint8_t v___x_424_; 
v___x_424_ = 0;
return v___x_424_;
}
}
}
else
{
lean_object* v_ks_425_; lean_object* v___x_426_; uint8_t v___x_427_; 
v_ks_425_ = lean_ctor_get(v_x_409_, 0);
v___x_426_ = lean_unsigned_to_nat(0u);
v___x_427_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_ks_425_, v___x_426_, v_x_411_);
return v___x_427_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_428_, lean_object* v_x_429_, lean_object* v_x_430_){
_start:
{
size_t v_x_997__boxed_431_; uint8_t v_res_432_; lean_object* v_r_433_; 
v_x_997__boxed_431_ = lean_unbox_usize(v_x_429_);
lean_dec(v_x_429_);
v_res_432_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg(v_x_428_, v_x_997__boxed_431_, v_x_430_);
lean_dec(v_x_430_);
lean_dec_ref(v_x_428_);
v_r_433_ = lean_box(v_res_432_);
return v_r_433_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(lean_object* v_x_434_, lean_object* v_x_435_){
_start:
{
uint64_t v___x_436_; size_t v___x_437_; uint8_t v___x_438_; 
v___x_436_ = l_Lean_instHashableMVarId_hash(v_x_435_);
v___x_437_ = lean_uint64_to_usize(v___x_436_);
v___x_438_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg(v_x_434_, v___x_437_, v_x_435_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg___boxed(lean_object* v_x_439_, lean_object* v_x_440_){
_start:
{
uint8_t v_res_441_; lean_object* v_r_442_; 
v_res_441_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(v_x_439_, v_x_440_);
lean_dec(v_x_440_);
lean_dec_ref(v_x_439_);
v_r_442_ = lean_box(v_res_441_);
return v_r_442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg(lean_object* v_mvarId_443_, lean_object* v___y_444_){
_start:
{
lean_object* v___x_446_; lean_object* v_mctx_447_; lean_object* v_eAssignment_448_; lean_object* v_dAssignment_449_; uint8_t v___x_450_; 
v___x_446_ = lean_st_ref_get(v___y_444_);
v_mctx_447_ = lean_ctor_get(v___x_446_, 0);
lean_inc_ref(v_mctx_447_);
lean_dec(v___x_446_);
v_eAssignment_448_ = lean_ctor_get(v_mctx_447_, 8);
lean_inc_ref(v_eAssignment_448_);
v_dAssignment_449_ = lean_ctor_get(v_mctx_447_, 9);
lean_inc_ref(v_dAssignment_449_);
lean_dec_ref(v_mctx_447_);
v___x_450_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(v_eAssignment_448_, v_mvarId_443_);
lean_dec_ref(v_eAssignment_448_);
if (v___x_450_ == 0)
{
uint8_t v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; 
v___x_451_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(v_dAssignment_449_, v_mvarId_443_);
lean_dec_ref(v_dAssignment_449_);
v___x_452_ = lean_box(v___x_451_);
v___x_453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_453_, 0, v___x_452_);
return v___x_453_;
}
else
{
lean_object* v___x_454_; lean_object* v___x_455_; 
lean_dec_ref(v_dAssignment_449_);
v___x_454_ = lean_box(v___x_450_);
v___x_455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
return v___x_455_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg___boxed(lean_object* v_mvarId_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg(v_mvarId_456_, v___y_457_);
lean_dec(v___y_457_);
lean_dec(v_mvarId_456_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1(lean_object* v_as_460_, size_t v_i_461_, size_t v_stop_462_, lean_object* v_b_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_){
_start:
{
uint8_t v___x_469_; 
v___x_469_ = lean_usize_dec_eq(v_i_461_, v_stop_462_);
if (v___x_469_ == 0)
{
lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_470_ = lean_array_uget_borrowed(v_as_460_, v_i_461_);
v___x_471_ = lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg(v___x_470_, v___y_465_);
if (lean_obj_tag(v___x_471_) == 0)
{
lean_object* v_a_472_; lean_object* v_a_474_; uint8_t v___x_478_; 
v_a_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc(v_a_472_);
lean_dec_ref_known(v___x_471_, 1);
v___x_478_ = lean_unbox(v_a_472_);
lean_dec(v_a_472_);
if (v___x_478_ == 0)
{
v_a_474_ = v_b_463_;
goto v___jp_473_;
}
else
{
lean_object* v___x_479_; 
lean_inc(v___x_470_);
v___x_479_ = lean_array_push(v_b_463_, v___x_470_);
v_a_474_ = v___x_479_;
goto v___jp_473_;
}
v___jp_473_:
{
size_t v___x_475_; size_t v___x_476_; 
v___x_475_ = ((size_t)1ULL);
v___x_476_ = lean_usize_add(v_i_461_, v___x_475_);
v_i_461_ = v___x_476_;
v_b_463_ = v_a_474_;
goto _start;
}
}
else
{
lean_object* v_a_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_487_; 
lean_dec_ref(v_b_463_);
v_a_480_ = lean_ctor_get(v___x_471_, 0);
v_isSharedCheck_487_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_487_ == 0)
{
v___x_482_ = v___x_471_;
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_a_480_);
lean_dec(v___x_471_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
lean_object* v___x_485_; 
if (v_isShared_483_ == 0)
{
v___x_485_ = v___x_482_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v_a_480_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
else
{
lean_object* v___x_488_; 
v___x_488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_488_, 0, v_b_463_);
return v___x_488_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1___boxed(lean_object* v_as_489_, lean_object* v_i_490_, lean_object* v_stop_491_, lean_object* v_b_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_){
_start:
{
size_t v_i_boxed_498_; size_t v_stop_boxed_499_; lean_object* v_res_500_; 
v_i_boxed_498_ = lean_unbox_usize(v_i_490_);
lean_dec(v_i_490_);
v_stop_boxed_499_ = lean_unbox_usize(v_stop_491_);
lean_dec(v_stop_491_);
v_res_500_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1(v_as_489_, v_i_boxed_498_, v_stop_boxed_499_, v_b_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
lean_dec_ref(v_as_489_);
return v_res_500_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0(lean_object* v___x_501_, lean_object* v___x_502_, lean_object* v_a_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v___x_509_; uint8_t v___x_510_; 
v___x_509_ = lean_mk_empty_array_with_capacity(v___x_501_);
v___x_510_ = lean_nat_dec_lt(v___x_501_, v___x_502_);
if (v___x_510_ == 0)
{
lean_object* v___x_511_; 
v___x_511_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_511_, 0, v___x_509_);
return v___x_511_;
}
else
{
uint8_t v___x_512_; 
v___x_512_ = lean_nat_dec_le(v___x_502_, v___x_502_);
if (v___x_512_ == 0)
{
if (v___x_510_ == 0)
{
lean_object* v___x_513_; 
v___x_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_513_, 0, v___x_509_);
return v___x_513_;
}
else
{
size_t v___x_514_; size_t v___x_515_; lean_object* v___x_516_; 
v___x_514_ = ((size_t)0ULL);
v___x_515_ = lean_usize_of_nat(v___x_502_);
v___x_516_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1(v_a_503_, v___x_514_, v___x_515_, v___x_509_, v___y_504_, v___y_505_, v___y_506_, v___y_507_);
return v___x_516_;
}
}
else
{
size_t v___x_517_; size_t v___x_518_; lean_object* v___x_519_; 
v___x_517_ = ((size_t)0ULL);
v___x_518_ = lean_usize_of_nat(v___x_502_);
v___x_519_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_getAssignedExprMVars_spec__1(v_a_503_, v___x_517_, v___x_518_, v___x_509_, v___y_504_, v___y_505_, v___y_506_, v___y_507_);
return v___x_519_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0___boxed(lean_object* v___x_520_, lean_object* v___x_521_, lean_object* v_a_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0(v___x_520_, v___x_521_, v_a_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
lean_dec(v___y_524_);
lean_dec_ref(v___y_523_);
lean_dec_ref(v_a_522_);
lean_dec(v___x_521_);
lean_dec(v___x_520_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars(lean_object* v_preState_529_, lean_object* v_postState_530_, lean_object* v_a_531_, lean_object* v_a_532_, lean_object* v_a_533_, lean_object* v_a_534_){
_start:
{
lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_536_ = ((lean_object*)(lp_batteries_Lean_Meta_getIntroducedExprMVars___closed__0));
v___x_537_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_preState_529_, v___x_536_, v_a_531_, v_a_532_, v_a_533_, v_a_534_);
if (lean_obj_tag(v___x_537_) == 0)
{
lean_object* v_a_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___f_541_; lean_object* v___x_542_; 
v_a_538_ = lean_ctor_get(v___x_537_, 0);
lean_inc(v_a_538_);
lean_dec_ref_known(v___x_537_, 1);
v___x_539_ = lean_unsigned_to_nat(0u);
v___x_540_ = lean_array_get_size(v_a_538_);
v___f_541_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_getAssignedExprMVars___lam__0___boxed), 8, 3);
lean_closure_set(v___f_541_, 0, v___x_539_);
lean_closure_set(v___f_541_, 1, v___x_540_);
lean_closure_set(v___f_541_, 2, v_a_538_);
v___x_542_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_postState_530_, v___f_541_, v_a_531_, v_a_532_, v_a_533_, v_a_534_);
return v___x_542_;
}
else
{
lean_dec_ref(v_postState_530_);
return v___x_537_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_getAssignedExprMVars___boxed(lean_object* v_preState_543_, lean_object* v_postState_544_, lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_, lean_object* v_a_548_, lean_object* v_a_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_batteries_Lean_Meta_getAssignedExprMVars(v_preState_543_, v_postState_544_, v_a_545_, v_a_546_, v_a_547_, v_a_548_);
lean_dec(v_a_548_);
lean_dec_ref(v_a_547_);
lean_dec(v_a_546_);
lean_dec_ref(v_a_545_);
return v_res_550_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0(lean_object* v_mvarId_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___redArg(v_mvarId_551_, v___y_553_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0___boxed(lean_object* v_mvarId_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_batteries_Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0(v_mvarId_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_);
lean_dec(v___y_562_);
lean_dec_ref(v___y_561_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
lean_dec(v_mvarId_558_);
return v_res_564_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0(lean_object* v_00_u03b2_565_, lean_object* v_x_566_, lean_object* v_x_567_){
_start:
{
uint8_t v___x_568_; 
v___x_568_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___redArg(v_x_566_, v_x_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0___boxed(lean_object* v_00_u03b2_569_, lean_object* v_x_570_, lean_object* v_x_571_){
_start:
{
uint8_t v_res_572_; lean_object* v_r_573_; 
v_res_572_ = lp_batteries_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0(v_00_u03b2_569_, v_x_570_, v_x_571_);
lean_dec(v_x_571_);
lean_dec_ref(v_x_570_);
v_r_573_ = lean_box(v_res_572_);
return v_r_573_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_574_, lean_object* v_x_575_, size_t v_x_576_, lean_object* v_x_577_){
_start:
{
uint8_t v___x_578_; 
v___x_578_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___redArg(v_x_575_, v_x_576_, v_x_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_579_, lean_object* v_x_580_, lean_object* v_x_581_, lean_object* v_x_582_){
_start:
{
size_t v_x_1217__boxed_583_; uint8_t v_res_584_; lean_object* v_r_585_; 
v_x_1217__boxed_583_ = lean_unbox_usize(v_x_581_);
lean_dec(v_x_581_);
v_res_584_ = lp_batteries_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1(v_00_u03b2_579_, v_x_580_, v_x_1217__boxed_583_, v_x_582_);
lean_dec(v_x_582_);
lean_dec_ref(v_x_580_);
v_r_585_ = lean_box(v_res_584_);
return v_r_585_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_586_, lean_object* v_keys_587_, lean_object* v_vals_588_, lean_object* v_heq_589_, lean_object* v_i_590_, lean_object* v_k_591_){
_start:
{
uint8_t v___x_592_; 
v___x_592_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___redArg(v_keys_587_, v_i_590_, v_k_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b2_593_, lean_object* v_keys_594_, lean_object* v_vals_595_, lean_object* v_heq_596_, lean_object* v_i_597_, lean_object* v_k_598_){
_start:
{
uint8_t v_res_599_; lean_object* v_r_600_; 
v_res_599_ = lp_batteries_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00Lean_Meta_getAssignedExprMVars_spec__0_spec__0_spec__1_spec__3(v_00_u03b2_593_, v_keys_594_, v_vals_595_, v_heq_596_, v_i_597_, v_k_598_);
lean_dec(v_k_598_);
lean_dec_ref(v_vals_595_);
lean_dec_ref(v_keys_594_);
v_r_600_ = lean_box(v_res_599_);
return v_r_600_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_MonadBacktrack(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_MonadBacktrack(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
}
#ifdef __cplusplus
}
#endif
