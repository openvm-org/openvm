// Lean compiler output
// Module: Aesop.Script.Util
// Imports: public import Init public meta import Init public import Aesop.Util.Basic import Batteries.Lean.Meta.SavedState import Aesop.Util.EqualUpToIds
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
lean_object* lp_aesop_Aesop_partitionGoalsAndMVars___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_tacticStatesEqualUpToIds_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__0_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__2_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_matchGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_matchGoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0(lean_object* v_goals_1_, lean_object* v_step_x3f_2_, lean_object* v_stepOrder_3_, lean_object* v_pos_4_, lean_object* v_h_5_, lean_object* v_____s_6_){
_start:
{
lean_object* v_g_7_; lean_object* v___x_8_; 
v_g_7_ = lean_array_fget_borrowed(v_goals_1_, v_pos_4_);
lean_inc(v_g_7_);
v___x_8_ = lean_apply_1(v_step_x3f_2_, v_g_7_);
if (lean_obj_tag(v___x_8_) == 1)
{
if (lean_obj_tag(v_____s_6_) == 1)
{
lean_object* v_val_9_; lean_object* v_snd_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_40_; 
v_val_9_ = lean_ctor_get(v_____s_6_, 0);
lean_inc(v_val_9_);
v_snd_10_ = lean_ctor_get(v_val_9_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v_val_9_);
if (v_isSharedCheck_40_ == 0)
{
lean_object* v_unused_41_; 
v_unused_41_ = lean_ctor_get(v_val_9_, 0);
lean_dec(v_unused_41_);
v___x_12_ = v_val_9_;
v_isShared_13_ = v_isSharedCheck_40_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_snd_10_);
lean_dec(v_val_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_40_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v_val_14_; lean_object* v_snd_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_38_; 
v_val_14_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_val_14_);
lean_dec_ref_known(v___x_8_, 1);
v_snd_15_ = lean_ctor_get(v_snd_10_, 1);
v_isSharedCheck_38_ = !lean_is_exclusive(v_snd_10_);
if (v_isSharedCheck_38_ == 0)
{
lean_object* v_unused_39_; 
v_unused_39_ = lean_ctor_get(v_snd_10_, 0);
lean_dec(v_unused_39_);
v___x_17_ = v_snd_10_;
v_isShared_18_ = v_isSharedCheck_38_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_snd_15_);
lean_dec(v_snd_10_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_38_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v___x_20_; uint8_t v___x_21_; 
lean_inc_ref(v_stepOrder_3_);
lean_inc(v_val_14_);
v___x_19_ = lean_apply_1(v_stepOrder_3_, v_val_14_);
v___x_20_ = lean_apply_1(v_stepOrder_3_, v_snd_15_);
v___x_21_ = lean_nat_dec_lt(v___x_19_, v___x_20_);
lean_dec(v___x_20_);
lean_dec(v___x_19_);
if (v___x_21_ == 0)
{
lean_object* v___x_22_; 
lean_del_object(v___x_17_);
lean_dec(v_val_14_);
lean_del_object(v___x_12_);
lean_dec(v_pos_4_);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v_____s_6_);
return v___x_22_;
}
else
{
lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_36_; 
v_isSharedCheck_36_ = !lean_is_exclusive(v_____s_6_);
if (v_isSharedCheck_36_ == 0)
{
lean_object* v_unused_37_; 
v_unused_37_ = lean_ctor_get(v_____s_6_, 0);
lean_dec(v_unused_37_);
v___x_24_ = v_____s_6_;
v_isShared_25_ = v_isSharedCheck_36_;
goto v_resetjp_23_;
}
else
{
lean_dec(v_____s_6_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_36_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_27_; 
lean_inc(v_g_7_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 1, v_val_14_);
lean_ctor_set(v___x_17_, 0, v_g_7_);
v___x_27_ = v___x_17_;
goto v_reusejp_26_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v_g_7_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v_val_14_);
v___x_27_ = v_reuseFailAlloc_35_;
goto v_reusejp_26_;
}
v_reusejp_26_:
{
lean_object* v___x_29_; 
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 1, v___x_27_);
lean_ctor_set(v___x_12_, 0, v_pos_4_);
v___x_29_ = v___x_12_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v_pos_4_);
lean_ctor_set(v_reuseFailAlloc_34_, 1, v___x_27_);
v___x_29_ = v_reuseFailAlloc_34_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
lean_object* v_firstStep_x3f_31_; 
if (v_isShared_25_ == 0)
{
lean_ctor_set(v___x_24_, 0, v___x_29_);
v_firstStep_x3f_31_ = v___x_24_;
goto v_reusejp_30_;
}
else
{
lean_object* v_reuseFailAlloc_33_; 
v_reuseFailAlloc_33_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_33_, 0, v___x_29_);
v_firstStep_x3f_31_ = v_reuseFailAlloc_33_;
goto v_reusejp_30_;
}
v_reusejp_30_:
{
lean_object* v___x_32_; 
v___x_32_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_32_, 0, v_firstStep_x3f_31_);
return v___x_32_;
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
lean_object* v_val_42_; lean_object* v___x_44_; uint8_t v_isShared_45_; uint8_t v_isSharedCheck_52_; 
lean_dec(v_____s_6_);
lean_dec_ref(v_stepOrder_3_);
v_val_42_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_52_ == 0)
{
v___x_44_ = v___x_8_;
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
else
{
lean_inc(v_val_42_);
lean_dec(v___x_8_);
v___x_44_ = lean_box(0);
v_isShared_45_ = v_isSharedCheck_52_;
goto v_resetjp_43_;
}
v_resetjp_43_:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v_firstStep_x3f_49_; 
lean_inc(v_g_7_);
v___x_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_46_, 0, v_g_7_);
lean_ctor_set(v___x_46_, 1, v_val_42_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v_pos_4_);
lean_ctor_set(v___x_47_, 1, v___x_46_);
if (v_isShared_45_ == 0)
{
lean_ctor_set(v___x_44_, 0, v___x_47_);
v_firstStep_x3f_49_ = v___x_44_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_47_);
v_firstStep_x3f_49_ = v_reuseFailAlloc_51_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
lean_object* v___x_50_; 
v___x_50_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_50_, 0, v_firstStep_x3f_49_);
return v___x_50_;
}
}
}
}
else
{
lean_object* v___x_53_; 
lean_dec(v___x_8_);
lean_dec(v_pos_4_);
lean_dec_ref(v_stepOrder_3_);
v___x_53_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_53_, 0, v_____s_6_);
return v___x_53_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0___boxed(lean_object* v_goals_54_, lean_object* v_step_x3f_55_, lean_object* v_stepOrder_56_, lean_object* v_pos_57_, lean_object* v_h_58_, lean_object* v_____s_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0(v_goals_54_, v_step_x3f_55_, v_stepOrder_56_, v_pos_57_, v_h_58_, v_____s_59_);
lean_dec_ref(v_goals_54_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f___redArg(lean_object* v_goals_80_, lean_object* v_step_x3f_81_, lean_object* v_stepOrder_82_){
_start:
{
lean_object* v___f_83_; lean_object* v___x_84_; lean_object* v_firstStep_x3f_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
lean_inc_ref(v_goals_80_);
v___f_83_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___lam__0___boxed), 6, 3);
lean_closure_set(v___f_83_, 0, v_goals_80_);
lean_closure_set(v___f_83_, 1, v_step_x3f_81_);
lean_closure_set(v___f_83_, 2, v_stepOrder_82_);
v___x_84_ = ((lean_object*)(lp_aesop_Aesop_Script_findFirstStep_x3f___redArg___closed__9));
v_firstStep_x3f_85_ = lean_box(0);
v___x_86_ = lean_unsigned_to_nat(0u);
v___x_87_ = lean_array_get_size(v_goals_80_);
lean_dec_ref(v_goals_80_);
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_89_, 0, v___x_86_);
lean_ctor_set(v___x_89_, 1, v___x_87_);
lean_ctor_set(v___x_89_, 2, v___x_88_);
v___x_90_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v___x_84_, v___x_89_, v___f_83_, v_firstStep_x3f_85_, v___x_86_, lean_box(0), lean_box(0));
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_findFirstStep_x3f(lean_object* v_00_u03b1_91_, lean_object* v_00_u03b2_92_, lean_object* v_goals_93_, lean_object* v_step_x3f_94_, lean_object* v_stepOrder_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_aesop_Aesop_Script_findFirstStep_x3f___redArg(v_goals_93_, v_step_x3f_94_, v_stepOrder_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0(lean_object* v___y_97_){
_start:
{
lean_inc(v___y_97_);
return v___y_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0___boxed(lean_object* v___y_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__0(v___y_98_);
lean_dec(v___y_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0(size_t v_sz_100_, size_t v_i_101_, lean_object* v_bs_102_){
_start:
{
uint8_t v___x_103_; 
v___x_103_ = lean_usize_dec_lt(v_i_101_, v_sz_100_);
if (v___x_103_ == 0)
{
return v_bs_102_;
}
else
{
lean_object* v_v_104_; lean_object* v_fst_105_; lean_object* v___x_106_; lean_object* v_bs_x27_107_; size_t v___x_108_; size_t v___x_109_; lean_object* v___x_110_; 
v_v_104_ = lean_array_uget_borrowed(v_bs_102_, v_i_101_);
v_fst_105_ = lean_ctor_get(v_v_104_, 0);
lean_inc(v_fst_105_);
v___x_106_ = lean_unsigned_to_nat(0u);
v_bs_x27_107_ = lean_array_uset(v_bs_102_, v_i_101_, v___x_106_);
v___x_108_ = ((size_t)1ULL);
v___x_109_ = lean_usize_add(v_i_101_, v___x_108_);
v___x_110_ = lean_array_uset(v_bs_x27_107_, v_i_101_, v_fst_105_);
v_i_101_ = v___x_109_;
v_bs_102_ = v___x_110_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0___boxed(lean_object* v_sz_112_, lean_object* v_i_113_, lean_object* v_bs_114_){
_start:
{
size_t v_sz_boxed_115_; size_t v_i_boxed_116_; lean_object* v_res_117_; 
v_sz_boxed_115_ = lean_unbox_usize(v_sz_112_);
lean_dec(v_sz_112_);
v_i_boxed_116_ = lean_unbox_usize(v_i_113_);
lean_dec(v_i_113_);
v_res_117_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0(v_sz_boxed_115_, v_i_boxed_116_, v_bs_114_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1(lean_object* v___f_118_, lean_object* v_goals_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_aesop_Aesop_partitionGoalsAndMVars___redArg(v___f_118_, v_goals_119_, v___y_120_, v___y_121_, v___y_122_, v___y_123_);
if (lean_obj_tag(v___x_125_) == 0)
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_137_; 
v_a_126_ = lean_ctor_get(v___x_125_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_125_);
if (v_isSharedCheck_137_ == 0)
{
v___x_128_ = v___x_125_;
v_isShared_129_ = v_isSharedCheck_137_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_125_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_137_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v_fst_130_; size_t v_sz_131_; size_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_135_; 
v_fst_130_ = lean_ctor_get(v_a_126_, 0);
lean_inc(v_fst_130_);
lean_dec(v_a_126_);
v_sz_131_ = lean_array_size(v_fst_130_);
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals_spec__0(v_sz_131_, v___x_132_, v_fst_130_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 0, v___x_133_);
v___x_135_ = v___x_128_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_133_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_a_138_ = lean_ctor_get(v___x_125_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_125_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_125_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_125_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1___boxed(lean_object* v___f_146_, lean_object* v_goals_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1(v___f_146_, v_goals_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec_ref(v_goals_147_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals(lean_object* v_state_155_, lean_object* v_goals_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v___f_162_; lean_object* v___f_163_; lean_object* v___x_164_; 
v___f_162_ = ((lean_object*)(lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___closed__0));
v___f_163_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___lam__1___boxed), 7, 2);
lean_closure_set(v___f_163_, 0, v___f_162_);
lean_closure_set(v___f_163_, 1, v_goals_156_);
v___x_164_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_state_155_, v___f_163_, v_a_157_, v_a_158_, v_a_159_, v_a_160_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals___boxed(lean_object* v_state_165_, lean_object* v_goals_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals(v_state_165_, v_goals_166_, v_a_167_, v_a_168_, v_a_169_, v_a_170_);
lean_dec(v_a_170_);
lean_dec_ref(v_a_169_);
lean_dec(v_a_168_);
lean_dec_ref(v_a_167_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_matchGoals(lean_object* v_postState_u2081_173_, lean_object* v_postState_u2082_174_, lean_object* v_goals_u2081_175_, lean_object* v_goals_u2082_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_){
_start:
{
lean_object* v___x_182_; 
lean_inc_ref(v_postState_u2081_173_);
v___x_182_ = lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals(v_postState_u2081_173_, v_goals_u2081_175_, v_a_177_, v_a_178_, v_a_179_, v_a_180_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v_a_183_; lean_object* v___x_184_; 
v_a_183_ = lean_ctor_get(v___x_182_, 0);
lean_inc(v_a_183_);
lean_dec_ref_known(v___x_182_, 1);
lean_inc_ref(v_postState_u2082_174_);
v___x_184_ = lp_aesop___private_Aesop_Script_Util_0__Aesop_Script_matchGoals_getProperGoals(v_postState_u2082_174_, v_goals_u2082_176_, v_a_177_, v_a_178_, v_a_179_, v_a_180_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_meta_185_; lean_object* v_meta_186_; lean_object* v_a_187_; lean_object* v_mctx_188_; lean_object* v_mctx_189_; lean_object* v___x_190_; uint8_t v___x_191_; lean_object* v___x_192_; 
v_meta_185_ = lean_ctor_get(v_postState_u2081_173_, 1);
lean_inc_ref(v_meta_185_);
lean_dec_ref(v_postState_u2081_173_);
v_meta_186_ = lean_ctor_get(v_postState_u2082_174_, 1);
lean_inc_ref(v_meta_186_);
lean_dec_ref(v_postState_u2082_174_);
v_a_187_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_187_);
lean_dec_ref_known(v___x_184_, 1);
v_mctx_188_ = lean_ctor_get(v_meta_185_, 0);
lean_inc_ref(v_mctx_188_);
lean_dec_ref(v_meta_185_);
v_mctx_189_ = lean_ctor_get(v_meta_186_, 0);
lean_inc_ref(v_mctx_189_);
lean_dec_ref(v_meta_186_);
v___x_190_ = lean_box(0);
v___x_191_ = 1;
v___x_192_ = lp_aesop_Aesop_tacticStatesEqualUpToIds_x27(v___x_190_, v_mctx_188_, v_mctx_189_, v_a_183_, v_a_187_, v___x_191_, v_a_177_, v_a_178_, v_a_179_, v_a_180_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_208_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_208_ == 0)
{
v___x_195_ = v___x_192_;
v_isShared_196_ = v_isSharedCheck_208_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_192_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_208_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v_fst_197_; uint8_t v___x_198_; 
v_fst_197_ = lean_ctor_get(v_a_193_, 0);
v___x_198_ = lean_unbox(v_fst_197_);
if (v___x_198_ == 0)
{
lean_object* v___x_200_; 
lean_dec(v_a_193_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 0, v___x_190_);
v___x_200_ = v___x_195_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v___x_190_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
else
{
lean_object* v_snd_202_; lean_object* v_equalMVarIds_203_; lean_object* v___x_204_; lean_object* v___x_206_; 
v_snd_202_ = lean_ctor_get(v_a_193_, 1);
lean_inc(v_snd_202_);
lean_dec(v_a_193_);
v_equalMVarIds_203_ = lean_ctor_get(v_snd_202_, 0);
lean_inc_ref(v_equalMVarIds_203_);
lean_dec(v_snd_202_);
v___x_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_204_, 0, v_equalMVarIds_203_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 0, v___x_204_);
v___x_206_ = v___x_195_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_204_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
v_a_209_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_192_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_192_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_a_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
else
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
lean_dec(v_a_183_);
lean_dec_ref(v_postState_u2082_174_);
lean_dec_ref(v_postState_u2081_173_);
v_a_217_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_184_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_184_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
else
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_232_; 
lean_dec_ref(v_goals_u2082_176_);
lean_dec_ref(v_postState_u2082_174_);
lean_dec_ref(v_postState_u2081_173_);
v_a_225_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_232_ == 0)
{
v___x_227_ = v___x_182_;
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_182_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_230_; 
if (v_isShared_228_ == 0)
{
v___x_230_ = v___x_227_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_225_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_matchGoals___boxed(lean_object* v_postState_u2081_233_, lean_object* v_postState_u2082_234_, lean_object* v_goals_u2081_235_, lean_object* v_goals_u2082_236_, lean_object* v_a_237_, lean_object* v_a_238_, lean_object* v_a_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_aesop_Aesop_Script_matchGoals(v_postState_u2081_233_, v_postState_u2082_234_, v_goals_u2081_235_, v_goals_u2082_236_, v_a_237_, v_a_238_, v_a_239_, v_a_240_);
lean_dec(v_a_240_);
lean_dec_ref(v_a_239_);
lean_dec(v_a_238_);
lean_dec_ref(v_a_237_);
return v_res_242_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_Util(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_Util(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_EqualUpToIds(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_Util(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_EqualUpToIds(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_Util(builtin);
}
#ifdef __cplusplus
}
#endif
