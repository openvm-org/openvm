// Lean compiler output
// Module: Mathlib.Tactic.GuardGoalNums
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Basic
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
static const lean_string_object lp_mathlib_guardGoalNums___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "guardGoalNums"};
static const lean_object* lp_mathlib_guardGoalNums___closed__0 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__0_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_guardGoalNums___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 159, 166, 158, 60, 122, 40, 126)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__1 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__1_value;
static const lean_string_object lp_mathlib_guardGoalNums___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_guardGoalNums___closed__2 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__2_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_guardGoalNums___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__3 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__3_value;
static const lean_string_object lp_mathlib_guardGoalNums___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "guard_goal_nums "};
static const lean_object* lp_mathlib_guardGoalNums___closed__4 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__4_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_guardGoalNums___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__5 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__5_value;
static const lean_string_object lp_mathlib_guardGoalNums___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_guardGoalNums___closed__6 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__6_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_guardGoalNums___closed__6_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__7 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__7_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_guardGoalNums___closed__7_value)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__8 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__8_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_guardGoalNums___closed__3_value),((lean_object*)&lp_mathlib_guardGoalNums___closed__5_value),((lean_object*)&lp_mathlib_guardGoalNums___closed__8_value)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__9 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__9_value;
static const lean_ctor_object lp_mathlib_guardGoalNums___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_guardGoalNums___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_guardGoalNums___closed__9_value)}};
static const lean_object* lp_mathlib_guardGoalNums___closed__10 = (const lean_object*)&lp_mathlib_guardGoalNums___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_guardGoalNums = (const lean_object*)&lp_mathlib_guardGoalNums___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "expected "};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " goals but found "};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__2_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_27_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v___x_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg(){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___closed__0);
v___x_30_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg___boxed(lean_object* v___y_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg();
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0(lean_object* v_00_u03b1_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg();
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___boxed(lean_object* v_00_u03b1_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0(v_00_u03b1_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
lean_dec(v___y_46_);
lean_dec_ref(v___y_45_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1(lean_object* v_msgData_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
lean_object* v___x_61_; lean_object* v_env_62_; lean_object* v___x_63_; lean_object* v_mctx_64_; lean_object* v_lctx_65_; lean_object* v_options_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_61_ = lean_st_ref_get(v___y_59_);
v_env_62_ = lean_ctor_get(v___x_61_, 0);
lean_inc_ref(v_env_62_);
lean_dec(v___x_61_);
v___x_63_ = lean_st_ref_get(v___y_57_);
v_mctx_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc_ref(v_mctx_64_);
lean_dec(v___x_63_);
v_lctx_65_ = lean_ctor_get(v___y_56_, 2);
v_options_66_ = lean_ctor_get(v___y_58_, 2);
lean_inc_ref(v_options_66_);
lean_inc_ref(v_lctx_65_);
v___x_67_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_67_, 0, v_env_62_);
lean_ctor_set(v___x_67_, 1, v_mctx_64_);
lean_ctor_set(v___x_67_, 2, v_lctx_65_);
lean_ctor_set(v___x_67_, 3, v_options_66_);
v___x_68_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v_msgData_55_);
v___x_69_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1___boxed(lean_object* v_msgData_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1(v_msgData_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(lean_object* v_msg_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v_ref_83_; lean_object* v___x_84_; lean_object* v_a_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_93_; 
v_ref_83_ = lean_ctor_get(v___y_80_, 5);
v___x_84_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1_spec__1(v_msg_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_);
v_a_85_ = lean_ctor_get(v___x_84_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v___x_84_);
if (v_isSharedCheck_93_ == 0)
{
v___x_87_ = v___x_84_;
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_a_85_);
lean_dec(v___x_84_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v___x_89_; lean_object* v___x_91_; 
lean_inc(v_ref_83_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v_ref_83_);
lean_ctor_set(v___x_89_, 1, v_a_85_);
if (v_isShared_88_ == 0)
{
lean_ctor_set_tag(v___x_87_, 1);
lean_ctor_set(v___x_87_, 0, v___x_89_);
v___x_91_ = v___x_87_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v___x_89_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg___boxed(lean_object* v_msg_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(v_msg_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v_res_100_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_102_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__0));
v___x_103_ = l_Lean_stringToMessageData(v___x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0(uint8_t v___x_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
if (v___x_104_ == 0)
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1, &lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1_once, _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___closed__1);
v___x_115_ = lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(v___x_114_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
return v___x_115_;
}
else
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = lean_box(0);
v___x_117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
return v___x_117_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___boxed(lean_object* v___x_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
uint8_t v___x_3159__boxed_128_; lean_object* v_res_129_; 
v___x_3159__boxed_128_ = lean_unbox(v___x_118_);
v_res_129_ = lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0(v___x_3159__boxed_128_, v___y_119_, v___y_120_, v___y_121_, v___y_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
lean_dec(v___y_122_);
lean_dec_ref(v___y_121_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
return v_res_129_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_131_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__0));
v___x_132_ = l_Lean_stringToMessageData(v___x_131_);
return v___x_132_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_134_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__2));
v___x_135_ = l_Lean_stringToMessageData(v___x_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1(lean_object* v_x_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v___x_146_; uint8_t v___x_147_; 
v___x_146_ = ((lean_object*)(lp_mathlib_guardGoalNums___closed__1));
lean_inc(v_x_136_);
v___x_147_ = l_Lean_Syntax_isOfKind(v_x_136_, v___x_146_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; 
lean_dec(v_x_136_);
v___x_148_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__0___redArg();
return v___x_148_;
}
else
{
lean_object* v___x_149_; 
v___x_149_ = l_Lean_Elab_Tactic_getGoals___redArg(v_a_138_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_151_; 
v_a_150_ = lean_ctor_get(v___x_149_, 0);
lean_inc(v_a_150_);
lean_dec_ref_known(v___x_149_, 1);
v___x_151_ = l_Lean_Elab_Tactic_saveState___redArg(v_a_138_, v_a_140_, v_a_142_, v_a_144_);
if (lean_obj_tag(v___x_151_) == 0)
{
lean_object* v_a_152_; lean_object* v___x_153_; lean_object* v_n_154_; lean_object* v___x_155_; lean_object* v___x_156_; uint8_t v___x_157_; lean_object* v___x_158_; lean_object* v___f_159_; lean_object* v___x_160_; 
v_a_152_ = lean_ctor_get(v___x_151_, 0);
lean_inc(v_a_152_);
lean_dec_ref_known(v___x_151_, 1);
v___x_153_ = lean_unsigned_to_nat(1u);
v_n_154_ = l_Lean_Syntax_getArg(v_x_136_, v___x_153_);
lean_dec(v_x_136_);
v___x_155_ = l_List_lengthTR___redArg(v_a_150_);
lean_dec(v_a_150_);
v___x_156_ = l_Lean_TSyntax_getNat(v_n_154_);
lean_dec(v_n_154_);
v___x_157_ = lean_nat_dec_eq(v___x_155_, v___x_156_);
v___x_158_ = lean_box(v___x_157_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_159_, 0, v___x_158_);
v___x_160_ = l_Lean_Elab_Tactic_withoutRecover___redArg(v___f_159_, v_a_137_, v_a_138_, v_a_139_, v_a_140_, v_a_141_, v_a_142_, v_a_143_, v_a_144_);
if (lean_obj_tag(v___x_160_) == 0)
{
lean_dec(v___x_156_);
lean_dec(v___x_155_);
lean_dec(v_a_152_);
return v___x_160_;
}
else
{
lean_object* v_a_161_; uint8_t v___y_163_; uint8_t v___x_191_; 
v_a_161_ = lean_ctor_get(v___x_160_, 0);
lean_inc(v_a_161_);
v___x_191_ = l_Lean_Exception_isInterrupt(v_a_161_);
if (v___x_191_ == 0)
{
uint8_t v___x_192_; 
v___x_192_ = l_Lean_Exception_isRuntime(v_a_161_);
v___y_163_ = v___x_192_;
goto v___jp_162_;
}
else
{
lean_dec(v_a_161_);
v___y_163_ = v___x_191_;
goto v___jp_162_;
}
v___jp_162_:
{
if (v___y_163_ == 0)
{
lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_189_; 
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; 
v_unused_190_ = lean_ctor_get(v___x_160_, 0);
lean_dec(v_unused_190_);
v___x_165_ = v___x_160_;
v_isShared_166_ = v_isSharedCheck_189_;
goto v_resetjp_164_;
}
else
{
lean_dec(v___x_160_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_189_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_167_; 
v___x_167_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_152_, v___y_163_, v_a_138_, v_a_139_, v_a_140_, v_a_141_, v_a_142_, v_a_143_, v_a_144_);
if (lean_obj_tag(v___x_167_) == 0)
{
lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_187_; 
v_isSharedCheck_187_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_187_ == 0)
{
lean_object* v_unused_188_; 
v_unused_188_ = lean_ctor_get(v___x_167_, 0);
lean_dec(v_unused_188_);
v___x_169_ = v___x_167_;
v_isShared_170_ = v_isSharedCheck_187_;
goto v_resetjp_168_;
}
else
{
lean_dec(v___x_167_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_187_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_174_; 
v___x_171_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1, &lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__1);
v___x_172_ = l_Nat_reprFast(v___x_156_);
if (v_isShared_170_ == 0)
{
lean_ctor_set_tag(v___x_169_, 3);
lean_ctor_set(v___x_169_, 0, v___x_172_);
v___x_174_ = v___x_169_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_172_);
v___x_174_ = v_reuseFailAlloc_186_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_181_; 
v___x_175_ = l_Lean_MessageData_ofFormat(v___x_174_);
v___x_176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_171_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
v___x_177_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3, &lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3_once, _init_lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___closed__3);
v___x_178_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_176_);
lean_ctor_set(v___x_178_, 1, v___x_177_);
v___x_179_ = l_Nat_reprFast(v___x_155_);
if (v_isShared_166_ == 0)
{
lean_ctor_set_tag(v___x_165_, 3);
lean_ctor_set(v___x_165_, 0, v___x_179_);
v___x_181_ = v___x_165_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_179_);
v___x_181_ = v_reuseFailAlloc_185_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = l_Lean_MessageData_ofFormat(v___x_181_);
v___x_183_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_178_);
lean_ctor_set(v___x_183_, 1, v___x_182_);
v___x_184_ = lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(v___x_183_, v_a_141_, v_a_142_, v_a_143_, v_a_144_);
return v___x_184_;
}
}
}
}
else
{
lean_del_object(v___x_165_);
lean_dec(v___x_156_);
lean_dec(v___x_155_);
return v___x_167_;
}
}
}
else
{
lean_dec(v___x_156_);
lean_dec(v___x_155_);
lean_dec(v_a_152_);
return v___x_160_;
}
}
}
}
else
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_200_; 
lean_dec(v_a_150_);
lean_dec(v_x_136_);
v_a_193_ = lean_ctor_get(v___x_151_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_151_);
if (v_isSharedCheck_200_ == 0)
{
v___x_195_ = v___x_151_;
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_151_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_a_193_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_dec(v_x_136_);
v_a_201_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_149_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_149_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1___boxed(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v_a_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib___aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1(v_x_209_, v_a_210_, v_a_211_, v_a_212_, v_a_213_, v_a_214_, v_a_215_, v_a_216_, v_a_217_);
lean_dec(v_a_217_);
lean_dec_ref(v_a_216_);
lean_dec(v_a_215_);
lean_dec_ref(v_a_214_);
lean_dec(v_a_213_);
lean_dec_ref(v_a_212_);
lean_dec(v_a_211_);
lean_dec_ref(v_a_210_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1(lean_object* v_00_u03b1_220_, lean_object* v_msg_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___redArg(v_msg_221_, v___y_226_, v___y_227_, v___y_228_, v___y_229_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1___boxed(lean_object* v_00_u03b1_232_, lean_object* v_msg_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Lean_throwError___at___00__aux__Mathlib__Tactic__GuardGoalNums______elabRules__guardGoalNums__1_spec__1(v_00_u03b1_232_, v_msg_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_, v___y_241_);
lean_dec(v___y_241_);
lean_dec_ref(v___y_240_);
lean_dec(v___y_239_);
lean_dec_ref(v___y_238_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
return v_res_243_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
}
#ifdef __cplusplus
}
#endif
