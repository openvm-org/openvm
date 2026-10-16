// Lean compiler output
// Module: Mathlib.Tactic.Constructor
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.SyntheticMVars public meta import Lean.Meta.Tactic.Constructor
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_constructor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticFconstructor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticFconstructor"};
static const lean_object* lp_mathlib_tacticFconstructor___closed__0 = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__0_value;
static const lean_ctor_object lp_mathlib_tacticFconstructor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticFconstructor___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 17, 112, 205, 163, 180, 179, 8)}};
static const lean_object* lp_mathlib_tacticFconstructor___closed__1 = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__1_value;
static const lean_string_object lp_mathlib_tacticFconstructor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "fconstructor"};
static const lean_object* lp_mathlib_tacticFconstructor___closed__2 = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__2_value;
static const lean_ctor_object lp_mathlib_tacticFconstructor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_tacticFconstructor___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tacticFconstructor___closed__3 = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__3_value;
static const lean_ctor_object lp_mathlib_tacticFconstructor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticFconstructor___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_tacticFconstructor___closed__3_value)}};
static const lean_object* lp_mathlib_tacticFconstructor___closed__4 = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tacticFconstructor = (const lean_object*)&lp_mathlib_tacticFconstructor___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticEconstructor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticEconstructor"};
static const lean_object* lp_mathlib_tacticEconstructor___closed__0 = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__0_value;
static const lean_ctor_object lp_mathlib_tacticEconstructor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticEconstructor___closed__0_value),LEAN_SCALAR_PTR_LITERAL(110, 138, 242, 41, 153, 63, 36, 250)}};
static const lean_object* lp_mathlib_tacticEconstructor___closed__1 = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__1_value;
static const lean_string_object lp_mathlib_tacticEconstructor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "econstructor"};
static const lean_object* lp_mathlib_tacticEconstructor___closed__2 = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__2_value;
static const lean_ctor_object lp_mathlib_tacticEconstructor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_tacticEconstructor___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tacticEconstructor___closed__3 = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__3_value;
static const lean_ctor_object lp_mathlib_tacticEconstructor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticEconstructor___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_tacticEconstructor___closed__3_value)}};
static const lean_object* lp_mathlib_tacticEconstructor___closed__4 = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tacticEconstructor = (const lean_object*)&lp_mathlib_tacticEconstructor___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_13_ = lean_box(0);
v___x_14_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_15_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
lean_ctor_set(v___x_15_, 1, v___x_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg(){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___closed__0);
v___x_18_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg___boxed(lean_object* v___y_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg();
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0(lean_object* v_00_u03b1_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg();
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___boxed(lean_object* v_00_u03b1_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0(v_00_u03b1_32_, v___y_33_, v___y_34_, v___y_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
lean_dec(v___y_38_);
lean_dec_ref(v___y_37_);
lean_dec(v___y_36_);
lean_dec_ref(v___y_35_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0(uint8_t v___x_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_45_, v___y_48_, v___y_49_, v___y_50_, v___y_51_);
if (lean_obj_tag(v___x_53_) == 0)
{
lean_object* v_a_54_; uint8_t v___x_55_; uint8_t v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v_a_54_ = lean_ctor_get(v___x_53_, 0);
lean_inc(v_a_54_);
lean_dec_ref_known(v___x_53_, 1);
v___x_55_ = 2;
v___x_56_ = 0;
v___x_57_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_57_, 0, v___x_55_);
lean_ctor_set_uint8(v___x_57_, 1, v___x_43_);
lean_ctor_set_uint8(v___x_57_, 2, v___x_56_);
lean_ctor_set_uint8(v___x_57_, 3, v___x_43_);
v___x_58_ = l_Lean_MVarId_constructor(v_a_54_, v___x_57_, v___y_48_, v___y_49_, v___y_50_, v___y_51_);
if (lean_obj_tag(v___x_58_) == 0)
{
lean_object* v_a_59_; lean_object* v___x_60_; 
v_a_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc(v_a_59_);
lean_dec_ref_known(v___x_58_, 1);
v___x_60_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_56_, v___y_46_, v___y_47_, v___y_48_, v___y_49_, v___y_50_, v___y_51_);
if (lean_obj_tag(v___x_60_) == 0)
{
lean_object* v___x_61_; 
lean_dec_ref_known(v___x_60_, 1);
v___x_61_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_59_, v___y_45_, v___y_48_, v___y_49_, v___y_50_, v___y_51_);
return v___x_61_;
}
else
{
lean_dec(v_a_59_);
return v___x_60_;
}
}
else
{
lean_object* v_a_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_69_; 
v_a_62_ = lean_ctor_get(v___x_58_, 0);
v_isSharedCheck_69_ = !lean_is_exclusive(v___x_58_);
if (v_isSharedCheck_69_ == 0)
{
v___x_64_ = v___x_58_;
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_a_62_);
lean_dec(v___x_58_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_67_; 
if (v_isShared_65_ == 0)
{
v___x_67_ = v___x_64_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v_a_62_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
else
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_77_; 
v_a_70_ = lean_ctor_get(v___x_53_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_53_);
if (v_isSharedCheck_77_ == 0)
{
v___x_72_ = v___x_53_;
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v___x_53_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_75_; 
if (v_isShared_73_ == 0)
{
v___x_75_ = v___x_72_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v_a_70_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0___boxed(lean_object* v___x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
uint8_t v___x_852__boxed_88_; lean_object* v_res_89_; 
v___x_852__boxed_88_ = lean_unbox(v___x_78_);
v_res_89_ = lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0(v___x_852__boxed_88_, v___y_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1(lean_object* v_x_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_, lean_object* v_a_95_, lean_object* v_a_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v___x_100_; uint8_t v___x_101_; 
v___x_100_ = ((lean_object*)(lp_mathlib_tacticFconstructor___closed__1));
v___x_101_ = l_Lean_Syntax_isOfKind(v_x_90_, v___x_100_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg();
return v___x_102_;
}
else
{
lean_object* v___x_103_; lean_object* v___f_104_; lean_object* v___x_105_; 
v___x_103_ = lean_box(v___x_101_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_104_, 0, v___x_103_);
v___x_105_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_104_, v_a_91_, v_a_92_, v_a_93_, v_a_94_, v_a_95_, v_a_96_, v_a_97_, v_a_98_);
return v___x_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1___boxed(lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1(v_x_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_, v_a_112_, v_a_113_, v_a_114_);
lean_dec(v_a_114_);
lean_dec_ref(v_a_113_);
lean_dec(v_a_112_);
lean_dec_ref(v_a_111_);
lean_dec(v_a_110_);
lean_dec_ref(v_a_109_);
lean_dec(v_a_108_);
lean_dec_ref(v_a_107_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0(uint8_t v___x_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_131_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_139_) == 0)
{
lean_object* v_a_140_; uint8_t v___x_141_; uint8_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_a_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_a_140_);
lean_dec_ref_known(v___x_139_, 1);
v___x_141_ = 1;
v___x_142_ = 0;
v___x_143_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_143_, 0, v___x_141_);
lean_ctor_set_uint8(v___x_143_, 1, v___x_129_);
lean_ctor_set_uint8(v___x_143_, 2, v___x_142_);
lean_ctor_set_uint8(v___x_143_, 3, v___x_129_);
v___x_144_ = l_Lean_MVarId_constructor(v_a_140_, v___x_143_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v_a_145_; lean_object* v___x_146_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_a_145_);
lean_dec_ref_known(v___x_144_, 1);
v___x_146_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_142_, v___y_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v___x_147_; 
lean_dec_ref_known(v___x_146_, 1);
v___x_147_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_145_, v___y_131_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
return v___x_147_;
}
else
{
lean_dec(v_a_145_);
return v___x_146_;
}
}
else
{
lean_object* v_a_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_155_; 
v_a_148_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_155_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_155_ == 0)
{
v___x_150_ = v___x_144_;
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_a_148_);
lean_dec(v___x_144_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_148_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
else
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
v_a_156_ = lean_ctor_get(v___x_139_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_139_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_139_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_139_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0___boxed(lean_object* v___x_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
uint8_t v___x_601__boxed_174_; lean_object* v_res_175_; 
v___x_601__boxed_174_ = lean_unbox(v___x_164_);
v_res_175_ = lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0(v___x_601__boxed_174_, v___y_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1(lean_object* v_x_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v___x_186_; uint8_t v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib_tacticEconstructor___closed__1));
v___x_187_ = l_Lean_Syntax_isOfKind(v_x_176_, v___x_186_);
if (v___x_187_ == 0)
{
lean_object* v___x_188_; 
v___x_188_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Constructor______elabRules__tacticFconstructor__1_spec__0___redArg();
return v___x_188_;
}
else
{
lean_object* v___x_189_; lean_object* v___f_190_; lean_object* v___x_191_; 
v___x_189_ = lean_box(v___x_187_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_190_, 0, v___x_189_);
v___x_191_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_190_, v_a_177_, v_a_178_, v_a_179_, v_a_180_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
return v___x_191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1___boxed(lean_object* v_x_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib___aux__Mathlib__Tactic__Constructor______elabRules__tacticEconstructor__1(v_x_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_);
lean_dec(v_a_200_);
lean_dec_ref(v_a_199_);
lean_dec(v_a_198_);
lean_dec_ref(v_a_197_);
lean_dec(v_a_196_);
lean_dec_ref(v_a_195_);
lean_dec(v_a_194_);
lean_dec_ref(v_a_193_);
return v_res_202_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Constructor(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_SyntheticMVars(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Constructor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin) {
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
res = initialize_Lean_Elab_SyntheticMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
}
#ifdef __cplusplus
}
#endif
