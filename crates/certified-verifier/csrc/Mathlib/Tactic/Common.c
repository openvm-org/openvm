// Lean compiler output
// Module: Mathlib.Tactic.Common
// Imports: public import Init public meta import Init public import Aesop public import Qq public import Plausible public import Batteries.Tactic.Basic public import Batteries.Tactic.Case public import Batteries.Tactic.HelpCmd public import Batteries.Tactic.Alias public import Batteries.Tactic.GeneralizeProofs public import Batteries.CodeAction public import LeanSearchClient public import Mathlib.Tactic.Linter.Lint public import Mathlib.Tactic.ApplyCongr public import Mathlib.Tactic.ApplyAt public import Mathlib.Tactic.ApplyWith public import Mathlib.Tactic.Basic public import Mathlib.Tactic.ByCases public import Mathlib.Tactic.ByContra public import Mathlib.Tactic.CasesM public import Mathlib.Tactic.Check public import Mathlib.Tactic.Choose public import Mathlib.Tactic.ClearExclamation public import Mathlib.Tactic.ClearExcept public import Mathlib.Tactic.Clear_ public import Mathlib.Tactic.ClickSuggestions public import Mathlib.Tactic.Coe public import Mathlib.Tactic.CongrExclamation public import Mathlib.Tactic.CongrM public import Mathlib.Tactic.Constructor public import Mathlib.Tactic.Contrapose public import Mathlib.Tactic.Conv public import Mathlib.Tactic.Convert public import Mathlib.Tactic.DefEqAbuse public import Mathlib.Tactic.DefEqTransformations public import Mathlib.Tactic.DeprecateTo public import Mathlib.Tactic.DepRewrite public import Mathlib.Tactic.DSimpPercent public import Mathlib.Tactic.ErwQuestion public import Mathlib.Tactic.Eqns public import Mathlib.Tactic.ExistsI public import Mathlib.Tactic.ExtractGoal public import Mathlib.Tactic.FailIfNoProgress public import Mathlib.Tactic.Find public import Mathlib.Tactic.FunProp public import Mathlib.Tactic.GCongr public import Mathlib.Tactic.GRewrite public import Mathlib.Tactic.GrindAttrs public import Mathlib.Tactic.GuardGoalNums public import Mathlib.Tactic.GuardHypNums public import Mathlib.Tactic.HigherOrder public import Mathlib.Tactic.Hint public import Mathlib.Tactic.InferParam public import Mathlib.Tactic.Inhabit public import Mathlib.Tactic.IrreducibleDef public import Mathlib.Tactic.Lift public import Mathlib.Tactic.Linter public import Mathlib.Tactic.MkIffOfInductiveProp public import Mathlib.Tactic.NthRewrite public import Mathlib.Tactic.Observe public import Mathlib.Tactic.OfNat public import Mathlib.Tactic.Push public import Mathlib.Tactic.RSuffices public import Mathlib.Tactic.Recover public import Mathlib.Tactic.Relation.Rfl public import Mathlib.Tactic.Rename public import Mathlib.Tactic.RenameBVar public import Mathlib.Tactic.Says public import Mathlib.Tactic.ScopedNS public import Mathlib.Tactic.Set public import Mathlib.Tactic.Setm public import Mathlib.Tactic.SimpIntro public import Mathlib.Tactic.SimpRw public import Mathlib.Tactic.Simproc.ExistsAndEq public import Mathlib.Tactic.Simps public import Mathlib.Tactic.SplitIfs public import Mathlib.Tactic.Spread public import Mathlib.Tactic.Subsingleton public import Mathlib.Tactic.Substs public import Mathlib.Tactic.SuccessIfFailWithMsg public import Mathlib.Tactic.SudoSetOption public import Mathlib.Tactic.SwapVar public import Mathlib.Tactic.Tauto public import Mathlib.Tactic.ToFun public import Mathlib.Tactic.TermCongr public import Mathlib.Tactic.ToExpr public import Mathlib.Tactic.ToLevel public import Mathlib.Tactic.Trace public import Mathlib.Tactic.UnsetOption public import Mathlib.Tactic.Use public import Mathlib.Tactic.Variable public import Mathlib.Tactic.Widget.Calc public import Mathlib.Tactic.Widget.CongrM public import Mathlib.Tactic.Widget.Conv public import Mathlib.Tactic.Widget.LibraryRewrite public import Mathlib.Tactic.WLOG public import Mathlib.Util.CountHeartbeats public import Mathlib.Util.PrintSorries public import Mathlib.Util.TransImports public import Mathlib.Util.WhatsNew public import Lean.Elab.Tactic.Try public meta import Lean.Meta.Tactic.Try.Collect
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0 = (const lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1 = (const lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1_value;
static const lean_string_object lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tauto"};
static const lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2 = (const lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2_value;
static const lean_string_object lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3 = (const lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3_value;
static const lean_array_object lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4 = (const lean_object*)&lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0 = (const lean_object*)&lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fun_prop"};
static const lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0 = (const lean_object*)&lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg(lean_object* v_a_8_){
_start:
{
lean_object* v___x_10_; lean_object* v_env_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_10_ = lean_st_ref_get(v_a_8_);
v_env_11_ = lean_ctor_get(v___x_10_, 0);
lean_inc_ref(v_env_11_);
lean_dec(v___x_10_);
v___x_12_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1));
v___x_13_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__2));
v___x_14_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3));
v___x_15_ = l_Lean_Parser_runParserCategory(v_env_11_, v___x_12_, v___x_13_, v___x_14_);
if (lean_obj_tag(v___x_15_) == 0)
{
lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_23_; 
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_23_ == 0)
{
lean_object* v_unused_24_; 
v_unused_24_ = lean_ctor_get(v___x_15_, 0);
lean_dec(v_unused_24_);
v___x_17_ = v___x_15_;
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
else
{
lean_dec(v___x_15_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_19_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4));
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v___x_19_);
v___x_21_ = v___x_17_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___x_19_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
else
{
lean_object* v_a_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_35_; 
v_a_25_ = lean_ctor_get(v___x_15_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_35_ == 0)
{
v___x_27_ = v___x_15_;
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_a_25_);
lean_dec(v___x_15_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_35_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_33_; 
v___x_29_ = lean_unsigned_to_nat(1u);
v___x_30_ = lean_mk_empty_array_with_capacity(v___x_29_);
v___x_31_ = lean_array_push(v___x_30_, v_a_25_);
if (v_isShared_28_ == 0)
{
lean_ctor_set_tag(v___x_27_, 0);
lean_ctor_set(v___x_27_, 0, v___x_31_);
v___x_33_ = v___x_27_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_31_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___redArg___boxed(lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib___auxTryTactic11400386961666083988___redArg(v_a_36_);
lean_dec(v_a_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988(lean_object* v___goal_39_, lean_object* v___info_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib___auxTryTactic11400386961666083988___redArg(v_a_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic11400386961666083988___boxed(lean_object* v___goal_47_, lean_object* v___info_48_, lean_object* v_a_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib___auxTryTactic11400386961666083988(v___goal_47_, v___info_48_, v_a_49_, v_a_50_, v_a_51_, v_a_52_);
lean_dec(v_a_52_);
lean_dec_ref(v_a_51_);
lean_dec(v_a_50_);
lean_dec_ref(v_a_49_);
lean_dec_ref(v___info_48_);
lean_dec(v___goal_47_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg(lean_object* v_a_56_){
_start:
{
lean_object* v___x_58_; lean_object* v_env_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_58_ = lean_st_ref_get(v_a_56_);
v_env_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc_ref(v_env_59_);
lean_dec(v___x_58_);
v___x_60_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1));
v___x_61_ = ((lean_object*)(lp_mathlib___auxTryTactic17471413729771528806___redArg___closed__0));
v___x_62_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3));
v___x_63_ = l_Lean_Parser_runParserCategory(v_env_59_, v___x_60_, v___x_61_, v___x_62_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_71_; 
v_isSharedCheck_71_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_71_ == 0)
{
lean_object* v_unused_72_; 
v_unused_72_ = lean_ctor_get(v___x_63_, 0);
lean_dec(v_unused_72_);
v___x_65_ = v___x_63_;
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
else
{
lean_dec(v___x_63_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_67_; lean_object* v___x_69_; 
v___x_67_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4));
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 0, v___x_67_);
v___x_69_ = v___x_65_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_67_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
else
{
lean_object* v_a_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_83_; 
v_a_73_ = lean_ctor_get(v___x_63_, 0);
v_isSharedCheck_83_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_83_ == 0)
{
v___x_75_ = v___x_63_;
v_isShared_76_ = v_isSharedCheck_83_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_a_73_);
lean_dec(v___x_63_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_83_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_81_; 
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_mk_empty_array_with_capacity(v___x_77_);
v___x_79_ = lean_array_push(v___x_78_, v_a_73_);
if (v_isShared_76_ == 0)
{
lean_ctor_set_tag(v___x_75_, 0);
lean_ctor_set(v___x_75_, 0, v___x_79_);
v___x_81_ = v___x_75_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v___x_79_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___redArg___boxed(lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib___auxTryTactic17471413729771528806___redArg(v_a_84_);
lean_dec(v_a_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806(lean_object* v___goal_87_, lean_object* v___info_88_, lean_object* v_a_89_, lean_object* v_a_90_, lean_object* v_a_91_, lean_object* v_a_92_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib___auxTryTactic17471413729771528806___redArg(v_a_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic17471413729771528806___boxed(lean_object* v___goal_95_, lean_object* v___info_96_, lean_object* v_a_97_, lean_object* v_a_98_, lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib___auxTryTactic17471413729771528806(v___goal_95_, v___info_96_, v_a_97_, v_a_98_, v_a_99_, v_a_100_);
lean_dec(v_a_100_);
lean_dec_ref(v_a_99_);
lean_dec(v_a_98_);
lean_dec_ref(v_a_97_);
lean_dec_ref(v___info_96_);
lean_dec(v___goal_95_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg(lean_object* v_a_104_){
_start:
{
lean_object* v___x_106_; lean_object* v_env_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_106_ = lean_st_ref_get(v_a_104_);
v_env_107_ = lean_ctor_get(v___x_106_, 0);
lean_inc_ref(v_env_107_);
lean_dec(v___x_106_);
v___x_108_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__1));
v___x_109_ = ((lean_object*)(lp_mathlib___auxTryTactic3259365975255618868___redArg___closed__0));
v___x_110_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__3));
v___x_111_ = l_Lean_Parser_runParserCategory(v_env_107_, v___x_108_, v___x_109_, v___x_110_);
if (lean_obj_tag(v___x_111_) == 0)
{
lean_object* v___x_113_; uint8_t v_isShared_114_; uint8_t v_isSharedCheck_119_; 
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_119_ == 0)
{
lean_object* v_unused_120_; 
v_unused_120_ = lean_ctor_get(v___x_111_, 0);
lean_dec(v_unused_120_);
v___x_113_ = v___x_111_;
v_isShared_114_ = v_isSharedCheck_119_;
goto v_resetjp_112_;
}
else
{
lean_dec(v___x_111_);
v___x_113_ = lean_box(0);
v_isShared_114_ = v_isSharedCheck_119_;
goto v_resetjp_112_;
}
v_resetjp_112_:
{
lean_object* v___x_115_; lean_object* v___x_117_; 
v___x_115_ = ((lean_object*)(lp_mathlib___auxTryTactic11400386961666083988___redArg___closed__4));
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 0, v___x_115_);
v___x_117_ = v___x_113_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v___x_115_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
else
{
lean_object* v_a_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_131_; 
v_a_121_ = lean_ctor_get(v___x_111_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_131_ == 0)
{
v___x_123_ = v___x_111_;
v_isShared_124_ = v_isSharedCheck_131_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_a_121_);
lean_dec(v___x_111_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_131_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_129_; 
v___x_125_ = lean_unsigned_to_nat(1u);
v___x_126_ = lean_mk_empty_array_with_capacity(v___x_125_);
v___x_127_ = lean_array_push(v___x_126_, v_a_121_);
if (v_isShared_124_ == 0)
{
lean_ctor_set_tag(v___x_123_, 0);
lean_ctor_set(v___x_123_, 0, v___x_127_);
v___x_129_ = v___x_123_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v___x_127_);
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
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___redArg___boxed(lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib___auxTryTactic3259365975255618868___redArg(v_a_132_);
lean_dec(v_a_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868(lean_object* v___goal_135_, lean_object* v___info_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_, lean_object* v_a_140_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib___auxTryTactic3259365975255618868___redArg(v_a_140_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic3259365975255618868___boxed(lean_object* v___goal_143_, lean_object* v___info_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib___auxTryTactic3259365975255618868(v___goal_143_, v___info_144_, v_a_145_, v_a_146_, v_a_147_, v_a_148_);
lean_dec(v_a_148_);
lean_dec_ref(v_a_147_);
lean_dec(v_a_146_);
lean_dec_ref(v_a_145_);
lean_dec_ref(v___info_144_);
lean_dec(v___goal_143_);
return v_res_150_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Case(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_GeneralizeProofs(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_CodeAction(uint8_t builtin);
lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Lint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DepRewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Find(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GrindAttrs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GuardHypNums(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Recover(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Setm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToLevel(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_PrintSorries(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_TransImports(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Try(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Case(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_GeneralizeProofs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Choose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DefEqAbuse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DepRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GrindAttrs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GuardHypNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Observe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Recover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Says(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Setm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Substs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToLevel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Variable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_PrintSorries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_TransImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Try_Collect(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Try_Collect(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_plausible_Plausible(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Case(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_GeneralizeProofs(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction(uint8_t builtin);
lean_object* initialize_LeanSearchClient_LeanSearchClient(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Lint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyAt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyWith(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByCases(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Check(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClearExclamation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClearExcept(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrExclamation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Constructor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Convert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DefEqAbuse(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DepRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ExistsI(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ExtractGoal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Find(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GrindAttrs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GuardGoalNums(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_GuardHypNums(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Hint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_IrreducibleDef(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NthRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Observe(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_OfNat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Recover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Rename(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_RenameBVar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Says(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Setm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SimpRw(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Substs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SudoSetOption(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SwapVar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToFun(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToLevel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_UnsetOption(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_WLOG(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CountHeartbeats(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_PrintSorries(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_TransImports(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Try(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Try_Collect(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Case(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_GeneralizeProofs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LeanSearchClient_LeanSearchClient(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyAt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyWith(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Choose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClearExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClearExcept(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrExclamation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Constructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Convert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DefEqAbuse(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DepRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ExistsI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ExtractGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Find(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GrindAttrs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GuardGoalNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_GuardHypNums(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Hint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_IrreducibleDef(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NthRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Observe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_OfNat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Recover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Rename(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_RenameBVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Says(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Setm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SimpRw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simproc_ExistsAndEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Substs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SuccessIfFailWithMsg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SudoSetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SwapVar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToLevel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_UnsetOption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Variable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_LibraryRewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_WLOG(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CountHeartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_PrintSorries(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_TransImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Try(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Try_Collect(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Common(builtin);
}
#ifdef __cplusplus
}
#endif
