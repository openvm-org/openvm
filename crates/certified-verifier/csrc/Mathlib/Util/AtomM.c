// Lean compiler output
// Module: Mathlib.Util.AtomM
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.Simp.Types public import Qq public import Qq.Typ
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_isDefEqSafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_isDefEqSafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0(lean_object* v_e_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; uint8_t v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_7_ = lean_box(0);
v___x_8_ = 1;
v___x_9_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_9_, 0, v_e_1_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
lean_ctor_set_uint8(v___x_9_, sizeof(void*)*2, v___x_8_);
v___x_10_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0___boxed(lean_object* v_e_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Mathlib_Tactic_AtomM_instInhabitedContext_default___lam__0(v_e_11_, v___y_12_, v___y_13_, v___y_14_, v___y_15_);
lean_dec(v___y_15_);
lean_dec_ref(v___y_14_);
lean_dec(v___y_13_);
lean_dec_ref(v___y_12_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t v_red_26_, lean_object* v_m_27_, lean_object* v_evalAtom_28_, lean_object* v_a_29_, lean_object* v_a_30_, lean_object* v_a_31_, lean_object* v_a_32_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_34_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___closed__0));
v___x_35_ = lean_st_mk_ref(v___x_34_);
v___x_36_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_36_, 0, v_evalAtom_28_);
lean_ctor_set_uint8(v___x_36_, sizeof(void*)*1, v_red_26_);
lean_inc(v_a_32_);
lean_inc_ref(v_a_31_);
lean_inc(v_a_30_);
lean_inc_ref(v_a_29_);
lean_inc(v___x_35_);
v___x_37_ = lean_apply_7(v_m_27_, v___x_36_, v___x_35_, v_a_29_, v_a_30_, v_a_31_, v_a_32_, lean_box(0));
if (lean_obj_tag(v___x_37_) == 0)
{
lean_object* v_a_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_46_; 
v_a_38_ = lean_ctor_get(v___x_37_, 0);
v_isSharedCheck_46_ = !lean_is_exclusive(v___x_37_);
if (v_isSharedCheck_46_ == 0)
{
v___x_40_ = v___x_37_;
v_isShared_41_ = v_isSharedCheck_46_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_a_38_);
lean_dec(v___x_37_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_46_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_42_; lean_object* v___x_44_; 
v___x_42_ = lean_st_ref_get(v___x_35_);
lean_dec(v___x_35_);
lean_dec(v___x_42_);
if (v_isShared_41_ == 0)
{
v___x_44_ = v___x_40_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v_a_38_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
else
{
lean_dec(v___x_35_);
return v___x_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg___boxed(lean_object* v_red_47_, lean_object* v_m_48_, lean_object* v_evalAtom_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
uint8_t v_red_boxed_55_; lean_object* v_res_56_; 
v_red_boxed_55_ = lean_unbox(v_red_47_);
v_res_56_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v_red_boxed_55_, v_m_48_, v_evalAtom_49_, v_a_50_, v_a_51_, v_a_52_, v_a_53_);
lean_dec(v_a_53_);
lean_dec_ref(v_a_52_);
lean_dec(v_a_51_);
lean_dec_ref(v_a_50_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run(lean_object* v_00_u03b1_57_, uint8_t v_red_58_, lean_object* v_m_59_, lean_object* v_evalAtom_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v_red_58_, v_m_59_, v_evalAtom_60_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___boxed(lean_object* v_00_u03b1_67_, lean_object* v_red_68_, lean_object* v_m_69_, lean_object* v_evalAtom_70_, lean_object* v_a_71_, lean_object* v_a_72_, lean_object* v_a_73_, lean_object* v_a_74_, lean_object* v_a_75_){
_start:
{
uint8_t v_red_boxed_76_; lean_object* v_res_77_; 
v_red_boxed_76_ = lean_unbox(v_red_68_);
v_res_77_ = lp_mathlib_Mathlib_Tactic_AtomM_run(v_00_u03b1_67_, v_red_boxed_76_, v_m_69_, v_evalAtom_70_, v_a_71_, v_a_72_, v_a_73_, v_a_74_);
lean_dec(v_a_74_);
lean_dec_ref(v_a_73_);
lean_dec(v_a_72_);
lean_dec_ref(v_a_71_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_isDefEqSafe(lean_object* v_a_78_, lean_object* v_b_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = l_Lean_Meta_isExprDefEq(v_a_78_, v_b_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_);
if (lean_obj_tag(v___x_85_) == 0)
{
return v___x_85_;
}
else
{
lean_object* v_a_86_; uint8_t v___y_88_; uint8_t v___x_98_; 
v_a_86_ = lean_ctor_get(v___x_85_, 0);
lean_inc(v_a_86_);
v___x_98_ = l_Lean_Exception_isInterrupt(v_a_86_);
if (v___x_98_ == 0)
{
uint8_t v___x_99_; 
v___x_99_ = l_Lean_Exception_isRuntime(v_a_86_);
v___y_88_ = v___x_99_;
goto v___jp_87_;
}
else
{
lean_dec(v_a_86_);
v___y_88_ = v___x_98_;
goto v___jp_87_;
}
v___jp_87_:
{
if (v___y_88_ == 0)
{
lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_96_; 
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_96_ == 0)
{
lean_object* v_unused_97_; 
v_unused_97_ = lean_ctor_get(v___x_85_, 0);
lean_dec(v_unused_97_);
v___x_90_ = v___x_85_;
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
else
{
lean_dec(v___x_85_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_92_ = lean_box(v___y_88_);
if (v_isShared_91_ == 0)
{
lean_ctor_set_tag(v___x_90_, 0);
lean_ctor_set(v___x_90_, 0, v___x_92_);
v___x_94_ = v___x_90_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_92_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
else
{
return v___x_85_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_isDefEqSafe___boxed(lean_object* v_a_100_, lean_object* v_b_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_Mathlib_Tactic_isDefEqSafe(v_a_100_, v_b_101_, v_a_102_, v_a_103_, v_a_104_, v_a_105_);
lean_dec(v_a_105_);
lean_dec_ref(v_a_104_);
lean_dec(v_a_103_);
lean_dec_ref(v_a_102_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg(lean_object* v___x_111_, lean_object* v_e_112_, lean_object* v_range_113_, lean_object* v_b_114_, lean_object* v_i_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_stop_122_; lean_object* v_step_123_; uint8_t v___x_124_; 
v_stop_122_ = lean_ctor_get(v_range_113_, 1);
v_step_123_ = lean_ctor_get(v_range_113_, 2);
v___x_124_ = lean_nat_dec_lt(v_i_115_, v_stop_122_);
if (v___x_124_ == 0)
{
lean_object* v___x_125_; 
lean_dec(v_i_115_);
lean_dec_ref(v_e_112_);
v___x_125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_125_, 0, v_b_114_);
return v___x_125_;
}
else
{
uint8_t v_red_126_; lean_object* v_keyedConfig_127_; uint8_t v_trackZetaDelta_128_; lean_object* v_zetaDeltaSet_129_; lean_object* v_lctx_130_; lean_object* v_localInstances_131_; lean_object* v_defEqCtx_x3f_132_; lean_object* v_synthPendingDepth_133_; lean_object* v_customCanUnfoldPredicate_x3f_134_; uint8_t v_univApprox_135_; uint8_t v_inTypeClassResolution_136_; uint8_t v_cacheInferType_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v_a_142_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
lean_dec_ref(v_b_114_);
v_red_126_ = lean_ctor_get_uint8(v___y_116_, sizeof(void*)*1);
v_keyedConfig_127_ = lean_ctor_get(v___y_117_, 0);
v_trackZetaDelta_128_ = lean_ctor_get_uint8(v___y_117_, sizeof(void*)*7);
v_zetaDeltaSet_129_ = lean_ctor_get(v___y_117_, 1);
v_lctx_130_ = lean_ctor_get(v___y_117_, 2);
v_localInstances_131_ = lean_ctor_get(v___y_117_, 3);
v_defEqCtx_x3f_132_ = lean_ctor_get(v___y_117_, 4);
v_synthPendingDepth_133_ = lean_ctor_get(v___y_117_, 5);
v_customCanUnfoldPredicate_x3f_134_ = lean_ctor_get(v___y_117_, 6);
v_univApprox_135_ = lean_ctor_get_uint8(v___y_117_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_136_ = lean_ctor_get_uint8(v___y_117_, sizeof(void*)*7 + 2);
v_cacheInferType_137_ = lean_ctor_get_uint8(v___y_117_, sizeof(void*)*7 + 3);
v___x_138_ = lean_box(0);
v___x_139_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___closed__0));
v___x_140_ = lean_array_fget_borrowed(v___x_111_, v_i_115_);
lean_inc_ref(v_keyedConfig_127_);
v___x_151_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_red_126_, v_keyedConfig_127_);
lean_inc(v_customCanUnfoldPredicate_x3f_134_);
lean_inc(v_synthPendingDepth_133_);
lean_inc(v_defEqCtx_x3f_132_);
lean_inc_ref(v_localInstances_131_);
lean_inc_ref(v_lctx_130_);
lean_inc(v_zetaDeltaSet_129_);
v___x_152_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_zetaDeltaSet_129_);
lean_ctor_set(v___x_152_, 2, v_lctx_130_);
lean_ctor_set(v___x_152_, 3, v_localInstances_131_);
lean_ctor_set(v___x_152_, 4, v_defEqCtx_x3f_132_);
lean_ctor_set(v___x_152_, 5, v_synthPendingDepth_133_);
lean_ctor_set(v___x_152_, 6, v_customCanUnfoldPredicate_x3f_134_);
lean_ctor_set_uint8(v___x_152_, sizeof(void*)*7, v_trackZetaDelta_128_);
lean_ctor_set_uint8(v___x_152_, sizeof(void*)*7 + 1, v_univApprox_135_);
lean_ctor_set_uint8(v___x_152_, sizeof(void*)*7 + 2, v_inTypeClassResolution_136_);
lean_ctor_set_uint8(v___x_152_, sizeof(void*)*7 + 3, v_cacheInferType_137_);
lean_inc(v___x_140_);
lean_inc_ref(v_e_112_);
v___x_153_ = lp_mathlib_Mathlib_Tactic_isDefEqSafe(v_e_112_, v___x_140_, v___x_152_, v___y_118_, v___y_119_, v___y_120_);
lean_dec_ref_known(v___x_152_, 7);
if (lean_obj_tag(v___x_153_) == 0)
{
lean_object* v_a_154_; uint8_t v___x_155_; 
v_a_154_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_a_154_);
lean_dec_ref_known(v___x_153_, 1);
v___x_155_ = lean_unbox(v_a_154_);
lean_dec(v_a_154_);
v_a_142_ = v___x_155_;
goto v___jp_141_;
}
else
{
if (lean_obj_tag(v___x_153_) == 0)
{
lean_object* v_a_156_; uint8_t v___x_157_; 
v_a_156_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_a_156_);
lean_dec_ref_known(v___x_153_, 1);
v___x_157_ = lean_unbox(v_a_156_);
lean_dec(v_a_156_);
v_a_142_ = v___x_157_;
goto v___jp_141_;
}
else
{
lean_object* v_a_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_165_; 
lean_dec(v_i_115_);
lean_dec_ref(v_e_112_);
v_a_158_ = lean_ctor_get(v___x_153_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_153_);
if (v_isSharedCheck_165_ == 0)
{
v___x_160_ = v___x_153_;
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_a_158_);
lean_dec(v___x_153_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_165_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v___x_163_; 
if (v_isShared_161_ == 0)
{
v___x_163_ = v___x_160_;
goto v_reusejp_162_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v_a_158_);
v___x_163_ = v_reuseFailAlloc_164_;
goto v_reusejp_162_;
}
v_reusejp_162_:
{
return v___x_163_;
}
}
}
}
v___jp_141_:
{
if (v_a_142_ == 0)
{
lean_object* v___x_143_; 
v___x_143_ = lean_nat_add(v_i_115_, v_step_123_);
lean_dec(v_i_115_);
v_b_114_ = v___x_139_;
v_i_115_ = v___x_143_;
goto _start;
}
else
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
lean_dec_ref(v_e_112_);
lean_inc(v___x_140_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v_i_115_);
lean_ctor_set(v___x_145_, 1, v___x_140_);
v___x_146_ = lean_box(v_a_142_);
v___x_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___x_145_);
v___x_148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___x_138_);
v___x_150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
return v___x_150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___boxed(lean_object* v___x_166_, lean_object* v_e_167_, lean_object* v_range_168_, lean_object* v_b_169_, lean_object* v_i_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg(v___x_166_, v_e_167_, v_range_168_, v_b_169_, v_i_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec_ref(v_range_168_);
lean_dec_ref(v___x_166_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd(lean_object* v_e_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_186_ = lean_st_ref_get(v_a_180_);
v___x_187_ = lean_unsigned_to_nat(0u);
v___x_188_ = lean_array_get_size(v___x_186_);
v___x_189_ = lean_unsigned_to_nat(1u);
v___x_190_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_190_, 0, v___x_187_);
lean_ctor_set(v___x_190_, 1, v___x_188_);
lean_ctor_set(v___x_190_, 2, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg___closed__0));
lean_inc_ref(v_e_178_);
v___x_192_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg(v___x_186_, v_e_178_, v___x_190_, v___x_191_, v___x_187_, v_a_179_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
lean_dec_ref_known(v___x_190_, 3);
lean_dec(v___x_186_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_220_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_220_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_220_ == 0)
{
v___x_195_ = v___x_192_;
v_isShared_196_ = v_isSharedCheck_220_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_192_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_220_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v_fst_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_218_; 
v_fst_197_ = lean_ctor_get(v_a_193_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v_a_193_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v_a_193_, 1);
lean_dec(v_unused_219_);
v___x_199_ = v_a_193_;
v_isShared_200_ = v_isSharedCheck_218_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_fst_197_);
lean_dec(v_a_193_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_218_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
if (lean_obj_tag(v_fst_197_) == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_204_; 
v___x_201_ = lean_st_ref_take(v_a_180_);
v___x_202_ = lean_array_get_size(v___x_201_);
lean_inc_ref(v_e_178_);
if (v_isShared_200_ == 0)
{
lean_ctor_set(v___x_199_, 1, v_e_178_);
lean_ctor_set(v___x_199_, 0, v___x_202_);
v___x_204_ = v___x_199_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v_e_178_);
v___x_204_ = v_reuseFailAlloc_213_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
lean_object* v___x_205_; lean_object* v___x_206_; uint8_t v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_205_ = lean_array_push(v___x_201_, v_e_178_);
v___x_206_ = lean_st_ref_set(v_a_180_, v___x_205_);
v___x_207_ = 0;
v___x_208_ = lean_box(v___x_207_);
v___x_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v___x_204_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 0, v___x_209_);
v___x_211_ = v___x_195_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_209_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
else
{
lean_object* v_val_214_; lean_object* v___x_216_; 
lean_del_object(v___x_199_);
lean_dec_ref(v_e_178_);
v_val_214_ = lean_ctor_get(v_fst_197_, 0);
lean_inc(v_val_214_);
lean_dec_ref_known(v_fst_197_, 1);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 0, v_val_214_);
v___x_216_ = v___x_195_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_val_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
}
}
else
{
lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
lean_dec_ref(v_e_178_);
v_a_221_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_192_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_192_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd___boxed(lean_object* v_e_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd(v_e_229_, v_a_230_, v_a_231_, v_a_232_, v_a_233_, v_a_234_, v_a_235_);
lean_dec(v_a_235_);
lean_dec_ref(v_a_234_);
lean_dec(v_a_233_);
lean_dec_ref(v_a_232_);
lean_dec(v_a_231_);
lean_dec_ref(v_a_230_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0(lean_object* v___x_238_, lean_object* v_e_239_, lean_object* v_range_240_, lean_object* v_b_241_, lean_object* v_i_242_, lean_object* v_hs_243_, lean_object* v_hl_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___redArg(v___x_238_, v_e_239_, v_range_240_, v_b_241_, v_i_242_, v___y_245_, v___y_247_, v___y_248_, v___y_249_, v___y_250_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0___boxed(lean_object* v___x_253_, lean_object* v_e_254_, lean_object* v_range_255_, lean_object* v_b_256_, lean_object* v_i_257_, lean_object* v_hs_258_, lean_object* v_hl_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_AtomM_containsThenAdd_spec__0(v___x_253_, v_e_254_, v_range_255_, v_b_256_, v_i_257_, v_hs_258_, v_hl_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec_ref(v_range_255_);
lean_dec_ref(v___x_253_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(lean_object* v_e_268_, lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd(v_e_268_, v_a_269_, v_a_270_, v_a_271_, v_a_272_, v_a_273_, v_a_274_);
if (lean_obj_tag(v___x_276_) == 0)
{
lean_object* v_a_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_302_; 
v_a_277_ = lean_ctor_get(v___x_276_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_276_);
if (v_isSharedCheck_302_ == 0)
{
v___x_279_ = v___x_276_;
v_isShared_280_ = v_isSharedCheck_302_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_a_277_);
lean_dec(v___x_276_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_302_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
lean_object* v_snd_281_; lean_object* v_fst_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_301_; 
v_snd_281_ = lean_ctor_get(v_a_277_, 1);
v_fst_282_ = lean_ctor_get(v_a_277_, 0);
v_isSharedCheck_301_ = !lean_is_exclusive(v_a_277_);
if (v_isSharedCheck_301_ == 0)
{
v___x_284_ = v_a_277_;
v_isShared_285_ = v_isSharedCheck_301_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_snd_281_);
lean_inc(v_fst_282_);
lean_dec(v_a_277_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_301_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v_fst_286_; lean_object* v_snd_287_; lean_object* v___x_289_; uint8_t v_isShared_290_; uint8_t v_isSharedCheck_300_; 
v_fst_286_ = lean_ctor_get(v_snd_281_, 0);
v_snd_287_ = lean_ctor_get(v_snd_281_, 1);
v_isSharedCheck_300_ = !lean_is_exclusive(v_snd_281_);
if (v_isSharedCheck_300_ == 0)
{
v___x_289_ = v_snd_281_;
v_isShared_290_ = v_isSharedCheck_300_;
goto v_resetjp_288_;
}
else
{
lean_inc(v_snd_287_);
lean_inc(v_fst_286_);
lean_dec(v_snd_281_);
v___x_289_ = lean_box(0);
v_isShared_290_ = v_isSharedCheck_300_;
goto v_resetjp_288_;
}
v_resetjp_288_:
{
lean_object* v___x_292_; 
if (v_isShared_290_ == 0)
{
v___x_292_ = v___x_289_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_fst_286_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v_snd_287_);
v___x_292_ = v_reuseFailAlloc_299_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
lean_object* v___x_294_; 
if (v_isShared_285_ == 0)
{
lean_ctor_set(v___x_284_, 1, v___x_292_);
v___x_294_ = v___x_284_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_fst_282_);
lean_ctor_set(v_reuseFailAlloc_298_, 1, v___x_292_);
v___x_294_ = v_reuseFailAlloc_298_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
lean_object* v___x_296_; 
if (v_isShared_280_ == 0)
{
lean_ctor_set(v___x_279_, 0, v___x_294_);
v___x_296_ = v___x_279_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_294_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
v_a_303_ = lean_ctor_get(v___x_276_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_276_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_276_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_276_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg___boxed(lean_object* v_e_311_, lean_object* v_a_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_, lean_object* v_a_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(v_e_311_, v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_);
lean_dec(v_a_317_);
lean_dec_ref(v_a_316_);
lean_dec(v_a_315_);
lean_dec_ref(v_a_314_);
lean_dec(v_a_313_);
lean_dec_ref(v_a_312_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ(lean_object* v_u_320_, lean_object* v_00_u03b1_321_, lean_object* v_e_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(v_e_322_, v_a_323_, v_a_324_, v_a_325_, v_a_326_, v_a_327_, v_a_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___boxed(lean_object* v_u_331_, lean_object* v_00_u03b1_332_, lean_object* v_e_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ(v_u_331_, v_00_u03b1_332_, v_e_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_, v_a_338_, v_a_339_);
lean_dec(v_a_339_);
lean_dec_ref(v_a_338_);
lean_dec(v_a_337_);
lean_dec_ref(v_a_336_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
lean_dec_ref(v_00_u03b1_332_);
lean_dec(v_u_331_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtom(lean_object* v_e_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAdd(v_e_342_, v_a_343_, v_a_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_359_; 
v_a_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_359_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v_snd_355_; lean_object* v___x_357_; 
v_snd_355_ = lean_ctor_get(v_a_351_, 1);
lean_inc(v_snd_355_);
lean_dec(v_a_351_);
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 0, v_snd_355_);
v___x_357_ = v___x_353_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v_snd_355_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
else
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_367_; 
v_a_360_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_367_ == 0)
{
v___x_362_ = v___x_350_;
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_350_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_365_; 
if (v_isShared_363_ == 0)
{
v___x_365_ = v___x_362_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_360_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtom___boxed(lean_object* v_e_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v_e_368_, v_a_369_, v_a_370_, v_a_371_, v_a_372_, v_a_373_, v_a_374_);
lean_dec(v_a_374_);
lean_dec_ref(v_a_373_);
lean_dec(v_a_372_);
lean_dec_ref(v_a_371_);
lean_dec(v_a_370_);
lean_dec_ref(v_a_369_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(lean_object* v_e_377_, lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtom(v_e_377_, v_a_378_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_);
if (lean_obj_tag(v___x_385_) == 0)
{
lean_object* v_a_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_402_; 
v_a_386_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_402_ == 0)
{
v___x_388_ = v___x_385_;
v_isShared_389_ = v_isSharedCheck_402_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_a_386_);
lean_dec(v___x_385_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_402_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v_fst_390_; lean_object* v_snd_391_; lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_401_; 
v_fst_390_ = lean_ctor_get(v_a_386_, 0);
v_snd_391_ = lean_ctor_get(v_a_386_, 1);
v_isSharedCheck_401_ = !lean_is_exclusive(v_a_386_);
if (v_isSharedCheck_401_ == 0)
{
v___x_393_ = v_a_386_;
v_isShared_394_ = v_isSharedCheck_401_;
goto v_resetjp_392_;
}
else
{
lean_inc(v_snd_391_);
lean_inc(v_fst_390_);
lean_dec(v_a_386_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_401_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v___x_396_; 
if (v_isShared_394_ == 0)
{
v___x_396_ = v___x_393_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v_fst_390_);
lean_ctor_set(v_reuseFailAlloc_400_, 1, v_snd_391_);
v___x_396_ = v_reuseFailAlloc_400_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
lean_object* v___x_398_; 
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_396_);
v___x_398_ = v___x_388_;
goto v_reusejp_397_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v___x_396_);
v___x_398_ = v_reuseFailAlloc_399_;
goto v_reusejp_397_;
}
v_reusejp_397_:
{
return v___x_398_;
}
}
}
}
}
else
{
lean_object* v_a_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_410_; 
v_a_403_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_410_ == 0)
{
v___x_405_ = v___x_385_;
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_a_403_);
lean_dec(v___x_385_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_408_; 
if (v_isShared_406_ == 0)
{
v___x_408_ = v___x_405_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_a_403_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg___boxed(lean_object* v_e_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(v_e_411_, v_a_412_, v_a_413_, v_a_414_, v_a_415_, v_a_416_, v_a_417_);
lean_dec(v_a_417_);
lean_dec_ref(v_a_416_);
lean_dec(v_a_415_);
lean_dec_ref(v_a_414_);
lean_dec(v_a_413_);
lean_dec_ref(v_a_412_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ(lean_object* v_u_420_, lean_object* v_00_u03b1_421_, lean_object* v_e_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_, lean_object* v_a_426_, lean_object* v_a_427_, lean_object* v_a_428_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___redArg(v_e_422_, v_a_423_, v_a_424_, v_a_425_, v_a_426_, v_a_427_, v_a_428_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ___boxed(lean_object* v_u_431_, lean_object* v_00_u03b1_432_, lean_object* v_e_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_, lean_object* v_a_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_Mathlib_Tactic_AtomM_addAtomQ(v_u_431_, v_00_u03b1_432_, v_e_433_, v_a_434_, v_a_435_, v_a_436_, v_a_437_, v_a_438_, v_a_439_);
lean_dec(v_a_439_);
lean_dec_ref(v_a_438_);
lean_dec(v_a_437_);
lean_dec_ref(v_a_436_);
lean_dec(v_a_435_);
lean_dec_ref(v_a_434_);
lean_dec_ref(v_00_u03b1_432_);
lean_dec(v_u_431_);
return v_res_441_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin) {
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
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_AtomM(builtin);
}
#ifdef __cplusplus
}
#endif
