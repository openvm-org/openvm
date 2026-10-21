// Lean compiler output
// Module: Aesop.Script.ScriptM
// Imports: public import Init public meta import Init public import Aesop.BaseM public import Aesop.Script.Step
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
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_ScriptT_run___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ST_Prim_mkRef___boxed, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ScriptT_run___redArg___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_ScriptT_run___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__0(lean_object* v_a_1_, lean_object* v_toPure_2_, lean_object* v_s_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_a_1_);
lean_ctor_set(v___x_4_, 1, v_s_3_);
v___x_5_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__1(lean_object* v_toPure_6_, lean_object* v_ref_7_, lean_object* v_inst_8_, lean_object* v_toBind_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___f_11_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ScriptT_run___redArg___lam__0), 3, 2);
lean_closure_set(v___f_11_, 0, v_a_10_);
lean_closure_set(v___f_11_, 1, v_toPure_6_);
v___x_12_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, lean_box(0));
lean_closure_set(v___x_12_, 2, v_ref_7_);
v___x_13_ = lean_apply_2(v_inst_8_, lean_box(0), v___x_12_);
v___x_14_ = lean_apply_4(v_toBind_9_, lean_box(0), lean_box(0), v___x_13_, v___f_11_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg___lam__2(lean_object* v_toPure_15_, lean_object* v_inst_16_, lean_object* v_toBind_17_, lean_object* v_x_18_, lean_object* v_ref_19_){
_start:
{
lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
lean_inc(v_toBind_17_);
lean_inc(v_ref_19_);
v___f_20_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ScriptT_run___redArg___lam__1), 5, 4);
lean_closure_set(v___f_20_, 0, v_toPure_15_);
lean_closure_set(v___f_20_, 1, v_ref_19_);
lean_closure_set(v___f_20_, 2, v_inst_16_);
lean_closure_set(v___f_20_, 3, v_toBind_17_);
v___x_21_ = lean_apply_1(v_x_18_, v_ref_19_);
v___x_22_ = lean_apply_4(v_toBind_17_, lean_box(0), lean_box(0), v___x_21_, v___f_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___redArg(lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_toApplicative_30_; lean_object* v_toBind_31_; lean_object* v_toPure_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___f_35_; lean_object* v___x_36_; 
v_toApplicative_30_ = lean_ctor_get(v_inst_27_, 0);
lean_inc_ref(v_toApplicative_30_);
v_toBind_31_ = lean_ctor_get(v_inst_27_, 1);
lean_inc_n(v_toBind_31_, 2);
lean_dec_ref(v_inst_27_);
v_toPure_32_ = lean_ctor_get(v_toApplicative_30_, 1);
lean_inc(v_toPure_32_);
lean_dec_ref(v_toApplicative_30_);
v___x_33_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___redArg___closed__1));
lean_inc(v_inst_28_);
v___x_34_ = lean_apply_2(v_inst_28_, lean_box(0), v___x_33_);
v___f_35_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ScriptT_run___redArg___lam__2), 5, 4);
lean_closure_set(v___f_35_, 0, v_toPure_32_);
lean_closure_set(v___f_35_, 1, v_inst_28_);
lean_closure_set(v___f_35_, 2, v_toBind_31_);
lean_closure_set(v___f_35_, 3, v_x_29_);
v___x_36_ = lean_apply_4(v_toBind_31_, lean_box(0), lean_box(0), v___x_34_, v___f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run(lean_object* v_m_37_, lean_object* v_00_u03b1_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_x_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_aesop_Aesop_ScriptT_run___redArg(v_inst_39_, v_inst_40_, v_x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___redArg___lam__0(lean_object* v_step_43_, lean_object* v_s_44_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_45_ = lean_box(0);
v___x_46_ = lean_array_push(v_s_44_, v_step_43_);
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v___x_45_);
lean_ctor_set(v___x_47_, 1, v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___redArg(lean_object* v_inst_48_, lean_object* v_step_49_){
_start:
{
lean_object* v_modifyGet_50_; lean_object* v___f_51_; lean_object* v___x_52_; 
v_modifyGet_50_ = lean_ctor_get(v_inst_48_, 2);
lean_inc(v_modifyGet_50_);
lean_dec_ref(v_inst_48_);
v___f_51_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordScriptStep___redArg___lam__0), 2, 1);
lean_closure_set(v___f_51_, 0, v_step_49_);
v___x_52_ = lean_apply_2(v_modifyGet_50_, lean_box(0), v___f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep(lean_object* v_m_53_, lean_object* v_inst_54_, lean_object* v_step_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_aesop_Aesop_recordScriptStep___redArg(v_inst_54_, v_step_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg___lam__0(lean_object* v_steps_57_, lean_object* v_s_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = lean_box(0);
v___x_60_ = l_Array_append___redArg(v_s_58_, v_steps_57_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_59_);
lean_ctor_set(v___x_61_, 1, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg___lam__0___boxed(lean_object* v_steps_62_, lean_object* v_s_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_aesop_Aesop_recordScriptSteps___redArg___lam__0(v_steps_62_, v_s_63_);
lean_dec_ref(v_steps_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps___redArg(lean_object* v_inst_65_, lean_object* v_steps_66_){
_start:
{
lean_object* v_modifyGet_67_; lean_object* v___f_68_; lean_object* v___x_69_; 
v_modifyGet_67_ = lean_ctor_get(v_inst_65_, 2);
lean_inc(v_modifyGet_67_);
lean_dec_ref(v_inst_65_);
v___f_68_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordScriptSteps___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_68_, 0, v_steps_66_);
v___x_69_ = lean_apply_2(v_modifyGet_67_, lean_box(0), v___f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptSteps(lean_object* v_m_70_, lean_object* v_inst_71_, lean_object* v_steps_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_aesop_Aesop_recordScriptSteps___redArg(v_inst_71_, v_steps_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(lean_object* v_step_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_77_ = lean_st_ref_take(v___y_75_);
v___x_78_ = lean_array_push(v___x_77_, v_step_74_);
v___x_79_ = lean_st_ref_set(v___y_75_, v___x_78_);
v___x_80_ = lean_box(0);
v___x_81_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg___boxed(lean_object* v_step_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(v_step_82_, v___y_83_);
lean_dec(v___y_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0(lean_object* v_step_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(v_step_86_, v___y_87_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___boxed(lean_object* v_step_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0(v_step_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec(v___y_96_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___redArg(lean_object* v_preGoal_104_, lean_object* v_postGoals_105_, lean_object* v_success_106_, lean_object* v_tacticBuilder_107_, lean_object* v_x_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = l_Lean_Meta_saveState___redArg(v_a_112_, v_a_114_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_object* v_a_117_; lean_object* v___x_118_; 
v_a_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_a_117_);
lean_dec_ref_known(v___x_116_, 1);
lean_inc(v_a_114_);
lean_inc_ref(v_a_113_);
lean_inc(v_a_112_);
lean_inc_ref(v_a_111_);
v___x_118_ = lean_apply_5(v_x_108_, v_a_111_, v_a_112_, v_a_113_, v_a_114_, lean_box(0));
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
lean_inc_n(v_a_119_, 2);
v___x_120_ = lean_apply_1(v_success_106_, v_a_119_);
v___x_121_ = lean_unbox(v___x_120_);
if (v___x_121_ == 0)
{
lean_dec(v_a_119_);
lean_dec(v_a_117_);
lean_dec_ref(v_tacticBuilder_107_);
lean_dec_ref(v_postGoals_105_);
lean_dec(v_preGoal_104_);
return v___x_118_;
}
else
{
lean_object* v___x_122_; 
lean_dec_ref_known(v___x_118_, 1);
v___x_122_ = l_Lean_Meta_saveState___redArg(v_a_112_, v_a_114_);
if (lean_obj_tag(v___x_122_) == 0)
{
lean_object* v_a_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_137_; 
v_a_123_ = lean_ctor_get(v___x_122_, 0);
lean_inc(v_a_123_);
lean_dec_ref_known(v___x_122_, 1);
lean_inc_n(v_a_119_, 2);
v___x_124_ = lean_apply_1(v_tacticBuilder_107_, v_a_119_);
v___x_125_ = lean_unsigned_to_nat(1u);
v___x_126_ = lean_mk_empty_array_with_capacity(v___x_125_);
v___x_127_ = lean_array_push(v___x_126_, v___x_124_);
v___x_128_ = lean_apply_1(v_postGoals_105_, v_a_119_);
v___x_129_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_129_, 0, v_a_117_);
lean_ctor_set(v___x_129_, 1, v_preGoal_104_);
lean_ctor_set(v___x_129_, 2, v___x_127_);
lean_ctor_set(v___x_129_, 3, v_a_123_);
lean_ctor_set(v___x_129_, 4, v___x_128_);
v___x_130_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(v___x_129_, v_a_109_);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_137_ == 0)
{
lean_object* v_unused_138_; 
v_unused_138_ = lean_ctor_get(v___x_130_, 0);
lean_dec(v_unused_138_);
v___x_132_ = v___x_130_;
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
else
{
lean_dec(v___x_130_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_135_; 
if (v_isShared_133_ == 0)
{
lean_ctor_set(v___x_132_, 0, v_a_119_);
v___x_135_ = v___x_132_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_a_119_);
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
lean_object* v_a_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_146_; 
lean_dec(v_a_119_);
lean_dec(v_a_117_);
lean_dec_ref(v_tacticBuilder_107_);
lean_dec_ref(v_postGoals_105_);
lean_dec(v_preGoal_104_);
v_a_139_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_146_ == 0)
{
v___x_141_ = v___x_122_;
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_a_139_);
lean_dec(v___x_122_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
lean_object* v___x_144_; 
if (v_isShared_142_ == 0)
{
v___x_144_ = v___x_141_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v_a_139_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
}
else
{
lean_dec(v_a_117_);
lean_dec_ref(v_tacticBuilder_107_);
lean_dec_ref(v_success_106_);
lean_dec_ref(v_postGoals_105_);
lean_dec(v_preGoal_104_);
return v___x_118_;
}
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
lean_dec_ref(v_x_108_);
lean_dec_ref(v_tacticBuilder_107_);
lean_dec_ref(v_success_106_);
lean_dec_ref(v_postGoals_105_);
lean_dec(v_preGoal_104_);
v_a_147_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_116_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_116_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___redArg___boxed(lean_object* v_preGoal_155_, lean_object* v_postGoals_156_, lean_object* v_success_157_, lean_object* v_tacticBuilder_158_, lean_object* v_x_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_aesop_Aesop_withScriptStep___redArg(v_preGoal_155_, v_postGoals_156_, v_success_157_, v_tacticBuilder_158_, v_x_159_, v_a_160_, v_a_161_, v_a_162_, v_a_163_, v_a_164_, v_a_165_);
lean_dec(v_a_165_);
lean_dec_ref(v_a_164_);
lean_dec(v_a_163_);
lean_dec_ref(v_a_162_);
lean_dec(v_a_161_);
lean_dec(v_a_160_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep(lean_object* v_00_u03b1_168_, lean_object* v_preGoal_169_, lean_object* v_postGoals_170_, lean_object* v_success_171_, lean_object* v_tacticBuilder_172_, lean_object* v_x_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_aesop_Aesop_withScriptStep___redArg(v_preGoal_169_, v_postGoals_170_, v_success_171_, v_tacticBuilder_172_, v_x_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_, v_a_178_, v_a_179_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withScriptStep___boxed(lean_object* v_00_u03b1_182_, lean_object* v_preGoal_183_, lean_object* v_postGoals_184_, lean_object* v_success_185_, lean_object* v_tacticBuilder_186_, lean_object* v_x_187_, lean_object* v_a_188_, lean_object* v_a_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_aesop_Aesop_withScriptStep(v_00_u03b1_182_, v_preGoal_183_, v_postGoals_184_, v_success_185_, v_tacticBuilder_186_, v_x_187_, v_a_188_, v_a_189_, v_a_190_, v_a_191_, v_a_192_, v_a_193_);
lean_dec(v_a_193_);
lean_dec_ref(v_a_192_);
lean_dec(v_a_191_);
lean_dec_ref(v_a_190_);
lean_dec(v_a_189_);
lean_dec(v_a_188_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___redArg(lean_object* v_preGoal_196_, lean_object* v_postGoals_197_, lean_object* v_tacticBuilder_198_, lean_object* v_x_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_, lean_object* v_a_203_, lean_object* v_a_204_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = l_Lean_Meta_saveState___redArg(v_a_202_, v_a_204_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_208_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc(v_a_207_);
lean_dec_ref_known(v___x_206_, 1);
lean_inc(v_a_204_);
lean_inc_ref(v_a_203_);
lean_inc(v_a_202_);
lean_inc_ref(v_a_201_);
v___x_208_ = lean_apply_5(v_x_199_, v_a_201_, v_a_202_, v_a_203_, v_a_204_, lean_box(0));
if (lean_obj_tag(v___x_208_) == 0)
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_243_; 
v_a_209_ = lean_ctor_get(v___x_208_, 0);
v_isSharedCheck_243_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_243_ == 0)
{
v___x_211_ = v___x_208_;
v_isShared_212_ = v_isSharedCheck_243_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_208_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_243_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
if (lean_obj_tag(v_a_209_) == 1)
{
lean_object* v_val_213_; lean_object* v___x_214_; 
lean_del_object(v___x_211_);
v_val_213_ = lean_ctor_get(v_a_209_, 0);
v___x_214_ = l_Lean_Meta_saveState___redArg(v_a_202_, v_a_204_);
if (lean_obj_tag(v___x_214_) == 0)
{
lean_object* v_a_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
v_a_215_ = lean_ctor_get(v___x_214_, 0);
lean_inc(v_a_215_);
lean_dec_ref_known(v___x_214_, 1);
lean_inc_n(v_val_213_, 2);
v___x_216_ = lean_apply_1(v_tacticBuilder_198_, v_val_213_);
v___x_217_ = lean_unsigned_to_nat(1u);
v___x_218_ = lean_mk_empty_array_with_capacity(v___x_217_);
v___x_219_ = lean_array_push(v___x_218_, v___x_216_);
v___x_220_ = lean_apply_1(v_postGoals_197_, v_val_213_);
v___x_221_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_221_, 0, v_a_207_);
lean_ctor_set(v___x_221_, 1, v_preGoal_196_);
lean_ctor_set(v___x_221_, 2, v___x_219_);
lean_ctor_set(v___x_221_, 3, v_a_215_);
lean_ctor_set(v___x_221_, 4, v___x_220_);
v___x_222_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_withScriptStep_spec__0___redArg(v___x_221_, v_a_200_);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_222_);
if (v_isSharedCheck_229_ == 0)
{
lean_object* v_unused_230_; 
v_unused_230_ = lean_ctor_get(v___x_222_, 0);
lean_dec(v_unused_230_);
v___x_224_ = v___x_222_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_dec(v___x_222_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 0, v_a_209_);
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_209_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
else
{
lean_object* v_a_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_238_; 
lean_dec_ref_known(v_a_209_, 1);
lean_dec(v_a_207_);
lean_dec_ref(v_tacticBuilder_198_);
lean_dec_ref(v_postGoals_197_);
lean_dec(v_preGoal_196_);
v_a_231_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_238_ == 0)
{
v___x_233_ = v___x_214_;
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_a_231_);
lean_dec(v___x_214_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_236_; 
if (v_isShared_234_ == 0)
{
v___x_236_ = v___x_233_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_a_231_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
else
{
lean_object* v___x_239_; lean_object* v___x_241_; 
lean_dec(v_a_209_);
lean_dec(v_a_207_);
lean_dec_ref(v_tacticBuilder_198_);
lean_dec_ref(v_postGoals_197_);
lean_dec(v_preGoal_196_);
v___x_239_ = lean_box(0);
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 0, v___x_239_);
v___x_241_ = v___x_211_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
else
{
lean_dec(v_a_207_);
lean_dec_ref(v_tacticBuilder_198_);
lean_dec_ref(v_postGoals_197_);
lean_dec(v_preGoal_196_);
return v___x_208_;
}
}
else
{
lean_object* v_a_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_251_; 
lean_dec_ref(v_x_199_);
lean_dec_ref(v_tacticBuilder_198_);
lean_dec_ref(v_postGoals_197_);
lean_dec(v_preGoal_196_);
v_a_244_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_251_ == 0)
{
v___x_246_ = v___x_206_;
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_a_244_);
lean_dec(v___x_206_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_251_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_249_; 
if (v_isShared_247_ == 0)
{
v___x_249_ = v___x_246_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_a_244_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___redArg___boxed(lean_object* v_preGoal_252_, lean_object* v_postGoals_253_, lean_object* v_tacticBuilder_254_, lean_object* v_x_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_preGoal_252_, v_postGoals_253_, v_tacticBuilder_254_, v_x_255_, v_a_256_, v_a_257_, v_a_258_, v_a_259_, v_a_260_);
lean_dec(v_a_260_);
lean_dec_ref(v_a_259_);
lean_dec(v_a_258_);
lean_dec_ref(v_a_257_);
lean_dec(v_a_256_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep(lean_object* v_00_u03b1_263_, lean_object* v_preGoal_264_, lean_object* v_postGoals_265_, lean_object* v_tacticBuilder_266_, lean_object* v_x_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_preGoal_264_, v_postGoals_265_, v_tacticBuilder_266_, v_x_267_, v_a_268_, v_a_270_, v_a_271_, v_a_272_, v_a_273_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_withOptScriptStep___boxed(lean_object* v_00_u03b1_276_, lean_object* v_preGoal_277_, lean_object* v_postGoals_278_, lean_object* v_tacticBuilder_279_, lean_object* v_x_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_aesop_Aesop_withOptScriptStep(v_00_u03b1_276_, v_preGoal_277_, v_postGoals_278_, v_tacticBuilder_279_, v_x_280_, v_a_281_, v_a_282_, v_a_283_, v_a_284_, v_a_285_, v_a_286_);
lean_dec(v_a_286_);
lean_dec_ref(v_a_285_);
lean_dec(v_a_284_);
lean_dec_ref(v_a_283_);
lean_dec(v_a_282_);
lean_dec(v_a_281_);
return v_res_288_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BaseM(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_Step(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BaseM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_BaseM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_Step(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BaseM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_Step(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_ScriptM(builtin);
}
#ifdef __cplusplus
}
#endif
