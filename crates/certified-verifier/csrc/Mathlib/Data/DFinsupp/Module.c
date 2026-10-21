// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Module
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.Pi public import Mathlib.Algebra.Module.LinearMap.Defs public import Mathlib.Algebra.Module.Pi public import Mathlib.Data.DFinsupp.Defs
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
lean_object* lp_mathlib_DFinsupp_mapRange___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_subtypeDomain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_coeFnLinearMap___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_coeFnLinearMap___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_coeFnLinearMap___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_c_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_apply_3(v_inst_1_, v_x_3_, v_c_2_, v_x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__1(lean_object* v_inst_6_, lean_object* v_c_7_, lean_object* v_v_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___x_10_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_9_, 0, v_inst_6_);
lean_closure_set(v___f_9_, 1, v_c_7_);
v___x_10_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_9_, v_v_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_12_, 0, v_inst_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass(lean_object* v_00_u03b9_13_, lean_object* v_00_u03b3_14_, lean_object* v_00_u03b2_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSMulZeroClass___boxed(lean_object* v_00_u03b9_19_, lean_object* v_00_u03b3_20_, lean_object* v_00_u03b2_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_DFinsupp_instSMulZeroClass(v_00_u03b9_19_, v_00_u03b3_20_, v_00_u03b2_21_, v_inst_22_, v_inst_23_);
lean_dec(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___redArg___lam__0(lean_object* v_inst_25_, lean_object* v_i_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_apply_3(v_inst_25_, v_i_26_, v___y_27_, v___y_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v___f_31_; lean_object* v___f_32_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_distribMulAction___redArg___lam__0), 4, 1);
lean_closure_set(v___f_31_, 0, v_inst_30_);
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSMulZeroClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_32_, 0, v___f_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction(lean_object* v_00_u03b9_33_, lean_object* v_00_u03b3_34_, lean_object* v_00_u03b2_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_DFinsupp_distribMulAction___redArg(v_inst_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction___boxed(lean_object* v_00_u03b9_40_, lean_object* v_00_u03b3_41_, lean_object* v_00_u03b2_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_DFinsupp_distribMulAction(v_00_u03b9_40_, v_00_u03b3_41_, v_00_u03b2_42_, v_inst_43_, v_inst_44_, v_inst_45_);
lean_dec_ref(v_inst_44_);
lean_dec_ref(v_inst_43_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module___redArg(lean_object* v_inst_47_){
_start:
{
lean_object* v___f_48_; lean_object* v___x_49_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_distribMulAction___redArg___lam__0), 4, 1);
lean_closure_set(v___f_48_, 0, v_inst_47_);
v___x_49_ = lp_mathlib_DFinsupp_distribMulAction___redArg(v___f_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module(lean_object* v_00_u03b9_50_, lean_object* v_00_u03b3_51_, lean_object* v_00_u03b2_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_DFinsupp_module___redArg(v_inst_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_module___boxed(lean_object* v_00_u03b9_57_, lean_object* v_00_u03b3_58_, lean_object* v_00_u03b2_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_DFinsupp_module(v_00_u03b9_57_, v_00_u03b3_58_, v_00_u03b2_59_, v_inst_60_, v_inst_61_, v_inst_62_);
lean_dec_ref(v_inst_61_);
lean_dec_ref(v_inst_60_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___lam__0(lean_object* v_f_64_, lean_object* v___y_65_){
_start:
{
lean_object* v_toFun_66_; lean_object* v___x_67_; 
v_toFun_66_ = lean_ctor_get(v_f_64_, 0);
lean_inc(v_toFun_66_);
lean_dec_ref(v_f_64_);
v___x_67_ = lean_apply_1(v_toFun_66_, v___y_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap(lean_object* v_00_u03b9_69_, lean_object* v_00_u03b3_70_, lean_object* v_00_u03b2_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___f_75_; 
v___f_75_ = ((lean_object*)(lp_mathlib_DFinsupp_coeFnLinearMap___closed__0));
return v___f_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnLinearMap___boxed(lean_object* v_00_u03b9_76_, lean_object* v_00_u03b3_77_, lean_object* v_00_u03b2_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_DFinsupp_coeFnLinearMap(v_00_u03b9_76_, v_00_u03b3_77_, v_00_u03b2_78_, v_inst_79_, v_inst_80_, v_inst_81_);
lean_dec(v_inst_81_);
lean_dec_ref(v_inst_80_);
lean_dec_ref(v_inst_79_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___redArg___lam__0(lean_object* v_inst_83_, lean_object* v_i_84_){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v_toZero_88_; 
v___x_85_ = lean_apply_1(v_inst_83_, v_i_84_);
v___x_86_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_85_);
lean_dec_ref(v___x_85_);
v___x_87_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_86_);
v_toZero_88_ = lean_ctor_get(v___x_87_, 0);
lean_inc(v_toZero_88_);
lean_dec_ref(v___x_87_);
return v_toZero_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___redArg(lean_object* v_inst_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v___f_91_; lean_object* v___x_92_; 
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_filterLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_91_, 0, v_inst_89_);
v___x_92_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_filter), 6, 5);
lean_closure_set(v___x_92_, 0, lean_box(0));
lean_closure_set(v___x_92_, 1, lean_box(0));
lean_closure_set(v___x_92_, 2, v___f_91_);
lean_closure_set(v___x_92_, 3, lean_box(0));
lean_closure_set(v___x_92_, 4, v_inst_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap(lean_object* v_00_u03b9_93_, lean_object* v_00_u03b3_94_, lean_object* v_00_u03b2_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_p_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_DFinsupp_filterLinearMap___redArg(v_inst_97_, v_inst_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterLinearMap___boxed(lean_object* v_00_u03b9_102_, lean_object* v_00_u03b3_103_, lean_object* v_00_u03b2_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_p_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_DFinsupp_filterLinearMap(v_00_u03b9_102_, v_00_u03b3_103_, v_00_u03b2_104_, v_inst_105_, v_inst_106_, v_inst_107_, v_p_108_, v_inst_109_);
lean_dec(v_inst_107_);
lean_dec_ref(v_inst_105_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap___redArg(lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___f_113_; lean_object* v___x_114_; 
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_filterLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_113_, 0, v_inst_111_);
v___x_114_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeDomain___boxed), 6, 5);
lean_closure_set(v___x_114_, 0, lean_box(0));
lean_closure_set(v___x_114_, 1, lean_box(0));
lean_closure_set(v___x_114_, 2, v___f_113_);
lean_closure_set(v___x_114_, 3, lean_box(0));
lean_closure_set(v___x_114_, 4, v_inst_112_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap(lean_object* v_00_u03b9_115_, lean_object* v_00_u03b3_116_, lean_object* v_00_u03b2_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_p_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_DFinsupp_subtypeDomainLinearMap___redArg(v_inst_119_, v_inst_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainLinearMap___boxed(lean_object* v_00_u03b9_124_, lean_object* v_00_u03b3_125_, lean_object* v_00_u03b2_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_p_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_DFinsupp_subtypeDomainLinearMap(v_00_u03b9_124_, v_00_u03b3_125_, v_00_u03b2_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_p_130_, v_inst_131_);
lean_dec(v_inst_129_);
lean_dec_ref(v_inst_127_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___redArg___lam__0(lean_object* v_inst_133_, lean_object* v_i_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v___x_137_; lean_object* v___x_24__overap_138_; lean_object* v___x_139_; 
v___x_137_ = lean_apply_1(v_inst_133_, v_i_134_);
v___x_24__overap_138_ = lp_mathlib_DFinsupp_distribMulAction___redArg(v___x_137_);
v___x_139_ = lean_apply_2(v___x_24__overap_138_, v___y_135_, v___y_136_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___redArg(lean_object* v_inst_140_){
_start:
{
lean_object* v___f_141_; lean_object* v___x_142_; 
v___f_141_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_distribMulAction_u2082___redArg___lam__0), 4, 1);
lean_closure_set(v___f_141_, 0, v_inst_140_);
v___x_142_ = lp_mathlib_DFinsupp_distribMulAction___redArg(v___f_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082(lean_object* v_00_u03b9_143_, lean_object* v_00_u03b3_144_, lean_object* v_00_u03b1_145_, lean_object* v_00_u03b4_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_DFinsupp_distribMulAction_u2082___redArg(v_inst_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_distribMulAction_u2082___boxed(lean_object* v_00_u03b9_151_, lean_object* v_00_u03b3_152_, lean_object* v_00_u03b1_153_, lean_object* v_00_u03b4_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_DFinsupp_distribMulAction_u2082(v_00_u03b9_151_, v_00_u03b3_152_, v_00_u03b1_153_, v_00_u03b4_154_, v_inst_155_, v_inst_156_, v_inst_157_);
lean_dec_ref(v_inst_156_);
lean_dec_ref(v_inst_155_);
return v_res_158_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
}
#ifdef __cplusplus
}
#endif
