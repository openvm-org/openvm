// Lean compiler output
// Module: Mathlib.Algebra.Group.Conj
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.End public import Mathlib.Algebra.Group.Semiconj.Units
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Quotient_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddConj_setoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsAddConj_setoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_slow_x2dfailing__instance__priority;
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Quotient_lift, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_ConjClasses_mkEquiv___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1 = (const lean_object*)&lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mkEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mkEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsConj_setoid___boxed(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_IsConj_setoid(v_00_u03b1_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddConj_setoid(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsAddConj_setoid___boxed(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_IsAddConj_setoid(v_00_u03b1_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg(lean_object* v_a_13_){
_start:
{
lean_inc(v_a_13_);
return v_a_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___redArg___boxed(lean_object* v_a_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_ConjClasses_mk___redArg(v_a_14_);
lean_dec(v_a_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_a_18_){
_start:
{
lean_inc(v_a_18_);
return v_a_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mk___boxed(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_a_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_ConjClasses_mk(v_00_u03b1_19_, v_inst_20_, v_a_21_);
lean_dec(v_a_21_);
lean_dec_ref(v_inst_20_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___redArg(lean_object* v_a_23_){
_start:
{
lean_inc(v_a_23_);
return v_a_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___redArg___boxed(lean_object* v_a_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_AddConjClasses_mk___redArg(v_a_24_);
lean_dec(v_a_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_a_28_){
_start:
{
lean_inc(v_a_28_);
return v_a_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mk___boxed(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_AddConjClasses_mk(v_00_u03b1_29_, v_inst_30_, v_a_31_);
lean_dec(v_a_31_);
lean_dec_ref(v_inst_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg(lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v_toOne_36_; 
v___x_34_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_33_);
v___x_35_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_34_);
v_toOne_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_toOne_36_);
lean_dec_ref(v___x_35_);
return v_toOne_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___redArg___boxed(lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_ConjClasses_instInhabited___redArg(v_inst_37_);
lean_dec_ref(v_inst_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_ConjClasses_instInhabited___redArg(v_inst_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instInhabited___boxed(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_ConjClasses_instInhabited(v_00_u03b1_42_, v_inst_43_);
lean_dec_ref(v_inst_43_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v_toZero_48_; 
v___x_46_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_45_);
v___x_47_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_46_);
v_toZero_48_ = lean_ctor_get(v___x_47_, 0);
lean_inc(v_toZero_48_);
lean_dec_ref(v___x_47_);
return v_toZero_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___redArg___boxed(lean_object* v_inst_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_AddConjClasses_instInhabited___redArg(v_inst_49_);
lean_dec_ref(v_inst_49_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_AddConjClasses_instInhabited___redArg(v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instInhabited___boxed(lean_object* v_00_u03b1_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddConjClasses_instInhabited(v_00_u03b1_54_, v_inst_55_);
lean_dec_ref(v_inst_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg(lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v_toOne_60_; 
v___x_58_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_57_);
v___x_59_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_58_);
v_toOne_60_ = lean_ctor_get(v___x_59_, 0);
lean_inc(v_toOne_60_);
lean_dec_ref(v___x_59_);
return v_toOne_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___redArg___boxed(lean_object* v_inst_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_ConjClasses_instOne___redArg(v_inst_61_);
lean_dec_ref(v_inst_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne(lean_object* v_00_u03b1_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_ConjClasses_instOne___redArg(v_inst_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instOne___boxed(lean_object* v_00_u03b1_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_ConjClasses_instOne(v_00_u03b1_66_, v_inst_67_);
lean_dec_ref(v_inst_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___redArg(lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v_toZero_72_; 
v___x_70_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_69_);
v___x_71_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_70_);
v_toZero_72_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toZero_72_);
lean_dec_ref(v___x_71_);
return v_toZero_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___redArg___boxed(lean_object* v_inst_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_AddConjClasses_instZero___redArg(v_inst_73_);
lean_dec_ref(v_inst_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_AddConjClasses_instZero___redArg(v_inst_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instZero___boxed(lean_object* v_00_u03b1_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_AddConjClasses_instZero(v_00_u03b1_78_, v_inst_79_);
lean_dec_ref(v_inst_79_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___redArg(lean_object* v_f_81_, lean_object* v_a_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_apply_1(v_f_81_, v_a_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map(lean_object* v_00_u03b1_84_, lean_object* v_00_u03b2_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_f_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lean_apply_1(v_f_88_, v_a_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_map___boxed(lean_object* v_00_u03b1_91_, lean_object* v_00_u03b2_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_f_95_, lean_object* v_a_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_ConjClasses_map(v_00_u03b1_91_, v_00_u03b2_92_, v_inst_93_, v_inst_94_, v_f_95_, v_a_96_);
lean_dec_ref(v_inst_94_);
lean_dec_ref(v_inst_93_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map___redArg(lean_object* v_f_98_, lean_object* v_a_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_apply_1(v_f_98_, v_a_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map(lean_object* v_00_u03b1_101_, lean_object* v_00_u03b2_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_f_105_, lean_object* v_a_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_apply_1(v_f_105_, v_a_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_map___boxed(lean_object* v_00_u03b1_108_, lean_object* v_00_u03b2_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_f_112_, lean_object* v_a_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_AddConjClasses_map(v_00_u03b1_108_, v_00_u03b2_109_, v_inst_110_, v_inst_111_, v_f_112_, v_a_113_);
lean_dec_ref(v_inst_111_);
lean_dec_ref(v_inst_110_);
return v_res_114_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_slow_x2dfailing__instance__priority(void){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_box(0);
return v___x_115_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___redArg(lean_object* v_inst_116_, lean_object* v_a_117_, lean_object* v_b_118_){
_start:
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = lean_apply_2(v_inst_116_, v_a_117_, v_b_118_);
v___x_120_ = lean_unbox(v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___redArg___boxed(lean_object* v_inst_121_, lean_object* v_a_122_, lean_object* v_b_123_){
_start:
{
uint8_t v_res_124_; lean_object* v_r_125_; 
v_res_124_ = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___redArg(v_inst_121_, v_a_122_, v_b_123_);
v_r_125_ = lean_box(v_res_124_);
return v_r_125_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_a_129_, lean_object* v_b_130_){
_start:
{
lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_131_ = lean_apply_2(v_inst_128_, v_a_129_, v_b_130_);
v___x_132_ = lean_unbox(v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1___boxed(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_a_136_, lean_object* v_b_137_){
_start:
{
uint8_t v_res_138_; lean_object* v_r_139_; 
v_res_138_ = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___aux__1(v_00_u03b1_133_, v_inst_134_, v_inst_135_, v_a_136_, v_b_137_);
lean_dec_ref(v_inst_134_);
v_r_139_ = lean_box(v_res_138_);
return v_r_139_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(lean_object* v_inst_140_, lean_object* v_a_141_, lean_object* v_b_142_){
_start:
{
lean_object* v___x_143_; uint8_t v___x_144_; 
v___x_143_ = lean_apply_2(v_inst_140_, v_a_141_, v_b_142_);
v___x_144_ = lean_unbox(v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg___boxed(lean_object* v_inst_145_, lean_object* v_a_146_, lean_object* v_b_147_){
_start:
{
uint8_t v_res_148_; lean_object* v_r_149_; 
v_res_148_ = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___redArg(v_inst_145_, v_a_146_, v_b_147_);
v_r_149_ = lean_box(v_res_148_);
return v_r_149_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(lean_object* v_00_u03b1_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_a_153_, lean_object* v_b_154_){
_start:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = lean_apply_2(v_inst_152_, v_a_153_, v_b_154_);
v___x_156_ = lean_unbox(v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj___boxed(lean_object* v_00_u03b1_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_a_160_, lean_object* v_b_161_){
_start:
{
uint8_t v_res_162_; lean_object* v_r_163_; 
v_res_162_ = lp_mathlib_ConjClasses_instDecidableEqOfDecidableRelIsConj(v_00_u03b1_157_, v_inst_158_, v_inst_159_, v_a_160_, v_b_161_);
lean_dec_ref(v_inst_158_);
v_r_163_ = lean_box(v_res_162_);
return v_r_163_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___redArg(lean_object* v_inst_164_, lean_object* v_a_165_, lean_object* v_b_166_){
_start:
{
lean_object* v___x_167_; uint8_t v___x_168_; 
v___x_167_ = lean_apply_2(v_inst_164_, v_a_165_, v_b_166_);
v___x_168_ = lean_unbox(v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___redArg___boxed(lean_object* v_inst_169_, lean_object* v_a_170_, lean_object* v_b_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___redArg(v_inst_169_, v_a_170_, v_b_171_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1(lean_object* v_00_u03b1_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_a_177_, lean_object* v_b_178_){
_start:
{
lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_179_ = lean_apply_2(v_inst_176_, v_a_177_, v_b_178_);
v___x_180_ = lean_unbox(v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1___boxed(lean_object* v_00_u03b1_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_a_184_, lean_object* v_b_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___aux__1(v_00_u03b1_181_, v_inst_182_, v_inst_183_, v_a_184_, v_b_185_);
lean_dec_ref(v_inst_182_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___redArg(lean_object* v_inst_188_, lean_object* v_a_189_, lean_object* v_b_190_){
_start:
{
lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_191_ = lean_apply_2(v_inst_188_, v_a_189_, v_b_190_);
v___x_192_ = lean_unbox(v___x_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___redArg___boxed(lean_object* v_inst_193_, lean_object* v_a_194_, lean_object* v_b_195_){
_start:
{
uint8_t v_res_196_; lean_object* v_r_197_; 
v_res_196_ = lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___redArg(v_inst_193_, v_a_194_, v_b_195_);
v_r_197_ = lean_box(v_res_196_);
return v_r_197_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_a_201_, lean_object* v_b_202_){
_start:
{
lean_object* v___x_203_; uint8_t v___x_204_; 
v___x_203_ = lean_apply_2(v_inst_200_, v_a_201_, v_b_202_);
v___x_204_ = lean_unbox(v___x_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj___boxed(lean_object* v_00_u03b1_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_a_208_, lean_object* v_b_209_){
_start:
{
uint8_t v_res_210_; lean_object* v_r_211_; 
v_res_210_ = lp_mathlib_AddConjClasses_instDecidableEqOfDecidableRelIsAddConj(v_00_u03b1_205_, v_inst_206_, v_inst_207_, v_a_208_, v_b_209_);
lean_dec_ref(v_inst_206_);
v_r_211_ = lean_box(v_res_210_);
return v_r_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv___redArg(lean_object* v_inst_216_){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_217_ = lean_alloc_closure((void*)(lp_mathlib_ConjClasses_mk___boxed), 3, 2);
lean_closure_set(v___x_217_, 0, lean_box(0));
lean_closure_set(v___x_217_, 1, v_inst_216_);
v___x_218_ = ((lean_object*)(lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1));
v___x_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_217_);
lean_ctor_set(v___x_219_, 1, v___x_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjClasses_mkEquiv(lean_object* v_00_u03b1_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___x_222_; 
v___x_222_ = lp_mathlib_ConjClasses_mkEquiv___redArg(v_inst_221_);
return v___x_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mkEquiv___redArg(lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_224_ = lean_alloc_closure((void*)(lp_mathlib_AddConjClasses_mk___boxed), 3, 2);
lean_closure_set(v___x_224_, 0, lean_box(0));
lean_closure_set(v___x_224_, 1, v_inst_223_);
v___x_225_ = ((lean_object*)(lp_mathlib_ConjClasses_mkEquiv___redArg___closed__1));
v___x_226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_224_);
lean_ctor_set(v___x_226_, 1, v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddConjClasses_mkEquiv(lean_object* v_00_u03b1_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_AddConjClasses_mkEquiv___redArg(v_inst_228_);
return v___x_229_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_slow_x2dfailing__instance__priority = _init_lp_mathlib_LibraryNote_slow_x2dfailing__instance__priority();
lean_mark_persistent(lp_mathlib_LibraryNote_slow_x2dfailing__instance__priority);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Conj(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Conj(builtin);
}
#ifdef __cplusplus
}
#endif
