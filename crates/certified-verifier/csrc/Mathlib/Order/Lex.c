// Lean compiler output
// Module: Mathlib.Order.Lex
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Defs
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
lean_object* lp_mathlib_Equiv_refl(lean_object*);
static lean_once_cell_t lp_mathlib_toLex___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_toLex___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_toLex(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ofLex(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instBEqLex___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instBEqLex___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instBEqLex___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instBEqLex___redArg___closed__0 = (const lean_object*)&lp_mathlib_instBEqLex___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toColex(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ofColex(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBEqColex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instBEqColex(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunColex___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunColex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Colex_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Colex_rec(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_toLex___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toLex(lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ofLex(lean_object* v_00_u03b1_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg___lam__0(lean_object* v_self_6_, lean_object* v___y_7_){
_start:
{
lean_object* v_toFun_8_; lean_object* v___x_9_; 
v_toFun_8_ = lean_ctor_get(v_self_6_, 0);
lean_inc(v_toFun_8_);
lean_dec_ref(v_self_6_);
v___x_9_ = lean_apply_1(v_toFun_8_, v___y_7_);
return v___x_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instBEqLex___redArg___lam__1(lean_object* v___f_10_, lean_object* v_inst_11_, lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; uint8_t v___x_18_; 
v___x_14_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
lean_inc(v___f_10_);
v___x_15_ = lean_apply_2(v___f_10_, v___x_14_, v_a_12_);
v___x_16_ = lean_apply_2(v___f_10_, v___x_14_, v_b_13_);
v___x_17_ = lean_apply_2(v_inst_11_, v___x_15_, v___x_16_);
v___x_18_ = lean_unbox(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg___lam__1___boxed(lean_object* v___f_19_, lean_object* v_inst_20_, lean_object* v_a_21_, lean_object* v_b_22_){
_start:
{
uint8_t v_res_23_; lean_object* v_r_24_; 
v_res_23_ = lp_mathlib_instBEqLex___redArg___lam__1(v___f_19_, v_inst_20_, v_a_21_, v_b_22_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___f_27_; lean_object* v___f_28_; 
v___f_27_ = ((lean_object*)(lp_mathlib_instBEqLex___redArg___closed__0));
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_instBEqLex___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_28_, 0, v___f_27_);
lean_closure_set(v___f_28_, 1, v_inst_26_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqLex(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_instBEqLex___redArg(v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___aux__1___redArg(lean_object* v_inst_32_, lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_35_ = lean_apply_2(v_inst_32_, v_a_33_, v_b_34_);
v___x_36_ = lean_unbox(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___aux__1___redArg___boxed(lean_object* v_inst_37_, lean_object* v_a_38_, lean_object* v_b_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_mathlib_instDecidableEqLex___aux__1___redArg(v_inst_37_, v_a_38_, v_b_39_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___aux__1(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_a_44_, lean_object* v_b_45_){
_start:
{
lean_object* v___x_46_; uint8_t v___x_47_; 
v___x_46_ = lean_apply_2(v_inst_43_, v_a_44_, v_b_45_);
v___x_47_ = lean_unbox(v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___aux__1___boxed(lean_object* v_00_u03b1_48_, lean_object* v_inst_49_, lean_object* v_a_50_, lean_object* v_b_51_){
_start:
{
uint8_t v_res_52_; lean_object* v_r_53_; 
v_res_52_ = lp_mathlib_instDecidableEqLex___aux__1(v_00_u03b1_48_, v_inst_49_, v_a_50_, v_b_51_);
v_r_53_ = lean_box(v_res_52_);
return v_r_53_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex___redArg(lean_object* v_inst_54_, lean_object* v_a_55_, lean_object* v_b_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = lean_apply_2(v_inst_54_, v_a_55_, v_b_56_);
v___x_58_ = lean_unbox(v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___redArg___boxed(lean_object* v_inst_59_, lean_object* v_a_60_, lean_object* v_b_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_mathlib_instDecidableEqLex___redArg(v_inst_59_, v_a_60_, v_b_61_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqLex(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_a_66_, lean_object* v_b_67_){
_start:
{
lean_object* v___x_68_; uint8_t v___x_69_; 
v___x_68_ = lean_apply_2(v_inst_65_, v_a_66_, v_b_67_);
v___x_69_ = lean_unbox(v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqLex___boxed(lean_object* v_00_u03b1_70_, lean_object* v_inst_71_, lean_object* v_a_72_, lean_object* v_b_73_){
_start:
{
uint8_t v_res_74_; lean_object* v_r_75_; 
v_res_74_ = lp_mathlib_instDecidableEqLex(v_00_u03b1_70_, v_inst_71_, v_a_72_, v_b_73_);
v_r_75_ = lean_box(v_res_74_);
return v_r_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___redArg(lean_object* v_inst_76_){
_start:
{
lean_inc(v_inst_76_);
return v_inst_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___redArg___boxed(lean_object* v_inst_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_instInhabitedLex___aux__1___redArg(v_inst_77_);
lean_dec(v_inst_77_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1(lean_object* v_00_u03b1_79_, lean_object* v_inst_80_){
_start:
{
lean_inc(v_inst_80_);
return v_inst_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___aux__1___boxed(lean_object* v_00_u03b1_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_instInhabitedLex___aux__1(v_00_u03b1_81_, v_inst_82_);
lean_dec(v_inst_82_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___redArg(lean_object* v_inst_84_){
_start:
{
lean_inc(v_inst_84_);
return v_inst_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___redArg___boxed(lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_instInhabitedLex___redArg(v_inst_85_);
lean_dec(v_inst_85_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_){
_start:
{
lean_inc(v_inst_88_);
return v_inst_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedLex___boxed(lean_object* v_00_u03b1_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_instInhabitedLex(v_00_u03b1_89_, v_inst_90_);
lean_dec(v_inst_90_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___redArg(lean_object* v_inst_92_){
_start:
{
lean_inc(v_inst_92_);
return v_inst_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___redArg___boxed(lean_object* v_inst_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_instUniqueLex___redArg(v_inst_93_);
lean_dec(v_inst_93_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex(lean_object* v_00_u03b1_95_, lean_object* v_inst_96_){
_start:
{
lean_inc(v_inst_96_);
return v_inst_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueLex___boxed(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_instUniqueLex(v_00_u03b1_97_, v_inst_98_);
lean_dec(v_inst_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex___redArg___lam__0(lean_object* v_H_100_, lean_object* v_f_101_){
_start:
{
lean_object* v___x_102_; lean_object* v_toFun_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_102_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
v_toFun_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_toFun_103_);
v___x_104_ = lean_apply_1(v_toFun_103_, v_f_101_);
v___x_105_ = lean_apply_1(v_H_100_, v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex___redArg(lean_object* v_H_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_instCoeFunLex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_107_, 0, v_H_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunLex(lean_object* v_00_u03b1_108_, lean_object* v_00_u03b3_109_, lean_object* v_H_110_){
_start:
{
lean_object* v___f_111_; 
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_instCoeFunLex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_111_, 0, v_H_110_);
return v___f_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_rec___redArg(lean_object* v_h_112_, lean_object* v_a_113_){
_start:
{
lean_object* v___x_114_; lean_object* v_toFun_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
v_toFun_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_toFun_115_);
v___x_116_ = lean_apply_1(v_toFun_115_, v_a_113_);
v___x_117_ = lean_apply_1(v_h_112_, v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_rec(lean_object* v_00_u03b1_118_, lean_object* v_00_u03b2_119_, lean_object* v_h_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_Lex_rec___redArg(v_h_120_, v_a_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toColex(lean_object* v___y_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ofColex(lean_object* v_00_u03b1_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqColex___redArg(lean_object* v_inst_127_){
_start:
{
lean_object* v___f_128_; lean_object* v___f_129_; 
v___f_128_ = ((lean_object*)(lp_mathlib_instBEqLex___redArg___closed__0));
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_instBEqLex___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_129_, 0, v___f_128_);
lean_closure_set(v___f_129_, 1, v_inst_127_);
return v___f_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instBEqColex(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_mathlib_instBEqColex___redArg(v_inst_131_);
return v___x_132_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___aux__1___redArg(lean_object* v_inst_133_, lean_object* v_a_134_, lean_object* v_b_135_){
_start:
{
lean_object* v___x_136_; uint8_t v___x_137_; 
v___x_136_ = lean_apply_2(v_inst_133_, v_a_134_, v_b_135_);
v___x_137_ = lean_unbox(v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___aux__1___redArg___boxed(lean_object* v_inst_138_, lean_object* v_a_139_, lean_object* v_b_140_){
_start:
{
uint8_t v_res_141_; lean_object* v_r_142_; 
v_res_141_ = lp_mathlib_instDecidableEqColex___aux__1___redArg(v_inst_138_, v_a_139_, v_b_140_);
v_r_142_ = lean_box(v_res_141_);
return v_r_142_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___aux__1(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_a_145_, lean_object* v_b_146_){
_start:
{
lean_object* v___x_147_; uint8_t v___x_148_; 
v___x_147_ = lean_apply_2(v_inst_144_, v_a_145_, v_b_146_);
v___x_148_ = lean_unbox(v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___aux__1___boxed(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_, lean_object* v_a_151_, lean_object* v_b_152_){
_start:
{
uint8_t v_res_153_; lean_object* v_r_154_; 
v_res_153_ = lp_mathlib_instDecidableEqColex___aux__1(v_00_u03b1_149_, v_inst_150_, v_a_151_, v_b_152_);
v_r_154_ = lean_box(v_res_153_);
return v_r_154_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex___redArg(lean_object* v_inst_155_, lean_object* v_a_156_, lean_object* v_b_157_){
_start:
{
lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_158_ = lean_apply_2(v_inst_155_, v_a_156_, v_b_157_);
v___x_159_ = lean_unbox(v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___redArg___boxed(lean_object* v_inst_160_, lean_object* v_a_161_, lean_object* v_b_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_mathlib_instDecidableEqColex___redArg(v_inst_160_, v_a_161_, v_b_162_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqColex(lean_object* v_00_u03b1_165_, lean_object* v_inst_166_, lean_object* v_a_167_, lean_object* v_b_168_){
_start:
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = lean_apply_2(v_inst_166_, v_a_167_, v_b_168_);
v___x_170_ = lean_unbox(v___x_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqColex___boxed(lean_object* v_00_u03b1_171_, lean_object* v_inst_172_, lean_object* v_a_173_, lean_object* v_b_174_){
_start:
{
uint8_t v_res_175_; lean_object* v_r_176_; 
v_res_175_ = lp_mathlib_instDecidableEqColex(v_00_u03b1_171_, v_inst_172_, v_a_173_, v_b_174_);
v_r_176_ = lean_box(v_res_175_);
return v_r_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___redArg(lean_object* v_inst_177_){
_start:
{
lean_inc(v_inst_177_);
return v_inst_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___redArg___boxed(lean_object* v_inst_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_instInhabitedColex___aux__1___redArg(v_inst_178_);
lean_dec(v_inst_178_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1(lean_object* v_00_u03b1_180_, lean_object* v_inst_181_){
_start:
{
lean_inc(v_inst_181_);
return v_inst_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___aux__1___boxed(lean_object* v_00_u03b1_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_instInhabitedColex___aux__1(v_00_u03b1_182_, v_inst_183_);
lean_dec(v_inst_183_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___redArg(lean_object* v_inst_185_){
_start:
{
lean_inc(v_inst_185_);
return v_inst_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___redArg___boxed(lean_object* v_inst_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_instInhabitedColex___redArg(v_inst_186_);
lean_dec(v_inst_186_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex(lean_object* v_00_u03b1_188_, lean_object* v_inst_189_){
_start:
{
lean_inc(v_inst_189_);
return v_inst_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedColex___boxed(lean_object* v_00_u03b1_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_instInhabitedColex(v_00_u03b1_190_, v_inst_191_);
lean_dec(v_inst_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___redArg(lean_object* v_inst_193_){
_start:
{
lean_inc(v_inst_193_);
return v_inst_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___redArg___boxed(lean_object* v_inst_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_instUniqueColex___redArg(v_inst_194_);
lean_dec(v_inst_194_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex(lean_object* v_00_u03b1_196_, lean_object* v_inst_197_){
_start:
{
lean_inc(v_inst_197_);
return v_inst_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueColex___boxed(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_instUniqueColex(v_00_u03b1_198_, v_inst_199_);
lean_dec(v_inst_199_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunColex___redArg(lean_object* v_H_201_){
_start:
{
lean_object* v___f_202_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_instCoeFunLex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_202_, 0, v_H_201_);
return v___f_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeFunColex(lean_object* v_00_u03b1_203_, lean_object* v_00_u03b3_204_, lean_object* v_H_205_){
_start:
{
lean_object* v___f_206_; 
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_instCoeFunLex___redArg___lam__0), 2, 1);
lean_closure_set(v___f_206_, 0, v_H_205_);
return v___f_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Colex_rec___redArg(lean_object* v_h_207_, lean_object* v_a_208_){
_start:
{
lean_object* v___x_209_; lean_object* v_toFun_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_209_ = lean_obj_once(&lp_mathlib_toLex___closed__0, &lp_mathlib_toLex___closed__0_once, _init_lp_mathlib_toLex___closed__0);
v_toFun_210_ = lean_ctor_get(v___x_209_, 0);
lean_inc(v_toFun_210_);
v___x_211_ = lean_apply_1(v_toFun_210_, v_a_208_);
v___x_212_ = lean_apply_1(v_h_207_, v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Colex_rec(lean_object* v_00_u03b1_213_, lean_object* v_00_u03b2_214_, lean_object* v_h_215_, lean_object* v_a_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lp_mathlib_Colex_rec___redArg(v_h_215_, v_a_216_);
return v___x_217_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
