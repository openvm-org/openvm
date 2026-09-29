// Lean compiler output
// Module: Mathlib.Algebra.Group.Subsemigroup.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Data.SetLike.Basic public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instSetLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subsemigroup_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subsemigroup_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instPartialOrder___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subsemigroup_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subsemigroup_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subsemigroup_instMin___closed__0 = (const lean_object*)&lp_mathlib_Subsemigroup_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubsemigroup_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubsemigroup_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubsemigroup_instMin___closed__0 = (const lean_object*)&lp_mathlib_AddSubsemigroup_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulMemClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulMemClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulMemClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_MulMemClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instSetLike(lean_object* v_M_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instSetLike___boxed(lean_object* v_M_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Subsemigroup_instSetLike(v_M_4_, v_inst_5_);
lean_dec(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instSetLike(lean_object* v_M_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instSetLike___boxed(lean_object* v_M_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_AddSubsemigroup_instSetLike(v_M_10_, v_inst_11_);
lean_dec(v_inst_11_);
return v_res_12_;
}
}
static lean_object* _init_lp_mathlib_Subsemigroup_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_box(0);
v___x_14_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instPartialOrder(lean_object* v_M_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Subsemigroup_instPartialOrder___closed__0, &lp_mathlib_Subsemigroup_instPartialOrder___closed__0_once, _init_lp_mathlib_Subsemigroup_instPartialOrder___closed__0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instPartialOrder___boxed(lean_object* v_M_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Subsemigroup_instPartialOrder(v_M_18_, v_inst_19_);
lean_dec(v_inst_19_);
return v_res_20_;
}
}
static lean_object* _init_lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_box(0);
v___x_22_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instPartialOrder(lean_object* v_M_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0, &lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0_once, _init_lp_mathlib_AddSubsemigroup_instPartialOrder___closed__0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instPartialOrder___boxed(lean_object* v_M_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddSubsemigroup_instPartialOrder(v_M_26_, v_inst_27_);
lean_dec(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_ofClass(lean_object* v_S_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_s_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_ofClass___boxed(lean_object* v_S_36_, lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_s_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Subsemigroup_ofClass(v_S_36_, v_M_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_s_41_);
lean_dec(v_s_41_);
lean_dec(v_inst_38_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_ofClass(lean_object* v_S_43_, lean_object* v_M_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_s_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_ofClass___boxed(lean_object* v_S_50_, lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_s_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddSubsemigroup_ofClass(v_S_50_, v_M_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_s_55_);
lean_dec(v_s_55_);
lean_dec(v_inst_52_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_copy(lean_object* v_M_57_, lean_object* v_inst_58_, lean_object* v_S_59_, lean_object* v_s_60_, lean_object* v_hs_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_box(0);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_copy___boxed(lean_object* v_M_63_, lean_object* v_inst_64_, lean_object* v_S_65_, lean_object* v_s_66_, lean_object* v_hs_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Subsemigroup_copy(v_M_63_, v_inst_64_, v_S_65_, v_s_66_, v_hs_67_);
lean_dec(v_inst_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_copy(lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_S_71_, lean_object* v_s_72_, lean_object* v_hs_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_copy___boxed(lean_object* v_M_75_, lean_object* v_inst_76_, lean_object* v_S_77_, lean_object* v_s_78_, lean_object* v_hs_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_AddSubsemigroup_copy(v_M_75_, v_inst_76_, v_S_77_, v_s_78_, v_hs_79_);
lean_dec(v_inst_76_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instTop(lean_object* v_M_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_box(0);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instTop___boxed(lean_object* v_M_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Subsemigroup_instTop(v_M_84_, v_inst_85_);
lean_dec(v_inst_85_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instTop(lean_object* v_M_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instTop___boxed(lean_object* v_M_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_AddSubsemigroup_instTop(v_M_90_, v_inst_91_);
lean_dec(v_inst_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instBot(lean_object* v_M_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_box(0);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instBot___boxed(lean_object* v_M_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Subsemigroup_instBot(v_M_96_, v_inst_97_);
lean_dec(v_inst_97_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instBot(lean_object* v_M_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instBot___boxed(lean_object* v_M_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_AddSubsemigroup_instBot(v_M_102_, v_inst_103_);
lean_dec(v_inst_103_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instInhabited(lean_object* v_M_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_box(0);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instInhabited___boxed(lean_object* v_M_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Subsemigroup_instInhabited(v_M_108_, v_inst_109_);
lean_dec(v_inst_109_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instInhabited(lean_object* v_M_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lean_box(0);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instInhabited___boxed(lean_object* v_M_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_AddSubsemigroup_instInhabited(v_M_114_, v_inst_115_);
lean_dec(v_inst_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin___lam__0(lean_object* v_S_u2081_117_, lean_object* v_S_u2082_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_box(0);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin(lean_object* v_M_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___f_123_; 
v___f_123_ = ((lean_object*)(lp_mathlib_Subsemigroup_instMin___closed__0));
return v___f_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemigroup_instMin___boxed(lean_object* v_M_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Subsemigroup_instMin(v_M_124_, v_inst_125_);
lean_dec(v_inst_125_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin___lam__0(lean_object* v_S_u2081_127_, lean_object* v_S_u2082_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_box(0);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin(lean_object* v_M_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___f_133_; 
v___f_133_ = ((lean_object*)(lp_mathlib_AddSubsemigroup_instMin___closed__0));
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubsemigroup_instMin___boxed(lean_object* v_M_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_AddSubsemigroup_instMin(v_M_134_, v_inst_135_);
lean_dec(v_inst_135_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_eqLocus(lean_object* v_M_137_, lean_object* v_N_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_f_141_, lean_object* v_g_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_eqLocus___boxed(lean_object* v_M_144_, lean_object* v_N_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_f_148_, lean_object* v_g_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_MulHom_eqLocus(v_M_144_, v_N_145_, v_inst_146_, v_inst_147_, v_f_148_, v_g_149_);
lean_dec(v_g_149_);
lean_dec(v_f_148_);
lean_dec(v_inst_147_);
lean_dec(v_inst_146_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_eqLocus(lean_object* v_M_151_, lean_object* v_N_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_f_155_, lean_object* v_g_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_box(0);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_eqLocus___boxed(lean_object* v_M_158_, lean_object* v_N_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_f_162_, lean_object* v_g_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_AddHom_eqLocus(v_M_158_, v_N_159_, v_inst_160_, v_inst_161_, v_f_162_, v_g_163_);
lean_dec(v_g_163_);
lean_dec(v_f_162_);
lean_dec(v_inst_161_);
lean_dec(v_inst_160_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___redArg___lam__0(lean_object* v_inst_165_, lean_object* v_a_166_, lean_object* v_b_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_apply_2(v_inst_165_, v_a_166_, v_b_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___redArg(lean_object* v_inst_169_){
_start:
{
lean_object* v___f_170_; 
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_170_, 0, v_inst_169_);
return v___f_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul(lean_object* v_M_171_, lean_object* v_A_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_hA_175_, lean_object* v_S_x27_176_){
_start:
{
lean_object* v___f_177_; 
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_177_, 0, v_inst_173_);
return v___f_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_mul___boxed(lean_object* v_M_178_, lean_object* v_A_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_hA_182_, lean_object* v_S_x27_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_MulMemClass_mul(v_M_178_, v_A_179_, v_inst_180_, v_inst_181_, v_hA_182_, v_S_x27_183_);
lean_dec(v_S_x27_183_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add___redArg(lean_object* v_inst_185_){
_start:
{
lean_object* v___f_186_; 
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_186_, 0, v_inst_185_);
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add(lean_object* v_M_187_, lean_object* v_A_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_hA_191_, lean_object* v_S_x27_192_){
_start:
{
lean_object* v___f_193_; 
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_193_, 0, v_inst_189_);
return v___f_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_add___boxed(lean_object* v_M_194_, lean_object* v_A_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_hA_198_, lean_object* v_S_x27_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_AddMemClass_add(v_M_194_, v_A_195_, v_inst_196_, v_inst_197_, v_hA_198_, v_S_x27_199_);
lean_dec(v_S_x27_199_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup___redArg(lean_object* v_inst_201_){
_start:
{
lean_object* v___f_202_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_202_, 0, v_inst_201_);
return v___f_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup(lean_object* v_M_203_, lean_object* v_inst_204_, lean_object* v_A_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_S_208_){
_start:
{
lean_object* v___f_209_; 
v___f_209_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_209_, 0, v_inst_204_);
return v___f_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toSemigroup___boxed(lean_object* v_M_210_, lean_object* v_inst_211_, lean_object* v_A_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_S_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_MulMemClass_toSemigroup(v_M_210_, v_inst_211_, v_A_212_, v_inst_213_, v_inst_214_, v_S_215_);
lean_dec(v_S_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v___f_218_; 
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_218_, 0, v_inst_217_);
return v___f_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup(lean_object* v_M_219_, lean_object* v_inst_220_, lean_object* v_A_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_S_224_){
_start:
{
lean_object* v___f_225_; 
v___f_225_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_225_, 0, v_inst_220_);
return v___f_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddSemigroup___boxed(lean_object* v_M_226_, lean_object* v_inst_227_, lean_object* v_A_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_S_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_AddMemClass_toAddSemigroup(v_M_226_, v_inst_227_, v_A_228_, v_inst_229_, v_inst_230_, v_S_231_);
lean_dec(v_S_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup___redArg(lean_object* v_inst_233_){
_start:
{
lean_object* v___f_234_; 
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_234_, 0, v_inst_233_);
return v___f_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup(lean_object* v_M_235_, lean_object* v_inst_236_, lean_object* v_A_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_S_240_){
_start:
{
lean_object* v___f_241_; 
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_241_, 0, v_inst_236_);
return v___f_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_toCommSemigroup___boxed(lean_object* v_M_242_, lean_object* v_inst_243_, lean_object* v_A_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_S_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_MulMemClass_toCommSemigroup(v_M_242_, v_inst_243_, v_A_244_, v_inst_245_, v_inst_246_, v_S_247_);
lean_dec(v_S_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup___redArg(lean_object* v_inst_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_250_, 0, v_inst_249_);
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup(lean_object* v_M_251_, lean_object* v_inst_252_, lean_object* v_A_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_S_256_){
_start:
{
lean_object* v___f_257_; 
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_257_, 0, v_inst_252_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_toAddCommSemigroup___boxed(lean_object* v_M_258_, lean_object* v_inst_259_, lean_object* v_A_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_S_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_AddMemClass_toAddCommSemigroup(v_M_258_, v_inst_259_, v_A_260_, v_inst_261_, v_inst_262_, v_S_263_);
lean_dec(v_S_263_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___lam__0(lean_object* v_self_265_){
_start:
{
lean_inc(v_self_265_);
return v_self_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___lam__0___boxed(lean_object* v_self_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_MulMemClass_subtype___lam__0(v_self_266_);
lean_dec(v_self_266_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype(lean_object* v_M_269_, lean_object* v_A_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_hA_273_, lean_object* v_S_x27_274_){
_start:
{
lean_object* v___f_275_; 
v___f_275_ = ((lean_object*)(lp_mathlib_MulMemClass_subtype___closed__0));
return v___f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulMemClass_subtype___boxed(lean_object* v_M_276_, lean_object* v_A_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_hA_280_, lean_object* v_S_x27_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_MulMemClass_subtype(v_M_276_, v_A_277_, v_inst_278_, v_inst_279_, v_hA_280_, v_S_x27_281_);
lean_dec(v_S_x27_281_);
lean_dec(v_inst_278_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_subtype(lean_object* v_M_283_, lean_object* v_A_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_hA_287_, lean_object* v_S_x27_288_){
_start:
{
lean_object* v___f_289_; 
v___f_289_ = ((lean_object*)(lp_mathlib_MulMemClass_subtype___closed__0));
return v___f_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMemClass_subtype___boxed(lean_object* v_M_290_, lean_object* v_A_291_, lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_hA_294_, lean_object* v_S_x27_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib_AddMemClass_subtype(v_M_290_, v_A_291_, v_inst_292_, v_inst_293_, v_hA_294_, v_S_x27_295_);
lean_dec(v_S_x27_295_);
lean_dec(v_inst_292_);
return v_res_296_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
