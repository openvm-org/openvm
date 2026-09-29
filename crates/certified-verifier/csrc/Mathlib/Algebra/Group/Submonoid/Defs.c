// Lean compiler output
// Module: Mathlib.Algebra.Group.Submonoid.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Subsemigroup.Defs public import Mathlib.Tactic.FastInstance public import Mathlib.Data.Set.Insert
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
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_MulMemClass_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instSetLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submonoid_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submonoid_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instPartialOrder___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddSubmonoid_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddSubmonoid_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submonoid_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submonoid_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submonoid_instMin___closed__0 = (const lean_object*)&lp_mathlib_Submonoid_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubmonoid_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubmonoid_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubmonoid_instMin___closed__0 = (const lean_object*)&lp_mathlib_AddSubmonoid_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocusM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocusM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocusM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocusM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SubmonoidClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubmonoidClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_SubmonoidClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_one(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_zero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMulOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instSetLike(lean_object* v_M_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instSetLike___boxed(lean_object* v_M_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Submonoid_instSetLike(v_M_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instSetLike(lean_object* v_M_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instSetLike___boxed(lean_object* v_M_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_AddSubmonoid_instSetLike(v_M_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
static lean_object* _init_lp_mathlib_Submonoid_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_box(0);
v___x_14_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instPartialOrder(lean_object* v_M_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Submonoid_instPartialOrder___closed__0, &lp_mathlib_Submonoid_instPartialOrder___closed__0_once, _init_lp_mathlib_Submonoid_instPartialOrder___closed__0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instPartialOrder___boxed(lean_object* v_M_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Submonoid_instPartialOrder(v_M_18_, v_inst_19_);
lean_dec_ref(v_inst_19_);
return v_res_20_;
}
}
static lean_object* _init_lp_mathlib_AddSubmonoid_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_box(0);
v___x_22_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instPartialOrder(lean_object* v_M_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_mathlib_AddSubmonoid_instPartialOrder___closed__0, &lp_mathlib_AddSubmonoid_instPartialOrder___closed__0_once, _init_lp_mathlib_AddSubmonoid_instPartialOrder___closed__0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instPartialOrder___boxed(lean_object* v_M_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddSubmonoid_instPartialOrder(v_M_26_, v_inst_27_);
lean_dec_ref(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_ofClass(lean_object* v_S_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_s_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_ofClass___boxed(lean_object* v_S_36_, lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_s_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Submonoid_ofClass(v_S_36_, v_M_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_s_41_);
lean_dec(v_s_41_);
lean_dec_ref(v_inst_38_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_ofClass(lean_object* v_S_43_, lean_object* v_M_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_s_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_ofClass___boxed(lean_object* v_S_50_, lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_s_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddSubmonoid_ofClass(v_S_50_, v_M_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_s_55_);
lean_dec(v_s_55_);
lean_dec_ref(v_inst_52_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy(lean_object* v_M_57_, lean_object* v_inst_58_, lean_object* v_S_59_, lean_object* v_s_60_, lean_object* v_hs_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_box(0);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy___boxed(lean_object* v_M_63_, lean_object* v_inst_64_, lean_object* v_S_65_, lean_object* v_s_66_, lean_object* v_hs_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Submonoid_copy(v_M_63_, v_inst_64_, v_S_65_, v_s_66_, v_hs_67_);
lean_dec_ref(v_inst_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_copy(lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_S_71_, lean_object* v_s_72_, lean_object* v_hs_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_copy___boxed(lean_object* v_M_75_, lean_object* v_inst_76_, lean_object* v_S_77_, lean_object* v_s_78_, lean_object* v_hs_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_AddSubmonoid_copy(v_M_75_, v_inst_76_, v_S_77_, v_s_78_, v_hs_79_);
lean_dec_ref(v_inst_76_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instTop(lean_object* v_M_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_box(0);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instTop___boxed(lean_object* v_M_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Submonoid_instTop(v_M_84_, v_inst_85_);
lean_dec_ref(v_inst_85_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instTop(lean_object* v_M_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instTop___boxed(lean_object* v_M_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_AddSubmonoid_instTop(v_M_90_, v_inst_91_);
lean_dec_ref(v_inst_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instBot(lean_object* v_M_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_box(0);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instBot___boxed(lean_object* v_M_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_Submonoid_instBot(v_M_96_, v_inst_97_);
lean_dec_ref(v_inst_97_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instBot(lean_object* v_M_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instBot___boxed(lean_object* v_M_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_AddSubmonoid_instBot(v_M_102_, v_inst_103_);
lean_dec_ref(v_inst_103_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instInhabited(lean_object* v_M_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_box(0);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instInhabited___boxed(lean_object* v_M_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Submonoid_instInhabited(v_M_108_, v_inst_109_);
lean_dec_ref(v_inst_109_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instInhabited(lean_object* v_M_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lean_box(0);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instInhabited___boxed(lean_object* v_M_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_AddSubmonoid_instInhabited(v_M_114_, v_inst_115_);
lean_dec_ref(v_inst_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin___lam__0(lean_object* v_S_u2081_117_, lean_object* v_S_u2082_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lean_box(0);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin(lean_object* v_M_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___f_123_; 
v___f_123_ = ((lean_object*)(lp_mathlib_Submonoid_instMin___closed__0));
return v___f_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instMin___boxed(lean_object* v_M_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Submonoid_instMin(v_M_124_, v_inst_125_);
lean_dec_ref(v_inst_125_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin___lam__0(lean_object* v_S_u2081_127_, lean_object* v_S_u2082_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_box(0);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin(lean_object* v_M_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___f_133_; 
v___f_133_ = ((lean_object*)(lp_mathlib_AddSubmonoid_instMin___closed__0));
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instMin___boxed(lean_object* v_M_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_AddSubmonoid_instMin(v_M_134_, v_inst_135_);
lean_dec_ref(v_inst_135_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instUniqueOfSubsingleton(lean_object* v_M_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lean_box(0);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_instUniqueOfSubsingleton___boxed(lean_object* v_M_141_, lean_object* v_inst_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Submonoid_instUniqueOfSubsingleton(v_M_141_, v_inst_142_, v_inst_143_);
lean_dec_ref(v_inst_142_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instUniqueOfSubsingleton(lean_object* v_M_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_box(0);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_instUniqueOfSubsingleton___boxed(lean_object* v_M_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_AddSubmonoid_instUniqueOfSubsingleton(v_M_149_, v_inst_150_, v_inst_151_);
lean_dec_ref(v_inst_150_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocusM(lean_object* v_M_153_, lean_object* v_N_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_f_157_, lean_object* v_g_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lean_box(0);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocusM___boxed(lean_object* v_M_160_, lean_object* v_N_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_g_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_MonoidHom_eqLocusM(v_M_160_, v_N_161_, v_inst_162_, v_inst_163_, v_f_164_, v_g_165_);
lean_dec(v_g_165_);
lean_dec(v_f_164_);
lean_dec_ref(v_inst_163_);
lean_dec_ref(v_inst_162_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocusM(lean_object* v_M_167_, lean_object* v_N_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_f_171_, lean_object* v_g_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lean_box(0);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocusM___boxed(lean_object* v_M_174_, lean_object* v_N_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_f_178_, lean_object* v_g_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_AddMonoidHom_eqLocusM(v_M_174_, v_N_175_, v_inst_176_, v_inst_177_, v_f_178_, v_g_179_);
lean_dec(v_g_179_);
lean_dec(v_f_178_);
lean_dec_ref(v_inst_177_);
lean_dec_ref(v_inst_176_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___redArg(lean_object* v_inst_181_){
_start:
{
lean_inc(v_inst_181_);
return v_inst_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___redArg___boxed(lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_OneMemClass_one___redArg(v_inst_182_);
lean_dec(v_inst_182_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one(lean_object* v_A_184_, lean_object* v_M_u2081_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_hA_188_, lean_object* v_S_x27_189_){
_start:
{
lean_inc(v_inst_187_);
return v_inst_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneMemClass_one___boxed(lean_object* v_A_190_, lean_object* v_M_u2081_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_hA_194_, lean_object* v_S_x27_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_OneMemClass_one(v_A_190_, v_M_u2081_191_, v_inst_192_, v_inst_193_, v_hA_194_, v_S_x27_195_);
lean_dec(v_S_x27_195_);
lean_dec(v_inst_193_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___redArg(lean_object* v_inst_197_){
_start:
{
lean_inc(v_inst_197_);
return v_inst_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___redArg___boxed(lean_object* v_inst_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_ZeroMemClass_zero___redArg(v_inst_198_);
lean_dec(v_inst_198_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero(lean_object* v_A_200_, lean_object* v_M_u2081_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_hA_204_, lean_object* v_S_x27_205_){
_start:
{
lean_inc(v_inst_203_);
return v_inst_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroMemClass_zero___boxed(lean_object* v_A_206_, lean_object* v_M_u2081_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_hA_210_, lean_object* v_S_x27_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_ZeroMemClass_zero(v_A_206_, v_M_u2081_207_, v_inst_208_, v_inst_209_, v_hA_210_, v_S_x27_211_);
lean_dec(v_S_x27_211_);
lean_dec(v_inst_209_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___redArg___lam__0(lean_object* v_toNPow_213_, lean_object* v_a_214_, lean_object* v_n_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lean_apply_2(v_toNPow_213_, v_n_215_, v_a_214_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v_toNPow_218_; lean_object* v___f_219_; 
v_toNPow_218_ = lean_ctor_get(v_inst_217_, 2);
lean_inc(v_toNPow_218_);
lean_dec_ref(v_inst_217_);
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_SubmonoidClass_instPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_219_, 0, v_toNPow_218_);
return v___f_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow(lean_object* v_M_220_, lean_object* v_inst_221_, lean_object* v_A_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_S_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_SubmonoidClass_instPow___redArg(v_inst_221_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_instPow___boxed(lean_object* v_M_227_, lean_object* v_inst_228_, lean_object* v_A_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_S_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_SubmonoidClass_instPow(v_M_227_, v_inst_228_, v_A_229_, v_inst_230_, v_inst_231_, v_S_232_);
lean_dec(v_S_232_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___redArg___lam__0(lean_object* v_toNSMul_234_, lean_object* v_n_235_, lean_object* v_a_236_){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = lean_apply_2(v_toNSMul_234_, v_n_235_, v_a_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___redArg(lean_object* v_inst_238_){
_start:
{
lean_object* v_toNSMul_239_; lean_object* v___f_240_; 
v_toNSMul_239_ = lean_ctor_get(v_inst_238_, 2);
lean_inc(v_toNSMul_239_);
lean_dec_ref(v_inst_238_);
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoidClass_instNSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_240_, 0, v_toNSMul_239_);
return v___f_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul(lean_object* v_M_241_, lean_object* v_inst_242_, lean_object* v_A_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_S_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_AddSubmonoidClass_instNSMul___redArg(v_inst_242_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_instNSMul___boxed(lean_object* v_M_248_, lean_object* v_inst_249_, lean_object* v_A_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_S_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_AddSubmonoidClass_instNSMul(v_M_248_, v_inst_249_, v_A_250_, v_inst_251_, v_inst_252_, v_S_253_);
lean_dec(v_S_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___redArg(lean_object* v_inst_255_){
_start:
{
lean_object* v___x_256_; lean_object* v_toOne_257_; lean_object* v_toMul_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_266_; 
v___x_256_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_255_);
v_toOne_257_ = lean_ctor_get(v___x_256_, 0);
v_toMul_258_ = lean_ctor_get(v___x_256_, 1);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_256_);
if (v_isSharedCheck_266_ == 0)
{
v___x_260_ = v___x_256_;
v_isShared_261_ = v_isSharedCheck_266_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_toMul_258_);
lean_inc(v_toOne_257_);
lean_dec(v___x_256_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_266_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___f_262_; lean_object* v___x_264_; 
v___f_262_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_262_, 0, v_toMul_258_);
if (v_isShared_261_ == 0)
{
lean_ctor_set(v___x_260_, 1, v___f_262_);
v___x_264_ = v___x_260_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_toOne_257_);
lean_ctor_set(v_reuseFailAlloc_265_, 1, v___f_262_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass(lean_object* v_M_267_, lean_object* v_inst_268_, lean_object* v_A_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_S_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_inst_268_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMulOneClass___boxed(lean_object* v_M_274_, lean_object* v_inst_275_, lean_object* v_A_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_S_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_SubmonoidClass_toMulOneClass(v_M_274_, v_inst_275_, v_A_276_, v_inst_277_, v_inst_278_, v_S_279_);
lean_dec(v_S_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; lean_object* v_toZero_283_; lean_object* v_toAdd_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_292_; 
v___x_282_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_281_);
v_toZero_283_ = lean_ctor_get(v___x_282_, 0);
v_toAdd_284_ = lean_ctor_get(v___x_282_, 1);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_292_ == 0)
{
v___x_286_ = v___x_282_;
v_isShared_287_ = v_isSharedCheck_292_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_toAdd_284_);
lean_inc(v_toZero_283_);
lean_dec(v___x_282_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_292_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___f_288_; lean_object* v___x_290_; 
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_MulMemClass_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_288_, 0, v_toAdd_284_);
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 1, v___f_288_);
v___x_290_ = v___x_286_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v_toZero_283_);
lean_ctor_set(v_reuseFailAlloc_291_, 1, v___f_288_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass(lean_object* v_M_293_, lean_object* v_inst_294_, lean_object* v_A_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_S_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(v_inst_294_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddZeroClass___boxed(lean_object* v_M_300_, lean_object* v_inst_301_, lean_object* v_A_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_S_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass(v_M_300_, v_inst_301_, v_A_302_, v_inst_303_, v_inst_304_, v_S_305_);
lean_dec(v_S_305_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v_toOne_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v_toMul_313_; lean_object* v___x_314_; lean_object* v___f_315_; lean_object* v___x_316_; 
v___x_308_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_307_);
lean_inc_ref(v___x_308_);
v___x_309_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_308_);
v_toOne_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_toOne_310_);
lean_dec_ref(v___x_309_);
v___x_311_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v___x_308_);
v___x_312_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_311_);
v_toMul_313_ = lean_ctor_get(v___x_312_, 1);
lean_inc(v_toMul_313_);
lean_dec_ref(v___x_312_);
v___x_314_ = lp_mathlib_SubmonoidClass_instPow___redArg(v_inst_307_);
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_315_, 0, v___x_314_);
v___x_316_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_316_, 0, v_toOne_310_);
lean_ctor_set(v___x_316_, 1, v_toMul_313_);
lean_ctor_set(v___x_316_, 2, v___f_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid(lean_object* v_M_317_, lean_object* v_inst_318_, lean_object* v_A_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_S_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_318_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toMonoid___boxed(lean_object* v_M_324_, lean_object* v_inst_325_, lean_object* v_A_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_S_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_SubmonoidClass_toMonoid(v_M_324_, v_inst_325_, v_A_326_, v_inst_327_, v_inst_328_, v_S_329_);
lean_dec(v_S_329_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v_toZero_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v_toAdd_337_; lean_object* v___x_338_; lean_object* v___f_339_; lean_object* v___x_340_; 
v___x_332_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_331_);
lean_inc_ref(v___x_332_);
v___x_333_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_332_);
v_toZero_334_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_toZero_334_);
lean_dec_ref(v___x_333_);
v___x_335_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(v___x_332_);
v___x_336_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_335_);
v_toAdd_337_ = lean_ctor_get(v___x_336_, 1);
lean_inc(v_toAdd_337_);
lean_dec_ref(v___x_336_);
v___x_338_ = lp_mathlib_AddSubmonoidClass_instNSMul___redArg(v_inst_331_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_339_, 0, v___x_338_);
v___x_340_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_340_, 0, v_toZero_334_);
lean_ctor_set(v___x_340_, 1, v_toAdd_337_);
lean_ctor_set(v___x_340_, 2, v___f_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid(lean_object* v_M_341_, lean_object* v_inst_342_, lean_object* v_A_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_S_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_342_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___boxed(lean_object* v_M_348_, lean_object* v_inst_349_, lean_object* v_A_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_S_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_mathlib_AddSubmonoidClass_toAddMonoid(v_M_348_, v_inst_349_, v_A_350_, v_inst_351_, v_inst_352_, v_S_353_);
lean_dec(v_S_353_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid___redArg(lean_object* v_inst_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid(lean_object* v_M_357_, lean_object* v_inst_358_, lean_object* v_A_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_S_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_358_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_toCommMonoid___boxed(lean_object* v_M_364_, lean_object* v_inst_365_, lean_object* v_A_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_S_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_SubmonoidClass_toCommMonoid(v_M_364_, v_inst_365_, v_A_366_, v_inst_367_, v_inst_368_, v_S_369_);
lean_dec(v_S_369_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid___redArg(lean_object* v_inst_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid(lean_object* v_M_373_, lean_object* v_inst_374_, lean_object* v_A_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_S_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_374_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_toAddCommMonoid___boxed(lean_object* v_M_380_, lean_object* v_inst_381_, lean_object* v_A_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_S_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_AddSubmonoidClass_toAddCommMonoid(v_M_380_, v_inst_381_, v_A_382_, v_inst_383_, v_inst_384_, v_S_385_);
lean_dec(v_S_385_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0(lean_object* v_self_387_){
_start:
{
lean_inc(v_self_387_);
return v_self_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object* v_self_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_SubmonoidClass_subtype___lam__0(v_self_388_);
lean_dec(v_self_388_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype(lean_object* v_M_391_, lean_object* v_A_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_hA_395_, lean_object* v_S_x27_396_){
_start:
{
lean_object* v___f_397_; 
v___f_397_ = ((lean_object*)(lp_mathlib_SubmonoidClass_subtype___closed__0));
return v___f_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubmonoidClass_subtype___boxed(lean_object* v_M_398_, lean_object* v_A_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_hA_402_, lean_object* v_S_x27_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_SubmonoidClass_subtype(v_M_398_, v_A_399_, v_inst_400_, v_inst_401_, v_hA_402_, v_S_x27_403_);
lean_dec(v_S_x27_403_);
lean_dec_ref(v_inst_400_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_subtype(lean_object* v_M_405_, lean_object* v_A_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_hA_409_, lean_object* v_S_x27_410_){
_start:
{
lean_object* v___f_411_; 
v___f_411_ = ((lean_object*)(lp_mathlib_SubmonoidClass_subtype___closed__0));
return v___f_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoidClass_subtype___boxed(lean_object* v_M_412_, lean_object* v_A_413_, lean_object* v_inst_414_, lean_object* v_inst_415_, lean_object* v_hA_416_, lean_object* v_S_x27_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_AddSubmonoidClass_subtype(v_M_412_, v_A_413_, v_inst_414_, v_inst_415_, v_hA_416_, v_S_x27_417_);
lean_dec(v_S_x27_417_);
lean_dec_ref(v_inst_414_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul___redArg___lam__0(lean_object* v_toMul_419_, lean_object* v_a_420_, lean_object* v_b_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lean_apply_2(v_toMul_419_, v_a_420_, v_b_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul___redArg(lean_object* v_inst_423_){
_start:
{
lean_object* v___x_424_; lean_object* v_toMul_425_; lean_object* v___f_426_; 
v___x_424_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_423_);
v_toMul_425_ = lean_ctor_get(v___x_424_, 1);
lean_inc(v_toMul_425_);
lean_dec_ref(v___x_424_);
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_426_, 0, v_toMul_425_);
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_mul(lean_object* v_M_427_, lean_object* v_inst_428_, lean_object* v_S_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Submonoid_mul___redArg(v_inst_428_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add___redArg___lam__0(lean_object* v_toAdd_431_, lean_object* v_a_432_, lean_object* v_b_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lean_apply_2(v_toAdd_431_, v_a_432_, v_b_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add___redArg(lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; lean_object* v_toAdd_437_; lean_object* v___f_438_; 
v___x_436_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_435_);
v_toAdd_437_ = lean_ctor_get(v___x_436_, 1);
lean_inc(v_toAdd_437_);
lean_dec_ref(v___x_436_);
v___f_438_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_438_, 0, v_toAdd_437_);
return v___f_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_add(lean_object* v_M_439_, lean_object* v_inst_440_, lean_object* v_S_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lp_mathlib_AddSubmonoid_add___redArg(v_inst_440_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_one___redArg(lean_object* v_inst_443_){
_start:
{
lean_object* v___x_444_; lean_object* v_toOne_445_; 
v___x_444_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_443_);
v_toOne_445_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_toOne_445_);
lean_dec_ref(v___x_444_);
return v_toOne_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_one(lean_object* v_M_446_, lean_object* v_inst_447_, lean_object* v_S_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_mathlib_Submonoid_one___redArg(v_inst_447_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_zero___redArg(lean_object* v_inst_450_){
_start:
{
lean_object* v___x_451_; lean_object* v_toZero_452_; 
v___x_451_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_450_);
v_toZero_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_toZero_452_);
lean_dec_ref(v___x_451_);
return v_toZero_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_zero(lean_object* v_M_453_, lean_object* v_inst_454_, lean_object* v_S_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_AddSubmonoid_zero___redArg(v_inst_454_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMulOneClass___redArg(lean_object* v_inst_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_inst_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMulOneClass(lean_object* v_M_459_, lean_object* v_inst_460_, lean_object* v_S_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_SubmonoidClass_toMulOneClass___redArg(v_inst_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddZeroClass___redArg(lean_object* v_inst_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(v_inst_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddZeroClass(lean_object* v_M_465_, lean_object* v_inst_466_, lean_object* v_S_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_AddSubmonoidClass_toAddZeroClass___redArg(v_inst_466_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMonoid___redArg(lean_object* v_inst_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toMonoid(lean_object* v_M_471_, lean_object* v_inst_472_, lean_object* v_S_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_472_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddMonoid___redArg(lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddMonoid(lean_object* v_M_477_, lean_object* v_inst_478_, lean_object* v_S_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_478_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toCommMonoid___redArg(lean_object* v_inst_481_){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toCommMonoid(lean_object* v_M_483_, lean_object* v_inst_484_, lean_object* v_S_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_484_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddCommMonoid___redArg(lean_object* v_inst_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toAddCommMonoid(lean_object* v_M_489_, lean_object* v_inst_490_, lean_object* v_S_491_){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_490_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subtype(lean_object* v_M_493_, lean_object* v_inst_494_, lean_object* v_S_495_){
_start:
{
lean_object* v___f_496_; 
v___f_496_ = ((lean_object*)(lp_mathlib_SubmonoidClass_subtype___closed__0));
return v___f_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_subtype___boxed(lean_object* v_M_497_, lean_object* v_inst_498_, lean_object* v_S_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_mathlib_Submonoid_subtype(v_M_497_, v_inst_498_, v_S_499_);
lean_dec_ref(v_inst_498_);
return v_res_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_subtype(lean_object* v_M_501_, lean_object* v_inst_502_, lean_object* v_S_503_){
_start:
{
lean_object* v___f_504_; 
v___f_504_ = ((lean_object*)(lp_mathlib_SubmonoidClass_subtype___closed__0));
return v___f_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_subtype___boxed(lean_object* v_M_505_, lean_object* v_inst_506_, lean_object* v_S_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_AddSubmonoid_subtype(v_M_505_, v_inst_506_, v_S_507_);
lean_dec_ref(v_inst_506_);
return v_res_508_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_Subsemigroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
