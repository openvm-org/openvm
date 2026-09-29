// Lean compiler output
// Module: Mathlib.Algebra.Field.Subfield.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Field.Subfield.Defs public import Mathlib.Algebra.GroupWithZero.Units.Lemmas public import Mathlib.Algebra.Ring.Subring.Basic public import Mathlib.RingTheory.SimpleRing.Basic
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
lean_object* lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Subfield_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
lean_object* lp_mathlib_Algebra_ofSubsemiring___redArg(lean_object*);
lean_object* lp_mathlib_Subfield_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Subsemiring_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subfield_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_instMin___closed__0 = (const lean_object*)&lp_mathlib_Subfield_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Subfield_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Subfield_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subfield_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Subfield_instCompleteLattice___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subfield_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_gi___closed__0 = (const lean_object*)&lp_mathlib_Subfield_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subfield_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subfield_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subfield_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Subfield_inclusion___closed__0_value;
static const lean_closure_object lp_mathlib_Subfield_inclusion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Subfield_inclusion___closed__0_value)} };
static const lean_object* lp_mathlib_Subfield_inclusion___closed__1 = (const lean_object*)&lp_mathlib_Subfield_inclusion___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_subfieldCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_subfieldCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop(lean_object* v_K_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instTop___boxed(lean_object* v_K_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Subfield_instTop(v_K_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited(lean_object* v_K_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInhabited___boxed(lean_object* v_K_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Subfield_instInhabited(v_K_10_, v_inst_11_);
lean_dec_ref(v_inst_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v_toRing_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v_toRing_14_ = lean_ctor_get(v_inst_13_, 0);
v___x_15_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_toRing_14_);
v___x_16_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v___x_15_);
v___x_17_ = lp_mathlib_Subsemiring_topEquiv(lean_box(0), v___x_16_);
lean_dec_ref(v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___redArg___boxed(lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Subfield_topEquiv___redArg(v_inst_18_);
lean_dec_ref(v_inst_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv(lean_object* v_K_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Subfield_topEquiv___redArg(v_inst_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_topEquiv___boxed(lean_object* v_K_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Subfield_topEquiv(v_K_23_, v_inst_24_);
lean_dec_ref(v_inst_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap(lean_object* v_K_26_, lean_object* v_L_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_f_30_, lean_object* v_s_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_box(0);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_comap___boxed(lean_object* v_K_33_, lean_object* v_L_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_f_37_, lean_object* v_s_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Subfield_comap(v_K_33_, v_L_34_, v_inst_35_, v_inst_36_, v_f_37_, v_s_38_);
lean_dec(v_f_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_35_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map(lean_object* v_K_40_, lean_object* v_L_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_f_44_, lean_object* v_s_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_map___boxed(lean_object* v_K_47_, lean_object* v_L_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_, lean_object* v_s_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Subfield_map(v_K_47_, v_L_48_, v_inst_49_, v_inst_50_, v_f_51_, v_s_52_);
lean_dec(v_f_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange(lean_object* v_K_54_, lean_object* v_L_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_box(0);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fieldRange___boxed(lean_object* v_K_60_, lean_object* v_L_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_f_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_RingHom_fieldRange(v_K_60_, v_L_61_, v_inst_62_, v_inst_63_, v_f_64_);
lean_dec(v_f_64_);
lean_dec_ref(v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0(lean_object* v_f_66_, lean_object* v___y_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_apply_1(v_f_66_, v___y_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___redArg(lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_f_71_){
_start:
{
lean_object* v___f_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_fintypeFieldRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_72_, 0, v_f_71_);
v___x_73_ = lp_mathlib_PLift_fintype___redArg(v_inst_69_);
v___x_74_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_70_, v___f_72_, v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange(lean_object* v_K_75_, lean_object* v_L_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_f_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_RingHom_fintypeFieldRange___redArg(v_inst_79_, v_inst_80_, v_f_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_fintypeFieldRange___boxed(lean_object* v_K_83_, lean_object* v_L_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_f_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_RingHom_fintypeFieldRange(v_K_83_, v_L_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_f_89_);
lean_dec_ref(v_inst_86_);
lean_dec_ref(v_inst_85_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___lam__0(lean_object* v_s_91_, lean_object* v_t_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lean_box(0);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin(lean_object* v_K_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___f_97_; 
v___f_97_ = ((lean_object*)(lp_mathlib_Subfield_instMin___closed__0));
return v___f_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMin___boxed(lean_object* v_K_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Subfield_instMin(v_K_98_, v_inst_99_);
lean_dec_ref(v_inst_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___lam__0(lean_object* v_S_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lean_box(0);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet(lean_object* v_K_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v___f_106_; 
v___f_106_ = ((lean_object*)(lp_mathlib_Subfield_instInfSet___closed__0));
return v___f_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instInfSet___boxed(lean_object* v_K_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Subfield_instInfSet(v_K_107_, v_inst_108_);
lean_dec_ref(v_inst_108_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg___lam__0(lean_object* v_x1_110_, lean_object* v_x2_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lean_box(0);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg(lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___f_116_; lean_object* v___x_117_; lean_object* v_toLattice_118_; lean_object* v_toBoundedOrder_119_; lean_object* v_toSupSet_120_; lean_object* v_toInfSet_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_148_; 
v___x_115_ = lp_mathlib_Subfield_instPartialOrder(lean_box(0), v_inst_114_);
v___f_116_ = ((lean_object*)(lp_mathlib_Subfield_instInfSet___closed__0));
v___x_117_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_115_, v___f_116_);
v_toLattice_118_ = lean_ctor_get(v___x_117_, 0);
v_toBoundedOrder_119_ = lean_ctor_get(v___x_117_, 3);
v_toSupSet_120_ = lean_ctor_get(v___x_117_, 1);
v_toInfSet_121_ = lean_ctor_get(v___x_117_, 2);
v_isSharedCheck_148_ = !lean_is_exclusive(v___x_117_);
if (v_isSharedCheck_148_ == 0)
{
v___x_123_ = v___x_117_;
v_isShared_124_ = v_isSharedCheck_148_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_toBoundedOrder_119_);
lean_inc(v_toInfSet_121_);
lean_inc(v_toSupSet_120_);
lean_inc(v_toLattice_118_);
lean_dec(v___x_117_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_148_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v_toSemilatticeSup_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_146_; 
v_toSemilatticeSup_125_ = lean_ctor_get(v_toLattice_118_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v_toLattice_118_);
if (v_isSharedCheck_146_ == 0)
{
lean_object* v_unused_147_; 
v_unused_147_ = lean_ctor_get(v_toLattice_118_, 1);
lean_dec(v_unused_147_);
v___x_127_ = v_toLattice_118_;
v_isShared_128_ = v_isSharedCheck_146_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_toSemilatticeSup_125_);
lean_dec(v_toLattice_118_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_146_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v_toOrderBot_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_144_; 
v_toOrderBot_129_ = lean_ctor_get(v_toBoundedOrder_119_, 1);
v_isSharedCheck_144_ = !lean_is_exclusive(v_toBoundedOrder_119_);
if (v_isSharedCheck_144_ == 0)
{
lean_object* v_unused_145_; 
v_unused_145_ = lean_ctor_get(v_toBoundedOrder_119_, 0);
lean_dec(v_unused_145_);
v___x_131_ = v_toBoundedOrder_119_;
v_isShared_132_ = v_isSharedCheck_144_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_toOrderBot_129_);
lean_dec(v_toBoundedOrder_119_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_144_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_136_; 
v___f_133_ = ((lean_object*)(lp_mathlib_Subfield_instCompleteLattice___redArg___closed__0));
v___x_134_ = lean_box(0);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 1, v___f_133_);
v___x_136_ = v___x_127_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_toSemilatticeSup_125_);
lean_ctor_set(v_reuseFailAlloc_143_, 1, v___f_133_);
v___x_136_ = v_reuseFailAlloc_143_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
lean_object* v___x_138_; 
if (v_isShared_132_ == 0)
{
lean_ctor_set(v___x_131_, 0, v___x_134_);
v___x_138_ = v___x_131_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_142_, 1, v_toOrderBot_129_);
v___x_138_ = v_reuseFailAlloc_142_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
lean_object* v___x_140_; 
if (v_isShared_124_ == 0)
{
lean_ctor_set(v___x_123_, 3, v___x_138_);
lean_ctor_set(v___x_123_, 0, v___x_136_);
v___x_140_ = v___x_123_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_136_);
lean_ctor_set(v_reuseFailAlloc_141_, 1, v_toSupSet_120_);
lean_ctor_set(v_reuseFailAlloc_141_, 2, v_toInfSet_121_);
lean_ctor_set(v_reuseFailAlloc_141_, 3, v___x_138_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___redArg___boxed(lean_object* v_inst_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Subfield_instCompleteLattice___redArg(v_inst_149_);
lean_dec_ref(v_inst_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice(lean_object* v_K_151_, lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Subfield_instCompleteLattice___redArg(v_inst_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instCompleteLattice___boxed(lean_object* v_K_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_Subfield_instCompleteLattice(v_K_154_, v_inst_155_);
lean_dec_ref(v_inst_155_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure(lean_object* v_K_157_, lean_object* v_inst_158_, lean_object* v_s_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_box(0);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_closure___boxed(lean_object* v_K_161_, lean_object* v_inst_162_, lean_object* v_s_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Subfield_closure(v_K_161_, v_inst_162_, v_s_163_);
lean_dec_ref(v_inst_162_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___lam__0(lean_object* v_s_165_, lean_object* v_x_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_box(0);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi(lean_object* v_K_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___f_171_; 
v___f_171_ = ((lean_object*)(lp_mathlib_Subfield_gi___closed__0));
return v___f_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_gi___boxed(lean_object* v_K_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Subfield_gi(v_K_172_, v_inst_173_);
lean_dec_ref(v_inst_173_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___redArg(lean_object* v_f_175_){
_start:
{
lean_object* v___f_176_; 
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_176_, 0, v_f_175_);
return v___f_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField(lean_object* v_K_177_, lean_object* v_L_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_f_181_){
_start:
{
lean_object* v___f_182_; 
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_182_, 0, v_f_181_);
return v___f_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_rangeRestrictField___boxed(lean_object* v_K_183_, lean_object* v_L_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_f_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_RingHom_rangeRestrictField(v_K_183_, v_L_184_, v_inst_185_, v_inst_186_, v_f_187_);
lean_dec_ref(v_inst_186_);
lean_dec_ref(v_inst_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField(lean_object* v_K_189_, lean_object* v_inst_190_, lean_object* v_L_191_, lean_object* v_inst_192_, lean_object* v_f_193_, lean_object* v_g_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_box(0);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_eqLocusField___boxed(lean_object* v_K_196_, lean_object* v_inst_197_, lean_object* v_L_198_, lean_object* v_inst_199_, lean_object* v_f_200_, lean_object* v_g_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_RingHom_eqLocusField(v_K_196_, v_inst_197_, v_L_198_, v_inst_199_, v_f_200_, v_g_201_);
lean_dec(v_g_201_);
lean_dec(v_f_200_);
lean_dec_ref(v_inst_199_);
lean_dec_ref(v_inst_197_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion(lean_object* v_K_206_, lean_object* v_inst_207_, lean_object* v_S_208_, lean_object* v_T_209_, lean_object* v_h_210_){
_start:
{
lean_object* v___f_211_; 
v___f_211_ = ((lean_object*)(lp_mathlib_Subfield_inclusion___closed__1));
return v___f_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_inclusion___boxed(lean_object* v_K_212_, lean_object* v_inst_213_, lean_object* v_S_214_, lean_object* v_T_215_, lean_object* v_h_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Subfield_inclusion(v_K_212_, v_inst_213_, v_S_214_, v_T_215_, v_h_216_);
lean_dec_ref(v_inst_213_);
return v_res_217_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_subfieldCongr___closed__0(void){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr(lean_object* v_K_219_, lean_object* v_inst_220_, lean_object* v_s_221_, lean_object* v_t_222_, lean_object* v_h_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_mathlib_RingEquiv_subfieldCongr___closed__0, &lp_mathlib_RingEquiv_subfieldCongr___closed__0_once, _init_lp_mathlib_RingEquiv_subfieldCongr___closed__0);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_subfieldCongr___boxed(lean_object* v_K_225_, lean_object* v_inst_226_, lean_object* v_s_227_, lean_object* v_t_228_, lean_object* v_h_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib_RingEquiv_subfieldCongr(v_K_225_, v_inst_226_, v_s_227_, v_t_228_, v_h_229_);
lean_dec_ref(v_inst_226_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg(lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; lean_object* v_toCommSemiring_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_232_ = lp_mathlib_Field_toSemifield___redArg(v_inst_231_);
v_toCommSemiring_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc_ref(v_toCommSemiring_233_);
lean_dec_ref(v___x_232_);
v___x_234_ = lp_mathlib_Algebra_id___redArg(v_toCommSemiring_233_);
v___x_235_ = lp_mathlib_Algebra_ofSubsemiring___redArg(v___x_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___redArg___boxed(lean_object* v_inst_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Subfield_toAlgebra___redArg(v_inst_236_);
lean_dec_ref(v_inst_236_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra(lean_object* v_K_238_, lean_object* v_inst_239_, lean_object* v_s_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Subfield_toAlgebra___redArg(v_inst_239_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_toAlgebra___boxed(lean_object* v_K_242_, lean_object* v_inst_243_, lean_object* v_s_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_Subfield_toAlgebra(v_K_242_, v_inst_243_, v_s_244_);
lean_dec_ref(v_inst_243_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(lean_object* v_K_246_, lean_object* v_inst_247_, lean_object* v_s_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_box(0);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure___boxed(lean_object* v_K_250_, lean_object* v_inst_251_, lean_object* v_s_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib___private_Mathlib_Algebra_Field_Subfield_Basic_0__Subfield_commClosure(v_K_250_, v_inst_251_, v_s_252_);
lean_dec_ref(v_inst_251_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg___lam__0(lean_object* v_inst_254_, lean_object* v_m_255_, lean_object* v_a_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lean_apply_2(v_inst_254_, v_m_255_, v_a_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___redArg(lean_object* v_inst_258_){
_start:
{
lean_object* v___f_259_; 
v___f_259_ = lean_alloc_closure((void*)(lp_mathlib_Subfield_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_259_, 0, v_inst_258_);
return v___f_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem(lean_object* v_K_260_, lean_object* v_inst_261_, lean_object* v_X_262_, lean_object* v_inst_263_, lean_object* v_F_264_){
_start:
{
lean_object* v___f_265_; 
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_Subfield_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_265_, 0, v_inst_263_);
return v___f_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulSubtypeMem___boxed(lean_object* v_K_266_, lean_object* v_inst_267_, lean_object* v_X_268_, lean_object* v_inst_269_, lean_object* v_F_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_Subfield_instSMulSubtypeMem(v_K_266_, v_inst_267_, v_X_268_, v_inst_269_, v_F_270_);
lean_dec_ref(v_inst_267_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v___f_273_; 
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_273_, 0, v_inst_272_);
return v___f_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem(lean_object* v_K_274_, lean_object* v_inst_275_, lean_object* v_X_276_, lean_object* v_inst_277_, lean_object* v_F_278_){
_start:
{
lean_object* v___f_279_; 
v___f_279_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_279_, 0, v_inst_277_);
return v___f_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionSubtypeMem___boxed(lean_object* v_K_280_, lean_object* v_inst_281_, lean_object* v_X_282_, lean_object* v_inst_283_, lean_object* v_F_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Subfield_instMulActionSubtypeMem(v_K_280_, v_inst_281_, v_X_282_, v_inst_283_, v_F_284_);
lean_dec_ref(v_inst_281_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___redArg(lean_object* v_inst_286_){
_start:
{
lean_object* v___f_287_; 
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_287_, 0, v_inst_286_);
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem(lean_object* v_K_288_, lean_object* v_inst_289_, lean_object* v_X_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_F_293_){
_start:
{
lean_object* v___f_294_; 
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_294_, 0, v_inst_292_);
return v___f_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instDistribMulActionSubtypeMem___boxed(lean_object* v_K_295_, lean_object* v_inst_296_, lean_object* v_X_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_F_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_Subfield_instDistribMulActionSubtypeMem(v_K_295_, v_inst_296_, v_X_297_, v_inst_298_, v_inst_299_, v_F_300_);
lean_dec_ref(v_inst_298_);
lean_dec_ref(v_inst_296_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___redArg(lean_object* v_inst_302_){
_start:
{
lean_object* v___f_303_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_303_, 0, v_inst_302_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(lean_object* v_K_304_, lean_object* v_inst_305_, lean_object* v_X_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_F_309_){
_start:
{
lean_object* v___f_310_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_310_, 0, v_inst_308_);
return v___f_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem___boxed(lean_object* v_K_311_, lean_object* v_inst_312_, lean_object* v_X_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_F_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Subfield_instMulDistribMulActionSubtypeMem(v_K_311_, v_inst_312_, v_X_313_, v_inst_314_, v_inst_315_, v_F_316_);
lean_dec_ref(v_inst_314_);
lean_dec_ref(v_inst_312_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___redArg(lean_object* v_inst_318_){
_start:
{
lean_object* v___f_319_; 
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_319_, 0, v_inst_318_);
return v___f_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(lean_object* v_K_320_, lean_object* v_inst_321_, lean_object* v_X_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_F_325_){
_start:
{
lean_object* v___f_326_; 
v___f_326_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_326_, 0, v_inst_324_);
return v___f_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instSMulWithZeroSubtypeMem___boxed(lean_object* v_K_327_, lean_object* v_inst_328_, lean_object* v_X_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_F_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_Subfield_instSMulWithZeroSubtypeMem(v_K_327_, v_inst_328_, v_X_329_, v_inst_330_, v_inst_331_, v_F_332_);
lean_dec(v_inst_330_);
lean_dec_ref(v_inst_328_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___redArg(lean_object* v_inst_334_){
_start:
{
lean_object* v___f_335_; 
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_335_, 0, v_inst_334_);
return v___f_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(lean_object* v_K_336_, lean_object* v_inst_337_, lean_object* v_X_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_F_341_){
_start:
{
lean_object* v___f_342_; 
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_342_, 0, v_inst_340_);
return v___f_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem___boxed(lean_object* v_K_343_, lean_object* v_inst_344_, lean_object* v_X_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_F_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_Subfield_instMulActionWithZeroSubtypeMem(v_K_343_, v_inst_344_, v_X_345_, v_inst_346_, v_inst_347_, v_F_348_);
lean_dec(v_inst_346_);
lean_dec_ref(v_inst_344_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___redArg(lean_object* v_inst_350_){
_start:
{
lean_object* v___f_351_; 
v___f_351_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_351_, 0, v_inst_350_);
return v___f_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem(lean_object* v_K_352_, lean_object* v_inst_353_, lean_object* v_X_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_F_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_358_, 0, v_inst_356_);
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instModuleSubtypeMem___boxed(lean_object* v_K_359_, lean_object* v_inst_360_, lean_object* v_X_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_F_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_Subfield_instModuleSubtypeMem(v_K_359_, v_inst_360_, v_X_361_, v_inst_362_, v_inst_363_, v_F_364_);
lean_dec_ref(v_inst_362_);
lean_dec_ref(v_inst_360_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___redArg(lean_object* v_inst_366_){
_start:
{
lean_object* v___f_367_; 
v___f_367_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_367_, 0, v_inst_366_);
return v___f_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(lean_object* v_K_368_, lean_object* v_inst_369_, lean_object* v_X_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_F_373_){
_start:
{
lean_object* v___f_374_; 
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_instSMulSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_374_, 0, v_inst_372_);
return v___f_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subfield_instMulSemiringActionSubtypeMem___boxed(lean_object* v_K_375_, lean_object* v_inst_376_, lean_object* v_X_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_F_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_Subfield_instMulSemiringActionSubtypeMem(v_K_375_, v_inst_376_, v_X_377_, v_inst_378_, v_inst_379_, v_F_380_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_376_);
return v_res_381_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
