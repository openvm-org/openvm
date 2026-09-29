// Lean compiler output
// Module: Mathlib.RingTheory.NonUnitalSubring.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Basic public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.GroupTheory.Subsemigroup.Center public import Mathlib.RingTheory.NonUnitalSubring.Defs public import Mathlib.RingTheory.NonUnitalSubsemiring.Basic
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
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_topEquiv___redArg(lean_object*);
lean_object* lp_mathlib_PLift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Set_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubring_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubringClass_subtype___lam__0(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubring_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubring_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubring_instMin___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubring_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubring_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubring_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubring_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubring_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center_instNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center_instNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerToMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerToMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centralizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centralizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closureNonUnitalCommRingOfComm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closureNonUnitalCommRingOfComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalSubring_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalSubring_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalSubring_gi___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalSubring_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_NonUnitalSubring_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NonUnitalSubring_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_fintypeRange___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg___closed__0 = (const lean_object*)&lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubringCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubringCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instTop(lean_object* v_R_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instTop___boxed(lean_object* v_R_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_NonUnitalSubring_instTop(v_R_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topEquiv___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_7_);
v___x_9_ = lp_mathlib_NonUnitalSubsemiring_topEquiv___redArg(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_topEquiv(lean_object* v_R_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_NonUnitalSubring_topEquiv___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_comap(lean_object* v_R_13_, lean_object* v_S_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_f_17_, lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_box(0);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_comap___boxed(lean_object* v_R_20_, lean_object* v_S_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_f_24_, lean_object* v_s_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_NonUnitalSubring_comap(v_R_20_, v_S_21_, v_inst_22_, v_inst_23_, v_f_24_, v_s_25_);
lean_dec(v_f_24_);
lean_dec_ref(v_inst_23_);
lean_dec_ref(v_inst_22_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_map(lean_object* v_R_27_, lean_object* v_S_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_, lean_object* v_s_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_box(0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_map___boxed(lean_object* v_R_34_, lean_object* v_S_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_f_38_, lean_object* v_s_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_NonUnitalSubring_map(v_R_34_, v_S_35_, v_inst_36_, v_inst_37_, v_f_38_, v_s_39_);
lean_dec(v_f_38_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_range(lean_object* v_R_41_, lean_object* v_S_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_f_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_range___boxed(lean_object* v_R_47_, lean_object* v_S_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_NonUnitalRingHom_range(v_R_47_, v_S_48_, v_inst_49_, v_inst_50_, v_f_51_);
lean_dec(v_f_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___redArg___lam__0(lean_object* v_f_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_apply_1(v_f_53_, v___y_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___redArg(lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_){
_start:
{
lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_fintypeRange___redArg___lam__0), 2, 1);
lean_closure_set(v___f_59_, 0, v_f_58_);
v___x_60_ = lp_mathlib_PLift_fintype___redArg(v_inst_56_);
v___x_61_ = lp_mathlib_Set_fintypeRange___redArg(v_inst_57_, v___f_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange(lean_object* v_R_62_, lean_object* v_S_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_NonUnitalRingHom_fintypeRange___redArg(v_inst_66_, v_inst_67_, v_f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_fintypeRange___boxed(lean_object* v_R_70_, lean_object* v_S_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_f_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_NonUnitalRingHom_fintypeRange(v_R_70_, v_S_71_, v_inst_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_f_76_);
lean_dec_ref(v_inst_73_);
lean_dec_ref(v_inst_72_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instBot(lean_object* v_R_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lean_box(0);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instBot___boxed(lean_object* v_R_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_NonUnitalSubring_instBot(v_R_81_, v_inst_82_);
lean_dec_ref(v_inst_82_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInhabited(lean_object* v_R_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_box(0);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInhabited___boxed(lean_object* v_R_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_NonUnitalSubring_instInhabited(v_R_87_, v_inst_88_);
lean_dec_ref(v_inst_88_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin___lam__0(lean_object* v_s_90_, lean_object* v_t_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_box(0);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin(lean_object* v_R_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___f_96_; 
v___f_96_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_instMin___closed__0));
return v___f_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instMin___boxed(lean_object* v_R_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_NonUnitalSubring_instMin(v_R_97_, v_inst_98_);
lean_dec_ref(v_inst_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet___lam__0(lean_object* v_s_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet(lean_object* v_R_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v___f_105_; 
v___f_105_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_instInfSet___closed__0));
return v___f_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instInfSet___boxed(lean_object* v_R_106_, lean_object* v_inst_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_NonUnitalSubring_instInfSet(v_R_106_, v_inst_107_);
lean_dec_ref(v_inst_107_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___lam__0(lean_object* v_x1_109_, lean_object* v_x2_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_box(0);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg(lean_object* v_inst_115_){
_start:
{
lean_object* v___x_116_; lean_object* v___f_117_; lean_object* v___x_118_; lean_object* v_toLattice_119_; lean_object* v_toSupSet_120_; lean_object* v_toInfSet_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_139_; 
v___x_116_ = lp_mathlib_NonUnitalSubring_instPartialOrder(lean_box(0), v_inst_115_);
v___f_117_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_instInfSet___closed__0));
v___x_118_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_116_, v___f_117_);
v_toLattice_119_ = lean_ctor_get(v___x_118_, 0);
v_toSupSet_120_ = lean_ctor_get(v___x_118_, 1);
v_toInfSet_121_ = lean_ctor_get(v___x_118_, 2);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_139_ == 0)
{
lean_object* v_unused_140_; 
v_unused_140_ = lean_ctor_get(v___x_118_, 3);
lean_dec(v_unused_140_);
v___x_123_ = v___x_118_;
v_isShared_124_ = v_isSharedCheck_139_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_toInfSet_121_);
lean_inc(v_toSupSet_120_);
lean_inc(v_toLattice_119_);
lean_dec(v___x_118_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_139_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v_toSemilatticeSup_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_137_; 
v_toSemilatticeSup_125_ = lean_ctor_get(v_toLattice_119_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v_toLattice_119_);
if (v_isSharedCheck_137_ == 0)
{
lean_object* v_unused_138_; 
v_unused_138_ = lean_ctor_get(v_toLattice_119_, 1);
lean_dec(v_unused_138_);
v___x_127_ = v_toLattice_119_;
v_isShared_128_ = v_isSharedCheck_137_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_toSemilatticeSup_125_);
lean_dec(v_toLattice_119_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_137_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___f_129_; lean_object* v___x_131_; 
v___f_129_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__0));
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 1, v___f_129_);
v___x_131_ = v___x_127_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_toSemilatticeSup_125_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v___f_129_);
v___x_131_ = v_reuseFailAlloc_136_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_132_; lean_object* v___x_134_; 
v___x_132_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___closed__1));
if (v_isShared_124_ == 0)
{
lean_ctor_set(v___x_123_, 3, v___x_132_);
lean_ctor_set(v___x_123_, 0, v___x_131_);
v___x_134_ = v___x_123_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_131_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v_toSupSet_120_);
lean_ctor_set(v_reuseFailAlloc_135_, 2, v_toInfSet_121_);
lean_ctor_set(v_reuseFailAlloc_135_, 3, v___x_132_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg___boxed(lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg(v_inst_141_);
lean_dec_ref(v_inst_141_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice(lean_object* v_R_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mathlib_NonUnitalSubring_instCompleteLattice___redArg(v_inst_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_instCompleteLattice___boxed(lean_object* v_R_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_NonUnitalSubring_instCompleteLattice(v_R_146_, v_inst_147_);
lean_dec_ref(v_inst_147_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center(lean_object* v_R_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lean_box(0);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center___boxed(lean_object* v_R_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_NonUnitalSubring_center(v_R_152_, v_inst_153_);
lean_dec_ref(v_inst_153_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center_instNonUnitalCommRing___redArg(lean_object* v_inst_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v_toAddCommGroup_158_; lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_178_; 
v___x_156_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_155_);
lean_inc_ref(v___x_156_);
v___x_157_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v___x_156_);
v_toAddCommGroup_158_ = lean_ctor_get(v___x_156_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_178_ == 0)
{
lean_object* v_unused_179_; 
v_unused_179_ = lean_ctor_get(v___x_156_, 1);
lean_dec(v_unused_179_);
v___x_160_ = v___x_156_;
v_isShared_161_ = v_isSharedCheck_178_;
goto v_resetjp_159_;
}
else
{
lean_inc(v_toAddCommGroup_158_);
lean_dec(v___x_156_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_178_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v_toAddCommMonoid_162_; lean_object* v_toMul_163_; lean_object* v_toNeg_164_; lean_object* v_toSub_165_; lean_object* v_toZSMul_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_176_; 
v_toAddCommMonoid_162_ = lean_ctor_get(v___x_157_, 0);
lean_inc_ref(v_toAddCommMonoid_162_);
v_toMul_163_ = lean_ctor_get(v___x_157_, 1);
lean_inc(v_toMul_163_);
lean_dec_ref(v___x_157_);
v_toNeg_164_ = lean_ctor_get(v_toAddCommGroup_158_, 1);
v_toSub_165_ = lean_ctor_get(v_toAddCommGroup_158_, 2);
v_toZSMul_166_ = lean_ctor_get(v_toAddCommGroup_158_, 3);
v_isSharedCheck_176_ = !lean_is_exclusive(v_toAddCommGroup_158_);
if (v_isSharedCheck_176_ == 0)
{
lean_object* v_unused_177_; 
v_unused_177_ = lean_ctor_get(v_toAddCommGroup_158_, 0);
lean_dec(v_unused_177_);
v___x_168_ = v_toAddCommGroup_158_;
v_isShared_169_ = v_isSharedCheck_176_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_toZSMul_166_);
lean_inc(v_toSub_165_);
lean_inc(v_toNeg_164_);
lean_dec(v_toAddCommGroup_158_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_176_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_171_; 
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 0, v_toAddCommMonoid_162_);
v___x_171_ = v___x_168_;
goto v_reusejp_170_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_toAddCommMonoid_162_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v_toNeg_164_);
lean_ctor_set(v_reuseFailAlloc_175_, 2, v_toSub_165_);
lean_ctor_set(v_reuseFailAlloc_175_, 3, v_toZSMul_166_);
v___x_171_ = v_reuseFailAlloc_175_;
goto v_reusejp_170_;
}
v_reusejp_170_:
{
lean_object* v___x_173_; 
if (v_isShared_161_ == 0)
{
lean_ctor_set(v___x_160_, 1, v_toMul_163_);
lean_ctor_set(v___x_160_, 0, v___x_171_);
v___x_173_ = v___x_160_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_171_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v_toMul_163_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_center_instNonUnitalCommRing(lean_object* v_R_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_NonUnitalSubring_center_instNonUnitalCommRing___redArg(v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr___redArg(lean_object* v_e_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr(lean_object* v_R_185_, lean_object* v_inst_186_, lean_object* v_S_187_, lean_object* v_inst_188_, lean_object* v_e_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_NonUnitalSubsemiring_centerCongr___redArg(v_e_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerCongr___boxed(lean_object* v_R_191_, lean_object* v_inst_192_, lean_object* v_S_193_, lean_object* v_inst_194_, lean_object* v_e_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_NonUnitalSubring_centerCongr(v_R_191_, v_inst_192_, v_S_193_, v_inst_194_, v_e_195_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_192_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerToMulOpposite___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_197_);
v___x_199_ = lp_mathlib_NonUnitalSubsemiring_centerToMulOpposite___redArg(v___x_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centerToMulOpposite(lean_object* v_R_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___x_202_; 
v___x_202_ = lp_mathlib_NonUnitalSubring_centerToMulOpposite___redArg(v_inst_201_);
return v___x_202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0(lean_object* v_toMul_203_, lean_object* v_x_204_, lean_object* v_inst_205_, lean_object* v_a_206_){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; uint8_t v___x_210_; 
lean_inc(v_toMul_203_);
lean_inc(v_x_204_);
lean_inc(v_a_206_);
v___x_207_ = lean_apply_2(v_toMul_203_, v_a_206_, v_x_204_);
v___x_208_ = lean_apply_2(v_toMul_203_, v_x_204_, v_a_206_);
v___x_209_ = lean_apply_2(v_inst_205_, v___x_207_, v___x_208_);
v___x_210_ = lean_unbox(v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0___boxed(lean_object* v_toMul_211_, lean_object* v_x_212_, lean_object* v_inst_213_, lean_object* v_a_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0(v_toMul_211_, v_x_212_, v_inst_213_, v_a_214_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg(lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_x_220_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v_toMul_223_; lean_object* v___f_224_; uint8_t v___x_225_; 
v___x_221_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_217_);
v___x_222_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_221_);
v_toMul_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_toMul_223_);
lean_dec_ref(v___x_222_);
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_224_, 0, v_toMul_223_);
lean_closure_set(v___f_224_, 1, v_x_220_);
lean_closure_set(v___f_224_, 2, v_inst_218_);
v___x_225_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_224_, v_inst_219_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg___boxed(lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_x_229_){
_start:
{
uint8_t v_res_230_; lean_object* v_r_231_; 
v_res_230_ = lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg(v_inst_226_, v_inst_227_, v_inst_228_, v_x_229_);
v_r_231_ = lean_box(v_res_230_);
return v_r_231_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_NonUnitalSubring_decidableMemCenter(lean_object* v_R_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_x_236_){
_start:
{
uint8_t v___x_237_; 
v___x_237_ = lp_mathlib_NonUnitalSubring_decidableMemCenter___redArg(v_inst_233_, v_inst_234_, v_inst_235_, v_x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_decidableMemCenter___boxed(lean_object* v_R_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_x_242_){
_start:
{
uint8_t v_res_243_; lean_object* v_r_244_; 
v_res_243_ = lp_mathlib_NonUnitalSubring_decidableMemCenter(v_R_238_, v_inst_239_, v_inst_240_, v_inst_241_, v_x_242_);
v_r_244_ = lean_box(v_res_243_);
return v_r_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centralizer(lean_object* v_R_245_, lean_object* v_inst_246_, lean_object* v_s_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lean_box(0);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_centralizer___boxed(lean_object* v_R_249_, lean_object* v_inst_250_, lean_object* v_s_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_NonUnitalSubring_centralizer(v_R_249_, v_inst_250_, v_s_251_);
lean_dec_ref(v_inst_250_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closure(lean_object* v_R_253_, lean_object* v_inst_254_, lean_object* v_s_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lean_box(0);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closure___boxed(lean_object* v_R_257_, lean_object* v_inst_258_, lean_object* v_s_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_NonUnitalSubring_closure(v_R_257_, v_inst_258_, v_s_259_);
lean_dec_ref(v_inst_258_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closureNonUnitalCommRingOfComm___redArg(lean_object* v_inst_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_closureNonUnitalCommRingOfComm(lean_object* v_R_263_, lean_object* v_inst_264_, lean_object* v_s_265_, lean_object* v_hcomm_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_NonUnitalSubringClass_toNonUnitalNonAssocRing___redArg(v_inst_264_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi___lam__0(lean_object* v_s_268_, lean_object* v_x_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lean_box(0);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi(lean_object* v_R_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v___f_274_; 
v___f_274_ = ((lean_object*)(lp_mathlib_NonUnitalSubring_gi___closed__0));
return v___f_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_gi___boxed(lean_object* v_R_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_NonUnitalSubring_gi(v_R_275_, v_inst_276_);
lean_dec_ref(v_inst_276_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prod(lean_object* v_R_278_, lean_object* v_S_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_s_282_, lean_object* v_t_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lean_box(0);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prod___boxed(lean_object* v_R_285_, lean_object* v_S_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_s_289_, lean_object* v_t_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_NonUnitalSubring_prod(v_R_285_, v_S_286_, v_inst_287_, v_inst_288_, v_s_289_, v_t_290_);
lean_dec_ref(v_inst_288_);
lean_dec_ref(v_inst_287_);
return v_res_291_;
}
}
static lean_object* _init_lp_mathlib_NonUnitalSubring_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prodEquiv(lean_object* v_R_293_, lean_object* v_S_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_s_297_, lean_object* v_t_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lean_obj_once(&lp_mathlib_NonUnitalSubring_prodEquiv___closed__0, &lp_mathlib_NonUnitalSubring_prodEquiv___closed__0_once, _init_lp_mathlib_NonUnitalSubring_prodEquiv___closed__0);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSubring_prodEquiv___boxed(lean_object* v_R_300_, lean_object* v_S_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_s_304_, lean_object* v_t_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_NonUnitalSubring_prodEquiv(v_R_300_, v_S_301_, v_inst_302_, v_inst_303_, v_s_304_, v_t_305_);
lean_dec_ref(v_inst_303_);
lean_dec_ref(v_inst_302_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg(lean_object* v_f_308_){
_start:
{
lean_object* v___f_309_; lean_object* v___f_310_; 
v___f_309_ = ((lean_object*)(lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg___closed__0));
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_NonUnitalRingHom_codRestrict___redArg___lam__0), 3, 2);
lean_closure_set(v___f_310_, 0, v___f_309_);
lean_closure_set(v___f_310_, 1, v_f_308_);
return v___f_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict(lean_object* v_R_311_, lean_object* v_S_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_f_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg(v_f_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_rangeRestrict___boxed(lean_object* v_R_317_, lean_object* v_S_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_f_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_NonUnitalRingHom_rangeRestrict(v_R_317_, v_S_318_, v_inst_319_, v_inst_320_, v_f_321_);
lean_dec_ref(v_inst_320_);
lean_dec_ref(v_inst_319_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqLocus(lean_object* v_R_323_, lean_object* v_S_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_f_327_, lean_object* v_g_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lean_box(0);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRingHom_eqLocus___boxed(lean_object* v_R_330_, lean_object* v_S_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_f_334_, lean_object* v_g_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_NonUnitalRingHom_eqLocus(v_R_330_, v_S_331_, v_inst_332_, v_inst_333_, v_f_334_, v_g_335_);
lean_dec(v_g_335_);
lean_dec(v_f_334_);
lean_dec_ref(v_inst_333_);
lean_dec_ref(v_inst_332_);
return v_res_336_;
}
}
static lean_object* _init_lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0(void){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubringCongr(lean_object* v_R_338_, lean_object* v_inst_339_, lean_object* v_s_340_, lean_object* v_t_341_, lean_object* v_h_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lean_obj_once(&lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0, &lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0_once, _init_lp_mathlib_RingEquiv_nonUnitalSubringCongr___closed__0);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_nonUnitalSubringCongr___boxed(lean_object* v_R_344_, lean_object* v_inst_345_, lean_object* v_s_346_, lean_object* v_t_347_, lean_object* v_h_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_RingEquiv_nonUnitalSubringCongr(v_R_344_, v_inst_345_, v_s_346_, v_t_347_, v_h_348_);
lean_dec_ref(v_inst_345_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__0(lean_object* v_f_350_, lean_object* v_x_351_){
_start:
{
lean_object* v___x_61__overap_352_; lean_object* v___x_353_; 
v___x_61__overap_352_ = lp_mathlib_NonUnitalRingHom_rangeRestrict___redArg(v_f_350_);
v___x_353_ = lean_apply_1(v___x_61__overap_352_, v_x_351_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1(lean_object* v_g_354_, lean_object* v_x_355_){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = lp_mathlib_NonUnitalSubringClass_subtype___lam__0(v_x_355_);
v___x_357_ = lean_apply_1(v_g_354_, v___x_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1___boxed(lean_object* v_g_358_, lean_object* v_x_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1(v_g_358_, v_x_359_);
lean_dec(v_x_359_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg(lean_object* v_g_361_, lean_object* v_f_362_){
_start:
{
lean_object* v___f_363_; lean_object* v___f_364_; lean_object* v___x_365_; 
v___f_363_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_363_, 0, v_f_362_);
v___f_364_ = lean_alloc_closure((void*)(lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_364_, 0, v_g_361_);
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v___f_363_);
lean_ctor_set(v___x_365_, 1, v___f_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27(lean_object* v_R_366_, lean_object* v_S_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_g_370_, lean_object* v_f_371_, lean_object* v_h_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_RingEquiv_ofLeftInverse_x27___redArg(v_g_370_, v_f_371_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_ofLeftInverse_x27___boxed(lean_object* v_R_374_, lean_object* v_S_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_g_378_, lean_object* v_f_379_, lean_object* v_h_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_RingEquiv_ofLeftInverse_x27(v_R_374_, v_S_375_, v_inst_376_, v_inst_377_, v_g_378_, v_f_379_, v_h_380_);
lean_dec_ref(v_inst_377_);
lean_dec_ref(v_inst_376_);
return v_res_381_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subsemigroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_NonUnitalSubring_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
