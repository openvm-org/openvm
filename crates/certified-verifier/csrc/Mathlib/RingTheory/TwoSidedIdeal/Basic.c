// Lean compiler output
// Module: Mathlib.RingTheory.TwoSidedIdeal.Basic
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Abel public import Mathlib.Algebra.Ring.Opposite public import Mathlib.GroupTheory.GroupAction.SubMulAction public import Mathlib.RingTheory.Congruence.Opposite
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_setLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_setLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeOrderEmbedding(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeOrderEmbedding___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__0 = (const lean_object*)&lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__0_value;
static const lean_closure_object lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__1 = (const lean_object*)&lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__1_value;
static const lean_ctor_object lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__0_value),((lean_object*)&lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__1_value)}};
static const lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__2 = (const lean_object*)&lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_addCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_op___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_unop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_opOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_opOrderIso(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_setLike(lean_object* v_R_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_setLike___boxed(lean_object* v_R_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_TwoSidedIdeal_setLike(v_R_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
static lean_object* _init_lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_box(0);
v___x_8_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instPartialOrder(lean_object* v_R_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0, &lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0_once, _init_lp_mathlib_TwoSidedIdeal_instPartialOrder___closed__0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instPartialOrder___boxed(lean_object* v_R_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_TwoSidedIdeal_instPartialOrder(v_R_12_, v_inst_13_);
lean_dec_ref(v_inst_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk___redArg(lean_object* v_c_15_){
_start:
{
return v_c_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_c_18_){
_start:
{
return v_c_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk___boxed(lean_object* v_R_19_, lean_object* v_inst_20_, lean_object* v_c_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_TwoSidedIdeal_mk(v_R_19_, v_inst_20_, v_c_21_);
lean_dec_ref(v_inst_20_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeOrderEmbedding(lean_object* v_R_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_box(0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeOrderEmbedding___boxed(lean_object* v_R_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_TwoSidedIdeal_coeOrderEmbedding(v_R_26_, v_inst_27_);
lean_dec_ref(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__0(lean_object* v_self_29_){
_start:
{
return v_self_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___lam__1(lean_object* v_ringCon_30_){
_start:
{
return v_ringCon_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon(lean_object* v_R_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = ((lean_object*)(lp_mathlib_TwoSidedIdeal_orderIsoRingCon___closed__2));
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_orderIsoRingCon___boxed(lean_object* v_R_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_TwoSidedIdeal_orderIsoRingCon(v_R_39_, v_inst_40_);
lean_dec_ref(v_inst_40_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk_x27(lean_object* v_R_42_, lean_object* v_inst_43_, lean_object* v_carrier_44_, lean_object* v_zero__mem_45_, lean_object* v_add__mem_46_, lean_object* v_neg__mem_47_, lean_object* v_mul__mem__left_48_, lean_object* v_mul__mem__right_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lean_box(0);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_mk_x27___boxed(lean_object* v_R_51_, lean_object* v_inst_52_, lean_object* v_carrier_53_, lean_object* v_zero__mem_54_, lean_object* v_add__mem_55_, lean_object* v_neg__mem_56_, lean_object* v_mul__mem__left_57_, lean_object* v_mul__mem__right_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_TwoSidedIdeal_mk_x27(v_R_51_, v_inst_52_, v_carrier_53_, v_zero__mem_54_, v_add__mem_55_, v_neg__mem_56_, v_mul__mem__left_57_, v_mul__mem__right_58_);
lean_dec_ref(v_inst_52_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg___lam__0(lean_object* v_toAdd_60_, lean_object* v_x_61_, lean_object* v_y_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_apply_2(v_toAdd_60_, v_x_61_, v_y_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg(lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v_toAdd_67_; lean_object* v___f_68_; 
v___x_65_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_64_);
v___x_66_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_65_);
v_toAdd_67_ = lean_ctor_get(v___x_66_, 1);
lean_inc(v_toAdd_67_);
lean_dec_ref(v___x_66_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_toAdd_67_);
return v___f_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instAddSubtypeMem(lean_object* v_R_69_, lean_object* v_inst_70_, lean_object* v_I_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg(v_inst_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem___redArg(lean_object* v_inst_73_){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v_toZero_76_; 
v___x_74_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_73_);
v___x_75_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v___x_74_);
v_toZero_76_ = lean_ctor_get(v___x_75_, 1);
lean_inc(v_toZero_76_);
lean_dec_ref(v___x_75_);
return v_toZero_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem(lean_object* v_R_77_, lean_object* v_inst_78_, lean_object* v_I_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem___redArg(v_inst_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg___lam__0(lean_object* v_toNSMul_81_, lean_object* v_n_82_, lean_object* v_x_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lean_apply_2(v_toNSMul_81_, v_n_82_, v_x_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg(lean_object* v_inst_85_){
_start:
{
lean_object* v_toAddCommGroup_86_; lean_object* v_toAddMonoid_87_; lean_object* v_toNSMul_88_; lean_object* v___f_89_; 
v_toAddCommGroup_86_ = lean_ctor_get(v_inst_85_, 0);
lean_inc_ref(v_toAddCommGroup_86_);
lean_dec_ref(v_inst_85_);
v_toAddMonoid_87_ = lean_ctor_get(v_toAddCommGroup_86_, 0);
lean_inc_ref(v_toAddMonoid_87_);
lean_dec_ref(v_toAddCommGroup_86_);
v_toNSMul_88_ = lean_ctor_get(v_toAddMonoid_87_, 2);
lean_inc(v_toNSMul_88_);
lean_dec_ref(v_toAddMonoid_87_);
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_89_, 0, v_toNSMul_88_);
return v___f_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem(lean_object* v_R_90_, lean_object* v_inst_91_, lean_object* v_I_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg(v_inst_91_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg___lam__0(lean_object* v_toNeg_94_, lean_object* v_x_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lean_apply_1(v_toNeg_94_, v_x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg(lean_object* v_inst_97_){
_start:
{
lean_object* v_toAddCommGroup_98_; lean_object* v___x_99_; lean_object* v_toNeg_100_; lean_object* v___f_101_; 
v_toAddCommGroup_98_ = lean_ctor_get(v_inst_97_, 0);
v___x_99_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_toAddCommGroup_98_);
v_toNeg_100_ = lean_ctor_get(v___x_99_, 1);
lean_inc(v_toNeg_100_);
lean_dec_ref(v___x_99_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg___lam__0), 2, 1);
lean_closure_set(v___f_101_, 0, v_toNeg_100_);
return v___f_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg___boxed(lean_object* v_inst_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg(v_inst_102_);
lean_dec_ref(v_inst_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem(lean_object* v_R_104_, lean_object* v_inst_105_, lean_object* v_I_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg(v_inst_105_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___boxed(lean_object* v_R_108_, lean_object* v_inst_109_, lean_object* v_I_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_TwoSidedIdeal_instNegSubtypeMem(v_R_108_, v_inst_109_, v_I_110_);
lean_dec_ref(v_inst_109_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg___lam__0(lean_object* v_toSub_112_, lean_object* v_x_113_, lean_object* v_y_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_apply_2(v_toSub_112_, v_x_113_, v_y_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg(lean_object* v_inst_116_){
_start:
{
lean_object* v_toAddCommGroup_117_; lean_object* v_toSub_118_; lean_object* v___f_119_; 
v_toAddCommGroup_117_ = lean_ctor_get(v_inst_116_, 0);
lean_inc_ref(v_toAddCommGroup_117_);
lean_dec_ref(v_inst_116_);
v_toSub_118_ = lean_ctor_get(v_toAddCommGroup_117_, 2);
lean_inc(v_toSub_118_);
lean_dec_ref(v_toAddCommGroup_117_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_119_, 0, v_toSub_118_);
return v___f_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSubSubtypeMem(lean_object* v_R_120_, lean_object* v_inst_121_, lean_object* v_I_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg(v_inst_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg___lam__0(lean_object* v_toZSMul_124_, lean_object* v_n_125_, lean_object* v_x_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_apply_2(v_toZSMul_124_, v_n_125_, v_x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg(lean_object* v_inst_128_){
_start:
{
lean_object* v_toAddCommGroup_129_; lean_object* v_toZSMul_130_; lean_object* v___f_131_; 
v_toAddCommGroup_129_ = lean_ctor_get(v_inst_128_, 0);
lean_inc_ref(v_toAddCommGroup_129_);
lean_dec_ref(v_inst_128_);
v_toZSMul_130_ = lean_ctor_get(v_toAddCommGroup_129_, 3);
lean_inc(v_toZSMul_130_);
lean_dec_ref(v_toAddCommGroup_129_);
v___f_131_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg___lam__0), 3, 1);
lean_closure_set(v___f_131_, 0, v_toZSMul_130_);
return v___f_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem(lean_object* v_R_132_, lean_object* v_inst_133_, lean_object* v_I_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg(v_inst_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_addCommGroup___redArg(lean_object* v_inst_136_){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
lean_inc_ref_n(v_inst_136_, 4);
v___x_137_ = lp_mathlib_TwoSidedIdeal_instAddSubtypeMem___redArg(v_inst_136_);
v___x_138_ = lp_mathlib_TwoSidedIdeal_instZeroSubtypeMem___redArg(v_inst_136_);
v___x_139_ = lp_mathlib_TwoSidedIdeal_instSMulNatSubtypeMem___redArg(v_inst_136_);
v___x_140_ = lp_mathlib_TwoSidedIdeal_instNegSubtypeMem___redArg(v_inst_136_);
v___x_141_ = lp_mathlib_TwoSidedIdeal_instSubSubtypeMem___redArg(v_inst_136_);
v___x_142_ = lp_mathlib_TwoSidedIdeal_instSMulIntSubtypeMem___redArg(v_inst_136_);
v___x_143_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___x_137_, v___x_138_, v___x_139_, v___x_140_, v___x_141_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_addCommGroup(lean_object* v_R_144_, lean_object* v_inst_145_, lean_object* v_I_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_TwoSidedIdeal_addCommGroup___redArg(v_inst_145_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0(lean_object* v_self_148_){
_start:
{
lean_inc(v_self_148_);
return v_self_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0___boxed(lean_object* v_self_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___lam__0(v_self_149_);
lean_dec(v_self_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom(lean_object* v_R_152_, lean_object* v_inst_153_, lean_object* v_I_154_){
_start:
{
lean_object* v___f_155_; 
v___f_155_ = ((lean_object*)(lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___closed__0));
return v___f_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_coeAddMonoidHom___boxed(lean_object* v_R_156_, lean_object* v_inst_157_, lean_object* v_I_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_TwoSidedIdeal_coeAddMonoidHom(v_R_156_, v_inst_157_, v_I_158_);
lean_dec_ref(v_inst_157_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_op(lean_object* v_R_160_, lean_object* v_inst_161_, lean_object* v_I_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lean_box(0);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_op___boxed(lean_object* v_R_164_, lean_object* v_inst_165_, lean_object* v_I_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_TwoSidedIdeal_op(v_R_164_, v_inst_165_, v_I_166_);
lean_dec_ref(v_inst_165_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_unop(lean_object* v_R_168_, lean_object* v_inst_169_, lean_object* v_I_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_box(0);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_unop___boxed(lean_object* v_R_172_, lean_object* v_inst_173_, lean_object* v_I_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_TwoSidedIdeal_unop(v_R_172_, v_inst_173_, v_I_174_);
lean_dec_ref(v_inst_173_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_opOrderIso___redArg(lean_object* v_inst_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
lean_inc_ref(v_inst_176_);
v___x_177_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_op___boxed), 3, 2);
lean_closure_set(v___x_177_, 0, lean_box(0));
lean_closure_set(v___x_177_, 1, v_inst_176_);
v___x_178_ = lean_alloc_closure((void*)(lp_mathlib_TwoSidedIdeal_unop___boxed), 3, 2);
lean_closure_set(v___x_178_, 0, lean_box(0));
lean_closure_set(v___x_178_, 1, v_inst_176_);
v___x_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_177_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TwoSidedIdeal_opOrderIso(lean_object* v_R_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_TwoSidedIdeal_opOrderIso___redArg(v_inst_181_);
return v___x_182_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Abel(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Abel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_SubMulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Congruence_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_TwoSidedIdeal_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
