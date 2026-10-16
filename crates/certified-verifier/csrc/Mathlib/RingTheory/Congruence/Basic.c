// Lean compiler output
// Module: Mathlib.RingTheory.Congruence.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Hom public import Mathlib.Algebra.Ring.Action.Basic public import Mathlib.GroupTheory.Congruence.Basic public import Mathlib.RingTheory.Congruence.Defs
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
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_RingCon_smulAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Setoid_completeLattice(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroOneClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_RingCon_mk_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_RingCon_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingCon_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingCon_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_RingCon_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_RingCon_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingCon_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_RingCon_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0;
static const lean_closure_object lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RingCon_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RingCon_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RingCon_gi___closed__0 = (const lean_object*)&lp_mathlib_RingCon_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_c_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toMul_6_; lean_object* v___x_7_; 
v___x_5_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_2_);
v_toMul_6_ = lean_ctor_get(v___x_5_, 1);
lean_inc(v_toMul_6_);
lean_dec_ref(v___x_5_);
v___x_7_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_smulAux___boxed), 9, 7);
lean_closure_set(v___x_7_, 0, lean_box(0));
lean_closure_set(v___x_7_, 1, v_inst_1_);
lean_closure_set(v___x_7_, 2, v_toMul_6_);
lean_closure_set(v___x_7_, 3, lean_box(0));
lean_closure_set(v___x_7_, 4, v_inst_3_);
lean_closure_set(v___x_7_, 5, v_c_4_);
lean_closure_set(v___x_7_, 6, lean_box(0));
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSMulQuotient(lean_object* v_00_u03b1_8_, lean_object* v_R_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_c_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_RingCon_instSMulQuotient___redArg(v_inst_10_, v_inst_11_, v_inst_12_, v_c_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_c_18_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_19_; lean_object* v___x_20_; lean_object* v_toAdd_21_; lean_object* v___x_22_; lean_object* v_toMulOneClass_23_; lean_object* v___x_24_; 
v_toNonUnitalNonAssocSemiring_19_ = lean_ctor_get(v_inst_16_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_19_);
v___x_20_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_toNonUnitalNonAssocSemiring_19_);
v_toAdd_21_ = lean_ctor_get(v___x_20_, 1);
lean_inc(v_toAdd_21_);
lean_dec_ref(v___x_20_);
v___x_22_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_16_);
v_toMulOneClass_23_ = lean_ctor_get(v___x_22_, 0);
lean_inc_ref(v_toMulOneClass_23_);
lean_dec_ref(v___x_22_);
v___x_24_ = lp_mathlib_RingCon_instSMulQuotient___redArg(v_toAdd_21_, v_toMulOneClass_23_, v_inst_17_, v_c_18_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower(lean_object* v_00_u03b1_25_, lean_object* v_R_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_c_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(v_inst_28_, v_inst_29_, v_c_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___boxed(lean_object* v_00_u03b1_33_, lean_object* v_R_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_c_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower(v_00_u03b1_33_, v_R_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_c_39_);
lean_dec_ref(v_inst_35_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___redArg(lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_c_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(v_inst_41_, v_inst_42_, v_c_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(lean_object* v_00_u03b1_45_, lean_object* v_R_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_c_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(v_inst_48_, v_inst_49_, v_c_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower___boxed(lean_object* v_00_u03b1_53_, lean_object* v_R_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_c_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_RingCon_instDistribMulActionQuotientOfIsScalarTower(v_00_u03b1_53_, v_R_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_c_59_);
lean_dec_ref(v_inst_55_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___redArg(lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_c_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_61_);
v___x_65_ = lp_mathlib_RingCon_instMulActionQuotientOfIsScalarTower___redArg(v___x_64_, v_inst_62_, v_c_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(lean_object* v_00_u03b1_66_, lean_object* v_R_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_c_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___redArg(v_inst_69_, v_inst_70_, v_c_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower___boxed(lean_object* v_00_u03b1_74_, lean_object* v_R_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_c_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_RingCon_instMulSemiringActionQuotientOfIsScalarTower(v_00_u03b1_74_, v_R_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_c_80_);
lean_dec_ref(v_inst_76_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient___redArg(lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_c_84_){
_start:
{
lean_object* v___x_85_; lean_object* v_toAdd_86_; lean_object* v___x_87_; lean_object* v_toMulOneClass_88_; lean_object* v_toSMul_89_; lean_object* v_algebraMap_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_101_; 
lean_inc_ref_n(v_inst_82_, 2);
v___x_85_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_82_);
v_toAdd_86_ = lean_ctor_get(v___x_85_, 1);
lean_inc(v_toAdd_86_);
lean_dec_ref(v___x_85_);
v___x_87_ = lp_mathlib_instMulZeroOneClassOfSemiring___redArg(v_inst_82_);
v_toMulOneClass_88_ = lean_ctor_get(v___x_87_, 0);
lean_inc_ref(v_toMulOneClass_88_);
lean_dec_ref(v___x_87_);
v_toSMul_89_ = lean_ctor_get(v_inst_83_, 0);
v_algebraMap_90_ = lean_ctor_get(v_inst_83_, 1);
v_isSharedCheck_101_ = !lean_is_exclusive(v_inst_83_);
if (v_isSharedCheck_101_ == 0)
{
v___x_92_ = v_inst_83_;
v_isShared_93_ = v_isSharedCheck_101_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_algebraMap_90_);
lean_inc(v_toSMul_89_);
lean_dec(v_inst_83_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_101_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___f_97_; lean_object* v___x_99_; 
v___x_94_ = lp_mathlib_RingCon_instSMulQuotient___redArg(v_toAdd_86_, v_toMulOneClass_88_, v_toSMul_89_, v_c_84_);
v___x_95_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_82_);
v___x_96_ = lp_mathlib_RingCon_mk_x27___redArg(v___x_95_, v_c_84_);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_97_, 0, v_algebraMap_90_);
lean_closure_set(v___f_97_, 1, v___x_96_);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 1, v___f_97_);
lean_ctor_set(v___x_92_, 0, v___x_94_);
v___x_99_ = v___x_92_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v___x_94_);
lean_ctor_set(v_reuseFailAlloc_100_, 1, v___f_97_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient(lean_object* v_00_u03b1_102_, lean_object* v_R_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_c_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_mathlib_RingCon_instAlgebraQuotient___redArg(v_inst_105_, v_inst_106_, v_c_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAlgebraQuotient___boxed(lean_object* v_00_u03b1_109_, lean_object* v_R_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_c_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_RingCon_instAlgebraQuotient(v_00_u03b1_109_, v_R_110_, v_inst_111_, v_inst_112_, v_inst_113_, v_c_114_);
lean_dec_ref(v_inst_111_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090___redArg(lean_object* v_inst_116_, lean_object* v_c_117_){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_116_);
v___x_119_ = lp_mathlib_RingCon_mk_x27___redArg(v___x_118_, v_c_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090(lean_object* v_00_u03b1_120_, lean_object* v_R_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_c_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_RingCon_mk_u2090___redArg(v_inst_123_, v_c_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_u2090___boxed(lean_object* v_00_u03b1_127_, lean_object* v_R_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_c_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_RingCon_mk_u2090(v_00_u03b1_127_, v_R_128_, v_inst_129_, v_inst_130_, v_inst_131_, v_c_132_);
lean_dec_ref(v_inst_131_);
lean_dec_ref(v_inst_129_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE(lean_object* v_R_134_, lean_object* v_inst_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_box(0);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instLE___boxed(lean_object* v_R_138_, lean_object* v_inst_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_RingCon_instLE(v_R_138_, v_inst_139_, v_inst_140_);
lean_dec(v_inst_140_);
lean_dec(v_inst_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___lam__0(lean_object* v_S_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet(lean_object* v_R_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___f_148_; 
v___f_148_ = ((lean_object*)(lp_mathlib_RingCon_instInfSet___closed__0));
return v___f_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInfSet___boxed(lean_object* v_R_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_RingCon_instInfSet(v_R_149_, v_inst_150_, v_inst_151_);
lean_dec(v_inst_151_);
lean_dec(v_inst_150_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder(lean_object* v_R_156_, lean_object* v_inst_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = ((lean_object*)(lp_mathlib_RingCon_instPartialOrder___closed__0));
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPartialOrder___boxed(lean_object* v_R_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_RingCon_instPartialOrder(v_R_160_, v_inst_161_, v_inst_162_);
lean_dec(v_inst_162_);
lean_dec(v_inst_161_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___lam__0(lean_object* v_c_164_, lean_object* v_d_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_box(0);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0(void){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Setoid_completeLattice(lean_box(0));
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg(lean_object* v_inst_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; lean_object* v___f_172_; lean_object* v___x_173_; lean_object* v_toLattice_174_; lean_object* v_toSupSet_175_; lean_object* v_toInfSet_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_198_; 
v___x_171_ = lp_mathlib_RingCon_instPartialOrder(lean_box(0), v_inst_169_, v_inst_170_);
v___f_172_ = ((lean_object*)(lp_mathlib_RingCon_instInfSet___closed__0));
v___x_173_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_171_, v___f_172_);
v_toLattice_174_ = lean_ctor_get(v___x_173_, 0);
v_toSupSet_175_ = lean_ctor_get(v___x_173_, 1);
v_toInfSet_176_ = lean_ctor_get(v___x_173_, 2);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_173_);
if (v_isSharedCheck_198_ == 0)
{
lean_object* v_unused_199_; 
v_unused_199_ = lean_ctor_get(v___x_173_, 3);
lean_dec(v_unused_199_);
v___x_178_ = v___x_173_;
v_isShared_179_ = v_isSharedCheck_198_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_toInfSet_176_);
lean_inc(v_toSupSet_175_);
lean_inc(v_toLattice_174_);
lean_dec(v___x_173_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_198_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v_toSemilatticeSup_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_196_; 
v_toSemilatticeSup_180_ = lean_ctor_get(v_toLattice_174_, 0);
v_isSharedCheck_196_ = !lean_is_exclusive(v_toLattice_174_);
if (v_isSharedCheck_196_ == 0)
{
lean_object* v_unused_197_; 
v_unused_197_ = lean_ctor_get(v_toLattice_174_, 1);
lean_dec(v_unused_197_);
v___x_182_ = v_toLattice_174_;
v_isShared_183_ = v_isSharedCheck_196_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_toSemilatticeSup_180_);
lean_dec(v_toLattice_174_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_196_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_184_; lean_object* v_toBoundedOrder_185_; lean_object* v_toOrderTop_186_; lean_object* v_toOrderBot_187_; lean_object* v___f_188_; lean_object* v___x_190_; 
v___x_184_ = lean_obj_once(&lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0, &lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0_once, _init_lp_mathlib_RingCon_instCompleteLattice___redArg___closed__0);
v_toBoundedOrder_185_ = lean_ctor_get(v___x_184_, 3);
v_toOrderTop_186_ = lean_ctor_get(v_toBoundedOrder_185_, 0);
v_toOrderBot_187_ = lean_ctor_get(v_toBoundedOrder_185_, 1);
v___f_188_ = ((lean_object*)(lp_mathlib_RingCon_instCompleteLattice___redArg___closed__1));
if (v_isShared_183_ == 0)
{
lean_ctor_set(v___x_182_, 1, v___f_188_);
v___x_190_ = v___x_182_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v_toSemilatticeSup_180_);
lean_ctor_set(v_reuseFailAlloc_195_, 1, v___f_188_);
v___x_190_ = v_reuseFailAlloc_195_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
lean_object* v___x_191_; lean_object* v___x_193_; 
lean_inc(v_toOrderBot_187_);
lean_inc(v_toOrderTop_186_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v_toOrderTop_186_);
lean_ctor_set(v___x_191_, 1, v_toOrderBot_187_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 3, v___x_191_);
lean_ctor_set(v___x_178_, 0, v___x_190_);
v___x_193_ = v___x_178_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_194_, 1, v_toSupSet_175_);
lean_ctor_set(v_reuseFailAlloc_194_, 2, v_toInfSet_176_);
lean_ctor_set(v_reuseFailAlloc_194_, 3, v___x_191_);
v___x_193_ = v_reuseFailAlloc_194_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
return v___x_193_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___redArg___boxed(lean_object* v_inst_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_RingCon_instCompleteLattice___redArg(v_inst_200_, v_inst_201_);
lean_dec(v_inst_201_);
lean_dec(v_inst_200_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice(lean_object* v_R_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_RingCon_instCompleteLattice___redArg(v_inst_204_, v_inst_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCompleteLattice___boxed(lean_object* v_R_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_RingCon_instCompleteLattice(v_R_207_, v_inst_208_, v_inst_209_);
lean_dec(v_inst_209_);
lean_dec(v_inst_208_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___lam__0(lean_object* v_r_211_, lean_object* v___h_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lean_box(0);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi(lean_object* v_R_215_, lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___f_218_; 
v___f_218_ = ((lean_object*)(lp_mathlib_RingCon_gi___closed__0));
return v___f_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_gi___boxed(lean_object* v_R_219_, lean_object* v_inst_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_RingCon_gi(v_R_219_, v_inst_220_, v_inst_221_);
lean_dec(v_inst_221_);
lean_dec(v_inst_220_);
return v_res_222_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Congruence_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
