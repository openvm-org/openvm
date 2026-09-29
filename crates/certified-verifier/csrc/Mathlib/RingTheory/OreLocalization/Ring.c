// Lean compiler output
// Module: Mathlib.RingTheory.OreLocalization.Ring
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Defs public import Mathlib.Algebra.Field.Defs public import Mathlib.RingTheory.OreLocalization.NonZeroDivisors
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
lean_object* lp_mathlib_OreLocalization_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_hsmul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_universalMulHom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_instMonoidWithZero___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_instAddMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommRing___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSemiring___redArg(lean_object* v_inst_1_, lean_object* v_S_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v_toAddCommMonoid_6_; lean_object* v_toMonoid_7_; lean_object* v___x_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_30_; 
v___x_4_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_1_);
lean_inc_ref(v_inst_3_);
v___x_5_ = lp_mathlib_OreLocalization_instMonoidWithZero___redArg(v___x_4_, v_S_2_, v_inst_3_);
v_toAddCommMonoid_6_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toAddCommMonoid_6_);
v_toMonoid_7_ = lean_ctor_get(v_inst_1_, 1);
lean_inc_ref(v_toMonoid_7_);
v___x_8_ = lp_mathlib_Semiring_toModule___redArg(v_inst_1_);
v_isSharedCheck_30_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_30_ == 0)
{
lean_object* v_unused_31_; lean_object* v_unused_32_; lean_object* v_unused_33_; 
v_unused_31_ = lean_ctor_get(v_inst_1_, 2);
lean_dec(v_unused_31_);
v_unused_32_ = lean_ctor_get(v_inst_1_, 1);
lean_dec(v_unused_32_);
v_unused_33_ = lean_ctor_get(v_inst_1_, 0);
lean_dec(v_unused_33_);
v___x_10_ = v_inst_1_;
v_isShared_11_ = v_isSharedCheck_30_;
goto v_resetjp_9_;
}
else
{
lean_dec(v_inst_1_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_30_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v_toMonoid_13_; lean_object* v_toZero_14_; lean_object* v_toAdd_15_; lean_object* v_toNSMul_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_28_; 
v___x_12_ = lp_mathlib_OreLocalization_instAddMonoid___redArg(v_toMonoid_7_, v_S_2_, v_inst_3_, v_toAddCommMonoid_6_, v___x_8_);
v_toMonoid_13_ = lean_ctor_get(v___x_5_, 0);
lean_inc_ref(v_toMonoid_13_);
v_toZero_14_ = lean_ctor_get(v___x_5_, 1);
lean_inc(v_toZero_14_);
lean_dec_ref(v___x_5_);
v_toAdd_15_ = lean_ctor_get(v___x_12_, 1);
v_toNSMul_16_ = lean_ctor_get(v___x_12_, 2);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_28_ == 0)
{
lean_object* v_unused_29_; 
v_unused_29_ = lean_ctor_get(v___x_12_, 0);
lean_dec(v_unused_29_);
v___x_18_ = v___x_12_;
v_isShared_19_ = v_isSharedCheck_28_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_toNSMul_16_);
lean_inc(v_toAdd_15_);
lean_dec(v___x_12_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_28_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_21_; 
lean_inc(v_toAdd_15_);
lean_inc(v_toZero_14_);
if (v_isShared_19_ == 0)
{
lean_ctor_set(v___x_18_, 0, v_toZero_14_);
v___x_21_ = v___x_18_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_toZero_14_);
lean_ctor_set(v_reuseFailAlloc_27_, 1, v_toAdd_15_);
lean_ctor_set(v_reuseFailAlloc_27_, 2, v_toNSMul_16_);
v___x_21_ = v_reuseFailAlloc_27_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
lean_object* v_toOne_22_; lean_object* v___x_23_; lean_object* v___x_25_; 
v_toOne_22_ = lean_ctor_get(v_toMonoid_13_, 0);
lean_inc(v_toOne_22_);
v___x_23_ = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(v___x_23_, 0, lean_box(0));
lean_closure_set(v___x_23_, 1, v_toOne_22_);
lean_closure_set(v___x_23_, 2, v_toZero_14_);
lean_closure_set(v___x_23_, 3, v_toAdd_15_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 2, v___x_23_);
lean_ctor_set(v___x_10_, 1, v_toMonoid_13_);
lean_ctor_set(v___x_10_, 0, v___x_21_);
v___x_25_ = v___x_10_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_21_);
lean_ctor_set(v_reuseFailAlloc_26_, 1, v_toMonoid_13_);
lean_ctor_set(v_reuseFailAlloc_26_, 2, v___x_23_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instSemiring(lean_object* v_R_34_, lean_object* v_inst_35_, lean_object* v_S_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_OreLocalization_instSemiring___redArg(v_inst_35_, v_S_36_, v_inst_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule___redArg(lean_object* v_inst_39_, lean_object* v_S_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_toMonoid_43_; lean_object* v___x_44_; 
v_toMonoid_43_ = lean_ctor_get(v_inst_39_, 1);
lean_inc_ref(v_toMonoid_43_);
lean_dec_ref(v_inst_39_);
v___x_44_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_smul___boxed), 8, 6);
lean_closure_set(v___x_44_, 0, lean_box(0));
lean_closure_set(v___x_44_, 1, v_toMonoid_43_);
lean_closure_set(v___x_44_, 2, v_S_40_);
lean_closure_set(v___x_44_, 3, v_inst_41_);
lean_closure_set(v___x_44_, 4, lean_box(0));
lean_closure_set(v___x_44_, 5, v_inst_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule(lean_object* v_R_45_, lean_object* v_inst_46_, lean_object* v_S_47_, lean_object* v_inst_48_, lean_object* v_X_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_OreLocalization_instModule___redArg(v_inst_46_, v_S_47_, v_inst_48_, v_inst_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModule___boxed(lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_S_55_, lean_object* v_inst_56_, lean_object* v_X_57_, lean_object* v_inst_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_OreLocalization_instModule(v_R_53_, v_inst_54_, v_S_55_, v_inst_56_, v_X_57_, v_inst_58_, v_inst_59_);
lean_dec_ref(v_inst_58_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower___redArg(lean_object* v_inst_61_, lean_object* v_S_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v_toMonoid_66_; lean_object* v___x_67_; 
v_toMonoid_66_ = lean_ctor_get(v_inst_61_, 1);
lean_inc_ref(v_toMonoid_66_);
lean_dec_ref(v_inst_61_);
v___x_67_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_hsmul___boxed), 10, 8);
lean_closure_set(v___x_67_, 0, lean_box(0));
lean_closure_set(v___x_67_, 1, lean_box(0));
lean_closure_set(v___x_67_, 2, lean_box(0));
lean_closure_set(v___x_67_, 3, v_toMonoid_66_);
lean_closure_set(v___x_67_, 4, v_S_62_);
lean_closure_set(v___x_67_, 5, v_inst_63_);
lean_closure_set(v___x_67_, 6, v_inst_64_);
lean_closure_set(v___x_67_, 7, v_inst_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower(lean_object* v_R_68_, lean_object* v_inst_69_, lean_object* v_S_70_, lean_object* v_inst_71_, lean_object* v_X_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_R_u2080_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_OreLocalization_instModuleOfIsScalarTower___redArg(v_inst_69_, v_S_70_, v_inst_71_, v_inst_74_, v_inst_78_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instModuleOfIsScalarTower___boxed(lean_object* v_R_82_, lean_object* v_inst_83_, lean_object* v_S_84_, lean_object* v_inst_85_, lean_object* v_X_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_R_u2080_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_OreLocalization_instModuleOfIsScalarTower(v_R_82_, v_inst_83_, v_S_84_, v_inst_85_, v_X_86_, v_inst_87_, v_inst_88_, v_R_u2080_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_inst_93_, v_inst_94_);
lean_dec(v_inst_91_);
lean_dec_ref(v_inst_90_);
lean_dec_ref(v_inst_87_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg___lam__0(lean_object* v_toOne_96_, lean_object* v_r_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v_r_97_);
lean_ctor_set(v___x_98_, 1, v_toOne_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg(lean_object* v_inst_99_){
_start:
{
lean_object* v_toMonoid_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v_toOne_103_; lean_object* v___f_104_; 
v_toMonoid_100_ = lean_ctor_get(v_inst_99_, 1);
v___x_101_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_100_);
v___x_102_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_101_);
v_toOne_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_toOne_103_);
lean_dec_ref(v___x_102_);
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_numeratorRingHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_104_, 0, v_toOne_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___redArg___boxed(lean_object* v_inst_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_OreLocalization_numeratorRingHom___redArg(v_inst_105_);
lean_dec_ref(v_inst_105_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom(lean_object* v_R_107_, lean_object* v_inst_108_, lean_object* v_S_109_, lean_object* v_inst_110_){
_start:
{
lean_object* v_toMonoid_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v_toOne_114_; lean_object* v___f_115_; 
v_toMonoid_111_ = lean_ctor_get(v_inst_108_, 1);
v___x_112_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_111_);
v___x_113_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_112_);
v_toOne_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc(v_toOne_114_);
lean_dec_ref(v___x_113_);
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_numeratorRingHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_115_, 0, v_toOne_114_);
return v___f_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_numeratorRingHom___boxed(lean_object* v_R_116_, lean_object* v_inst_117_, lean_object* v_S_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_OreLocalization_numeratorRingHom(v_R_116_, v_inst_117_, v_S_118_, v_inst_119_);
lean_dec_ref(v_inst_119_);
lean_dec_ref(v_inst_117_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra___redArg(lean_object* v_inst_121_, lean_object* v_S_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v_toSMul_125_; lean_object* v_algebraMap_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_141_; 
v_toSMul_125_ = lean_ctor_get(v_inst_124_, 0);
v_algebraMap_126_ = lean_ctor_get(v_inst_124_, 1);
v_isSharedCheck_141_ = !lean_is_exclusive(v_inst_124_);
if (v_isSharedCheck_141_ == 0)
{
v___x_128_ = v_inst_124_;
v_isShared_129_ = v_isSharedCheck_141_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_algebraMap_126_);
lean_inc(v_toSMul_125_);
lean_dec(v_inst_124_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_141_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v_toMonoid_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v_toOne_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___f_136_; lean_object* v___f_137_; lean_object* v___x_139_; 
v_toMonoid_130_ = lean_ctor_get(v_inst_121_, 1);
v___x_131_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_130_);
v___x_132_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_131_);
v_toOne_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc(v_toOne_133_);
lean_dec_ref(v___x_132_);
v___x_134_ = lp_mathlib_Semiring_toModule___redArg(v_inst_121_);
v___x_135_ = lp_mathlib_OreLocalization_instModuleOfIsScalarTower___redArg(v_inst_121_, v_S_122_, v_inst_123_, v___x_134_, v_toSMul_125_);
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_OreLocalization_numeratorRingHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_136_, 0, v_toOne_133_);
v___f_137_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_137_, 0, v_algebraMap_126_);
lean_closure_set(v___f_137_, 1, v___f_136_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 1, v___f_137_);
lean_ctor_set(v___x_128_, 0, v___x_135_);
v___x_139_ = v___x_128_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v___x_135_);
lean_ctor_set(v_reuseFailAlloc_140_, 1, v___f_137_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra(lean_object* v_R_142_, lean_object* v_inst_143_, lean_object* v_S_144_, lean_object* v_inst_145_, lean_object* v_R_u2080_146_, lean_object* v_inst_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_OreLocalization_instAlgebra___redArg(v_inst_143_, v_S_144_, v_inst_145_, v_inst_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instAlgebra___boxed(lean_object* v_R_150_, lean_object* v_inst_151_, lean_object* v_S_152_, lean_object* v_inst_153_, lean_object* v_R_u2080_154_, lean_object* v_inst_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_OreLocalization_instAlgebra(v_R_150_, v_inst_151_, v_S_152_, v_inst_153_, v_R_u2080_154_, v_inst_155_, v_inst_156_);
lean_dec_ref(v_inst_155_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___redArg(lean_object* v_inst_158_, lean_object* v_f_159_, lean_object* v_fS_160_){
_start:
{
lean_object* v_toMonoid_161_; lean_object* v___x_162_; 
v_toMonoid_161_ = lean_ctor_get(v_inst_158_, 1);
v___x_162_ = lp_mathlib_OreLocalization_universalMulHom___redArg(v_toMonoid_161_, v_f_159_, v_fS_160_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___redArg___boxed(lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_fS_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_OreLocalization_universalHom___redArg(v_inst_163_, v_f_164_, v_fS_165_);
lean_dec_ref(v_inst_163_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom(lean_object* v_R_167_, lean_object* v_inst_168_, lean_object* v_S_169_, lean_object* v_inst_170_, lean_object* v_T_171_, lean_object* v_inst_172_, lean_object* v_f_173_, lean_object* v_fS_174_, lean_object* v_hf_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_OreLocalization_universalHom___redArg(v_inst_172_, v_f_173_, v_fS_174_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_universalHom___boxed(lean_object* v_R_177_, lean_object* v_inst_178_, lean_object* v_S_179_, lean_object* v_inst_180_, lean_object* v_T_181_, lean_object* v_inst_182_, lean_object* v_f_183_, lean_object* v_fS_184_, lean_object* v_hf_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_OreLocalization_universalHom(v_R_177_, v_inst_178_, v_S_179_, v_inst_180_, v_T_181_, v_inst_182_, v_f_183_, v_fS_184_, v_hf_185_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_180_);
lean_dec_ref(v_inst_178_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instRing___redArg(lean_object* v_inst_187_, lean_object* v_S_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v_toSemiring_190_; lean_object* v___x_191_; lean_object* v_toMonoid_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v_toNeg_197_; lean_object* v_toSub_198_; lean_object* v_toZSMul_199_; lean_object* v_toNatCast_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v_toSemiring_190_ = lean_ctor_get(v_inst_187_, 0);
lean_inc_ref_n(v_toSemiring_190_, 2);
lean_inc_ref(v_inst_189_);
v___x_191_ = lp_mathlib_OreLocalization_instSemiring___redArg(v_toSemiring_190_, v_S_188_, v_inst_189_);
v_toMonoid_192_ = lean_ctor_get(v_toSemiring_190_, 1);
lean_inc_ref(v_toMonoid_192_);
v___x_193_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_187_);
v___x_194_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_193_);
lean_dec_ref(v___x_193_);
v___x_195_ = lp_mathlib_Semiring_toModule___redArg(v_toSemiring_190_);
lean_dec_ref(v_toSemiring_190_);
v___x_196_ = lp_mathlib_OreLocalization_instAddGroupOreLocalization___redArg(v_toMonoid_192_, v_S_188_, v_inst_189_, v___x_194_, v___x_195_);
v_toNeg_197_ = lean_ctor_get(v___x_196_, 1);
lean_inc_n(v_toNeg_197_, 2);
v_toSub_198_ = lean_ctor_get(v___x_196_, 2);
lean_inc(v_toSub_198_);
v_toZSMul_199_ = lean_ctor_get(v___x_196_, 3);
lean_inc(v_toZSMul_199_);
lean_dec_ref(v___x_196_);
v_toNatCast_200_ = lean_ctor_get(v___x_191_, 2);
lean_inc(v_toNatCast_200_);
v___x_201_ = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(v___x_201_, 0, lean_box(0));
lean_closure_set(v___x_201_, 1, v_toNatCast_200_);
lean_closure_set(v___x_201_, 2, v_toNeg_197_);
v___x_202_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_202_, 0, v___x_191_);
lean_ctor_set(v___x_202_, 1, v_toNeg_197_);
lean_ctor_set(v___x_202_, 2, v_toSub_198_);
lean_ctor_set(v___x_202_, 3, v_toZSMul_199_);
lean_ctor_set(v___x_202_, 4, v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instRing(lean_object* v_R_203_, lean_object* v_inst_204_, lean_object* v_S_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_OreLocalization_instRing___redArg(v_inst_204_, v_S_205_, v_inst_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommSemiring___redArg(lean_object* v_inst_208_, lean_object* v_S_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_OreLocalization_instSemiring___redArg(v_inst_208_, v_S_209_, v_inst_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommSemiring(lean_object* v_R_212_, lean_object* v_inst_213_, lean_object* v_S_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_OreLocalization_instSemiring___redArg(v_inst_213_, v_S_214_, v_inst_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommRing___redArg(lean_object* v_inst_217_, lean_object* v_S_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_OreLocalization_instRing___redArg(v_inst_217_, v_S_218_, v_inst_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OreLocalization_instCommRing(lean_object* v_R_221_, lean_object* v_inst_222_, lean_object* v_S_223_, lean_object* v_inst_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lp_mathlib_OreLocalization_instRing___redArg(v_inst_222_, v_S_223_, v_inst_224_);
return v___x_225_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_NonZeroDivisors(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_NonZeroDivisors(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OreLocalization_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
}
#ifdef __cplusplus
}
#endif
