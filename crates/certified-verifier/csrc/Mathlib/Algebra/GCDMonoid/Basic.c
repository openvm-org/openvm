// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Associated
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
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Units_mk0___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Associates_mk___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizeHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalizeHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueNormalizationMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueNormalizationMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueStrongNormalizationMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instUniqueStrongNormalizationMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v_toMul_6_; lean_object* v___x_7_; lean_object* v_val_8_; lean_object* v___x_9_; 
v___x_4_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_1_);
v___x_5_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_4_);
v_toMul_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toMul_6_);
lean_dec_ref(v___x_5_);
lean_inc(v_x_3_);
v___x_7_ = lean_apply_1(v_inst_2_, v_x_3_);
v_val_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_val_8_);
lean_dec_ref(v___x_7_);
v___x_9_ = lean_apply_2(v_toMul_6_, v_x_3_, v_val_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_x_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_normalize___redArg(v_inst_11_, v_inst_12_, v_x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___redArg(lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_normalize___redArg(v_inst_15_, v_inst_16_, v_a_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_normalize___redArg(v_inst_20_, v_inst_21_, v_a_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizeHom___redArg(lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_24_);
v___x_27_ = lean_alloc_closure((void*)(lp_mathlib_normalize), 4, 3);
lean_closure_set(v___x_27_, 0, lean_box(0));
lean_closure_set(v___x_27_, 1, v___x_26_);
lean_closure_set(v___x_27_, 2, v_inst_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalizeHom(lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_normalizeHom___redArg(v_inst_29_, v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v_toStrongNormalizationMonoid_33_; lean_object* v_toGCDMonoid_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_41_; 
v_toStrongNormalizationMonoid_33_ = lean_ctor_get(v_inst_32_, 0);
v_toGCDMonoid_34_ = lean_ctor_get(v_inst_32_, 1);
v_isSharedCheck_41_ = !lean_is_exclusive(v_inst_32_);
if (v_isSharedCheck_41_ == 0)
{
v___x_36_ = v_inst_32_;
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_toGCDMonoid_34_);
lean_inc(v_toStrongNormalizationMonoid_33_);
lean_dec(v_inst_32_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_41_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_39_; 
if (v_isShared_37_ == 0)
{
v___x_39_ = v___x_36_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v_toStrongNormalizationMonoid_33_);
lean_ctor_set(v_reuseFailAlloc_40_, 1, v_toGCDMonoid_34_);
v___x_39_ = v_reuseFailAlloc_40_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
return v___x_39_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid___redArg(v_inst_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid___boxed(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_instNormalizedGCDMonoidOfStrongNormalizedGCDMonoid(v_00_u03b1_46_, v_inst_47_, v_inst_48_);
lean_dec_ref(v_inst_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0(lean_object* v_toMonoid_50_, lean_object* v_x_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v_toOne_54_; lean_object* v___x_56_; uint8_t v_isShared_57_; uint8_t v_isSharedCheck_61_; 
v___x_52_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_50_);
v___x_53_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_52_);
v_toOne_54_ = lean_ctor_get(v___x_53_, 0);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_53_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v___x_53_, 1);
lean_dec(v_unused_62_);
v___x_56_ = v___x_53_;
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
else
{
lean_inc(v_toOne_54_);
lean_dec(v___x_53_);
v___x_56_ = lean_box(0);
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
v_resetjp_55_:
{
lean_object* v___x_59_; 
lean_inc(v_toOne_54_);
if (v_isShared_57_ == 0)
{
lean_ctor_set(v___x_56_, 1, v_toOne_54_);
v___x_59_ = v___x_56_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_toOne_54_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v_toOne_54_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0___boxed(lean_object* v_toMonoid_63_, lean_object* v_x_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0(v_toMonoid_63_, v_x_64_);
lean_dec(v_x_64_);
lean_dec_ref(v_toMonoid_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg(lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; lean_object* v_toMonoid_68_; lean_object* v___f_69_; 
v___x_67_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_66_);
v_toMonoid_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc_ref(v_toMonoid_68_);
lean_dec_ref(v___x_67_);
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_instStrongNormalizationMonoid___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_69_, 0, v_toMonoid_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizationMonoid(lean_object* v_00_u03b1_70_, lean_object* v_inst_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_71_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits___redArg(lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NormalizationMonoid_ofUniqueUnits(lean_object* v_00_u03b1_76_, lean_object* v_inst_77_, lean_object* v_inst_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueNormalizationMonoid___redArg(lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueNormalizationMonoid(lean_object* v_00_u03b1_82_, lean_object* v_inst_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_83_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueStrongNormalizationMonoid___redArg(lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueStrongNormalizationMonoid(lean_object* v_00_u03b1_88_, lean_object* v_inst_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits___redArg(lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; lean_object* v_toMonoid_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
lean_inc_ref(v_inst_92_);
v___x_93_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_92_);
v_toMonoid_94_ = lean_ctor_get(v___x_93_, 0);
lean_inc_ref(v_toMonoid_94_);
v___x_95_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v_inst_92_);
v___x_96_ = lean_alloc_closure((void*)(lp_mathlib_Associates_out), 4, 3);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, v___x_93_);
lean_closure_set(v___x_96_, 2, v___x_95_);
v___x_97_ = lean_alloc_closure((void*)(lp_mathlib_Associates_mk___boxed), 3, 2);
lean_closure_set(v___x_97_, 0, lean_box(0));
lean_closure_set(v___x_97_, 1, v_toMonoid_94_);
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_96_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesEquivOfUniqueUnits(lean_object* v_00_u03b1_99_, lean_object* v_inst_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_associatesEquivOfUniqueUnits___redArg(v_inst_100_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0(lean_object* v_inst_103_, lean_object* v_toZero_104_, lean_object* v___x_105_, lean_object* v___x_106_, lean_object* v_x_107_){
_start:
{
lean_object* v___x_108_; uint8_t v___x_109_; 
lean_inc(v_x_107_);
v___x_108_ = lean_apply_2(v_inst_103_, v_x_107_, v_toZero_104_);
v___x_109_ = lean_unbox(v___x_108_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; lean_object* v_val_111_; lean_object* v_inv_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_119_; 
v___x_110_ = lp_mathlib_Units_mk0___redArg(v___x_105_, v_x_107_);
v_val_111_ = lean_ctor_get(v___x_110_, 0);
v_inv_112_ = lean_ctor_get(v___x_110_, 1);
v_isSharedCheck_119_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_119_ == 0)
{
v___x_114_ = v___x_110_;
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_inv_112_);
lean_inc(v_val_111_);
lean_dec(v___x_110_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_119_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_117_; 
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 1, v_val_111_);
lean_ctor_set(v___x_114_, 0, v_inv_112_);
v___x_117_ = v___x_114_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v_inv_112_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v_val_111_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
else
{
lean_object* v_toMonoid_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v_toOne_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_130_; 
lean_dec(v_x_107_);
lean_dec_ref(v___x_105_);
v_toMonoid_120_ = lean_ctor_get(v___x_106_, 0);
v___x_121_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_120_);
v___x_122_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_121_);
v_toOne_123_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_130_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_130_ == 0)
{
lean_object* v_unused_131_; 
v_unused_131_ = lean_ctor_get(v___x_122_, 1);
lean_dec(v_unused_131_);
v___x_125_ = v___x_122_;
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_toOne_123_);
lean_dec(v___x_122_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_130_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_128_; 
lean_inc(v_toOne_123_);
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 1, v_toOne_123_);
v___x_128_ = v___x_125_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v_toOne_123_);
lean_ctor_set(v_reuseFailAlloc_129_, 1, v_toOne_123_);
v___x_128_ = v_reuseFailAlloc_129_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
return v___x_128_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0___boxed(lean_object* v_inst_132_, lean_object* v_toZero_133_, lean_object* v___x_134_, lean_object* v___x_135_, lean_object* v_x_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0(v_inst_132_, v_toZero_133_, v___x_134_, v___x_135_, v_x_136_);
lean_dec_ref(v___x_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__1(lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_toZero_140_, lean_object* v_a_141_, lean_object* v_b_142_){
_start:
{
lean_object* v___x_147_; uint8_t v___x_148_; 
lean_inc_ref(v_inst_139_);
lean_inc(v_toZero_140_);
v___x_147_ = lean_apply_2(v_inst_139_, v_a_141_, v_toZero_140_);
v___x_148_ = lean_unbox(v___x_147_);
if (v___x_148_ == 0)
{
lean_dec(v_b_142_);
lean_dec(v_toZero_140_);
lean_dec_ref(v_inst_139_);
goto v___jp_143_;
}
else
{
lean_object* v___x_149_; uint8_t v___x_150_; 
lean_inc(v_toZero_140_);
v___x_149_ = lean_apply_2(v_inst_139_, v_b_142_, v_toZero_140_);
v___x_150_ = lean_unbox(v___x_149_);
if (v___x_150_ == 0)
{
lean_dec(v_toZero_140_);
goto v___jp_143_;
}
else
{
lean_dec_ref(v_inst_138_);
return v_toZero_140_;
}
}
v___jp_143_:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v_toOne_146_; 
v___x_144_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v_inst_138_);
v___x_145_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_144_);
lean_dec_ref(v___x_144_);
v_toOne_146_ = lean_ctor_get(v___x_145_, 0);
lean_inc(v_toOne_146_);
lean_dec_ref(v___x_145_);
return v_toOne_146_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__2(lean_object* v_inst_151_, lean_object* v_toZero_152_, lean_object* v_inst_153_, lean_object* v_a_154_, lean_object* v_b_155_){
_start:
{
lean_object* v___x_156_; uint8_t v___x_157_; 
lean_inc_ref(v_inst_151_);
lean_inc(v_toZero_152_);
v___x_156_ = lean_apply_2(v_inst_151_, v_a_154_, v_toZero_152_);
v___x_157_ = lean_unbox(v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; uint8_t v___x_159_; 
lean_inc(v_toZero_152_);
v___x_158_ = lean_apply_2(v_inst_151_, v_b_155_, v_toZero_152_);
v___x_159_ = lean_unbox(v___x_158_);
if (v___x_159_ == 0)
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v_toOne_162_; 
lean_dec(v_toZero_152_);
v___x_160_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v_inst_153_);
v___x_161_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_160_);
lean_dec_ref(v___x_160_);
v_toOne_162_ = lean_ctor_get(v___x_161_, 0);
lean_inc(v_toOne_162_);
lean_dec_ref(v___x_161_);
return v_toOne_162_;
}
else
{
lean_dec_ref(v_inst_153_);
return v_toZero_152_;
}
}
else
{
lean_dec(v_b_155_);
lean_dec_ref(v_inst_153_);
lean_dec_ref(v_inst_151_);
return v_toZero_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg(lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v_toCommMonoidWithZero_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v_toMonoidWithZero_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v_toZero_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_182_; 
v_toCommMonoidWithZero_165_ = lean_ctor_get(v_inst_163_, 0);
lean_inc_ref(v_toCommMonoidWithZero_165_);
v___x_166_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_toCommMonoidWithZero_165_);
lean_inc_ref(v_inst_163_);
v___x_167_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_inst_163_);
v_toMonoidWithZero_168_ = lean_ctor_get(v___x_167_, 0);
lean_inc_ref(v_toMonoidWithZero_168_);
v___x_169_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_toMonoidWithZero_168_);
v___x_170_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_169_);
v_toZero_171_ = lean_ctor_get(v___x_170_, 1);
v_isSharedCheck_182_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_182_ == 0)
{
lean_object* v_unused_183_; 
v_unused_183_ = lean_ctor_get(v___x_170_, 0);
lean_dec(v_unused_183_);
v___x_173_ = v___x_170_;
v_isShared_174_ = v_isSharedCheck_182_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_toZero_171_);
lean_dec(v___x_170_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_182_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___f_175_; lean_object* v___f_176_; lean_object* v___f_177_; lean_object* v___x_179_; 
lean_inc_n(v_toZero_171_, 2);
lean_inc_ref_n(v_inst_164_, 2);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_175_, 0, v_inst_164_);
lean_closure_set(v___f_175_, 1, v_toZero_171_);
lean_closure_set(v___f_175_, 2, v___x_167_);
lean_closure_set(v___f_175_, 3, v___x_166_);
lean_inc_ref(v_inst_163_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__1), 5, 3);
lean_closure_set(v___f_176_, 0, v_inst_163_);
lean_closure_set(v___f_176_, 1, v_inst_164_);
lean_closure_set(v___f_176_, 2, v_toZero_171_);
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg___lam__2), 5, 3);
lean_closure_set(v___f_177_, 0, v_inst_164_);
lean_closure_set(v___f_177_, 1, v_toZero_171_);
lean_closure_set(v___f_177_, 2, v_inst_163_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 1, v___f_177_);
lean_ctor_set(v___x_173_, 0, v___f_176_);
v___x_179_ = v___x_173_;
goto v_reusejp_178_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v___f_176_);
lean_ctor_set(v_reuseFailAlloc_181_, 1, v___f_177_);
v___x_179_ = v_reuseFailAlloc_181_;
goto v_reusejp_178_;
}
v_reusejp_178_:
{
lean_object* v___x_180_; 
v___x_180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_180_, 0, v___f_175_);
lean_ctor_set(v___x_180_, 1, v___x_179_);
return v___x_180_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid(lean_object* v_G_u2080_184_, lean_object* v_inst_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_CommGroupWithZero_instStrongNormalizedGCDMonoid___redArg(v_inst_185_, v_inst_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___redArg(lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; lean_object* v_gcd_190_; lean_object* v_lcm_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_200_; 
v___x_189_ = lean_box(0);
v_gcd_190_ = lean_ctor_get(v_inst_188_, 0);
v_lcm_191_ = lean_ctor_get(v_inst_188_, 1);
v_isSharedCheck_200_ = !lean_is_exclusive(v_inst_188_);
if (v_isSharedCheck_200_ == 0)
{
v___x_193_ = v_inst_188_;
v_isShared_194_ = v_isSharedCheck_200_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_lcm_191_);
lean_inc(v_gcd_190_);
lean_dec(v_inst_188_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_200_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_198_; 
v___x_195_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_195_, 0, lean_box(0));
lean_closure_set(v___x_195_, 1, lean_box(0));
lean_closure_set(v___x_195_, 2, v___x_189_);
lean_closure_set(v___x_195_, 3, v___x_189_);
lean_closure_set(v___x_195_, 4, lean_box(0));
lean_closure_set(v___x_195_, 5, v___x_189_);
lean_closure_set(v___x_195_, 6, v_gcd_190_);
lean_closure_set(v___x_195_, 7, lean_box(0));
v___x_196_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_196_, 0, lean_box(0));
lean_closure_set(v___x_196_, 1, lean_box(0));
lean_closure_set(v___x_196_, 2, v___x_189_);
lean_closure_set(v___x_196_, 3, v___x_189_);
lean_closure_set(v___x_196_, 4, lean_box(0));
lean_closure_set(v___x_196_, 5, v___x_189_);
lean_closure_set(v___x_196_, 6, v_lcm_191_);
lean_closure_set(v___x_196_, 7, lean_box(0));
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 1, v___x_196_);
lean_ctor_set(v___x_193_, 0, v___x_195_);
v___x_198_ = v___x_193_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_195_);
lean_ctor_set(v_reuseFailAlloc_199_, 1, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_Associates_instGCDMonoid___redArg(v_inst_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instGCDMonoid___boxed(lean_object* v_00_u03b1_205_, lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Associates_instGCDMonoid(v_00_u03b1_205_, v_inst_206_, v_inst_207_);
lean_dec_ref(v_inst_206_);
return v_res_208_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associated(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associated(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
