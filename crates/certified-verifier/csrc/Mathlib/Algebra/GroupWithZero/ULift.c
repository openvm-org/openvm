// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.ULift
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.ULift public import Mathlib.Algebra.GroupWithZero.InjSurj
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
lean_object* lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_ULift_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ULift_inv___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroOneClass___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; lean_object* v_toMul_3_; lean_object* v_toZero_4_; lean_object* v_toMulOneClass_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_23_; 
lean_inc_ref(v_inst_1_);
v___x_2_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_1_);
v_toMul_3_ = lean_ctor_get(v___x_2_, 0);
lean_inc(v_toMul_3_);
v_toZero_4_ = lean_ctor_get(v___x_2_, 1);
lean_inc(v_toZero_4_);
lean_dec_ref(v___x_2_);
v_toMulOneClass_5_ = lean_ctor_get(v_inst_1_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_23_ == 0)
{
lean_object* v_unused_24_; 
v_unused_24_ = lean_ctor_get(v_inst_1_, 1);
lean_dec(v_unused_24_);
v___x_7_ = v_inst_1_;
v_isShared_8_ = v_isSharedCheck_23_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_toMulOneClass_5_);
lean_dec(v_inst_1_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_23_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v___x_9_; lean_object* v_toOne_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_21_; 
v___x_9_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_5_);
v_toOne_10_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_21_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_21_ == 0)
{
lean_object* v_unused_22_; 
v_unused_22_ = lean_ctor_get(v___x_9_, 1);
lean_dec(v_unused_22_);
v___x_12_ = v___x_9_;
v_isShared_13_ = v_isSharedCheck_21_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_toOne_10_);
lean_dec(v___x_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_21_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___f_14_; lean_object* v___x_16_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_toMul_3_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 1, v___f_14_);
v___x_16_ = v___x_12_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v_toOne_10_);
lean_ctor_set(v_reuseFailAlloc_20_, 1, v___f_14_);
v___x_16_ = v_reuseFailAlloc_20_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
lean_object* v___x_18_; 
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 1, v_toZero_4_);
lean_ctor_set(v___x_7_, 0, v___x_16_);
v___x_18_ = v___x_7_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v___x_16_);
lean_ctor_set(v_reuseFailAlloc_19_, 1, v_toZero_4_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulZeroOneClass(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_ULift_mulZeroOneClass___redArg(v_inst_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero___redArg___lam__0(lean_object* v_toNPow_28_, lean_object* v_n_29_, lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_apply_2(v_toNPow_28_, v_n_29_, v_x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v_toMul_35_; lean_object* v_toZero_36_; lean_object* v_toMulOneClass_37_; lean_object* v___x_38_; lean_object* v_toMonoid_39_; lean_object* v___x_41_; uint8_t v_isShared_42_; uint8_t v_isSharedCheck_59_; 
lean_inc_ref(v_inst_32_);
v___x_33_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_32_);
lean_inc_ref(v___x_33_);
v___x_34_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_33_);
v_toMul_35_ = lean_ctor_get(v___x_34_, 0);
lean_inc(v_toMul_35_);
v_toZero_36_ = lean_ctor_get(v___x_34_, 1);
lean_inc(v_toZero_36_);
lean_dec_ref(v___x_34_);
v_toMulOneClass_37_ = lean_ctor_get(v___x_33_, 0);
lean_inc_ref(v_toMulOneClass_37_);
lean_dec_ref(v___x_33_);
v___x_38_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_37_);
v_toMonoid_39_ = lean_ctor_get(v_inst_32_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v_inst_32_);
if (v_isSharedCheck_59_ == 0)
{
lean_object* v_unused_60_; 
v_unused_60_ = lean_ctor_get(v_inst_32_, 1);
lean_dec(v_unused_60_);
v___x_41_ = v_inst_32_;
v_isShared_42_ = v_isSharedCheck_59_;
goto v_resetjp_40_;
}
else
{
lean_inc(v_toMonoid_39_);
lean_dec(v_inst_32_);
v___x_41_ = lean_box(0);
v_isShared_42_ = v_isSharedCheck_59_;
goto v_resetjp_40_;
}
v_resetjp_40_:
{
lean_object* v_toOne_43_; lean_object* v_toNPow_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_56_; 
v_toOne_43_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_toOne_43_);
lean_dec_ref(v___x_38_);
v_toNPow_44_ = lean_ctor_get(v_toMonoid_39_, 2);
v_isSharedCheck_56_ = !lean_is_exclusive(v_toMonoid_39_);
if (v_isSharedCheck_56_ == 0)
{
lean_object* v_unused_57_; lean_object* v_unused_58_; 
v_unused_57_ = lean_ctor_get(v_toMonoid_39_, 1);
lean_dec(v_unused_57_);
v_unused_58_ = lean_ctor_get(v_toMonoid_39_, 0);
lean_dec(v_unused_58_);
v___x_46_ = v_toMonoid_39_;
v_isShared_47_ = v_isSharedCheck_56_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_toNPow_44_);
lean_dec(v_toMonoid_39_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_56_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_51_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_48_, 0, v_toMul_35_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_49_, 0, v_toNPow_44_);
if (v_isShared_47_ == 0)
{
lean_ctor_set(v___x_46_, 2, v___f_49_);
lean_ctor_set(v___x_46_, 1, v___f_48_);
lean_ctor_set(v___x_46_, 0, v_toOne_43_);
v___x_51_ = v___x_46_;
goto v_reusejp_50_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_toOne_43_);
lean_ctor_set(v_reuseFailAlloc_55_, 1, v___f_48_);
lean_ctor_set(v_reuseFailAlloc_55_, 2, v___f_49_);
v___x_51_ = v_reuseFailAlloc_55_;
goto v_reusejp_50_;
}
v_reusejp_50_:
{
lean_object* v___x_53_; 
if (v_isShared_42_ == 0)
{
lean_ctor_set(v___x_41_, 1, v_toZero_36_);
lean_ctor_set(v___x_41_, 0, v___x_51_);
v___x_53_ = v___x_41_;
goto v_reusejp_52_;
}
else
{
lean_object* v_reuseFailAlloc_54_; 
v_reuseFailAlloc_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_54_, 0, v___x_51_);
lean_ctor_set(v_reuseFailAlloc_54_, 1, v_toZero_36_);
v___x_53_ = v_reuseFailAlloc_54_;
goto v_reusejp_52_;
}
v_reusejp_52_:
{
return v___x_53_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_monoidWithZero(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_ULift_monoidWithZero___redArg(v_inst_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoidWithZero___redArg(lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v_toMul_68_; lean_object* v_toZero_69_; lean_object* v_toMulOneClass_70_; lean_object* v___x_71_; lean_object* v_toMonoid_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_92_; 
v___x_65_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_64_);
lean_inc_ref(v___x_65_);
v___x_66_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_65_);
lean_inc_ref(v___x_66_);
v___x_67_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_66_);
v_toMul_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc(v_toMul_68_);
v_toZero_69_ = lean_ctor_get(v___x_67_, 1);
lean_inc(v_toZero_69_);
lean_dec_ref(v___x_67_);
v_toMulOneClass_70_ = lean_ctor_get(v___x_66_, 0);
lean_inc_ref(v_toMulOneClass_70_);
lean_dec_ref(v___x_66_);
v___x_71_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_70_);
v_toMonoid_72_ = lean_ctor_get(v___x_65_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_65_);
if (v_isSharedCheck_92_ == 0)
{
lean_object* v_unused_93_; 
v_unused_93_ = lean_ctor_get(v___x_65_, 1);
lean_dec(v_unused_93_);
v___x_74_ = v___x_65_;
v_isShared_75_ = v_isSharedCheck_92_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_toMonoid_72_);
lean_dec(v___x_65_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_92_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v_toOne_76_; lean_object* v_toNPow_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_89_; 
v_toOne_76_ = lean_ctor_get(v___x_71_, 0);
lean_inc(v_toOne_76_);
lean_dec_ref(v___x_71_);
v_toNPow_77_ = lean_ctor_get(v_toMonoid_72_, 2);
v_isSharedCheck_89_ = !lean_is_exclusive(v_toMonoid_72_);
if (v_isSharedCheck_89_ == 0)
{
lean_object* v_unused_90_; lean_object* v_unused_91_; 
v_unused_90_ = lean_ctor_get(v_toMonoid_72_, 1);
lean_dec(v_unused_90_);
v_unused_91_ = lean_ctor_get(v_toMonoid_72_, 0);
lean_dec(v_unused_91_);
v___x_79_ = v_toMonoid_72_;
v_isShared_80_ = v_isSharedCheck_89_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_toNPow_77_);
lean_dec(v_toMonoid_72_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_89_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___f_81_; lean_object* v___f_82_; lean_object* v___x_84_; 
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_81_, 0, v_toMul_68_);
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_82_, 0, v_toNPow_77_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 2, v___f_82_);
lean_ctor_set(v___x_79_, 1, v___f_81_);
lean_ctor_set(v___x_79_, 0, v_toOne_76_);
v___x_84_ = v___x_79_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_toOne_76_);
lean_ctor_set(v_reuseFailAlloc_88_, 1, v___f_81_);
lean_ctor_set(v_reuseFailAlloc_88_, 2, v___f_82_);
v___x_84_ = v_reuseFailAlloc_88_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
lean_object* v___x_86_; 
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 1, v_toZero_69_);
lean_ctor_set(v___x_74_, 0, v___x_84_);
v___x_86_ = v___x_74_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
lean_ctor_set(v_reuseFailAlloc_87_, 1, v_toZero_69_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commMonoidWithZero(lean_object* v_00_u03b1_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_ULift_commMonoidWithZero___redArg(v_inst_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero___redArg___lam__0(lean_object* v_toZPow_97_, lean_object* v_n_98_, lean_object* v_x_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_apply_2(v_toZPow_97_, v_n_98_, v_x_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero___redArg(lean_object* v_inst_101_){
_start:
{
lean_object* v_toMonoidWithZero_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v_toMul_105_; lean_object* v_toZero_106_; lean_object* v_toMulOneClass_107_; lean_object* v___x_108_; lean_object* v_toOne_109_; lean_object* v___x_110_; lean_object* v_toMonoid_111_; lean_object* v___x_113_; uint8_t v_isShared_114_; uint8_t v_isSharedCheck_144_; 
v_toMonoidWithZero_102_ = lean_ctor_get(v_inst_101_, 0);
lean_inc_ref_n(v_toMonoidWithZero_102_, 2);
v___x_103_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_toMonoidWithZero_102_);
lean_inc_ref(v___x_103_);
v___x_104_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_103_);
v_toMul_105_ = lean_ctor_get(v___x_104_, 0);
lean_inc(v_toMul_105_);
v_toZero_106_ = lean_ctor_get(v___x_104_, 1);
lean_inc(v_toZero_106_);
lean_dec_ref(v___x_104_);
v_toMulOneClass_107_ = lean_ctor_get(v___x_103_, 0);
lean_inc_ref(v_toMulOneClass_107_);
lean_dec_ref(v___x_103_);
v___x_108_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_107_);
v_toOne_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_toOne_109_);
lean_dec_ref(v___x_108_);
v___x_110_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v_inst_101_);
v_toMonoid_111_ = lean_ctor_get(v_toMonoidWithZero_102_, 0);
v_isSharedCheck_144_ = !lean_is_exclusive(v_toMonoidWithZero_102_);
if (v_isSharedCheck_144_ == 0)
{
lean_object* v_unused_145_; 
v_unused_145_ = lean_ctor_get(v_toMonoidWithZero_102_, 1);
lean_dec(v_unused_145_);
v___x_113_ = v_toMonoidWithZero_102_;
v_isShared_114_ = v_isSharedCheck_144_;
goto v_resetjp_112_;
}
else
{
lean_inc(v_toMonoid_111_);
lean_dec(v_toMonoidWithZero_102_);
v___x_113_ = lean_box(0);
v_isShared_114_ = v_isSharedCheck_144_;
goto v_resetjp_112_;
}
v_resetjp_112_:
{
lean_object* v_toInv_115_; lean_object* v_toDiv_116_; lean_object* v_toZPow_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_142_; 
v_toInv_115_ = lean_ctor_get(v___x_110_, 1);
v_toDiv_116_ = lean_ctor_get(v___x_110_, 2);
v_toZPow_117_ = lean_ctor_get(v___x_110_, 3);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_142_ == 0)
{
lean_object* v_unused_143_; 
v_unused_143_ = lean_ctor_get(v___x_110_, 0);
lean_dec(v_unused_143_);
v___x_119_ = v___x_110_;
v_isShared_120_ = v_isSharedCheck_142_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_toZPow_117_);
lean_inc(v_toDiv_116_);
lean_inc(v_toInv_115_);
lean_dec(v___x_110_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_142_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v_toNPow_121_; lean_object* v___x_123_; uint8_t v_isShared_124_; uint8_t v_isSharedCheck_139_; 
v_toNPow_121_ = lean_ctor_get(v_toMonoid_111_, 2);
v_isSharedCheck_139_ = !lean_is_exclusive(v_toMonoid_111_);
if (v_isSharedCheck_139_ == 0)
{
lean_object* v_unused_140_; lean_object* v_unused_141_; 
v_unused_140_ = lean_ctor_get(v_toMonoid_111_, 1);
lean_dec(v_unused_140_);
v_unused_141_ = lean_ctor_get(v_toMonoid_111_, 0);
lean_dec(v_unused_141_);
v___x_123_ = v_toMonoid_111_;
v_isShared_124_ = v_isSharedCheck_139_;
goto v_resetjp_122_;
}
else
{
lean_inc(v_toNPow_121_);
lean_dec(v_toMonoid_111_);
v___x_123_ = lean_box(0);
v_isShared_124_ = v_isSharedCheck_139_;
goto v_resetjp_122_;
}
v_resetjp_122_:
{
lean_object* v___f_125_; lean_object* v___f_126_; lean_object* v___f_127_; lean_object* v___f_128_; lean_object* v___f_129_; lean_object* v___x_131_; 
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_125_, 0, v_toMul_105_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_ULift_groupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_126_, 0, v_toZPow_117_);
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_127_, 0, v_toInv_115_);
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_128_, 0, v_toDiv_116_);
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_129_, 0, v_toNPow_121_);
if (v_isShared_124_ == 0)
{
lean_ctor_set(v___x_123_, 2, v___f_129_);
lean_ctor_set(v___x_123_, 1, v___f_125_);
lean_ctor_set(v___x_123_, 0, v_toOne_109_);
v___x_131_ = v___x_123_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_toOne_109_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v___f_125_);
lean_ctor_set(v_reuseFailAlloc_138_, 2, v___f_129_);
v___x_131_ = v_reuseFailAlloc_138_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_133_; 
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 1, v_toZero_106_);
lean_ctor_set(v___x_113_, 0, v___x_131_);
v___x_133_ = v___x_113_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v___x_131_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_toZero_106_);
v___x_133_ = v_reuseFailAlloc_137_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
lean_object* v___x_135_; 
if (v_isShared_120_ == 0)
{
lean_ctor_set(v___x_119_, 3, v___f_126_);
lean_ctor_set(v___x_119_, 2, v___f_128_);
lean_ctor_set(v___x_119_, 1, v___f_127_);
lean_ctor_set(v___x_119_, 0, v___x_133_);
v___x_135_ = v___x_119_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_133_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v___f_127_);
lean_ctor_set(v_reuseFailAlloc_136_, 2, v___f_128_);
lean_ctor_set(v_reuseFailAlloc_136_, 3, v___f_126_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_groupWithZero(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_ULift_groupWithZero___redArg(v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroupWithZero___redArg(lean_object* v_inst_149_){
_start:
{
lean_object* v___x_150_; lean_object* v_toMonoidWithZero_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v_toMul_154_; lean_object* v_toZero_155_; lean_object* v_toMulOneClass_156_; lean_object* v___x_157_; lean_object* v_toOne_158_; lean_object* v___x_159_; lean_object* v_toMonoid_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_193_; 
v___x_150_ = lp_mathlib_CommGroupWithZero_toGroupWithZero___redArg(v_inst_149_);
v_toMonoidWithZero_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref_n(v_toMonoidWithZero_151_, 2);
v___x_152_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_toMonoidWithZero_151_);
lean_inc_ref(v___x_152_);
v___x_153_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_152_);
v_toMul_154_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_toMul_154_);
v_toZero_155_ = lean_ctor_get(v___x_153_, 1);
lean_inc(v_toZero_155_);
lean_dec_ref(v___x_153_);
v_toMulOneClass_156_ = lean_ctor_get(v___x_152_, 0);
lean_inc_ref(v_toMulOneClass_156_);
lean_dec_ref(v___x_152_);
v___x_157_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_156_);
v_toOne_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc(v_toOne_158_);
lean_dec_ref(v___x_157_);
v___x_159_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_150_);
v_toMonoid_160_ = lean_ctor_get(v_toMonoidWithZero_151_, 0);
v_isSharedCheck_193_ = !lean_is_exclusive(v_toMonoidWithZero_151_);
if (v_isSharedCheck_193_ == 0)
{
lean_object* v_unused_194_; 
v_unused_194_ = lean_ctor_get(v_toMonoidWithZero_151_, 1);
lean_dec(v_unused_194_);
v___x_162_ = v_toMonoidWithZero_151_;
v_isShared_163_ = v_isSharedCheck_193_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_toMonoid_160_);
lean_dec(v_toMonoidWithZero_151_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_193_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v_toInv_164_; lean_object* v_toDiv_165_; lean_object* v_toZPow_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_191_; 
v_toInv_164_ = lean_ctor_get(v___x_159_, 1);
v_toDiv_165_ = lean_ctor_get(v___x_159_, 2);
v_toZPow_166_ = lean_ctor_get(v___x_159_, 3);
v_isSharedCheck_191_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_191_ == 0)
{
lean_object* v_unused_192_; 
v_unused_192_ = lean_ctor_get(v___x_159_, 0);
lean_dec(v_unused_192_);
v___x_168_ = v___x_159_;
v_isShared_169_ = v_isSharedCheck_191_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_toZPow_166_);
lean_inc(v_toDiv_165_);
lean_inc(v_toInv_164_);
lean_dec(v___x_159_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_191_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v_toNPow_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_188_; 
v_toNPow_170_ = lean_ctor_get(v_toMonoid_160_, 2);
v_isSharedCheck_188_ = !lean_is_exclusive(v_toMonoid_160_);
if (v_isSharedCheck_188_ == 0)
{
lean_object* v_unused_189_; lean_object* v_unused_190_; 
v_unused_189_ = lean_ctor_get(v_toMonoid_160_, 1);
lean_dec(v_unused_189_);
v_unused_190_ = lean_ctor_get(v_toMonoid_160_, 0);
lean_dec(v_unused_190_);
v___x_172_ = v_toMonoid_160_;
v_isShared_173_ = v_isSharedCheck_188_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_toNPow_170_);
lean_dec(v_toMonoid_160_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_188_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___f_174_; lean_object* v___f_175_; lean_object* v___f_176_; lean_object* v___f_177_; lean_object* v___f_178_; lean_object* v___x_180_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_174_, 0, v_toMul_154_);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_ULift_groupWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_175_, 0, v_toZPow_166_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_ULift_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_176_, 0, v_toInv_164_);
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_ULift_mul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_177_, 0, v_toDiv_165_);
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_ULift_monoidWithZero___redArg___lam__0), 3, 1);
lean_closure_set(v___f_178_, 0, v_toNPow_170_);
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 2, v___f_178_);
lean_ctor_set(v___x_172_, 1, v___f_174_);
lean_ctor_set(v___x_172_, 0, v_toOne_158_);
v___x_180_ = v___x_172_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_toOne_158_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v___f_174_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v___f_178_);
v___x_180_ = v_reuseFailAlloc_187_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
lean_object* v___x_182_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 1, v_toZero_155_);
lean_ctor_set(v___x_162_, 0, v___x_180_);
v___x_182_ = v___x_162_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_toZero_155_);
v___x_182_ = v_reuseFailAlloc_186_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
lean_object* v___x_184_; 
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 3, v___f_175_);
lean_ctor_set(v___x_168_, 2, v___f_177_);
lean_ctor_set(v___x_168_, 1, v___f_176_);
lean_ctor_set(v___x_168_, 0, v___x_182_);
v___x_184_ = v___x_168_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_182_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v___f_176_);
lean_ctor_set(v_reuseFailAlloc_185_, 2, v___f_177_);
lean_ctor_set(v_reuseFailAlloc_185_, 3, v___f_175_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_commGroupWithZero(lean_object* v_00_u03b1_195_, lean_object* v_inst_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_ULift_commGroupWithZero___redArg(v_inst_196_);
return v___x_197_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
