// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.Algebra.GroupWithZero.WithZero
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
lean_object* lp_mathlib_Prod_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_divMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_mulMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMulOneClass___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidWithZeroHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_divMonoidWithZeroHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_divMonoidWithZeroHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroClass___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v_toMul_3_; lean_object* v_toZero_4_; lean_object* v___x_6_; uint8_t v_isShared_7_; uint8_t v_isSharedCheck_21_; 
v_toMul_3_ = lean_ctor_get(v_inst_1_, 0);
v_toZero_4_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_21_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_21_ == 0)
{
v___x_6_ = v_inst_1_;
v_isShared_7_ = v_isSharedCheck_21_;
goto v_resetjp_5_;
}
else
{
lean_inc(v_toZero_4_);
lean_inc(v_toMul_3_);
lean_dec(v_inst_1_);
v___x_6_ = lean_box(0);
v_isShared_7_ = v_isSharedCheck_21_;
goto v_resetjp_5_;
}
v_resetjp_5_:
{
lean_object* v_toMul_8_; lean_object* v_toZero_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_20_; 
v_toMul_8_ = lean_ctor_get(v_inst_2_, 0);
v_toZero_9_ = lean_ctor_get(v_inst_2_, 1);
v_isSharedCheck_20_ = !lean_is_exclusive(v_inst_2_);
if (v_isSharedCheck_20_ == 0)
{
v___x_11_ = v_inst_2_;
v_isShared_12_ = v_isSharedCheck_20_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_toZero_9_);
lean_inc(v_toMul_8_);
lean_dec(v_inst_2_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_20_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___f_13_; lean_object* v___x_15_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_13_, 0, v_toMul_3_);
lean_closure_set(v___f_13_, 1, v_toMul_8_);
if (v_isShared_7_ == 0)
{
lean_ctor_set(v___x_6_, 1, v_toZero_9_);
lean_ctor_set(v___x_6_, 0, v_toZero_4_);
v___x_15_ = v___x_6_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v_toZero_4_);
lean_ctor_set(v_reuseFailAlloc_19_, 1, v_toZero_9_);
v___x_15_ = v_reuseFailAlloc_19_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
lean_object* v___x_17_; 
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 1, v___x_15_);
lean_ctor_set(v___x_11_, 0, v___f_13_);
v___x_17_ = v___x_11_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___f_13_);
lean_ctor_set(v_reuseFailAlloc_18_, 1, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroClass(lean_object* v_M_u2080_22_, lean_object* v_N_u2080_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Prod_instMulZeroClass___redArg(v_inst_24_, v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroupWithZero___redArg(lean_object* v_inst_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v_toSemigroup_29_; lean_object* v_toSemigroup_30_; lean_object* v___x_31_; lean_object* v_toZero_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_50_; 
v_toSemigroup_29_ = lean_ctor_get(v_inst_27_, 0);
lean_inc(v_toSemigroup_29_);
v_toSemigroup_30_ = lean_ctor_get(v_inst_28_, 0);
lean_inc(v_toSemigroup_30_);
v___x_31_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_27_);
v_toZero_32_ = lean_ctor_get(v___x_31_, 1);
v_isSharedCheck_50_ = !lean_is_exclusive(v___x_31_);
if (v_isSharedCheck_50_ == 0)
{
lean_object* v_unused_51_; 
v_unused_51_ = lean_ctor_get(v___x_31_, 0);
lean_dec(v_unused_51_);
v___x_34_ = v___x_31_;
v_isShared_35_ = v_isSharedCheck_50_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_toZero_32_);
lean_dec(v___x_31_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_50_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___x_36_; lean_object* v_toZero_37_; lean_object* v___x_39_; uint8_t v_isShared_40_; uint8_t v_isSharedCheck_48_; 
v___x_36_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_28_);
v_toZero_37_ = lean_ctor_get(v___x_36_, 1);
v_isSharedCheck_48_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_48_ == 0)
{
lean_object* v_unused_49_; 
v_unused_49_ = lean_ctor_get(v___x_36_, 0);
lean_dec(v_unused_49_);
v___x_39_ = v___x_36_;
v_isShared_40_ = v_isSharedCheck_48_;
goto v_resetjp_38_;
}
else
{
lean_inc(v_toZero_37_);
lean_dec(v___x_36_);
v___x_39_ = lean_box(0);
v_isShared_40_ = v_isSharedCheck_48_;
goto v_resetjp_38_;
}
v_resetjp_38_:
{
lean_object* v___f_41_; lean_object* v___x_43_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_41_, 0, v_toSemigroup_29_);
lean_closure_set(v___f_41_, 1, v_toSemigroup_30_);
if (v_isShared_40_ == 0)
{
lean_ctor_set(v___x_39_, 0, v_toZero_32_);
v___x_43_ = v___x_39_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_47_; 
v_reuseFailAlloc_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_47_, 0, v_toZero_32_);
lean_ctor_set(v_reuseFailAlloc_47_, 1, v_toZero_37_);
v___x_43_ = v_reuseFailAlloc_47_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
lean_object* v___x_45_; 
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 1, v___x_43_);
lean_ctor_set(v___x_34_, 0, v___f_41_);
v___x_45_ = v___x_34_;
goto v_reusejp_44_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___f_41_);
lean_ctor_set(v_reuseFailAlloc_46_, 1, v___x_43_);
v___x_45_ = v_reuseFailAlloc_46_;
goto v_reusejp_44_;
}
v_reusejp_44_:
{
return v___x_45_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroupWithZero(lean_object* v_M_u2080_52_, lean_object* v_N_u2080_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Prod_instSemigroupWithZero___redArg(v_inst_54_, v_inst_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroOneClass___redArg(lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_toMulOneClass_59_; lean_object* v_toMulOneClass_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v_toZero_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_80_; 
v_toMulOneClass_59_ = lean_ctor_get(v_inst_57_, 0);
v_toMulOneClass_60_ = lean_ctor_get(v_inst_58_, 0);
lean_inc_ref(v_toMulOneClass_60_);
lean_inc_ref(v_toMulOneClass_59_);
v___x_61_ = lp_mathlib_Prod_instMulOneClass___redArg(v_toMulOneClass_59_, v_toMulOneClass_60_);
v___x_62_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_57_);
v_toZero_63_ = lean_ctor_get(v___x_62_, 1);
v_isSharedCheck_80_ = !lean_is_exclusive(v___x_62_);
if (v_isSharedCheck_80_ == 0)
{
lean_object* v_unused_81_; 
v_unused_81_ = lean_ctor_get(v___x_62_, 0);
lean_dec(v_unused_81_);
v___x_65_ = v___x_62_;
v_isShared_66_ = v_isSharedCheck_80_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_toZero_63_);
lean_dec(v___x_62_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_80_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_67_; lean_object* v_toZero_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_78_; 
v___x_67_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_58_);
v_toZero_68_ = lean_ctor_get(v___x_67_, 1);
v_isSharedCheck_78_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_78_ == 0)
{
lean_object* v_unused_79_; 
v_unused_79_ = lean_ctor_get(v___x_67_, 0);
lean_dec(v_unused_79_);
v___x_70_ = v___x_67_;
v_isShared_71_ = v_isSharedCheck_78_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_toZero_68_);
lean_dec(v___x_67_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_78_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
lean_object* v___x_73_; 
if (v_isShared_71_ == 0)
{
lean_ctor_set(v___x_70_, 0, v_toZero_63_);
v___x_73_ = v___x_70_;
goto v_reusejp_72_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v_toZero_63_);
lean_ctor_set(v_reuseFailAlloc_77_, 1, v_toZero_68_);
v___x_73_ = v_reuseFailAlloc_77_;
goto v_reusejp_72_;
}
v_reusejp_72_:
{
lean_object* v___x_75_; 
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 1, v___x_73_);
lean_ctor_set(v___x_65_, 0, v___x_61_);
v___x_75_ = v___x_65_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v___x_61_);
lean_ctor_set(v_reuseFailAlloc_76_, 1, v___x_73_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulZeroOneClass(lean_object* v_M_u2080_82_, lean_object* v_N_u2080_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Prod_instMulZeroOneClass___redArg(v_inst_84_, v_inst_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoidWithZero___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v_toMonoid_89_; lean_object* v_toMonoid_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v_toZero_94_; lean_object* v___x_96_; uint8_t v_isShared_97_; uint8_t v_isSharedCheck_112_; 
v_toMonoid_89_ = lean_ctor_get(v_inst_87_, 0);
v_toMonoid_90_ = lean_ctor_get(v_inst_88_, 0);
lean_inc_ref(v_toMonoid_90_);
lean_inc_ref(v_toMonoid_89_);
v___x_91_ = lp_mathlib_Prod_instMonoid___redArg(v_toMonoid_89_, v_toMonoid_90_);
v___x_92_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_87_);
v___x_93_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_92_);
v_toZero_94_ = lean_ctor_get(v___x_93_, 1);
v_isSharedCheck_112_ = !lean_is_exclusive(v___x_93_);
if (v_isSharedCheck_112_ == 0)
{
lean_object* v_unused_113_; 
v_unused_113_ = lean_ctor_get(v___x_93_, 0);
lean_dec(v_unused_113_);
v___x_96_ = v___x_93_;
v_isShared_97_ = v_isSharedCheck_112_;
goto v_resetjp_95_;
}
else
{
lean_inc(v_toZero_94_);
lean_dec(v___x_93_);
v___x_96_ = lean_box(0);
v_isShared_97_ = v_isSharedCheck_112_;
goto v_resetjp_95_;
}
v_resetjp_95_:
{
lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v_toZero_100_; lean_object* v___x_102_; uint8_t v_isShared_103_; uint8_t v_isSharedCheck_110_; 
v___x_98_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_88_);
v___x_99_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_98_);
v_toZero_100_ = lean_ctor_get(v___x_99_, 1);
v_isSharedCheck_110_ = !lean_is_exclusive(v___x_99_);
if (v_isSharedCheck_110_ == 0)
{
lean_object* v_unused_111_; 
v_unused_111_ = lean_ctor_get(v___x_99_, 0);
lean_dec(v_unused_111_);
v___x_102_ = v___x_99_;
v_isShared_103_ = v_isSharedCheck_110_;
goto v_resetjp_101_;
}
else
{
lean_inc(v_toZero_100_);
lean_dec(v___x_99_);
v___x_102_ = lean_box(0);
v_isShared_103_ = v_isSharedCheck_110_;
goto v_resetjp_101_;
}
v_resetjp_101_:
{
lean_object* v___x_105_; 
if (v_isShared_103_ == 0)
{
lean_ctor_set(v___x_102_, 0, v_toZero_94_);
v___x_105_ = v___x_102_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v_toZero_94_);
lean_ctor_set(v_reuseFailAlloc_109_, 1, v_toZero_100_);
v___x_105_ = v_reuseFailAlloc_109_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
lean_object* v___x_107_; 
if (v_isShared_97_ == 0)
{
lean_ctor_set(v___x_96_, 1, v___x_105_);
lean_ctor_set(v___x_96_, 0, v___x_91_);
v___x_107_ = v___x_96_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v___x_91_);
lean_ctor_set(v_reuseFailAlloc_108_, 1, v___x_105_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoidWithZero(lean_object* v_M_u2080_114_, lean_object* v_N_u2080_115_, lean_object* v_inst_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_Prod_instMonoidWithZero___redArg(v_inst_116_, v_inst_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoidWithZero___redArg(lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_toCommMonoid_121_; lean_object* v_toCommMonoid_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v_toZero_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_146_; 
v_toCommMonoid_121_ = lean_ctor_get(v_inst_119_, 0);
v_toCommMonoid_122_ = lean_ctor_get(v_inst_120_, 0);
lean_inc_ref(v_toCommMonoid_122_);
lean_inc_ref(v_toCommMonoid_121_);
v___x_123_ = lp_mathlib_Prod_instMonoid___redArg(v_toCommMonoid_121_, v_toCommMonoid_122_);
v___x_124_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_119_);
v___x_125_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_124_);
v___x_126_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_125_);
v_toZero_127_ = lean_ctor_get(v___x_126_, 1);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_146_ == 0)
{
lean_object* v_unused_147_; 
v_unused_147_ = lean_ctor_get(v___x_126_, 0);
lean_dec(v_unused_147_);
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_146_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_toZero_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_146_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v_toZero_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_144_; 
v___x_131_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_120_);
v___x_132_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_131_);
v___x_133_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_132_);
v_toZero_134_ = lean_ctor_get(v___x_133_, 1);
v_isSharedCheck_144_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_144_ == 0)
{
lean_object* v_unused_145_; 
v_unused_145_ = lean_ctor_get(v___x_133_, 0);
lean_dec(v_unused_145_);
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_144_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_toZero_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_144_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_139_; 
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 0, v_toZero_127_);
v___x_139_ = v___x_136_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_toZero_127_);
lean_ctor_set(v_reuseFailAlloc_143_, 1, v_toZero_134_);
v___x_139_ = v_reuseFailAlloc_143_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
lean_object* v___x_141_; 
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 1, v___x_139_);
lean_ctor_set(v___x_129_, 0, v___x_123_);
v___x_141_ = v___x_129_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_123_);
lean_ctor_set(v_reuseFailAlloc_142_, 1, v___x_139_);
v___x_141_ = v_reuseFailAlloc_142_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
return v___x_141_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoidWithZero(lean_object* v_M_u2080_148_, lean_object* v_N_u2080_149_, lean_object* v_inst_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_Prod_instCommMonoidWithZero___redArg(v_inst_150_, v_inst_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidWithZeroHom___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v_toCommMonoid_154_; lean_object* v___x_155_; 
v_toCommMonoid_154_ = lean_ctor_get(v_inst_153_, 0);
lean_inc_ref(v_toCommMonoid_154_);
lean_dec_ref(v_inst_153_);
v___x_155_ = lp_mathlib_mulMonoidHom___redArg(v_toCommMonoid_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidWithZeroHom(lean_object* v_M_u2080_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_mulMonoidWithZeroHom___redArg(v_inst_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_divMonoidWithZeroHom___redArg(lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v_inst_159_);
v___x_161_ = lp_mathlib_divMonoidHom___redArg(v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_divMonoidWithZeroHom(lean_object* v_M_u2080_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_divMonoidWithZeroHom___redArg(v_inst_163_);
return v___x_164_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
