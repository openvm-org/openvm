// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Algebra.Group.Pi.Basic
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
lean_object* lp_mathlib_Pi_mulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instOne___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_monoid___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Pi_semigroup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toMul_6_; lean_object* v___x_7_; 
v___x_5_ = lean_apply_1(v_inst_1_, v_i_2_);
v_toMul_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toMul_6_);
lean_dec_ref(v___x_5_);
v___x_7_ = lean_apply_2(v_toMul_6_, v___y_3_, v___y_4_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg___lam__1(lean_object* v_inst_8_, lean_object* v_i_9_){
_start:
{
lean_object* v___x_10_; lean_object* v_toZero_11_; 
v___x_10_ = lean_apply_1(v_inst_8_, v_i_9_);
v_toZero_11_ = lean_ctor_get(v___x_10_, 1);
lean_inc(v_toZero_11_);
lean_dec_ref(v___x_10_);
return v_toZero_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass___redArg(lean_object* v_inst_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___f_14_; lean_object* v___f_15_; lean_object* v___f_16_; lean_object* v___x_17_; 
lean_inc_ref(v_inst_12_);
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_13_, 0, v_inst_12_);
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_14_, 0, v_inst_12_);
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_15_, 0, v___f_13_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_16_, 0, v___f_14_);
v___x_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_17_, 0, v___f_15_);
lean_ctor_set(v___x_17_, 1, v___f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroClass(lean_object* v_00_u03b9_18_, lean_object* v_00_u03b1_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Pi_mulZeroClass___redArg(v_inst_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single___redArg(lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_i_24_){
_start:
{
lean_object* v___f_25_; lean_object* v___x_26_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_25_, 0, v_inst_22_);
v___x_26_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 5);
lean_closure_set(v___x_26_, 0, lean_box(0));
lean_closure_set(v___x_26_, 1, lean_box(0));
lean_closure_set(v___x_26_, 2, v___f_25_);
lean_closure_set(v___x_26_, 3, v_inst_23_);
lean_closure_set(v___x_26_, 4, v_i_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_single(lean_object* v_00_u03b9_27_, lean_object* v_00_u03b1_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_i_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_MulHom_single___redArg(v_inst_29_, v_inst_30_, v_i_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0(lean_object* v_inst_33_, lean_object* v_i_34_){
_start:
{
lean_object* v___x_35_; lean_object* v_toMulOneClass_36_; 
v___x_35_ = lean_apply_1(v_inst_33_, v_i_34_);
v_toMulOneClass_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc_ref(v_toMulOneClass_36_);
lean_dec_ref(v___x_35_);
return v_toMulOneClass_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1(lean_object* v_inst_37_, lean_object* v_i_38_){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = lean_apply_1(v_inst_37_, v_i_38_);
v___x_40_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; lean_object* v___x_43_; lean_object* v_toOne_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_57_; 
lean_inc_ref(v_inst_41_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
v___x_43_ = lp_mathlib_Pi_mulOneClass___redArg(v___f_42_);
v_toOne_44_ = lean_ctor_get(v___x_43_, 0);
v_isSharedCheck_57_ = !lean_is_exclusive(v___x_43_);
if (v_isSharedCheck_57_ == 0)
{
lean_object* v_unused_58_; 
v_unused_58_ = lean_ctor_get(v___x_43_, 1);
lean_dec(v_unused_58_);
v___x_46_ = v___x_43_;
v_isShared_47_ = v_isSharedCheck_57_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_toOne_44_);
lean_dec(v___x_43_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_57_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___f_50_; lean_object* v___f_51_; lean_object* v___f_52_; lean_object* v___x_54_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroOneClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_48_, 0, v_inst_41_);
lean_inc_ref(v___f_48_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_49_, 0, v___f_48_);
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_50_, 0, v___f_49_);
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_51_, 0, v___f_48_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_52_, 0, v___f_51_);
if (v_isShared_47_ == 0)
{
lean_ctor_set(v___x_46_, 1, v___f_50_);
v___x_54_ = v___x_46_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v_toOne_44_);
lean_ctor_set(v_reuseFailAlloc_56_, 1, v___f_50_);
v___x_54_ = v_reuseFailAlloc_56_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
lean_object* v___x_55_; 
v___x_55_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
lean_ctor_set(v___x_55_, 1, v___f_52_);
return v___x_55_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulZeroOneClass(lean_object* v_00_u03b9_59_, lean_object* v_00_u03b1_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_Pi_mulZeroOneClass___redArg(v_inst_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__0(lean_object* v_inst_63_, lean_object* v_i_64_){
_start:
{
lean_object* v___x_65_; lean_object* v_toMonoid_66_; 
v___x_65_ = lean_apply_1(v_inst_63_, v_i_64_);
v_toMonoid_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc_ref(v_toMonoid_66_);
lean_dec_ref(v___x_65_);
return v_toMonoid_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg___lam__1(lean_object* v_inst_67_, lean_object* v_i_68_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_69_ = lean_apply_1(v_inst_67_, v_i_68_);
v___x_70_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_69_);
v___x_71_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero___redArg(lean_object* v_inst_72_){
_start:
{
lean_object* v___f_73_; lean_object* v___f_74_; lean_object* v___x_75_; lean_object* v___f_76_; lean_object* v___f_77_; lean_object* v___x_78_; 
lean_inc_ref(v_inst_72_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__0), 2, 1);
lean_closure_set(v___f_73_, 0, v_inst_72_);
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__1), 2, 1);
lean_closure_set(v___f_74_, 0, v_inst_72_);
v___x_75_ = lp_mathlib_Pi_monoid___redArg(v___f_73_);
v___f_76_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_76_, 0, v___f_74_);
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_77_, 0, v___f_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_75_);
lean_ctor_set(v___x_78_, 1, v___f_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoidWithZero(lean_object* v_00_u03b9_79_, lean_object* v_00_u03b1_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Pi_monoidWithZero___redArg(v_inst_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0(lean_object* v_inst_83_, lean_object* v_i_84_){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = lean_apply_1(v_inst_83_, v_i_84_);
v___x_86_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero___redArg(lean_object* v_inst_87_){
_start:
{
lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v_toMonoid_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_100_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoidWithZero___redArg___lam__0), 2, 1);
lean_closure_set(v___f_88_, 0, v_inst_87_);
lean_inc_ref(v___f_88_);
v___x_89_ = lp_mathlib_Pi_monoidWithZero___redArg(v___f_88_);
v_toMonoid_90_ = lean_ctor_get(v___x_89_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v___x_89_);
if (v_isSharedCheck_100_ == 0)
{
lean_object* v_unused_101_; 
v_unused_101_ = lean_ctor_get(v___x_89_, 1);
lean_dec(v_unused_101_);
v___x_92_ = v___x_89_;
v_isShared_93_ = v_isSharedCheck_100_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_toMonoid_90_);
lean_dec(v___x_89_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_100_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___f_94_; lean_object* v___f_95_; lean_object* v___f_96_; lean_object* v___x_98_; 
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoidWithZero___redArg___lam__1), 2, 1);
lean_closure_set(v___f_94_, 0, v___f_88_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_95_, 0, v___f_94_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_96_, 0, v___f_95_);
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 1, v___f_96_);
v___x_98_ = v___x_92_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_toMonoid_90_);
lean_ctor_set(v_reuseFailAlloc_99_, 1, v___f_96_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoidWithZero(lean_object* v_00_u03b9_102_, lean_object* v_00_u03b1_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_Pi_commMonoidWithZero___redArg(v_inst_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__0(lean_object* v_inst_106_, lean_object* v_i_107_, lean_object* v___y_108_, lean_object* v___y_109_){
_start:
{
lean_object* v___x_110_; lean_object* v_toSemigroup_111_; lean_object* v___x_112_; 
v___x_110_ = lean_apply_1(v_inst_106_, v_i_107_);
v_toSemigroup_111_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_toSemigroup_111_);
lean_dec_ref(v___x_110_);
v___x_112_ = lean_apply_2(v_toSemigroup_111_, v___y_108_, v___y_109_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg___lam__1(lean_object* v_inst_113_, lean_object* v_i_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = lean_apply_1(v_inst_113_, v_i_114_);
v___x_116_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v___x_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero___redArg(lean_object* v_inst_117_){
_start:
{
lean_object* v___f_118_; lean_object* v___f_119_; lean_object* v___x_120_; lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___x_123_; 
lean_inc_ref(v_inst_117_);
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_Pi_semigroupWithZero___redArg___lam__0), 4, 1);
lean_closure_set(v___f_118_, 0, v_inst_117_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_Pi_semigroupWithZero___redArg___lam__1), 2, 1);
lean_closure_set(v___f_119_, 0, v_inst_117_);
v___x_120_ = lp_mathlib_Pi_semigroup___redArg(v___f_118_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulZeroClass___redArg___lam__1), 2, 1);
lean_closure_set(v___f_121_, 0, v___f_119_);
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_122_, 0, v___f_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_120_);
lean_ctor_set(v___x_123_, 1, v___f_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroupWithZero(lean_object* v_00_u03b9_124_, lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_Pi_semigroupWithZero___redArg(v_inst_126_);
return v___x_127_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
