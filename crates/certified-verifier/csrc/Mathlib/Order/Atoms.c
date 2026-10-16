// Lean compiler output
// Module: Mathlib.Order.Atoms
// Imports: public import Init public meta import Init public import Mathlib.Order.ConditionallyCompletePartialOrder.Indexed public import Mathlib.Order.ModularLattice public import Mathlib.Order.SuccPred.Basic public import Mathlib.Tactic.Nontriviality.Core
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
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_lattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_lattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc_ref(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___redArg(v_inst_2_);
lean_dec_ref(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_inst_6_){
_start:
{
lean_inc_ref(v_inst_5_);
return v_inst_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra___boxed(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_CompleteBooleanAlgebra_toCompleteAtomicBooleanAlgebra(v_00_u03b1_7_, v_inst_8_, v_inst_9_);
lean_dec_ref(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg___lam__0(lean_object* v_toSupSet_11_, lean_object* v_S_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_apply_1(v_toSupSet_11_, lean_box(0));
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v_toCompleteLattice_15_; lean_object* v___x_16_; lean_object* v_toSupSet_17_; lean_object* v___f_18_; lean_object* v___x_19_; 
v_toCompleteLattice_15_ = lean_ctor_get(v_inst_14_, 0);
lean_inc_ref(v_toCompleteLattice_15_);
lean_dec_ref(v_inst_14_);
v___x_16_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v_toCompleteLattice_15_);
v_toSupSet_17_ = lean_ctor_get(v___x_16_, 1);
lean_inc(v_toSupSet_17_);
lean_dec_ref(v___x_16_);
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_18_, 0, v_toSupSet_17_);
v___x_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_19_, 0, lean_box(0));
lean_ctor_set(v___x_19_, 1, v___f_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_CompleteAtomicBooleanAlgebra_toSetOfIsAtom___redArg(v_inst_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_box(0);
v___x_25_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_25_, 0, v_inst_23_);
lean_ctor_set(v___x_25_, 1, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_IsSimpleOrder_preorder___redArg(v_inst_27_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_preorder___boxed(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_IsSimpleOrder_preorder(v_00_u03b1_31_, v_inst_32_, v_inst_33_, v_inst_34_);
lean_dec_ref(v_inst_33_);
return v_res_35_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0(lean_object* v_inst_36_, lean_object* v_toOrderBot_37_, lean_object* v_toOrderTop_38_, lean_object* v_a_39_, lean_object* v_b_40_){
_start:
{
lean_object* v___x_41_; uint8_t v___x_42_; 
lean_inc_ref(v_inst_36_);
v___x_41_ = lean_apply_2(v_inst_36_, v_a_39_, v_toOrderBot_37_);
v___x_42_ = lean_unbox(v___x_41_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_43_ = lean_apply_2(v_inst_36_, v_b_40_, v_toOrderTop_38_);
v___x_44_ = lean_unbox(v___x_43_);
return v___x_44_;
}
else
{
uint8_t v___x_45_; 
lean_dec(v_b_40_);
lean_dec(v_toOrderTop_38_);
lean_dec_ref(v_inst_36_);
v___x_45_ = lean_unbox(v___x_41_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_46_, lean_object* v_toOrderBot_47_, lean_object* v_toOrderTop_48_, lean_object* v_a_49_, lean_object* v_b_50_){
_start:
{
uint8_t v_res_51_; lean_object* v_r_52_; 
v_res_51_ = lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0(v_inst_46_, v_toOrderBot_47_, v_toOrderTop_48_, v_a_49_, v_b_50_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2(lean_object* v___f_53_, lean_object* v_inst_54_, lean_object* v_a_55_, lean_object* v_b_56_){
_start:
{
uint8_t v___x_57_; 
lean_inc(v_b_56_);
lean_inc(v_a_55_);
v___x_57_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_53_, v_a_55_, v_b_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; uint8_t v___x_59_; 
v___x_58_ = lean_apply_2(v_inst_54_, v_a_55_, v_b_56_);
v___x_59_ = lean_unbox(v___x_58_);
if (v___x_59_ == 0)
{
uint8_t v___x_60_; 
v___x_60_ = 2;
return v___x_60_;
}
else
{
uint8_t v___x_61_; 
v___x_61_ = 1;
return v___x_61_;
}
}
else
{
uint8_t v___x_62_; 
lean_dec(v_b_56_);
lean_dec(v_a_55_);
lean_dec_ref(v_inst_54_);
v___x_62_ = 0;
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2___boxed(lean_object* v___f_63_, lean_object* v_inst_64_, lean_object* v_a_65_, lean_object* v_b_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2(v___f_63_, v_inst_64_, v_a_65_, v_b_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__1(lean_object* v_inst_69_, lean_object* v_toOrderBot_70_, lean_object* v_toOrderTop_71_, lean_object* v_a_72_, lean_object* v_b_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
lean_inc_ref(v_inst_69_);
lean_inc(v_a_72_);
v___x_74_ = lean_apply_2(v_inst_69_, v_a_72_, v_toOrderBot_70_);
v___x_75_ = lean_unbox(v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; uint8_t v___x_77_; 
lean_inc(v_b_73_);
v___x_76_ = lean_apply_2(v_inst_69_, v_b_73_, v_toOrderTop_71_);
v___x_77_ = lean_unbox(v___x_76_);
if (v___x_77_ == 0)
{
lean_dec(v_b_73_);
return v_a_72_;
}
else
{
lean_dec(v_a_72_);
return v_b_73_;
}
}
else
{
lean_dec(v_a_72_);
lean_dec(v_toOrderTop_71_);
lean_dec_ref(v_inst_69_);
return v_b_73_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__3(lean_object* v_inst_78_, lean_object* v_toOrderBot_79_, lean_object* v_toOrderTop_80_, lean_object* v_a_81_, lean_object* v_b_82_){
_start:
{
lean_object* v___x_83_; uint8_t v___x_84_; 
lean_inc_ref(v_inst_78_);
lean_inc(v_a_81_);
v___x_83_ = lean_apply_2(v_inst_78_, v_a_81_, v_toOrderBot_79_);
v___x_84_ = lean_unbox(v___x_83_);
if (v___x_84_ == 0)
{
lean_object* v___x_85_; uint8_t v___x_86_; 
lean_inc(v_b_82_);
v___x_85_ = lean_apply_2(v_inst_78_, v_b_82_, v_toOrderTop_80_);
v___x_86_ = lean_unbox(v___x_85_);
if (v___x_86_ == 0)
{
lean_dec(v_a_81_);
return v_b_82_;
}
else
{
lean_dec(v_b_82_);
return v_a_81_;
}
}
else
{
lean_dec(v_b_82_);
lean_dec(v_toOrderTop_80_);
lean_dec_ref(v_inst_78_);
return v_a_81_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_){
_start:
{
lean_object* v_toOrderTop_90_; lean_object* v_toOrderBot_91_; lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___f_94_; lean_object* v___f_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v_toOrderTop_90_ = lean_ctor_get(v_inst_88_, 0);
lean_inc_n(v_toOrderTop_90_, 3);
v_toOrderBot_91_ = lean_ctor_get(v_inst_88_, 1);
lean_inc_n(v_toOrderBot_91_, 3);
lean_dec_ref(v_inst_88_);
lean_inc_ref_n(v_inst_89_, 4);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_92_, 0, v_inst_89_);
lean_closure_set(v___f_92_, 1, v_toOrderBot_91_);
lean_closure_set(v___f_92_, 2, v_toOrderTop_90_);
lean_inc_ref_n(v___f_92_, 2);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_93_, 0, v___f_92_);
lean_closure_set(v___f_93_, 1, v_inst_89_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__1), 5, 3);
lean_closure_set(v___f_94_, 0, v_inst_89_);
lean_closure_set(v___f_94_, 1, v_toOrderBot_91_);
lean_closure_set(v___f_94_, 2, v_toOrderTop_90_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_linearOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_95_, 0, v_inst_89_);
lean_closure_set(v___f_95_, 1, v_toOrderBot_91_);
lean_closure_set(v___f_95_, 2, v_toOrderTop_90_);
lean_inc_ref(v_inst_87_);
v___x_96_ = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, v_inst_87_);
lean_closure_set(v___x_96_, 2, v___f_92_);
v___x_97_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_97_, 0, v_inst_87_);
lean_ctor_set(v___x_97_, 1, v___f_95_);
lean_ctor_set(v___x_97_, 2, v___f_94_);
lean_ctor_set(v___x_97_, 3, v___f_93_);
lean_ctor_set(v___x_97_, 4, v___f_92_);
lean_ctor_set(v___x_97_, 5, v_inst_89_);
lean_ctor_set(v___x_97_, 6, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_linearOrder(lean_object* v_00_u03b1_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_IsSimpleOrder_linearOrder___redArg(v_inst_99_, v_inst_100_, v_inst_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_lattice___redArg(lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = lp_mathlib_IsSimpleOrder_linearOrder___redArg(v_inst_105_, v_inst_106_, v_inst_104_);
v___x_108_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_107_);
lean_dec_ref(v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_lattice(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_IsSimpleOrder_lattice___redArg(v_inst_110_, v_inst_111_, v_inst_112_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___redArg(lean_object* v_inst_115_){
_start:
{
lean_inc_ref(v_inst_115_);
return v_inst_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___redArg___boxed(lean_object* v_inst_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_IsSimpleOrder_distribLattice___redArg(v_inst_116_);
lean_dec_ref(v_inst_116_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice(lean_object* v_00_u03b1_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_inc_ref(v_inst_119_);
return v_inst_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_distribLattice___boxed(lean_object* v_00_u03b1_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_IsSimpleOrder_distribLattice(v_00_u03b1_122_, v_inst_123_, v_inst_124_, v_inst_125_);
lean_dec_ref(v_inst_124_);
lean_dec_ref(v_inst_123_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0(lean_object* v_toOrderBot_127_, lean_object* v_toOrderTop_128_, uint8_t v_x_129_){
_start:
{
if (v_x_129_ == 0)
{
lean_inc(v_toOrderBot_127_);
return v_toOrderBot_127_;
}
else
{
lean_inc(v_toOrderTop_128_);
return v_toOrderTop_128_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0___boxed(lean_object* v_toOrderBot_130_, lean_object* v_toOrderTop_131_, lean_object* v_x_132_){
_start:
{
uint8_t v_x_boxed_133_; lean_object* v_res_134_; 
v_x_boxed_133_ = lean_unbox(v_x_132_);
v_res_134_ = lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0(v_toOrderBot_130_, v_toOrderTop_131_, v_x_boxed_133_);
lean_dec(v_toOrderTop_131_);
lean_dec(v_toOrderBot_130_);
return v_res_134_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1(lean_object* v_inst_135_, lean_object* v_toOrderTop_136_, lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; uint8_t v___x_139_; 
v___x_138_ = lean_apply_2(v_inst_135_, v_x_137_, v_toOrderTop_136_);
v___x_139_ = lean_unbox(v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1___boxed(lean_object* v_inst_140_, lean_object* v_toOrderTop_141_, lean_object* v_x_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1(v_inst_140_, v_toOrderTop_141_, v_x_142_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool___redArg(lean_object* v_inst_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v_toOrderTop_147_; lean_object* v_toOrderBot_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_157_; 
v_toOrderTop_147_ = lean_ctor_get(v_inst_146_, 0);
v_toOrderBot_148_ = lean_ctor_get(v_inst_146_, 1);
v_isSharedCheck_157_ = !lean_is_exclusive(v_inst_146_);
if (v_isSharedCheck_157_ == 0)
{
v___x_150_ = v_inst_146_;
v_isShared_151_ = v_isSharedCheck_157_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_toOrderBot_148_);
lean_inc(v_toOrderTop_147_);
lean_dec(v_inst_146_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_157_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___f_152_; lean_object* v___f_153_; lean_object* v___x_155_; 
lean_inc(v_toOrderTop_147_);
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_152_, 0, v_toOrderBot_148_);
lean_closure_set(v___f_152_, 1, v_toOrderTop_147_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_equivBool___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_153_, 0, v_inst_145_);
lean_closure_set(v___f_153_, 1, v_toOrderTop_147_);
if (v_isShared_151_ == 0)
{
lean_ctor_set(v___x_150_, 1, v___f_152_);
lean_ctor_set(v___x_150_, 0, v___f_153_);
v___x_155_ = v___x_150_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___f_153_);
lean_ctor_set(v_reuseFailAlloc_156_, 1, v___f_152_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_equivBool(lean_object* v_00_u03b1_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lp_mathlib_IsSimpleOrder_equivBool___redArg(v_inst_159_, v_inst_161_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool___redArg(lean_object* v_inst_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_IsSimpleOrder_equivBool___redArg(v_inst_164_, v_inst_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_IsSimpleOrder_equivBool___redArg(v_inst_168_, v_inst_170_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_orderIsoBool___boxed(lean_object* v_00_u03b1_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_IsSimpleOrder_orderIsoBool(v_00_u03b1_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_inst_177_);
lean_dec_ref(v_inst_175_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0(lean_object* v_inst_179_, lean_object* v_toOrderBot_180_, lean_object* v_toOrderTop_181_, lean_object* v_x_182_){
_start:
{
lean_object* v___x_183_; uint8_t v___x_184_; 
lean_inc(v_toOrderBot_180_);
v___x_183_ = lean_apply_2(v_inst_179_, v_x_182_, v_toOrderBot_180_);
v___x_184_ = lean_unbox(v___x_183_);
if (v___x_184_ == 0)
{
return v_toOrderBot_180_;
}
else
{
lean_dec(v_toOrderBot_180_);
lean_inc(v_toOrderTop_181_);
return v_toOrderTop_181_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0___boxed(lean_object* v_inst_185_, lean_object* v_toOrderBot_186_, lean_object* v_toOrderTop_187_, lean_object* v_x_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0(v_inst_185_, v_toOrderBot_186_, v_toOrderTop_187_, v_x_188_);
lean_dec(v_toOrderTop_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__1(lean_object* v_inst_190_, lean_object* v_toOrderTop_191_, lean_object* v_toOrderBot_192_, lean_object* v_x_193_, lean_object* v_y_194_){
_start:
{
lean_object* v___x_195_; uint8_t v___x_196_; 
lean_inc_ref(v_inst_190_);
lean_inc(v_toOrderTop_191_);
v___x_195_ = lean_apply_2(v_inst_190_, v_x_193_, v_toOrderTop_191_);
v___x_196_ = lean_unbox(v___x_195_);
if (v___x_196_ == 0)
{
lean_dec(v_y_194_);
lean_dec(v_toOrderTop_191_);
lean_dec_ref(v_inst_190_);
return v_toOrderBot_192_;
}
else
{
lean_object* v___x_197_; uint8_t v___x_198_; 
lean_inc(v_toOrderBot_192_);
v___x_197_ = lean_apply_2(v_inst_190_, v_y_194_, v_toOrderBot_192_);
v___x_198_ = lean_unbox(v___x_197_);
if (v___x_198_ == 0)
{
lean_dec(v_toOrderTop_191_);
return v_toOrderBot_192_;
}
else
{
lean_dec(v_toOrderBot_192_);
return v_toOrderTop_191_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__2(lean_object* v_toSemilatticeSup_199_, lean_object* v___f_200_, lean_object* v_x_201_, lean_object* v_y_202_){
_start:
{
lean_object* v_sup_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v_sup_203_ = lean_ctor_get(v_toSemilatticeSup_199_, 1);
lean_inc(v_sup_203_);
lean_dec_ref(v_toSemilatticeSup_199_);
v___x_204_ = lean_apply_1(v___f_200_, v_x_201_);
v___x_205_ = lean_apply_2(v_sup_203_, v_y_202_, v___x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_){
_start:
{
lean_object* v_toOrderTop_209_; lean_object* v_toOrderBot_210_; lean_object* v_toSemilatticeSup_211_; lean_object* v___f_212_; lean_object* v___f_213_; lean_object* v___f_214_; lean_object* v___x_215_; 
v_toOrderTop_209_ = lean_ctor_get(v_inst_208_, 0);
lean_inc_n(v_toOrderTop_209_, 3);
v_toOrderBot_210_ = lean_ctor_get(v_inst_208_, 1);
lean_inc_n(v_toOrderBot_210_, 3);
lean_dec_ref(v_inst_208_);
v_toSemilatticeSup_211_ = lean_ctor_get(v_inst_207_, 0);
lean_inc_ref(v_inst_206_);
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_212_, 0, v_inst_206_);
lean_closure_set(v___f_212_, 1, v_toOrderBot_210_);
lean_closure_set(v___f_212_, 2, v_toOrderTop_209_);
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__1), 5, 3);
lean_closure_set(v___f_213_, 0, v_inst_206_);
lean_closure_set(v___f_213_, 1, v_toOrderTop_209_);
lean_closure_set(v___f_213_, 2, v_toOrderBot_210_);
lean_inc_ref(v___f_212_);
lean_inc_ref(v_toSemilatticeSup_211_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg___lam__2), 4, 2);
lean_closure_set(v___f_214_, 0, v_toSemilatticeSup_211_);
lean_closure_set(v___f_214_, 1, v___f_212_);
v___x_215_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_215_, 0, v_inst_207_);
lean_ctor_set(v___x_215_, 1, v___f_212_);
lean_ctor_set(v___x_215_, 2, v___f_213_);
lean_ctor_set(v___x_215_, 3, v___f_214_);
lean_ctor_set(v___x_215_, 4, v_toOrderTop_209_);
lean_ctor_set(v___x_215_, 5, v_toOrderBot_210_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsSimpleOrder_booleanAlgebra(lean_object* v_00_u03b1_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lp_mathlib_IsSimpleOrder_booleanAlgebra___redArg(v_inst_217_, v_inst_218_, v_inst_219_);
return v___x_221_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Indexed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Indexed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Atoms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompletePartialOrder_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Atoms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Atoms(builtin);
}
#ifdef __cplusplus
}
#endif
