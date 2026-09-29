// Lean compiler output
// Module: Mathlib.Order.SuccPred.LinearLocallyFinite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Nat public import Mathlib.Basic.Countable.Basic public import Mathlib.Data.Finset.Max public import Mathlib.Data.Fintype.Pigeonhole public import Mathlib.Logic.Encodable.Basic public import Mathlib.Order.Interval.Finset.Defs public import Mathlib.Order.SuccPred.Archimedean
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
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Order_pred___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lp_mathlib_Order_succ___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toZ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg(lean_object* v_this_2_, lean_object* v_a_3_){
_start:
{
lean_object* v___x_4_; lean_object* v_toFun_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_4_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0, &lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0);
v_toFun_5_ = lean_ctor_get(v___x_4_, 0);
lean_inc_n(v_toFun_5_, 2);
v___x_6_ = lean_apply_1(v_toFun_5_, v_a_3_);
v___x_7_ = lean_apply_1(v_this_2_, v___x_6_);
v___x_8_ = lean_apply_1(v_toFun_5_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1(lean_object* v_00_u03b9_9_, lean_object* v_inst_10_, lean_object* v_this_11_, lean_object* v_a_12_){
_start:
{
lean_object* v___x_13_; lean_object* v_toFun_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_13_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0, &lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___redArg___closed__0);
v_toFun_14_ = lean_ctor_get(v___x_13_, 0);
lean_inc_n(v_toFun_14_, 2);
v___x_15_ = lean_apply_1(v_toFun_14_, v_a_12_);
v___x_16_ = lean_apply_1(v_this_11_, v___x_15_);
v___x_17_ = lean_apply_1(v_toFun_14_, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1___boxed(lean_object* v_00_u03b9_18_, lean_object* v_inst_19_, lean_object* v_this_20_, lean_object* v_a_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib___private_Mathlib_Order_SuccPred_LinearLocallyFinite_0__LinearLocallyFiniteOrder_predOrder___aux__1(v_00_u03b9_18_, v_inst_19_, v_this_20_, v_a_21_);
lean_dec_ref(v_inst_19_);
return v_res_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__0(lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_i0_25_, lean_object* v_toDecidableEq_26_, lean_object* v_i_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v_toPartialOrder_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; uint8_t v___x_35_; 
v___x_29_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_23_);
v___x_30_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_29_);
v_toPartialOrder_31_ = lean_ctor_get(v___x_30_, 0);
lean_inc_ref(v_toPartialOrder_31_);
lean_dec_ref(v___x_30_);
v___x_32_ = lean_alloc_closure((void*)(lp_mathlib_Order_pred___boxed), 4, 3);
lean_closure_set(v___x_32_, 0, lean_box(0));
lean_closure_set(v___x_32_, 1, v_toPartialOrder_31_);
lean_closure_set(v___x_32_, 2, v_inst_24_);
v___x_33_ = lp_mathlib_Nat_iterate___redArg(v___x_32_, v_a_28_, v_i0_25_);
v___x_34_ = lean_apply_2(v_toDecidableEq_26_, v___x_33_, v_i_27_);
v___x_35_ = lean_unbox(v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__0___boxed(lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_i0_38_, lean_object* v_toDecidableEq_39_, lean_object* v_i_40_, lean_object* v_a_41_){
_start:
{
uint8_t v_res_42_; lean_object* v_r_43_; 
v_res_42_ = lp_mathlib_toZ___redArg___lam__0(v_inst_36_, v_inst_37_, v_i0_38_, v_toDecidableEq_39_, v_i_40_, v_a_41_);
lean_dec_ref(v_inst_36_);
v_r_43_ = lean_box(v_res_42_);
return v_r_43_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_toZ___redArg___lam__1(lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_i0_46_, lean_object* v_toDecidableEq_47_, lean_object* v_i_48_, lean_object* v_a_49_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v_toPartialOrder_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; uint8_t v___x_56_; 
v___x_50_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_44_);
v___x_51_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_50_);
v_toPartialOrder_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc_ref(v_toPartialOrder_52_);
lean_dec_ref(v___x_51_);
v___x_53_ = lean_alloc_closure((void*)(lp_mathlib_Order_succ___boxed), 4, 3);
lean_closure_set(v___x_53_, 0, lean_box(0));
lean_closure_set(v___x_53_, 1, v_toPartialOrder_52_);
lean_closure_set(v___x_53_, 2, v_inst_45_);
v___x_54_ = lp_mathlib_Nat_iterate___redArg(v___x_53_, v_a_49_, v_i0_46_);
v___x_55_ = lean_apply_2(v_toDecidableEq_47_, v___x_54_, v_i_48_);
v___x_56_ = lean_unbox(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg___lam__1___boxed(lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_i0_59_, lean_object* v_toDecidableEq_60_, lean_object* v_i_61_, lean_object* v_a_62_){
_start:
{
uint8_t v_res_63_; lean_object* v_r_64_; 
v_res_63_ = lp_mathlib_toZ___redArg___lam__1(v_inst_57_, v_inst_58_, v_i0_59_, v_toDecidableEq_60_, v_i_61_, v_a_62_);
lean_dec_ref(v_inst_57_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ___redArg(lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_i0_68_, lean_object* v_i_69_){
_start:
{
lean_object* v_toDecidableLE_70_; lean_object* v_toDecidableEq_71_; lean_object* v___x_72_; uint8_t v___x_73_; 
v_toDecidableLE_70_ = lean_ctor_get(v_inst_65_, 4);
v_toDecidableEq_71_ = lean_ctor_get(v_inst_65_, 5);
lean_inc_ref(v_toDecidableEq_71_);
lean_inc_ref(v_toDecidableLE_70_);
lean_inc(v_i_69_);
lean_inc(v_i0_68_);
v___x_72_ = lean_apply_2(v_toDecidableLE_70_, v_i0_68_, v_i_69_);
v___x_73_ = lean_unbox(v___x_72_);
if (v___x_73_ == 0)
{
lean_object* v___f_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
lean_dec(v_inst_66_);
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_toZ___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_74_, 0, v_inst_65_);
lean_closure_set(v___f_74_, 1, v_inst_67_);
lean_closure_set(v___f_74_, 2, v_i0_68_);
lean_closure_set(v___f_74_, 3, v_toDecidableEq_71_);
lean_closure_set(v___f_74_, 4, v_i_69_);
v___x_75_ = lp_mathlib_Nat_findX___redArg(v___f_74_);
v___x_76_ = lean_nat_to_int(v___x_75_);
v___x_77_ = lean_int_neg(v___x_76_);
lean_dec(v___x_76_);
return v___x_77_;
}
else
{
lean_object* v___f_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
lean_dec(v_inst_67_);
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_toZ___redArg___lam__1___boxed), 6, 5);
lean_closure_set(v___f_78_, 0, v_inst_65_);
lean_closure_set(v___f_78_, 1, v_inst_66_);
lean_closure_set(v___f_78_, 2, v_i0_68_);
lean_closure_set(v___f_78_, 3, v_toDecidableEq_71_);
lean_closure_set(v___f_78_, 4, v_i_69_);
v___x_79_ = lp_mathlib_Nat_findX___redArg(v___f_78_);
v___x_80_ = lean_nat_to_int(v___x_79_);
return v___x_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_toZ(lean_object* v_00_u03b9_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_i0_86_, lean_object* v_i_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_toZ___redArg(v_inst_82_, v_inst_83_, v_inst_85_, v_i0_86_, v_i_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0(lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_i_93_){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lp_mathlib_toZ___redArg(v_inst_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_i_93_);
v___x_95_ = l_Int_toNat(v___x_94_);
lean_dec(v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1(lean_object* v_toPartialOrder_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_n_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_alloc_closure((void*)(lp_mathlib_Order_succ___boxed), 4, 3);
lean_closure_set(v___x_100_, 0, lean_box(0));
lean_closure_set(v___x_100_, 1, v_toPartialOrder_96_);
lean_closure_set(v___x_100_, 2, v_inst_97_);
v___x_101_ = lp_mathlib_Nat_iterate___redArg(v___x_100_, v_n_99_, v_inst_98_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_toPartialOrder_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_117_; 
v___x_106_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_102_);
v___x_107_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_106_);
v_toPartialOrder_108_ = lean_ctor_get(v___x_107_, 0);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_107_);
if (v_isSharedCheck_117_ == 0)
{
lean_object* v_unused_118_; 
v_unused_118_ = lean_ctor_get(v___x_107_, 1);
lean_dec(v_unused_118_);
v___x_110_ = v___x_107_;
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_toPartialOrder_108_);
lean_dec(v___x_107_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___f_112_; lean_object* v___f_113_; lean_object* v___x_115_; 
lean_inc(v_inst_105_);
lean_inc(v_inst_103_);
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(v___f_112_, 0, v_inst_102_);
lean_closure_set(v___f_112_, 1, v_inst_103_);
lean_closure_set(v___f_112_, 2, v_inst_104_);
lean_closure_set(v___f_112_, 3, v_inst_105_);
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(v___f_113_, 0, v_toPartialOrder_108_);
lean_closure_set(v___f_113_, 1, v_inst_103_);
lean_closure_set(v___f_113_, 2, v_inst_105_);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 1, v___f_113_);
lean_ctor_set(v___x_110_, 0, v___f_112_);
v___x_115_ = v___x_110_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v___f_112_);
lean_ctor_set(v_reuseFailAlloc_116_, 1, v___f_113_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoNatOfLinearSuccPredArch(lean_object* v_00_u03b9_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v___x_126_; 
v___x_126_ = lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg(v_inst_120_, v_inst_121_, v_inst_122_, v_inst_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v_toPartialOrder_133_; lean_object* v___x_135_; uint8_t v_isShared_136_; uint8_t v_isSharedCheck_142_; 
v___x_131_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_127_);
v___x_132_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_131_);
v_toPartialOrder_133_ = lean_ctor_get(v___x_132_, 0);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_132_);
if (v_isSharedCheck_142_ == 0)
{
lean_object* v_unused_143_; 
v_unused_143_ = lean_ctor_get(v___x_132_, 1);
lean_dec(v_unused_143_);
v___x_135_ = v___x_132_;
v_isShared_136_ = v_isSharedCheck_142_;
goto v_resetjp_134_;
}
else
{
lean_inc(v_toPartialOrder_133_);
lean_dec(v___x_132_);
v___x_135_ = lean_box(0);
v_isShared_136_ = v_isSharedCheck_142_;
goto v_resetjp_134_;
}
v_resetjp_134_:
{
lean_object* v___f_137_; lean_object* v___f_138_; lean_object* v___x_140_; 
lean_inc(v_inst_130_);
lean_inc(v_inst_128_);
v___f_137_ = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__0), 5, 4);
lean_closure_set(v___f_137_, 0, v_inst_127_);
lean_closure_set(v___f_137_, 1, v_inst_128_);
lean_closure_set(v___f_137_, 2, v_inst_129_);
lean_closure_set(v___f_137_, 3, v_inst_130_);
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_orderIsoNatOfLinearSuccPredArch___redArg___lam__1), 4, 3);
lean_closure_set(v___f_138_, 0, v_toPartialOrder_133_);
lean_closure_set(v___f_138_, 1, v_inst_128_);
lean_closure_set(v___f_138_, 2, v_inst_130_);
if (v_isShared_136_ == 0)
{
lean_ctor_set(v___x_135_, 1, v___f_138_);
lean_ctor_set(v___x_135_, 0, v___f_137_);
v___x_140_ = v___x_135_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___f_137_);
lean_ctor_set(v_reuseFailAlloc_141_, 1, v___f_138_);
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
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch(lean_object* v_00_u03b9_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_orderIsoRangeOfLinearSuccPredArch___redArg(v_inst_145_, v_inst_146_, v_inst_147_, v_inst_149_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderIsoRangeOfLinearSuccPredArch___boxed(lean_object* v_00_u03b9_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_orderIsoRangeOfLinearSuccPredArch(v_00_u03b9_152_, v_inst_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_inst_158_);
lean_dec(v_inst_158_);
return v_res_159_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Countable_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Countable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Countable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Countable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pigeonhole(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_Archimedean(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SuccPred_LinearLocallyFinite(builtin);
}
#ifdef __cplusplus
}
#endif
