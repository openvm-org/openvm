// Lean compiler output
// Module: Mathlib.Order.GaloisConnection.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Image public import Mathlib.Order.CompleteLattice.Basic public import Mathlib.Order.WithBot
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
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_unbotD___redArg(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_untopD___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giSSupIic___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giSSupIic(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gi__sSup__Iic___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gi__sSup__Iic(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciIciSInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciIciSInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gci__Ici__sInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gci__Ici__sInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_toGaloisCoinsertion___at___00gciMapBicompl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciMapBicompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciMapBicompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciMapOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_gciMapOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_toGaloisInsertion___at___00giMapBicompl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giMapBicompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giMapBicompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giMapOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_giMapOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___redArg___lam__0(lean_object* v_e_1_, lean_object* v_b_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_e_1_, v_b_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___redArg(lean_object* v_e_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_toGaloisInsertion___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_e_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion(lean_object* v_00_u03b1_7_, lean_object* v_00_u03b2_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_e_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_toGaloisInsertion___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_e_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisInsertion___boxed(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_e_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_OrderIso_toGaloisInsertion(v_00_u03b1_13_, v_00_u03b2_14_, v_inst_15_, v_inst_16_, v_e_17_);
lean_dec_ref(v_inst_16_);
lean_dec_ref(v_inst_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___redArg___lam__0(lean_object* v___x_19_, lean_object* v_b_20_, lean_object* v_x_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_19_, v_b_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___redArg(lean_object* v_e_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___f_25_; 
v___x_24_ = lp_mathlib_Equiv_symm___redArg(v_e_23_);
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_toGaloisCoinsertion___redArg___lam__0), 3, 1);
lean_closure_set(v___f_25_, 0, v___x_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion(lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_e_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_OrderIso_toGaloisCoinsertion___redArg(v_e_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toGaloisCoinsertion___boxed(lean_object* v_00_u03b1_32_, lean_object* v_00_u03b2_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_e_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_OrderIso_toGaloisCoinsertion(v_00_u03b1_32_, v_00_u03b2_33_, v_inst_34_, v_inst_35_, v_e_36_);
lean_dec_ref(v_inst_35_);
lean_dec_ref(v_inst_34_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg___lam__0(lean_object* v_inst_38_, lean_object* v_u_39_, lean_object* v_l_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
lean_object* v_sup_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v_sup_43_ = lean_ctor_get(v_inst_38_, 1);
lean_inc(v_sup_43_);
lean_dec_ref(v_inst_38_);
lean_inc(v_u_39_);
v___x_44_ = lean_apply_1(v_u_39_, v_a_41_);
v___x_45_ = lean_apply_1(v_u_39_, v_b_42_);
v___x_46_ = lean_apply_2(v_sup_43_, v___x_44_, v___x_45_);
v___x_47_ = lean_apply_1(v_l_40_, v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg(lean_object* v_l_48_, lean_object* v_u_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; lean_object* v___x_53_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg___lam__0), 5, 3);
lean_closure_set(v___f_52_, 0, v_inst_51_);
lean_closure_set(v___f_52_, 1, v_u_49_);
lean_closure_set(v___f_52_, 2, v_l_48_);
v___x_53_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_53_, 0, v_inst_50_);
lean_ctor_set(v___x_53_, 1, v___f_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup(lean_object* v_00_u03b1_54_, lean_object* v_00_u03b2_55_, lean_object* v_l_56_, lean_object* v_u_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_gi_60_){
_start:
{
lean_object* v___f_61_; lean_object* v___x_62_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftSemilatticeSup___redArg___lam__0), 5, 3);
lean_closure_set(v___f_61_, 0, v_inst_59_);
lean_closure_set(v___f_61_, 1, v_u_57_);
lean_closure_set(v___f_61_, 2, v_l_56_);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v_inst_58_);
lean_ctor_set(v___x_62_, 1, v___f_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeSup___boxed(lean_object* v_00_u03b1_63_, lean_object* v_00_u03b2_64_, lean_object* v_l_65_, lean_object* v_u_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_gi_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_GaloisInsertion_liftSemilatticeSup(v_00_u03b1_63_, v_00_u03b2_64_, v_l_65_, v_u_66_, v_inst_67_, v_inst_68_, v_gi_69_);
lean_dec(v_gi_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg___lam__0(lean_object* v_inst_71_, lean_object* v_u_72_, lean_object* v_l_73_, lean_object* v_a_74_, lean_object* v_b_75_){
_start:
{
lean_object* v_inf_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_inf_76_ = lean_ctor_get(v_inst_71_, 1);
lean_inc(v_inf_76_);
lean_dec_ref(v_inst_71_);
lean_inc(v_u_72_);
v___x_77_ = lean_apply_1(v_u_72_, v_a_74_);
v___x_78_ = lean_apply_1(v_u_72_, v_b_75_);
v___x_79_ = lean_apply_2(v_inf_76_, v___x_77_, v___x_78_);
v___x_80_ = lean_apply_1(v_l_73_, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg(lean_object* v_l_81_, lean_object* v_u_82_, lean_object* v_inst_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___f_85_; lean_object* v___x_86_; 
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg___lam__0), 5, 3);
lean_closure_set(v___f_85_, 0, v_inst_84_);
lean_closure_set(v___f_85_, 1, v_u_82_);
lean_closure_set(v___f_85_, 2, v_l_81_);
v___x_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_86_, 0, v_inst_83_);
lean_ctor_set(v___x_86_, 1, v___f_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf(lean_object* v_00_u03b1_87_, lean_object* v_00_u03b2_88_, lean_object* v_l_89_, lean_object* v_u_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_gi_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___redArg(v_l_89_, v_u_90_, v_inst_91_, v_inst_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeInf___boxed(lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_l_97_, lean_object* v_u_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_gi_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeInf(v_00_u03b1_95_, v_00_u03b2_96_, v_l_97_, v_u_98_, v_inst_99_, v_inst_100_, v_gi_101_);
lean_dec(v_gi_101_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg___lam__0(lean_object* v_inst_103_, lean_object* v_u_104_, lean_object* v_gi_105_, lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
lean_object* v_inf_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v_inf_108_ = lean_ctor_get(v_inst_103_, 1);
lean_inc(v_inf_108_);
lean_dec_ref(v_inst_103_);
lean_inc(v_u_104_);
v___x_109_ = lean_apply_1(v_u_104_, v_a_106_);
v___x_110_ = lean_apply_1(v_u_104_, v_b_107_);
v___x_111_ = lean_apply_2(v_inf_108_, v___x_109_, v___x_110_);
v___x_112_ = lean_apply_2(v_gi_105_, v___x_111_, lean_box(0));
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg(lean_object* v_u_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_gi_116_){
_start:
{
lean_object* v___f_117_; lean_object* v___x_118_; 
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg___lam__0), 5, 3);
lean_closure_set(v___f_117_, 0, v_inst_115_);
lean_closure_set(v___f_117_, 1, v_u_113_);
lean_closure_set(v___f_117_, 2, v_gi_116_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v_inst_114_);
lean_ctor_set(v___x_118_, 1, v___f_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf(lean_object* v_00_u03b1_119_, lean_object* v_00_u03b2_120_, lean_object* v_l_121_, lean_object* v_u_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_gi_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___x_127_; 
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftSemilatticeInf___redArg___lam__0), 5, 3);
lean_closure_set(v___f_126_, 0, v_inst_124_);
lean_closure_set(v___f_126_, 1, v_u_122_);
lean_closure_set(v___f_126_, 2, v_gi_125_);
v___x_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_127_, 0, v_inst_123_);
lean_ctor_set(v___x_127_, 1, v___f_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftSemilatticeInf___boxed(lean_object* v_00_u03b1_128_, lean_object* v_00_u03b2_129_, lean_object* v_l_130_, lean_object* v_u_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_gi_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_GaloisInsertion_liftSemilatticeInf(v_00_u03b1_128_, v_00_u03b2_129_, v_l_130_, v_u_131_, v_inst_132_, v_inst_133_, v_gi_134_);
lean_dec(v_l_130_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg___lam__0(lean_object* v_inst_136_, lean_object* v_u_137_, lean_object* v_gi_138_, lean_object* v_a_139_, lean_object* v_b_140_){
_start:
{
lean_object* v_sup_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v_sup_141_ = lean_ctor_get(v_inst_136_, 1);
lean_inc(v_sup_141_);
lean_dec_ref(v_inst_136_);
lean_inc(v_u_137_);
v___x_142_ = lean_apply_1(v_u_137_, v_a_139_);
v___x_143_ = lean_apply_1(v_u_137_, v_b_140_);
v___x_144_ = lean_apply_2(v_sup_141_, v___x_142_, v___x_143_);
v___x_145_ = lean_apply_2(v_gi_138_, v___x_144_, lean_box(0));
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(lean_object* v_u_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_gi_149_){
_start:
{
lean_object* v___f_150_; lean_object* v___x_151_; 
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg___lam__0), 5, 3);
lean_closure_set(v___f_150_, 0, v_inst_148_);
lean_closure_set(v___f_150_, 1, v_u_146_);
lean_closure_set(v___f_150_, 2, v_gi_149_);
v___x_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_151_, 0, v_inst_147_);
lean_ctor_set(v___x_151_, 1, v___f_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup(lean_object* v_00_u03b1_152_, lean_object* v_00_u03b2_153_, lean_object* v_l_154_, lean_object* v_u_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_gi_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(v_u_155_, v_inst_156_, v_inst_157_, v_gi_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___boxed(lean_object* v_00_u03b1_160_, lean_object* v_00_u03b2_161_, lean_object* v_l_162_, lean_object* v_u_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_gi_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeSup(v_00_u03b1_160_, v_00_u03b2_161_, v_l_162_, v_u_163_, v_inst_164_, v_inst_165_, v_gi_166_);
lean_dec(v_l_162_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0(lean_object* v_u_168_, lean_object* v_inf_169_, lean_object* v_gi_170_, lean_object* v_a_171_, lean_object* v_b_172_){
_start:
{
lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
lean_inc(v_u_168_);
v___x_173_ = lean_apply_1(v_u_168_, v_a_171_);
v___x_174_ = lean_apply_1(v_u_168_, v_b_172_);
v___x_175_ = lean_apply_2(v_inf_169_, v___x_173_, v___x_174_);
v___x_176_ = lean_apply_2(v_gi_170_, v___x_175_, lean_box(0));
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1(lean_object* v_toSemilatticeSup_177_, lean_object* v_u_178_, lean_object* v_l_179_, lean_object* v_a_180_, lean_object* v_b_181_){
_start:
{
lean_object* v_sup_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v_sup_182_ = lean_ctor_get(v_toSemilatticeSup_177_, 1);
lean_inc(v_sup_182_);
lean_dec_ref(v_toSemilatticeSup_177_);
lean_inc(v_u_178_);
v___x_183_ = lean_apply_1(v_u_178_, v_a_180_);
v___x_184_ = lean_apply_1(v_u_178_, v_b_181_);
v___x_185_ = lean_apply_2(v_sup_182_, v___x_183_, v___x_184_);
v___x_186_ = lean_apply_1(v_l_179_, v___x_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice___redArg(lean_object* v_l_187_, lean_object* v_u_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_gi_191_){
_start:
{
lean_object* v_toSemilatticeSup_192_; lean_object* v_inf_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_203_; 
v_toSemilatticeSup_192_ = lean_ctor_get(v_inst_190_, 0);
v_inf_193_ = lean_ctor_get(v_inst_190_, 1);
v_isSharedCheck_203_ = !lean_is_exclusive(v_inst_190_);
if (v_isSharedCheck_203_ == 0)
{
v___x_195_ = v_inst_190_;
v_isShared_196_ = v_isSharedCheck_203_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_inf_193_);
lean_inc(v_toSemilatticeSup_192_);
lean_dec(v_inst_190_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_203_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___f_197_; lean_object* v___f_198_; lean_object* v___x_199_; lean_object* v___x_201_; 
lean_inc(v_u_188_);
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_197_, 0, v_u_188_);
lean_closure_set(v___f_197_, 1, v_inf_193_);
lean_closure_set(v___f_197_, 2, v_gi_191_);
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1), 5, 3);
lean_closure_set(v___f_198_, 0, v_toSemilatticeSup_192_);
lean_closure_set(v___f_198_, 1, v_u_188_);
lean_closure_set(v___f_198_, 2, v_l_187_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v_inst_189_);
lean_ctor_set(v___x_199_, 1, v___f_198_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 1, v___f_197_);
lean_ctor_set(v___x_195_, 0, v___x_199_);
v___x_201_ = v___x_195_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v___x_199_);
lean_ctor_set(v_reuseFailAlloc_202_, 1, v___f_197_);
v___x_201_ = v_reuseFailAlloc_202_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
return v___x_201_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftLattice(lean_object* v_00_u03b1_204_, lean_object* v_00_u03b2_205_, lean_object* v_l_206_, lean_object* v_u_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_gi_210_){
_start:
{
lean_object* v_toSemilatticeSup_211_; lean_object* v_inf_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_222_; 
v_toSemilatticeSup_211_ = lean_ctor_get(v_inst_209_, 0);
v_inf_212_ = lean_ctor_get(v_inst_209_, 1);
v_isSharedCheck_222_ = !lean_is_exclusive(v_inst_209_);
if (v_isSharedCheck_222_ == 0)
{
v___x_214_ = v_inst_209_;
v_isShared_215_ = v_isSharedCheck_222_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_inf_212_);
lean_inc(v_toSemilatticeSup_211_);
lean_dec(v_inst_209_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_222_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___f_216_; lean_object* v___f_217_; lean_object* v___x_218_; lean_object* v___x_220_; 
lean_inc(v_u_207_);
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_216_, 0, v_u_207_);
lean_closure_set(v___f_216_, 1, v_inf_212_);
lean_closure_set(v___f_216_, 2, v_gi_210_);
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1), 5, 3);
lean_closure_set(v___f_217_, 0, v_toSemilatticeSup_211_);
lean_closure_set(v___f_217_, 1, v_u_207_);
lean_closure_set(v___f_217_, 2, v_l_206_);
v___x_218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_218_, 0, v_inst_208_);
lean_ctor_set(v___x_218_, 1, v___f_217_);
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v___f_216_);
lean_ctor_set(v___x_214_, 0, v___x_218_);
v___x_220_ = v___x_214_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_218_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v___f_216_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
return v___x_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice___redArg___lam__0(lean_object* v_u_223_, lean_object* v_inf_224_, lean_object* v_l_225_, lean_object* v_a_226_, lean_object* v_b_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
lean_inc(v_u_223_);
v___x_228_ = lean_apply_1(v_u_223_, v_a_226_);
v___x_229_ = lean_apply_1(v_u_223_, v_b_227_);
v___x_230_ = lean_apply_2(v_inf_224_, v___x_228_, v___x_229_);
v___x_231_ = lean_apply_1(v_l_225_, v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice___redArg(lean_object* v_l_232_, lean_object* v_u_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_gi_236_){
_start:
{
lean_object* v_toSemilatticeSup_237_; lean_object* v_inf_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_247_; 
v_toSemilatticeSup_237_ = lean_ctor_get(v_inst_235_, 0);
v_inf_238_ = lean_ctor_get(v_inst_235_, 1);
v_isSharedCheck_247_ = !lean_is_exclusive(v_inst_235_);
if (v_isSharedCheck_247_ == 0)
{
v___x_240_ = v_inst_235_;
v_isShared_241_ = v_isSharedCheck_247_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_inf_238_);
lean_inc(v_toSemilatticeSup_237_);
lean_dec(v_inst_235_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_247_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___f_242_; lean_object* v___x_243_; lean_object* v___x_245_; 
lean_inc(v_u_233_);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_242_, 0, v_u_233_);
lean_closure_set(v___f_242_, 1, v_inf_238_);
lean_closure_set(v___f_242_, 2, v_l_232_);
v___x_243_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(v_u_233_, v_inst_234_, v_toSemilatticeSup_237_, v_gi_236_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 1, v___f_242_);
lean_ctor_set(v___x_240_, 0, v___x_243_);
v___x_245_ = v___x_240_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_246_, 1, v___f_242_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftLattice(lean_object* v_00_u03b1_248_, lean_object* v_00_u03b2_249_, lean_object* v_l_250_, lean_object* v_u_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_gi_254_){
_start:
{
lean_object* v_toSemilatticeSup_255_; lean_object* v_inf_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_265_; 
v_toSemilatticeSup_255_ = lean_ctor_get(v_inst_253_, 0);
v_inf_256_ = lean_ctor_get(v_inst_253_, 1);
v_isSharedCheck_265_ = !lean_is_exclusive(v_inst_253_);
if (v_isSharedCheck_265_ == 0)
{
v___x_258_ = v_inst_253_;
v_isShared_259_ = v_isSharedCheck_265_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_inf_256_);
lean_inc(v_toSemilatticeSup_255_);
lean_dec(v_inst_253_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_265_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___f_260_; lean_object* v___x_261_; lean_object* v___x_263_; 
lean_inc(v_u_251_);
v___f_260_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_260_, 0, v_u_251_);
lean_closure_set(v___f_260_, 1, v_inf_256_);
lean_closure_set(v___f_260_, 2, v_l_250_);
v___x_261_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(v_u_251_, v_inst_252_, v_toSemilatticeSup_255_, v_gi_254_);
if (v_isShared_259_ == 0)
{
lean_ctor_set(v___x_258_, 1, v___f_260_);
lean_ctor_set(v___x_258_, 0, v___x_261_);
v___x_263_ = v___x_258_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_261_);
lean_ctor_set(v_reuseFailAlloc_264_, 1, v___f_260_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop___redArg(lean_object* v_inst_266_, lean_object* v_gi_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lean_apply_2(v_gi_267_, v_inst_266_, lean_box(0));
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop(lean_object* v_00_u03b1_269_, lean_object* v_00_u03b2_270_, lean_object* v_l_271_, lean_object* v_u_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_gi_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lean_apply_2(v_gi_276_, v_inst_275_, lean_box(0));
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftOrderTop___boxed(lean_object* v_00_u03b1_278_, lean_object* v_00_u03b2_279_, lean_object* v_l_280_, lean_object* v_u_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_gi_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_GaloisInsertion_liftOrderTop(v_00_u03b1_278_, v_00_u03b2_279_, v_l_280_, v_u_281_, v_inst_282_, v_inst_283_, v_inst_284_, v_gi_285_);
lean_dec_ref(v_inst_283_);
lean_dec_ref(v_inst_282_);
lean_dec(v_u_281_);
lean_dec(v_l_280_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot___redArg(lean_object* v_inst_287_, lean_object* v_gi_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lean_apply_2(v_gi_288_, v_inst_287_, lean_box(0));
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot(lean_object* v_00_u03b1_290_, lean_object* v_00_u03b2_291_, lean_object* v_l_292_, lean_object* v_u_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_gi_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lean_apply_2(v_gi_297_, v_inst_296_, lean_box(0));
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftOrderBot___boxed(lean_object* v_00_u03b1_299_, lean_object* v_00_u03b2_300_, lean_object* v_l_301_, lean_object* v_u_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_gi_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_GaloisCoinsertion_liftOrderBot(v_00_u03b1_299_, v_00_u03b2_300_, v_l_301_, v_u_302_, v_inst_303_, v_inst_304_, v_inst_305_, v_gi_306_);
lean_dec_ref(v_inst_304_);
lean_dec_ref(v_inst_303_);
lean_dec(v_u_302_);
lean_dec(v_l_301_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder___redArg(lean_object* v_l_308_, lean_object* v_inst_309_, lean_object* v_gi_310_){
_start:
{
lean_object* v_toOrderTop_311_; lean_object* v_toOrderBot_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_321_; 
v_toOrderTop_311_ = lean_ctor_get(v_inst_309_, 0);
v_toOrderBot_312_ = lean_ctor_get(v_inst_309_, 1);
v_isSharedCheck_321_ = !lean_is_exclusive(v_inst_309_);
if (v_isSharedCheck_321_ == 0)
{
v___x_314_ = v_inst_309_;
v_isShared_315_ = v_isSharedCheck_321_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_toOrderBot_312_);
lean_inc(v_toOrderTop_311_);
lean_dec(v_inst_309_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_321_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_319_; 
v___x_316_ = lean_apply_2(v_gi_310_, v_toOrderTop_311_, lean_box(0));
v___x_317_ = lean_apply_1(v_l_308_, v_toOrderBot_312_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 1, v___x_317_);
lean_ctor_set(v___x_314_, 0, v___x_316_);
v___x_319_ = v___x_314_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_316_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v___x_317_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder(lean_object* v_00_u03b1_322_, lean_object* v_00_u03b2_323_, lean_object* v_l_324_, lean_object* v_u_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_gi_329_){
_start:
{
lean_object* v_toOrderTop_330_; lean_object* v_toOrderBot_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_340_; 
v_toOrderTop_330_ = lean_ctor_get(v_inst_328_, 0);
v_toOrderBot_331_ = lean_ctor_get(v_inst_328_, 1);
v_isSharedCheck_340_ = !lean_is_exclusive(v_inst_328_);
if (v_isSharedCheck_340_ == 0)
{
v___x_333_ = v_inst_328_;
v_isShared_334_ = v_isSharedCheck_340_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_toOrderBot_331_);
lean_inc(v_toOrderTop_330_);
lean_dec(v_inst_328_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_340_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_338_; 
v___x_335_ = lean_apply_2(v_gi_329_, v_toOrderTop_330_, lean_box(0));
v___x_336_ = lean_apply_1(v_l_324_, v_toOrderBot_331_);
if (v_isShared_334_ == 0)
{
lean_ctor_set(v___x_333_, 1, v___x_336_);
lean_ctor_set(v___x_333_, 0, v___x_335_);
v___x_338_ = v___x_333_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_335_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v___x_336_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftBoundedOrder___boxed(lean_object* v_00_u03b1_341_, lean_object* v_00_u03b2_342_, lean_object* v_l_343_, lean_object* v_u_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_gi_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_GaloisInsertion_liftBoundedOrder(v_00_u03b1_341_, v_00_u03b2_342_, v_l_343_, v_u_344_, v_inst_345_, v_inst_346_, v_inst_347_, v_gi_348_);
lean_dec_ref(v_inst_346_);
lean_dec_ref(v_inst_345_);
lean_dec(v_u_344_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder___redArg(lean_object* v_l_350_, lean_object* v_inst_351_, lean_object* v_gi_352_){
_start:
{
lean_object* v_toOrderTop_353_; lean_object* v_toOrderBot_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_363_; 
v_toOrderTop_353_ = lean_ctor_get(v_inst_351_, 0);
v_toOrderBot_354_ = lean_ctor_get(v_inst_351_, 1);
v_isSharedCheck_363_ = !lean_is_exclusive(v_inst_351_);
if (v_isSharedCheck_363_ == 0)
{
v___x_356_ = v_inst_351_;
v_isShared_357_ = v_isSharedCheck_363_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_toOrderBot_354_);
lean_inc(v_toOrderTop_353_);
lean_dec(v_inst_351_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_363_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_361_; 
v___x_358_ = lean_apply_2(v_gi_352_, v_toOrderBot_354_, lean_box(0));
v___x_359_ = lean_apply_1(v_l_350_, v_toOrderTop_353_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 1, v___x_358_);
lean_ctor_set(v___x_356_, 0, v___x_359_);
v___x_361_ = v___x_356_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v___x_359_);
lean_ctor_set(v_reuseFailAlloc_362_, 1, v___x_358_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder(lean_object* v_00_u03b1_364_, lean_object* v_00_u03b2_365_, lean_object* v_l_366_, lean_object* v_u_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_gi_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_GaloisCoinsertion_liftBoundedOrder___redArg(v_l_366_, v_inst_370_, v_gi_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftBoundedOrder___boxed(lean_object* v_00_u03b1_373_, lean_object* v_00_u03b2_374_, lean_object* v_l_375_, lean_object* v_u_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_gi_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_GaloisCoinsertion_liftBoundedOrder(v_00_u03b1_373_, v_00_u03b2_374_, v_l_375_, v_u_376_, v_inst_377_, v_inst_378_, v_inst_379_, v_gi_380_);
lean_dec_ref(v_inst_378_);
lean_dec_ref(v_inst_377_);
lean_dec(v_u_376_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__2(lean_object* v_toSupSet_382_, lean_object* v_l_383_, lean_object* v_s_384_){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_385_ = lean_apply_1(v_toSupSet_382_, lean_box(0));
v___x_386_ = lean_apply_1(v_l_383_, v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__0(lean_object* v_toInfSet_387_, lean_object* v_gi_388_, lean_object* v_s_389_){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = lean_apply_1(v_toInfSet_387_, lean_box(0));
v___x_391_ = lean_apply_2(v_gi_388_, v___x_390_, lean_box(0));
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg(lean_object* v_l_392_, lean_object* v_u_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_gi_396_){
_start:
{
lean_object* v___x_397_; lean_object* v_toBoundedOrder_398_; lean_object* v_toLattice_399_; lean_object* v_toOrderTop_400_; lean_object* v_toOrderBot_401_; lean_object* v___x_403_; uint8_t v_isShared_404_; uint8_t v_isSharedCheck_428_; 
lean_inc_ref(v_inst_395_);
v___x_397_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_395_);
v_toBoundedOrder_398_ = lean_ctor_get(v_inst_395_, 3);
lean_inc_ref(v_toBoundedOrder_398_);
v_toLattice_399_ = lean_ctor_get(v_inst_395_, 0);
lean_inc_ref(v_toLattice_399_);
v_toOrderTop_400_ = lean_ctor_get(v_toBoundedOrder_398_, 0);
v_toOrderBot_401_ = lean_ctor_get(v_toBoundedOrder_398_, 1);
v_isSharedCheck_428_ = !lean_is_exclusive(v_toBoundedOrder_398_);
if (v_isSharedCheck_428_ == 0)
{
v___x_403_ = v_toBoundedOrder_398_;
v_isShared_404_ = v_isSharedCheck_428_;
goto v_resetjp_402_;
}
else
{
lean_inc(v_toOrderBot_401_);
lean_inc(v_toOrderTop_400_);
lean_dec(v_toBoundedOrder_398_);
v___x_403_ = lean_box(0);
v_isShared_404_ = v_isSharedCheck_428_;
goto v_resetjp_402_;
}
v_resetjp_402_:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_408_; 
lean_inc(v_gi_396_);
v___x_405_ = lean_apply_2(v_gi_396_, v_toOrderTop_400_, lean_box(0));
lean_inc(v_l_392_);
v___x_406_ = lean_apply_1(v_l_392_, v_toOrderBot_401_);
if (v_isShared_404_ == 0)
{
lean_ctor_set(v___x_403_, 1, v___x_406_);
lean_ctor_set(v___x_403_, 0, v___x_405_);
v___x_408_ = v___x_403_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v___x_405_);
lean_ctor_set(v_reuseFailAlloc_427_, 1, v___x_406_);
v___x_408_ = v_reuseFailAlloc_427_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v_toSemilatticeSup_409_; lean_object* v_inf_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_426_; 
v_toSemilatticeSup_409_ = lean_ctor_get(v_toLattice_399_, 0);
v_inf_410_ = lean_ctor_get(v_toLattice_399_, 1);
v_isSharedCheck_426_ = !lean_is_exclusive(v_toLattice_399_);
if (v_isSharedCheck_426_ == 0)
{
v___x_412_ = v_toLattice_399_;
v_isShared_413_ = v_isSharedCheck_426_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_inf_410_);
lean_inc(v_toSemilatticeSup_409_);
lean_dec(v_toLattice_399_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_426_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___f_414_; lean_object* v___f_415_; lean_object* v___x_416_; lean_object* v___x_418_; 
lean_inc(v_gi_396_);
lean_inc(v_u_393_);
v___f_414_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_414_, 0, v_u_393_);
lean_closure_set(v___f_414_, 1, v_inf_410_);
lean_closure_set(v___f_414_, 2, v_gi_396_);
lean_inc(v_l_392_);
v___f_415_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1), 5, 3);
lean_closure_set(v___f_415_, 0, v_toSemilatticeSup_409_);
lean_closure_set(v___f_415_, 1, v_u_393_);
lean_closure_set(v___f_415_, 2, v_l_392_);
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v_inst_394_);
lean_ctor_set(v___x_416_, 1, v___f_415_);
if (v_isShared_413_ == 0)
{
lean_ctor_set(v___x_412_, 1, v___f_414_);
lean_ctor_set(v___x_412_, 0, v___x_416_);
v___x_418_ = v___x_412_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v___x_416_);
lean_ctor_set(v_reuseFailAlloc_425_, 1, v___f_414_);
v___x_418_ = v_reuseFailAlloc_425_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
lean_object* v___x_419_; lean_object* v_toSupSet_420_; lean_object* v_toInfSet_421_; lean_object* v___f_422_; lean_object* v___f_423_; lean_object* v___x_424_; 
v___x_419_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_395_);
v_toSupSet_420_ = lean_ctor_get(v___x_419_, 1);
lean_inc(v_toSupSet_420_);
lean_dec_ref(v___x_419_);
v_toInfSet_421_ = lean_ctor_get(v___x_397_, 1);
lean_inc(v_toInfSet_421_);
lean_dec_ref(v___x_397_);
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__2), 3, 2);
lean_closure_set(v___f_422_, 0, v_toSupSet_420_);
lean_closure_set(v___f_422_, 1, v_l_392_);
v___f_423_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__0), 3, 2);
lean_closure_set(v___f_423_, 0, v_toInfSet_421_);
lean_closure_set(v___f_423_, 1, v_gi_396_);
v___x_424_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_424_, 0, v___x_418_);
lean_ctor_set(v___x_424_, 1, v___f_422_);
lean_ctor_set(v___x_424_, 2, v___f_423_);
lean_ctor_set(v___x_424_, 3, v___x_408_);
return v___x_424_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_liftCompleteLattice(lean_object* v_00_u03b1_429_, lean_object* v_00_u03b2_430_, lean_object* v_l_431_, lean_object* v_u_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_gi_435_){
_start:
{
lean_object* v___x_436_; lean_object* v_toBoundedOrder_437_; lean_object* v_toLattice_438_; lean_object* v_toOrderTop_439_; lean_object* v_toOrderBot_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_467_; 
lean_inc_ref(v_inst_434_);
v___x_436_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_434_);
v_toBoundedOrder_437_ = lean_ctor_get(v_inst_434_, 3);
lean_inc_ref(v_toBoundedOrder_437_);
v_toLattice_438_ = lean_ctor_get(v_inst_434_, 0);
lean_inc_ref(v_toLattice_438_);
v_toOrderTop_439_ = lean_ctor_get(v_toBoundedOrder_437_, 0);
v_toOrderBot_440_ = lean_ctor_get(v_toBoundedOrder_437_, 1);
v_isSharedCheck_467_ = !lean_is_exclusive(v_toBoundedOrder_437_);
if (v_isSharedCheck_467_ == 0)
{
v___x_442_ = v_toBoundedOrder_437_;
v_isShared_443_ = v_isSharedCheck_467_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_toOrderBot_440_);
lean_inc(v_toOrderTop_439_);
lean_dec(v_toBoundedOrder_437_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_467_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_447_; 
lean_inc(v_gi_435_);
v___x_444_ = lean_apply_2(v_gi_435_, v_toOrderTop_439_, lean_box(0));
lean_inc(v_l_431_);
v___x_445_ = lean_apply_1(v_l_431_, v_toOrderBot_440_);
if (v_isShared_443_ == 0)
{
lean_ctor_set(v___x_442_, 1, v___x_445_);
lean_ctor_set(v___x_442_, 0, v___x_444_);
v___x_447_ = v___x_442_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v___x_444_);
lean_ctor_set(v_reuseFailAlloc_466_, 1, v___x_445_);
v___x_447_ = v_reuseFailAlloc_466_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
lean_object* v_toSemilatticeSup_448_; lean_object* v_inf_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_465_; 
v_toSemilatticeSup_448_ = lean_ctor_get(v_toLattice_438_, 0);
v_inf_449_ = lean_ctor_get(v_toLattice_438_, 1);
v_isSharedCheck_465_ = !lean_is_exclusive(v_toLattice_438_);
if (v_isSharedCheck_465_ == 0)
{
v___x_451_ = v_toLattice_438_;
v_isShared_452_ = v_isSharedCheck_465_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_inf_449_);
lean_inc(v_toSemilatticeSup_448_);
lean_dec(v_toLattice_438_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_465_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___f_453_; lean_object* v___f_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
lean_inc(v_gi_435_);
lean_inc(v_u_432_);
v___f_453_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_453_, 0, v_u_432_);
lean_closure_set(v___f_453_, 1, v_inf_449_);
lean_closure_set(v___f_453_, 2, v_gi_435_);
lean_inc(v_l_431_);
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftLattice___redArg___lam__1), 5, 3);
lean_closure_set(v___f_454_, 0, v_toSemilatticeSup_448_);
lean_closure_set(v___f_454_, 1, v_u_432_);
lean_closure_set(v___f_454_, 2, v_l_431_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v_inst_433_);
lean_ctor_set(v___x_455_, 1, v___f_454_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 1, v___f_453_);
lean_ctor_set(v___x_451_, 0, v___x_455_);
v___x_457_ = v___x_451_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v___x_455_);
lean_ctor_set(v_reuseFailAlloc_464_, 1, v___f_453_);
v___x_457_ = v_reuseFailAlloc_464_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
lean_object* v___x_458_; lean_object* v_toSupSet_459_; lean_object* v_toInfSet_460_; lean_object* v___f_461_; lean_object* v___f_462_; lean_object* v___x_463_; 
v___x_458_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_434_);
v_toSupSet_459_ = lean_ctor_get(v___x_458_, 1);
lean_inc(v_toSupSet_459_);
lean_dec_ref(v___x_458_);
v_toInfSet_460_ = lean_ctor_get(v___x_436_, 1);
lean_inc(v_toInfSet_460_);
lean_dec_ref(v___x_436_);
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__2), 3, 2);
lean_closure_set(v___f_461_, 0, v_toSupSet_459_);
lean_closure_set(v___f_461_, 1, v_l_431_);
v___f_462_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_liftCompleteLattice___redArg___lam__0), 3, 2);
lean_closure_set(v___f_462_, 0, v_toInfSet_460_);
lean_closure_set(v___f_462_, 1, v_gi_435_);
v___x_463_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_463_, 0, v___x_457_);
lean_ctor_set(v___x_463_, 1, v___f_461_);
lean_ctor_set(v___x_463_, 2, v___f_462_);
lean_ctor_set(v___x_463_, 3, v___x_447_);
return v___x_463_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__1(lean_object* v_toSupSet_468_, lean_object* v_gi_469_, lean_object* v_s_470_){
_start:
{
lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_471_ = lean_apply_1(v_toSupSet_468_, lean_box(0));
v___x_472_ = lean_apply_2(v_gi_469_, v___x_471_, lean_box(0));
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__0(lean_object* v_toInfSet_473_, lean_object* v_l_474_, lean_object* v_s_475_){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_476_ = lean_apply_1(v_toInfSet_473_, lean_box(0));
v___x_477_ = lean_apply_1(v_l_474_, v___x_476_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg(lean_object* v_l_478_, lean_object* v_u_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_gi_482_){
_start:
{
lean_object* v___x_483_; lean_object* v_toLattice_484_; lean_object* v_toBoundedOrder_485_; lean_object* v___x_486_; lean_object* v_toSemilatticeSup_487_; lean_object* v_inf_488_; lean_object* v___x_490_; uint8_t v_isShared_491_; uint8_t v_isSharedCheck_503_; 
lean_inc_ref(v_inst_481_);
v___x_483_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_481_);
v_toLattice_484_ = lean_ctor_get(v_inst_481_, 0);
lean_inc_ref(v_toLattice_484_);
v_toBoundedOrder_485_ = lean_ctor_get(v_inst_481_, 3);
lean_inc(v_gi_482_);
lean_inc_ref(v_toBoundedOrder_485_);
lean_inc(v_l_478_);
v___x_486_ = lp_mathlib_GaloisCoinsertion_liftBoundedOrder___redArg(v_l_478_, v_toBoundedOrder_485_, v_gi_482_);
v_toSemilatticeSup_487_ = lean_ctor_get(v_toLattice_484_, 0);
v_inf_488_ = lean_ctor_get(v_toLattice_484_, 1);
v_isSharedCheck_503_ = !lean_is_exclusive(v_toLattice_484_);
if (v_isSharedCheck_503_ == 0)
{
v___x_490_ = v_toLattice_484_;
v_isShared_491_ = v_isSharedCheck_503_;
goto v_resetjp_489_;
}
else
{
lean_inc(v_inf_488_);
lean_inc(v_toSemilatticeSup_487_);
lean_dec(v_toLattice_484_);
v___x_490_ = lean_box(0);
v_isShared_491_ = v_isSharedCheck_503_;
goto v_resetjp_489_;
}
v_resetjp_489_:
{
lean_object* v___f_492_; lean_object* v___x_493_; lean_object* v___x_495_; 
lean_inc(v_l_478_);
lean_inc(v_u_479_);
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftLattice___redArg___lam__0), 5, 3);
lean_closure_set(v___f_492_, 0, v_u_479_);
lean_closure_set(v___f_492_, 1, v_inf_488_);
lean_closure_set(v___f_492_, 2, v_l_478_);
lean_inc(v_gi_482_);
v___x_493_ = lp_mathlib_GaloisCoinsertion_liftSemilatticeSup___redArg(v_u_479_, v_inst_480_, v_toSemilatticeSup_487_, v_gi_482_);
if (v_isShared_491_ == 0)
{
lean_ctor_set(v___x_490_, 1, v___f_492_);
lean_ctor_set(v___x_490_, 0, v___x_493_);
v___x_495_ = v___x_490_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_493_);
lean_ctor_set(v_reuseFailAlloc_502_, 1, v___f_492_);
v___x_495_ = v_reuseFailAlloc_502_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
lean_object* v_toSupSet_496_; lean_object* v___x_497_; lean_object* v_toInfSet_498_; lean_object* v___f_499_; lean_object* v___f_500_; lean_object* v___x_501_; 
v_toSupSet_496_ = lean_ctor_get(v___x_483_, 1);
lean_inc(v_toSupSet_496_);
lean_dec_ref(v___x_483_);
v___x_497_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_481_);
v_toInfSet_498_ = lean_ctor_get(v___x_497_, 1);
lean_inc(v_toInfSet_498_);
lean_dec_ref(v___x_497_);
v___f_499_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__1), 3, 2);
lean_closure_set(v___f_499_, 0, v_toSupSet_496_);
lean_closure_set(v___f_499_, 1, v_gi_482_);
v___f_500_ = lean_alloc_closure((void*)(lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg___lam__0), 3, 2);
lean_closure_set(v___f_500_, 0, v_toInfSet_498_);
lean_closure_set(v___f_500_, 1, v_l_478_);
v___x_501_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_501_, 0, v___x_495_);
lean_ctor_set(v___x_501_, 1, v___f_499_);
lean_ctor_set(v___x_501_, 2, v___f_500_);
lean_ctor_set(v___x_501_, 3, v___x_486_);
return v___x_501_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisCoinsertion_liftCompleteLattice(lean_object* v_00_u03b1_504_, lean_object* v_00_u03b2_505_, lean_object* v_l_506_, lean_object* v_u_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_gi_510_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lp_mathlib_GaloisCoinsertion_liftCompleteLattice___redArg(v_l_506_, v_u_507_, v_inst_508_, v_inst_509_, v_gi_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giSSupIic___redArg(lean_object* v_inst_512_){
_start:
{
lean_object* v_toSupSet_513_; lean_object* v___f_514_; 
v_toSupSet_513_ = lean_ctor_get(v_inst_512_, 1);
lean_inc(v_toSupSet_513_);
lean_dec_ref(v_inst_512_);
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_514_, 0, v_toSupSet_513_);
return v___f_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giSSupIic(lean_object* v_00_u03b1_515_, lean_object* v_inst_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lp_mathlib_giSSupIic___redArg(v_inst_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gi__sSup__Iic___redArg(lean_object* v_inst_518_){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lp_mathlib_giSSupIic___redArg(v_inst_518_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gi__sSup__Iic(lean_object* v_00_u03b1_520_, lean_object* v_inst_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_giSSupIic___redArg(v_inst_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciIciSInf___redArg(lean_object* v_inst_523_){
_start:
{
lean_object* v_toInfSet_524_; lean_object* v___x_525_; lean_object* v___f_526_; 
v_toInfSet_524_ = lean_ctor_get(v_inst_523_, 1);
lean_inc(v_toInfSet_524_);
lean_dec_ref(v_inst_523_);
v___x_525_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_525_, 0, lean_box(0));
lean_closure_set(v___x_525_, 1, lean_box(0));
lean_closure_set(v___x_525_, 2, lean_box(0));
lean_closure_set(v___x_525_, 3, v_toInfSet_524_);
lean_closure_set(v___x_525_, 4, lean_box(0));
v___f_526_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_526_, 0, v___x_525_);
return v___f_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciIciSInf(lean_object* v_00_u03b1_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_gciIciSInf___redArg(v_inst_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gci__Ici__sInf___redArg(lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_gciIciSInf___redArg(v_inst_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gci__Ici__sInf(lean_object* v_00_u03b1_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib_gciIciSInf___redArg(v_inst_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0(lean_object* v_inst_535_, lean_object* v_o_536_, lean_object* v_x_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_mathlib_WithBot_unbotD___redArg(v_inst_535_, v_o_536_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0___boxed(lean_object* v_inst_539_, lean_object* v_o_540_, lean_object* v_x_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0(v_inst_539_, v_o_540_, v_x_541_);
lean_dec(v_inst_539_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___redArg(lean_object* v_inst_543_){
_start:
{
lean_object* v___f_544_; 
v___f_544_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_544_, 0, v_inst_543_);
return v___f_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot(lean_object* v_00_u03b1_545_, lean_object* v_inst_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v___f_548_; 
v___f_548_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_giUnbotDBot___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_548_, 0, v_inst_547_);
return v___f_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_giUnbotDBot___boxed(lean_object* v_00_u03b1_549_, lean_object* v_inst_550_, lean_object* v_inst_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_mathlib_WithBot_giUnbotDBot(v_00_u03b1_549_, v_inst_550_, v_inst_551_);
lean_dec_ref(v_inst_550_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg___lam__0(lean_object* v_inst_553_, lean_object* v_o_554_, lean_object* v_x_555_){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lp_mathlib_WithTop_untopD___redArg(v_inst_553_, v_o_554_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg___lam__0___boxed(lean_object* v_inst_557_, lean_object* v_o_558_, lean_object* v_x_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_WithTop_giUntopDTop___redArg___lam__0(v_inst_557_, v_o_558_, v_x_559_);
lean_dec(v_inst_557_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___redArg(lean_object* v_inst_561_){
_start:
{
lean_object* v___f_562_; 
v___f_562_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_giUntopDTop___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_562_, 0, v_inst_561_);
return v___f_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop(lean_object* v_00_u03b1_563_, lean_object* v_inst_564_, lean_object* v_inst_565_){
_start:
{
lean_object* v___f_566_; 
v___f_566_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_giUntopDTop___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_566_, 0, v_inst_565_);
return v___f_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_giUntopDTop___boxed(lean_object* v_00_u03b1_567_, lean_object* v_inst_568_, lean_object* v_inst_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_WithTop_giUntopDTop(v_00_u03b1_567_, v_inst_568_, v_inst_569_);
lean_dec_ref(v_inst_568_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_toGaloisCoinsertion___at___00gciMapBicompl_spec__0(lean_object* v_00_u03b3_571_, lean_object* v_00_u03b4_572_, lean_object* v_00_u03b1_573_, lean_object* v_00_u03b2_574_, lean_object* v_l_575_, lean_object* v_u_576_, lean_object* v_gc_577_, lean_object* v_h_578_){
_start:
{
lean_object* v___x_579_; 
v___x_579_ = lean_box(0);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciMapBicompl(lean_object* v_00_u03b1_580_, lean_object* v_00_u03b2_581_, lean_object* v_00_u03b3_582_, lean_object* v_00_u03b4_583_, lean_object* v_f_584_, lean_object* v_g_585_, lean_object* v_hf_586_, lean_object* v_hg_587_){
_start:
{
lean_object* v___x_588_; 
v___x_588_ = lean_box(0);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciMapBicompl___boxed(lean_object* v_00_u03b1_589_, lean_object* v_00_u03b2_590_, lean_object* v_00_u03b3_591_, lean_object* v_00_u03b4_592_, lean_object* v_f_593_, lean_object* v_g_594_, lean_object* v_hf_595_, lean_object* v_hg_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_gciMapBicompl(v_00_u03b1_589_, v_00_u03b2_590_, v_00_u03b3_591_, v_00_u03b4_592_, v_f_593_, v_g_594_, v_hf_595_, v_hg_596_);
lean_dec(v_g_594_);
lean_dec(v_f_593_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciMapOnFun(lean_object* v_00_u03b1_598_, lean_object* v_00_u03b2_599_, lean_object* v_f_600_, lean_object* v_hf_601_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lean_box(0);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_gciMapOnFun___boxed(lean_object* v_00_u03b1_603_, lean_object* v_00_u03b2_604_, lean_object* v_f_605_, lean_object* v_hf_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_mathlib_gciMapOnFun(v_00_u03b1_603_, v_00_u03b2_604_, v_f_605_, v_hf_606_);
lean_dec(v_f_605_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisConnection_toGaloisInsertion___at___00giMapBicompl_spec__0(lean_object* v_00_u03b1_608_, lean_object* v_00_u03b2_609_, lean_object* v_00_u03b3_610_, lean_object* v_00_u03b4_611_, lean_object* v_l_612_, lean_object* v_u_613_, lean_object* v_gc_614_, lean_object* v_h_615_){
_start:
{
lean_object* v___x_616_; 
v___x_616_ = lean_box(0);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giMapBicompl(lean_object* v_00_u03b1_617_, lean_object* v_00_u03b2_618_, lean_object* v_00_u03b3_619_, lean_object* v_00_u03b4_620_, lean_object* v_f_621_, lean_object* v_g_622_, lean_object* v_hf_623_, lean_object* v_hg_624_){
_start:
{
lean_object* v___x_625_; 
v___x_625_ = lean_box(0);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giMapBicompl___boxed(lean_object* v_00_u03b1_626_, lean_object* v_00_u03b2_627_, lean_object* v_00_u03b3_628_, lean_object* v_00_u03b4_629_, lean_object* v_f_630_, lean_object* v_g_631_, lean_object* v_hf_632_, lean_object* v_hg_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_giMapBicompl(v_00_u03b1_626_, v_00_u03b2_627_, v_00_u03b3_628_, v_00_u03b4_629_, v_f_630_, v_g_631_, v_hf_632_, v_hg_633_);
lean_dec(v_g_631_);
lean_dec(v_f_630_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giMapOnFun(lean_object* v_00_u03b1_635_, lean_object* v_00_u03b2_636_, lean_object* v_f_637_, lean_object* v_hf_638_){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = lean_box(0);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_giMapOnFun___boxed(lean_object* v_00_u03b1_640_, lean_object* v_00_u03b2_641_, lean_object* v_f_642_, lean_object* v_hf_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_giMapOnFun(v_00_u03b1_640_, v_00_u03b2_641_, v_f_642_, v_hf_643_);
lean_dec(v_f_642_);
return v_res_644_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_GaloisConnection_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
