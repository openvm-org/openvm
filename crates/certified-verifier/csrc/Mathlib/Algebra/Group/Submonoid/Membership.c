// Lean compiler output
// Module: Mathlib.Algebra.Group.Submonoid.Membership
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Group.Multiset.Defs public import Mathlib.Algebra.FreeMonoid.Basic public import Mathlib.Algebra.Group.Idempotent public import Mathlib.Algebra.Group.Nat.Hom public import Mathlib.Algebra.Group.Submonoid.MulOpposite public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.Int.Basic public import Mathlib.Algebra.Group.Int.Defs
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Int_natMod(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_powersHom___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_findX___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powers(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powers___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submonoid_pow___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submonoid_pow___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_log___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_multiples(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_multiples___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powers(lean_object* v_M_1_, lean_object* v_inst_2_, lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powers___boxed(lean_object* v_M_5_, lean_object* v_inst_6_, lean_object* v_n_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Submonoid_powers(v_M_5_, v_inst_6_, v_n_7_);
lean_dec(v_n_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__0(lean_object* v_inst_9_, lean_object* v_n_10_, lean_object* v_x_11_){
_start:
{
lean_object* v_toNPow_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_toNPow_12_ = lean_ctor_get(v_inst_9_, 2);
lean_inc(v_toNPow_12_);
lean_dec_ref(v_inst_9_);
v___x_13_ = lean_unsigned_to_nat(1u);
v___x_14_ = lean_nat_sub(v_n_10_, v___x_13_);
v___x_15_ = lean_apply_2(v_toNPow_12_, v___x_14_, v_x_11_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__0___boxed(lean_object* v_inst_16_, lean_object* v_n_17_, lean_object* v_x_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Submonoid_groupPowers___redArg___lam__0(v_inst_16_, v_n_17_, v_x_18_);
lean_dec(v_n_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__1(lean_object* v_inst_20_, lean_object* v_n_21_, lean_object* v_z_22_, lean_object* v_x_23_){
_start:
{
lean_object* v_toNPow_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v_toNPow_24_ = lean_ctor_get(v_inst_20_, 2);
lean_inc(v_toNPow_24_);
lean_dec_ref(v_inst_20_);
v___x_25_ = lean_nat_to_int(v_n_21_);
v___x_26_ = lp_mathlib_Int_natMod(v_z_22_, v___x_25_);
lean_dec(v___x_25_);
v___x_27_ = lean_apply_2(v_toNPow_24_, v___x_26_, v_x_23_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg___lam__1___boxed(lean_object* v_inst_28_, lean_object* v_n_29_, lean_object* v_z_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Submonoid_groupPowers___redArg___lam__1(v_inst_28_, v_n_29_, v_z_30_, v_x_31_);
lean_dec(v_z_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___redArg(lean_object* v_inst_33_, lean_object* v_n_34_){
_start:
{
lean_object* v___f_35_; lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
lean_inc(v_n_34_);
lean_inc_ref_n(v_inst_33_, 2);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_groupPowers___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_35_, 0, v_inst_33_);
lean_closure_set(v___f_35_, 1, v_n_34_);
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_groupPowers___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_36_, 0, v_inst_33_);
lean_closure_set(v___f_36_, 1, v_n_34_);
v___x_37_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_33_);
lean_inc_ref(v___f_35_);
lean_inc_ref(v___x_37_);
v___x_38_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_38_, 0, lean_box(0));
lean_closure_set(v___x_38_, 1, v___x_37_);
lean_closure_set(v___x_38_, 2, v___f_35_);
v___x_39_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_39_, 0, v___x_37_);
lean_ctor_set(v___x_39_, 1, v___f_35_);
lean_ctor_set(v___x_39_, 2, v___x_38_);
lean_ctor_set(v___x_39_, 3, v___f_36_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers(lean_object* v_M_40_, lean_object* v_inst_41_, lean_object* v_x_42_, lean_object* v_n_43_, lean_object* v_hpos_44_, lean_object* v_hx_45_){
_start:
{
lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
lean_inc(v_n_43_);
lean_inc_ref_n(v_inst_41_, 2);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_groupPowers___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_46_, 0, v_inst_41_);
lean_closure_set(v___f_46_, 1, v_n_43_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_groupPowers___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_47_, 0, v_inst_41_);
lean_closure_set(v___f_47_, 1, v_n_43_);
v___x_48_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_inst_41_);
lean_inc_ref(v___f_46_);
lean_inc_ref(v___x_48_);
v___x_49_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_49_, 0, lean_box(0));
lean_closure_set(v___x_49_, 1, v___x_48_);
lean_closure_set(v___x_49_, 2, v___f_46_);
v___x_50_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_50_, 0, v___x_48_);
lean_ctor_set(v___x_50_, 1, v___f_46_);
lean_ctor_set(v___x_50_, 2, v___x_49_);
lean_ctor_set(v___x_50_, 3, v___f_47_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_groupPowers___boxed(lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_x_53_, lean_object* v_n_54_, lean_object* v_hpos_55_, lean_object* v_hx_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_Submonoid_groupPowers(v_M_51_, v_inst_52_, v_x_53_, v_n_54_, v_hpos_55_, v_hx_56_);
lean_dec(v_x_53_);
return v_res_57_;
}
}
static lean_object* _init_lp_mathlib_Submonoid_pow___redArg___closed__0(void){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pow___redArg(lean_object* v_inst_59_, lean_object* v_n_60_, lean_object* v_m_61_){
_start:
{
lean_object* v___x_62_; lean_object* v_toFun_63_; lean_object* v___x_64_; lean_object* v_toFun_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_62_ = lp_mathlib_powersHom___redArg(v_inst_59_);
v_toFun_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_toFun_63_);
lean_dec_ref(v___x_62_);
v___x_64_ = lean_obj_once(&lp_mathlib_Submonoid_pow___redArg___closed__0, &lp_mathlib_Submonoid_pow___redArg___closed__0_once, _init_lp_mathlib_Submonoid_pow___redArg___closed__0);
v_toFun_65_ = lean_ctor_get(v___x_64_, 0);
v___x_66_ = lean_apply_1(v_toFun_63_, v_n_60_);
lean_inc(v_toFun_65_);
v___x_67_ = lean_apply_1(v_toFun_65_, v_m_61_);
v___x_68_ = lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(v___x_66_, v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pow(lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_n_71_, lean_object* v_m_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Submonoid_pow___redArg(v_inst_70_, v_n_71_, v_m_72_);
return v___x_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Submonoid_log___redArg___lam__0(lean_object* v_toNPow_74_, lean_object* v_n_75_, lean_object* v_inst_76_, lean_object* v_p_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; uint8_t v___x_81_; 
v___x_79_ = lean_apply_2(v_toNPow_74_, v_a_78_, v_n_75_);
v___x_80_ = lean_apply_2(v_inst_76_, v___x_79_, v_p_77_);
v___x_81_ = lean_unbox(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log___redArg___lam__0___boxed(lean_object* v_toNPow_82_, lean_object* v_n_83_, lean_object* v_inst_84_, lean_object* v_p_85_, lean_object* v_a_86_){
_start:
{
uint8_t v_res_87_; lean_object* v_r_88_; 
v_res_87_ = lp_mathlib_Submonoid_log___redArg___lam__0(v_toNPow_82_, v_n_83_, v_inst_84_, v_p_85_, v_a_86_);
v_r_88_ = lean_box(v_res_87_);
return v_r_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log___redArg(lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_n_91_, lean_object* v_p_92_){
_start:
{
lean_object* v_toNPow_93_; lean_object* v___f_94_; lean_object* v___x_95_; 
v_toNPow_93_ = lean_ctor_get(v_inst_89_, 2);
lean_inc(v_toNPow_93_);
lean_dec_ref(v_inst_89_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_log___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_94_, 0, v_toNPow_93_);
lean_closure_set(v___f_94_, 1, v_n_91_);
lean_closure_set(v___f_94_, 2, v_inst_90_);
lean_closure_set(v___f_94_, 3, v_p_92_);
v___x_95_ = lp_mathlib_Nat_findX___redArg(v___f_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_log(lean_object* v_M_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_n_99_, lean_object* v_p_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Submonoid_log___redArg(v_inst_97_, v_inst_98_, v_n_99_, v_p_100_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0(lean_object* v_inst_103_, lean_object* v_n_104_, lean_object* v_m_105_){
_start:
{
lean_object* v___x_106_; lean_object* v_toFun_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0, &lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0_once, _init_lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0___closed__0);
v_toFun_107_ = lean_ctor_get(v___x_106_, 0);
lean_inc(v_toFun_107_);
v___x_108_ = lean_apply_1(v_toFun_107_, v_m_105_);
v___x_109_ = lp_mathlib_Submonoid_pow___redArg(v_inst_103_, v_n_104_, v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg___lam__1(lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_n_112_, lean_object* v_m_113_){
_start:
{
lean_object* v___x_114_; lean_object* v_toFun_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_114_ = lean_obj_once(&lp_mathlib_Submonoid_pow___redArg___closed__0, &lp_mathlib_Submonoid_pow___redArg___closed__0_once, _init_lp_mathlib_Submonoid_pow___redArg___closed__0);
v_toFun_115_ = lean_ctor_get(v___x_114_, 0);
v___x_116_ = lp_mathlib_Submonoid_log___redArg(v_inst_110_, v_inst_111_, v_n_112_, v_m_113_);
lean_inc(v_toFun_115_);
v___x_117_ = lean_apply_1(v_toFun_115_, v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv___redArg(lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_n_120_){
_start:
{
lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___x_123_; 
lean_inc(v_n_120_);
lean_inc_ref(v_inst_118_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_powLogEquiv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_121_, 0, v_inst_118_);
lean_closure_set(v___f_121_, 1, v_n_120_);
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_powLogEquiv___redArg___lam__1), 4, 3);
lean_closure_set(v___f_122_, 0, v_inst_118_);
lean_closure_set(v___f_122_, 1, v_inst_119_);
lean_closure_set(v___f_122_, 2, v_n_120_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___f_121_);
lean_ctor_set(v___x_123_, 1, v___f_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_powLogEquiv(lean_object* v_M_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_n_127_, lean_object* v_h_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_Submonoid_powLogEquiv___redArg(v_inst_125_, v_inst_126_, v_n_127_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_multiples(lean_object* v_A_130_, lean_object* v_inst_131_, lean_object* v_x_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lean_box(0);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_multiples___boxed(lean_object* v_A_134_, lean_object* v_inst_135_, lean_object* v_x_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_AddSubmonoid_multiples(v_A_134_, v_inst_135_, v_x_136_);
lean_dec(v_x_136_);
lean_dec_ref(v_inst_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0(lean_object* v_inst_138_, lean_object* v_n_139_, lean_object* v_x_140_){
_start:
{
lean_object* v_toNSMul_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_toNSMul_141_ = lean_ctor_get(v_inst_138_, 2);
lean_inc(v_toNSMul_141_);
lean_dec_ref(v_inst_138_);
v___x_142_ = lean_unsigned_to_nat(1u);
v___x_143_ = lean_nat_sub(v_n_139_, v___x_142_);
v___x_144_ = lean_apply_2(v_toNSMul_141_, v___x_143_, v_x_140_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0___boxed(lean_object* v_inst_145_, lean_object* v_n_146_, lean_object* v_x_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0(v_inst_145_, v_n_146_, v_x_147_);
lean_dec(v_n_146_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1(lean_object* v_inst_149_, lean_object* v_n_150_, lean_object* v_z_151_, lean_object* v_x_152_){
_start:
{
lean_object* v_toNSMul_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v_toNSMul_153_ = lean_ctor_get(v_inst_149_, 2);
lean_inc(v_toNSMul_153_);
lean_dec_ref(v_inst_149_);
v___x_154_ = lean_nat_to_int(v_n_150_);
v___x_155_ = lp_mathlib_Int_natMod(v_z_151_, v___x_154_);
lean_dec(v___x_154_);
v___x_156_ = lean_apply_2(v_toNSMul_153_, v___x_155_, v_x_152_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1___boxed(lean_object* v_inst_157_, lean_object* v_n_158_, lean_object* v_z_159_, lean_object* v_x_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1(v_inst_157_, v_n_158_, v_z_159_, v_x_160_);
lean_dec(v_z_159_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___redArg(lean_object* v_inst_162_, lean_object* v_n_163_){
_start:
{
lean_object* v___f_164_; lean_object* v___f_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
lean_inc(v_n_163_);
lean_inc_ref_n(v_inst_162_, 2);
v___f_164_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_164_, 0, v_inst_162_);
lean_closure_set(v___f_164_, 1, v_n_163_);
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_addGroupMultiples___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_165_, 0, v_inst_162_);
lean_closure_set(v___f_165_, 1, v_n_163_);
v___x_166_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_inst_162_);
lean_inc_ref(v___f_164_);
lean_inc_ref(v___x_166_);
v___x_167_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_167_, 0, lean_box(0));
lean_closure_set(v___x_167_, 1, v___x_166_);
lean_closure_set(v___x_167_, 2, v___f_164_);
v___x_168_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_168_, 0, v___x_166_);
lean_ctor_set(v___x_168_, 1, v___f_164_);
lean_ctor_set(v___x_168_, 2, v___x_167_);
lean_ctor_set(v___x_168_, 3, v___f_165_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples(lean_object* v_M_169_, lean_object* v_inst_170_, lean_object* v_x_171_, lean_object* v_n_172_, lean_object* v_hpos_173_, lean_object* v_hx_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_AddSubmonoid_addGroupMultiples___redArg(v_inst_170_, v_n_172_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_addGroupMultiples___boxed(lean_object* v_M_176_, lean_object* v_inst_177_, lean_object* v_x_178_, lean_object* v_n_179_, lean_object* v_hpos_180_, lean_object* v_hx_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_AddSubmonoid_addGroupMultiples(v_M_176_, v_inst_177_, v_x_178_, v_n_179_, v_hpos_180_, v_hx_181_);
lean_dec(v_x_178_);
return v_res_182_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Idempotent(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Idempotent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_FreeMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
}
#ifdef __cplusplus
}
#endif
