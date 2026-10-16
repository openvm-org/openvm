// Lean compiler output
// Module: Mathlib.Data.Finset.Powerset
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Card public import Mathlib.Data.Finset.Lattice.Union public import Mathlib.Data.Multiset.Powerset public import Mathlib.Data.Set.Pairwise.Lattice
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
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_powersetAux___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_erase___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDexistsMultiset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_powersetCardAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_powerset___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_powerset___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_powerset___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_powerset___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_ssubsets___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powersetCard___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powersetCard(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg___lam__0(lean_object* v_val_1_, lean_object* v_nodup_2_){
_start:
{
lean_inc(v_val_1_);
return v_val_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg___lam__0___boxed(lean_object* v_val_3_, lean_object* v_nodup_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Finset_powerset___redArg___lam__0(v_val_3_, v_nodup_4_);
lean_dec(v_val_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset___redArg(lean_object* v_s_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = ((lean_object*)(lp_mathlib_Finset_powerset___redArg___closed__0));
v___x_9_ = lp_mathlib_Multiset_powersetAux___redArg(v_s_7_);
v___x_10_ = lp_mathlib_Multiset_pmap___redArg(v___f_8_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_powerset(lean_object* v_00_u03b1_11_, lean_object* v_s_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Finset_powerset___redArg(v_s_12_);
return v___x_13_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0(lean_object* v_inst_14_, lean_object* v_a_15_, lean_object* v_h_16_){
_start:
{
lean_object* v___x_17_; uint8_t v___x_18_; 
v___x_17_ = lean_apply_2(v_inst_14_, v_a_15_, lean_box(0));
v___x_18_ = lean_unbox(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed(lean_object* v_inst_19_, lean_object* v_a_20_, lean_object* v_h_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0(v_inst_19_, v_a_20_, v_h_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg(lean_object* v_s_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___f_26_; lean_object* v___x_27_; uint8_t v___x_28_; 
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_26_, 0, v_inst_25_);
v___x_27_ = lp_mathlib_Finset_powerset___redArg(v_s_24_);
v___x_28_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v___x_27_, v___f_26_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___boxed(lean_object* v_s_29_, lean_object* v_inst_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg(v_s_29_, v_inst_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets(lean_object* v_00_u03b1_33_, lean_object* v_s_34_, lean_object* v_p_35_, lean_object* v_inst_36_){
_start:
{
uint8_t v___x_37_; 
v___x_37_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg(v_s_34_, v_inst_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets___boxed(lean_object* v_00_u03b1_38_, lean_object* v_s_39_, lean_object* v_p_40_, lean_object* v_inst_41_){
_start:
{
uint8_t v_res_42_; lean_object* v_r_43_; 
v_res_42_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets(v_00_u03b1_38_, v_s_39_, v_p_40_, v_inst_41_);
v_r_43_ = lean_box(v_res_42_);
return v_r_43_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg(lean_object* v_s_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; lean_object* v___x_47_; uint8_t v___x_48_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_46_, 0, v_inst_45_);
v___x_47_ = lp_mathlib_Finset_powerset___redArg(v_s_44_);
v___x_48_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v___x_47_, v___f_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg___boxed(lean_object* v_s_49_, lean_object* v_inst_50_){
_start:
{
uint8_t v_res_51_; lean_object* v_r_52_; 
v_res_51_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg(v_s_49_, v_inst_50_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets(lean_object* v_00_u03b1_53_, lean_object* v_s_54_, lean_object* v_p_55_, lean_object* v_inst_56_){
_start:
{
uint8_t v___x_57_; 
v___x_57_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg(v_s_54_, v_inst_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets___boxed(lean_object* v_00_u03b1_58_, lean_object* v_s_59_, lean_object* v_p_60_, lean_object* v_inst_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets(v_00_u03b1_58_, v_s_59_, v_p_60_, v_inst_61_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0(lean_object* v_inst_64_, lean_object* v_t_65_, lean_object* v_h_66_){
_start:
{
lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_67_ = lean_apply_1(v_inst_64_, v_t_65_);
v___x_68_ = lean_unbox(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0___boxed(lean_object* v_inst_69_, lean_object* v_t_70_, lean_object* v_h_71_){
_start:
{
uint8_t v_res_72_; lean_object* v_r_73_; 
v_res_72_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0(v_inst_69_, v_t_70_, v_h_71_);
v_r_73_ = lean_box(v_res_72_);
return v_r_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg(lean_object* v_s_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v___f_76_; uint8_t v___x_77_; 
v___f_76_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_76_, 0, v_inst_75_);
v___x_77_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg(v_s_74_, v___f_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___boxed(lean_object* v_s_78_, lean_object* v_inst_79_){
_start:
{
uint8_t v_res_80_; lean_object* v_r_81_; 
v_res_80_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg(v_s_78_, v_inst_79_);
v_r_81_ = lean_box(v_res_80_);
return v_r_81_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27(lean_object* v_00_u03b1_82_, lean_object* v_s_83_, lean_object* v_p_84_, lean_object* v_inst_85_){
_start:
{
uint8_t v___x_86_; 
v___x_86_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg(v_s_83_, v_inst_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___boxed(lean_object* v_00_u03b1_87_, lean_object* v_s_88_, lean_object* v_p_89_, lean_object* v_inst_90_){
_start:
{
uint8_t v_res_91_; lean_object* v_r_92_; 
v_res_91_ = lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27(v_00_u03b1_87_, v_s_88_, v_p_89_, v_inst_90_);
v_r_92_ = lean_box(v_res_91_);
return v_r_92_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg(lean_object* v_s_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; uint8_t v___x_96_; 
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets_x27___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_95_, 0, v_inst_94_);
v___x_96_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets___redArg(v_s_93_, v___f_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg___boxed(lean_object* v_s_97_, lean_object* v_inst_98_){
_start:
{
uint8_t v_res_99_; lean_object* v_r_100_; 
v_res_99_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg(v_s_97_, v_inst_98_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27(lean_object* v_00_u03b1_101_, lean_object* v_s_102_, lean_object* v_p_103_, lean_object* v_inst_104_){
_start:
{
uint8_t v___x_105_; 
v___x_105_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___redArg(v_s_102_, v_inst_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27___boxed(lean_object* v_00_u03b1_106_, lean_object* v_s_107_, lean_object* v_p_108_, lean_object* v_inst_109_){
_start:
{
uint8_t v_res_110_; lean_object* v_r_111_; 
v_res_110_ = lp_mathlib_Finset_decidableForallOfDecidableSubsets_x27(v_00_u03b1_106_, v_s_107_, v_p_108_, v_inst_109_);
v_r_111_ = lean_box(v_res_110_);
return v_r_111_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_ssubsets___redArg___lam__0(lean_object* v_inst_112_, lean_object* v_a_113_, lean_object* v_b_114_){
_start:
{
uint8_t v___x_115_; 
v___x_115_ = l_List_decidablePerm___redArg(v_inst_112_, v_a_113_, v_b_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets___redArg___lam__0___boxed(lean_object* v_inst_116_, lean_object* v_a_117_, lean_object* v_b_118_){
_start:
{
uint8_t v_res_119_; lean_object* v_r_120_; 
v_res_119_ = lp_mathlib_Finset_ssubsets___redArg___lam__0(v_inst_116_, v_a_117_, v_b_118_);
v_r_120_ = lean_box(v_res_119_);
return v_r_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets___redArg(lean_object* v_inst_121_, lean_object* v_s_122_){
_start:
{
lean_object* v___f_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_Finset_ssubsets___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_123_, 0, v_inst_121_);
lean_inc(v_s_122_);
v___x_124_ = lp_mathlib_Finset_powerset___redArg(v_s_122_);
v___x_125_ = lp_mathlib_Multiset_erase___redArg(v___f_123_, v___x_124_, v_s_122_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_ssubsets(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_s_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_Finset_ssubsets___redArg(v_inst_127_, v_s_128_);
return v___x_129_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(lean_object* v_inst_130_, lean_object* v_s_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___f_133_; lean_object* v___x_134_; uint8_t v___x_135_; 
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_133_, 0, v_inst_132_);
v___x_134_ = lp_mathlib_Finset_ssubsets___redArg(v_inst_130_, v_s_131_);
v___x_135_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v___x_134_, v___f_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg___boxed(lean_object* v_inst_136_, lean_object* v_s_137_, lean_object* v_inst_138_){
_start:
{
uint8_t v_res_139_; lean_object* v_r_140_; 
v_res_139_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(v_inst_136_, v_s_137_, v_inst_138_);
v_r_140_ = lean_box(v_res_139_);
return v_r_140_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets(lean_object* v_00_u03b1_141_, lean_object* v_inst_142_, lean_object* v_s_143_, lean_object* v_p_144_, lean_object* v_inst_145_){
_start:
{
uint8_t v___x_146_; 
v___x_146_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(v_inst_142_, v_s_143_, v_inst_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___boxed(lean_object* v_00_u03b1_147_, lean_object* v_inst_148_, lean_object* v_s_149_, lean_object* v_p_150_, lean_object* v_inst_151_){
_start:
{
uint8_t v_res_152_; lean_object* v_r_153_; 
v_res_152_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets(v_00_u03b1_147_, v_inst_148_, v_s_149_, v_p_150_, v_inst_151_);
v_r_153_ = lean_box(v_res_152_);
return v_r_153_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(lean_object* v_inst_154_, lean_object* v_s_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v___f_157_; lean_object* v___x_158_; uint8_t v___x_159_; 
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsOfDecidableSubsets___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_157_, 0, v_inst_156_);
v___x_158_ = lp_mathlib_Finset_ssubsets___redArg(v_inst_154_, v_s_155_);
v___x_159_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v___x_158_, v___f_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg___boxed(lean_object* v_inst_160_, lean_object* v_s_161_, lean_object* v_inst_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(v_inst_160_, v_s_161_, v_inst_162_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets(lean_object* v_00_u03b1_165_, lean_object* v_inst_166_, lean_object* v_s_167_, lean_object* v_p_168_, lean_object* v_inst_169_){
_start:
{
uint8_t v___x_170_; 
v___x_170_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(v_inst_166_, v_s_167_, v_inst_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets___boxed(lean_object* v_00_u03b1_171_, lean_object* v_inst_172_, lean_object* v_s_173_, lean_object* v_p_174_, lean_object* v_inst_175_){
_start:
{
uint8_t v_res_176_; lean_object* v_r_177_; 
v_res_176_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets(v_00_u03b1_171_, v_inst_172_, v_s_173_, v_p_174_, v_inst_175_);
v_r_177_ = lean_box(v_res_176_);
return v_r_177_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___redArg(lean_object* v_inst_178_, lean_object* v_s_179_, lean_object* v_hu_180_){
_start:
{
uint8_t v___x_181_; 
v___x_181_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(v_inst_178_, v_s_179_, v_hu_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___redArg___boxed(lean_object* v_inst_182_, lean_object* v_s_183_, lean_object* v_hu_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___redArg(v_inst_182_, v_s_183_, v_hu_184_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_s_189_, lean_object* v_p_190_, lean_object* v_hu_191_){
_start:
{
uint8_t v___x_192_; 
v___x_192_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets___redArg(v_inst_188_, v_s_189_, v_hu_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27___boxed(lean_object* v_00_u03b1_193_, lean_object* v_inst_194_, lean_object* v_s_195_, lean_object* v_p_196_, lean_object* v_hu_197_){
_start:
{
uint8_t v_res_198_; lean_object* v_r_199_; 
v_res_198_ = lp_mathlib_Finset_decidableExistsOfDecidableSSubsets_x27(v_00_u03b1_193_, v_inst_194_, v_s_195_, v_p_196_, v_hu_197_);
v_r_199_ = lean_box(v_res_198_);
return v_r_199_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___redArg(lean_object* v_inst_200_, lean_object* v_s_201_, lean_object* v_hu_202_){
_start:
{
uint8_t v___x_203_; 
v___x_203_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(v_inst_200_, v_s_201_, v_hu_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___redArg___boxed(lean_object* v_inst_204_, lean_object* v_s_205_, lean_object* v_hu_206_){
_start:
{
uint8_t v_res_207_; lean_object* v_r_208_; 
v_res_207_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___redArg(v_inst_204_, v_s_205_, v_hu_206_);
v_r_208_ = lean_box(v_res_207_);
return v_r_208_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_s_211_, lean_object* v_p_212_, lean_object* v_hu_213_){
_start:
{
uint8_t v___x_214_; 
v___x_214_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets___redArg(v_inst_210_, v_s_211_, v_hu_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27___boxed(lean_object* v_00_u03b1_215_, lean_object* v_inst_216_, lean_object* v_s_217_, lean_object* v_p_218_, lean_object* v_hu_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_mathlib_Finset_decidableForallOfDecidableSSubsets_x27(v_00_u03b1_215_, v_inst_216_, v_s_217_, v_p_218_, v_hu_219_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_powersetCard___redArg(lean_object* v_n_222_, lean_object* v_s_223_){
_start:
{
lean_object* v___f_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___f_224_ = ((lean_object*)(lp_mathlib_Finset_powerset___redArg___closed__0));
v___x_225_ = lp_mathlib_Multiset_powersetCardAux___redArg(v_n_222_, v_s_223_);
v___x_226_ = lp_mathlib_Multiset_pmap___redArg(v___f_224_, v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_powersetCard(lean_object* v_00_u03b1_227_, lean_object* v_n_228_, lean_object* v_s_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_Finset_powersetCard___redArg(v_n_228_, v_s_229_);
return v___x_230_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Lattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Pairwise_Lattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Union(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Pairwise_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
}
#ifdef __cplusplus
}
#endif
