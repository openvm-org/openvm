// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Map
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Lattice public import Mathlib.Algebra.Group.TypeTags.Hom
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__0_value;
static const lean_ctor_object lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__0_value),((lean_object*)&lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__0_value)}};
static const lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOfEquivOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOfEquivOfLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_comapAddSubgroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_comapAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mapSubgroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mapSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapAddSubgroup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapAddSubgroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_subgroupCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_subgroupCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_comap(lean_object* v_G_1_, lean_object* v_inst_2_, lean_object* v_N_3_, lean_object* v_inst_4_, lean_object* v_f_5_, lean_object* v_H_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_comap___boxed(lean_object* v_G_8_, lean_object* v_inst_9_, lean_object* v_N_10_, lean_object* v_inst_11_, lean_object* v_f_12_, lean_object* v_H_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Subgroup_comap(v_G_8_, v_inst_9_, v_N_10_, v_inst_11_, v_f_12_, v_H_13_);
lean_dec(v_f_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_9_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_comap(lean_object* v_G_15_, lean_object* v_inst_16_, lean_object* v_N_17_, lean_object* v_inst_18_, lean_object* v_f_19_, lean_object* v_H_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_comap___boxed(lean_object* v_G_22_, lean_object* v_inst_23_, lean_object* v_N_24_, lean_object* v_inst_25_, lean_object* v_f_26_, lean_object* v_H_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_AddSubgroup_comap(v_G_22_, v_inst_23_, v_N_24_, v_inst_25_, v_f_26_, v_H_27_);
lean_dec(v_f_26_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_23_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_map(lean_object* v_G_29_, lean_object* v_inst_30_, lean_object* v_N_31_, lean_object* v_inst_32_, lean_object* v_f_33_, lean_object* v_H_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_map___boxed(lean_object* v_G_36_, lean_object* v_inst_37_, lean_object* v_N_38_, lean_object* v_inst_39_, lean_object* v_f_40_, lean_object* v_H_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Subgroup_map(v_G_36_, v_inst_37_, v_N_38_, v_inst_39_, v_f_40_, v_H_41_);
lean_dec(v_f_40_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_37_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_map(lean_object* v_G_43_, lean_object* v_inst_44_, lean_object* v_N_45_, lean_object* v_inst_46_, lean_object* v_f_47_, lean_object* v_H_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_box(0);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_map___boxed(lean_object* v_G_50_, lean_object* v_inst_51_, lean_object* v_N_52_, lean_object* v_inst_53_, lean_object* v_f_54_, lean_object* v_H_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_AddSubgroup_map(v_G_50_, v_inst_51_, v_N_52_, v_inst_53_, v_f_54_, v_H_55_);
lean_dec(v_f_54_);
lean_dec_ref(v_inst_53_);
lean_dec_ref(v_inst_51_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOf(lean_object* v_G_57_, lean_object* v_inst_58_, lean_object* v_H_59_, lean_object* v_K_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_box(0);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOf___boxed(lean_object* v_G_62_, lean_object* v_inst_63_, lean_object* v_H_64_, lean_object* v_K_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Subgroup_subgroupOf(v_G_62_, v_inst_63_, v_H_64_, v_K_65_);
lean_dec_ref(v_inst_63_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOf(lean_object* v_G_67_, lean_object* v_inst_68_, lean_object* v_H_69_, lean_object* v_K_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_box(0);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOf___boxed(lean_object* v_G_72_, lean_object* v_inst_73_, lean_object* v_H_74_, lean_object* v_K_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_AddSubgroup_addSubgroupOf(v_G_72_, v_inst_73_, v_H_74_, v_K_75_);
lean_dec_ref(v_inst_73_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0(lean_object* v_g_77_){
_start:
{
lean_inc(v_g_77_);
return v_g_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0___boxed(lean_object* v_g_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_mathlib_Subgroup_subgroupOfEquivOfLe___lam__0(v_g_78_);
lean_dec(v_g_78_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe(lean_object* v_G_83_, lean_object* v_inst_84_, lean_object* v_H_85_, lean_object* v_K_86_, lean_object* v_h_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = ((lean_object*)(lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__1));
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subgroupOfEquivOfLe___boxed(lean_object* v_G_89_, lean_object* v_inst_90_, lean_object* v_H_91_, lean_object* v_K_92_, lean_object* v_h_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Subgroup_subgroupOfEquivOfLe(v_G_89_, v_inst_90_, v_H_91_, v_K_92_, v_h_93_);
lean_dec_ref(v_inst_90_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOfEquivOfLe(lean_object* v_G_95_, lean_object* v_inst_96_, lean_object* v_H_97_, lean_object* v_K_98_, lean_object* v_h_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = ((lean_object*)(lp_mathlib_Subgroup_subgroupOfEquivOfLe___closed__1));
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_addSubgroupOfEquivOfLe___boxed(lean_object* v_G_101_, lean_object* v_inst_102_, lean_object* v_H_103_, lean_object* v_K_104_, lean_object* v_h_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_AddSubgroup_addSubgroupOfEquivOfLe(v_G_101_, v_inst_102_, v_H_103_, v_K_104_, v_h_105_);
lean_dec_ref(v_inst_102_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0(lean_object* v_f_107_, lean_object* v___y_108_){
_start:
{
lean_object* v_toFun_109_; lean_object* v___x_110_; 
v_toFun_109_ = lean_ctor_get(v_f_107_, 0);
lean_inc(v_toFun_109_);
lean_dec_ref(v_f_107_);
v___x_110_ = lean_apply_1(v_toFun_109_, v___y_108_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1(lean_object* v___x_111_, lean_object* v___y_112_){
_start:
{
lean_object* v_toFun_113_; lean_object* v___x_114_; 
v_toFun_113_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_toFun_113_);
lean_dec_ref(v___x_111_);
v___x_114_ = lean_apply_1(v_toFun_113_, v___y_112_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup___redArg(lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_f_117_){
_start:
{
lean_object* v___f_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___f_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
lean_inc_ref(v_f_117_);
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_118_, 0, v_f_117_);
lean_inc_ref(v_inst_116_);
lean_inc_ref(v_inst_115_);
v___x_119_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_comap___boxed), 6, 5);
lean_closure_set(v___x_119_, 0, lean_box(0));
lean_closure_set(v___x_119_, 1, v_inst_115_);
lean_closure_set(v___x_119_, 2, lean_box(0));
lean_closure_set(v___x_119_, 3, v_inst_116_);
lean_closure_set(v___x_119_, 4, v___f_118_);
v___x_120_ = lp_mathlib_Equiv_symm___redArg(v_f_117_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1), 2, 1);
lean_closure_set(v___f_121_, 0, v___x_120_);
v___x_122_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_comap___boxed), 6, 5);
lean_closure_set(v___x_122_, 0, lean_box(0));
lean_closure_set(v___x_122_, 1, v_inst_116_);
lean_closure_set(v___x_122_, 2, lean_box(0));
lean_closure_set(v___x_122_, 3, v_inst_115_);
lean_closure_set(v___x_122_, 4, v___f_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_119_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_comapSubgroup(lean_object* v_G_124_, lean_object* v_inst_125_, lean_object* v_H_126_, lean_object* v_inst_127_, lean_object* v_f_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_MulEquiv_comapSubgroup___redArg(v_inst_125_, v_inst_127_, v_f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_comapAddSubgroup___redArg(lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_f_132_){
_start:
{
lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___f_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
lean_inc_ref(v_f_132_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_133_, 0, v_f_132_);
lean_inc_ref(v_inst_131_);
lean_inc_ref(v_inst_130_);
v___x_134_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_comap___boxed), 6, 5);
lean_closure_set(v___x_134_, 0, lean_box(0));
lean_closure_set(v___x_134_, 1, v_inst_130_);
lean_closure_set(v___x_134_, 2, lean_box(0));
lean_closure_set(v___x_134_, 3, v_inst_131_);
lean_closure_set(v___x_134_, 4, v___f_133_);
v___x_135_ = lp_mathlib_Equiv_symm___redArg(v_f_132_);
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1), 2, 1);
lean_closure_set(v___f_136_, 0, v___x_135_);
v___x_137_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_comap___boxed), 6, 5);
lean_closure_set(v___x_137_, 0, lean_box(0));
lean_closure_set(v___x_137_, 1, v_inst_131_);
lean_closure_set(v___x_137_, 2, lean_box(0));
lean_closure_set(v___x_137_, 3, v_inst_130_);
lean_closure_set(v___x_137_, 4, v___f_136_);
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_134_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_comapAddSubgroup(lean_object* v_G_139_, lean_object* v_inst_140_, lean_object* v_H_141_, lean_object* v_inst_142_, lean_object* v_f_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_AddEquiv_comapAddSubgroup___redArg(v_inst_140_, v_inst_142_, v_f_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mapSubgroup___redArg(lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_f_147_){
_start:
{
lean_object* v___f_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
lean_inc_ref(v_f_147_);
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_148_, 0, v_f_147_);
lean_inc_ref(v_inst_146_);
lean_inc_ref(v_inst_145_);
v___x_149_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_map___boxed), 6, 5);
lean_closure_set(v___x_149_, 0, lean_box(0));
lean_closure_set(v___x_149_, 1, v_inst_145_);
lean_closure_set(v___x_149_, 2, lean_box(0));
lean_closure_set(v___x_149_, 3, v_inst_146_);
lean_closure_set(v___x_149_, 4, v___f_148_);
v___x_150_ = lp_mathlib_Equiv_symm___redArg(v_f_147_);
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1), 2, 1);
lean_closure_set(v___f_151_, 0, v___x_150_);
v___x_152_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_map___boxed), 6, 5);
lean_closure_set(v___x_152_, 0, lean_box(0));
lean_closure_set(v___x_152_, 1, v_inst_146_);
lean_closure_set(v___x_152_, 2, lean_box(0));
lean_closure_set(v___x_152_, 3, v_inst_145_);
lean_closure_set(v___x_152_, 4, v___f_151_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_149_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mapSubgroup(lean_object* v_G_154_, lean_object* v_inst_155_, lean_object* v_H_156_, lean_object* v_inst_157_, lean_object* v_f_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_MulEquiv_mapSubgroup___redArg(v_inst_155_, v_inst_157_, v_f_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapAddSubgroup___redArg(lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_f_162_){
_start:
{
lean_object* v___f_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___f_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
lean_inc_ref(v_f_162_);
v___f_163_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_163_, 0, v_f_162_);
lean_inc_ref(v_inst_161_);
lean_inc_ref(v_inst_160_);
v___x_164_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_map___boxed), 6, 5);
lean_closure_set(v___x_164_, 0, lean_box(0));
lean_closure_set(v___x_164_, 1, v_inst_160_);
lean_closure_set(v___x_164_, 2, lean_box(0));
lean_closure_set(v___x_164_, 3, v_inst_161_);
lean_closure_set(v___x_164_, 4, v___f_163_);
v___x_165_ = lp_mathlib_Equiv_symm___redArg(v_f_162_);
v___f_166_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_comapSubgroup___redArg___lam__1), 2, 1);
lean_closure_set(v___f_166_, 0, v___x_165_);
v___x_167_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_map___boxed), 6, 5);
lean_closure_set(v___x_167_, 0, lean_box(0));
lean_closure_set(v___x_167_, 1, v_inst_161_);
lean_closure_set(v___x_167_, 2, lean_box(0));
lean_closure_set(v___x_167_, 3, v_inst_160_);
lean_closure_set(v___x_167_, 4, v___f_166_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_164_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mapAddSubgroup(lean_object* v_G_169_, lean_object* v_inst_170_, lean_object* v_H_171_, lean_object* v_inst_172_, lean_object* v_f_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_AddEquiv_mapAddSubgroup___redArg(v_inst_170_, v_inst_172_, v_f_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap___redArg(lean_object* v_f_175_){
_start:
{
lean_object* v___f_176_; 
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_176_, 0, v_f_175_);
return v___f_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap(lean_object* v_G_177_, lean_object* v_G_x27_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_f_181_, lean_object* v_H_x27_182_){
_start:
{
lean_object* v___f_183_; 
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_183_, 0, v_f_181_);
return v___f_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupComap___boxed(lean_object* v_G_184_, lean_object* v_G_x27_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_f_188_, lean_object* v_H_x27_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_MonoidHom_subgroupComap(v_G_184_, v_G_x27_185_, v_inst_186_, v_inst_187_, v_f_188_, v_H_x27_189_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap___redArg(lean_object* v_f_191_){
_start:
{
lean_object* v___f_192_; 
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_192_, 0, v_f_191_);
return v___f_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap(lean_object* v_G_193_, lean_object* v_G_x27_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_f_197_, lean_object* v_H_x27_198_){
_start:
{
lean_object* v___f_199_; 
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_199_, 0, v_f_197_);
return v___f_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupComap___boxed(lean_object* v_G_200_, lean_object* v_G_x27_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_f_204_, lean_object* v_H_x27_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_AddMonoidHom_addSubgroupComap(v_G_200_, v_G_x27_201_, v_inst_202_, v_inst_203_, v_f_204_, v_H_x27_205_);
lean_dec_ref(v_inst_203_);
lean_dec_ref(v_inst_202_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap___redArg(lean_object* v_f_207_){
_start:
{
lean_object* v___f_208_; 
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_208_, 0, v_f_207_);
return v___f_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap(lean_object* v_G_209_, lean_object* v_G_x27_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_f_213_, lean_object* v_H_214_){
_start:
{
lean_object* v___f_215_; 
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_215_, 0, v_f_213_);
return v___f_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_subgroupMap___boxed(lean_object* v_G_216_, lean_object* v_G_x27_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_f_220_, lean_object* v_H_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_MonoidHom_subgroupMap(v_G_216_, v_G_x27_217_, v_inst_218_, v_inst_219_, v_f_220_, v_H_221_);
lean_dec_ref(v_inst_219_);
lean_dec_ref(v_inst_218_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap___redArg(lean_object* v_f_223_){
_start:
{
lean_object* v___f_224_; 
v___f_224_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_224_, 0, v_f_223_);
return v___f_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap(lean_object* v_G_225_, lean_object* v_G_x27_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_f_229_, lean_object* v_H_230_){
_start:
{
lean_object* v___f_231_; 
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_231_, 0, v_f_229_);
return v___f_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubgroupMap___boxed(lean_object* v_G_232_, lean_object* v_G_x27_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_f_236_, lean_object* v_H_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_AddMonoidHom_addSubgroupMap(v_G_232_, v_G_x27_233_, v_inst_234_, v_inst_235_, v_f_236_, v_H_237_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_234_);
return v_res_238_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_subgroupCongr___closed__0(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupCongr(lean_object* v_G_240_, lean_object* v_inst_241_, lean_object* v_H_242_, lean_object* v_K_243_, lean_object* v_h_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_obj_once(&lp_mathlib_MulEquiv_subgroupCongr___closed__0, &lp_mathlib_MulEquiv_subgroupCongr___closed__0_once, _init_lp_mathlib_MulEquiv_subgroupCongr___closed__0);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupCongr___boxed(lean_object* v_G_246_, lean_object* v_inst_247_, lean_object* v_H_248_, lean_object* v_K_249_, lean_object* v_h_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_MulEquiv_subgroupCongr(v_G_246_, v_inst_247_, v_H_248_, v_K_249_, v_h_250_);
lean_dec_ref(v_inst_247_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupCongr(lean_object* v_G_252_, lean_object* v_inst_253_, lean_object* v_H_254_, lean_object* v_K_255_, lean_object* v_h_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lean_obj_once(&lp_mathlib_MulEquiv_subgroupCongr___closed__0, &lp_mathlib_MulEquiv_subgroupCongr___closed__0_once, _init_lp_mathlib_MulEquiv_subgroupCongr___closed__0);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupCongr___boxed(lean_object* v_G_258_, lean_object* v_inst_259_, lean_object* v_H_260_, lean_object* v_K_261_, lean_object* v_h_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_AddEquiv_addSubgroupCongr(v_G_258_, v_inst_259_, v_H_260_, v_K_261_, v_h_262_);
lean_dec_ref(v_inst_259_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap___redArg(lean_object* v_e_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_MulEquiv_submonoidMap___redArg(v_e_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap(lean_object* v_G_266_, lean_object* v_G_x27_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_e_270_, lean_object* v_H_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_MulEquiv_submonoidMap___redArg(v_e_270_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_subgroupMap___boxed(lean_object* v_G_273_, lean_object* v_G_x27_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_e_277_, lean_object* v_H_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_MulEquiv_subgroupMap(v_G_273_, v_G_x27_274_, v_inst_275_, v_inst_276_, v_e_277_, v_H_278_);
lean_dec_ref(v_inst_276_);
lean_dec_ref(v_inst_275_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap___redArg(lean_object* v_e_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap(lean_object* v_G_282_, lean_object* v_G_x27_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_e_286_, lean_object* v_H_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_286_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubgroupMap___boxed(lean_object* v_G_289_, lean_object* v_G_x27_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_e_293_, lean_object* v_H_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_AddEquiv_addSubgroupMap(v_G_289_, v_G_x27_290_, v_inst_291_, v_inst_292_, v_e_293_, v_H_294_);
lean_dec_ref(v_inst_292_);
lean_dec_ref(v_inst_291_);
return v_res_295_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
}
#ifdef __cplusplus
}
#endif
