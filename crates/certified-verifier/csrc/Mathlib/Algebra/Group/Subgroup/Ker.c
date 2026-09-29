// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Ker
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Map public import Mathlib.Tactic.ApplyFun
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
lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_SubgroupClass_subtype___lam__0___boxed(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_ofLeftInverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubgroupClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_ofLeftInverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemKer___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemKer___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemKer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemKer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemKer___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemKer___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemKer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemKer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocus(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocus___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_MapSubtype_orderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_MapSubtype_orderIso___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_MapSubtype_orderIso___closed__0_value;
static const lean_closure_object lp_mathlib_Subgroup_MapSubtype_orderIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_MapSubtype_orderIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_MapSubtype_orderIso___closed__1_value;
static const lean_ctor_object lp_mathlib_Subgroup_MapSubtype_orderIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Subgroup_MapSubtype_orderIso___closed__1_value),((lean_object*)&lp_mathlib_Subgroup_MapSubtype_orderIso___closed__0_value)}};
static const lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___closed__2 = (const lean_object*)&lp_mathlib_Subgroup_MapSubtype_orderIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__0_value;
static const lean_closure_object lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__1 = (const lean_object*)&lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__1_value;
static const lean_ctor_object lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__1_value),((lean_object*)&lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__0_value)}};
static const lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__2 = (const lean_object*)&lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_range(lean_object* v_G_1_, lean_object* v_inst_2_, lean_object* v_N_3_, lean_object* v_inst_4_, lean_object* v_f_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_range___boxed(lean_object* v_G_7_, lean_object* v_inst_8_, lean_object* v_N_9_, lean_object* v_inst_10_, lean_object* v_f_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_MonoidHom_range(v_G_7_, v_inst_8_, v_N_9_, v_inst_10_, v_f_11_);
lean_dec(v_f_11_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_8_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_range(lean_object* v_G_13_, lean_object* v_inst_14_, lean_object* v_N_15_, lean_object* v_inst_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_range___boxed(lean_object* v_G_19_, lean_object* v_inst_20_, lean_object* v_N_21_, lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_AddMonoidHom_range(v_G_19_, v_inst_20_, v_N_21_, v_inst_22_, v_f_23_);
lean_dec(v_f_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_20_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict___redArg(lean_object* v_f_25_){
_start:
{
lean_object* v___f_26_; 
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_26_, 0, v_f_25_);
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict(lean_object* v_G_27_, lean_object* v_inst_28_, lean_object* v_N_29_, lean_object* v_inst_30_, lean_object* v_f_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_32_, 0, v_f_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_rangeRestrict___boxed(lean_object* v_G_33_, lean_object* v_inst_34_, lean_object* v_N_35_, lean_object* v_inst_36_, lean_object* v_f_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_MonoidHom_rangeRestrict(v_G_33_, v_inst_34_, v_N_35_, v_inst_36_, v_f_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_34_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict___redArg(lean_object* v_f_39_){
_start:
{
lean_object* v___f_40_; 
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_40_, 0, v_f_39_);
return v___f_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict(lean_object* v_G_41_, lean_object* v_inst_42_, lean_object* v_N_43_, lean_object* v_inst_44_, lean_object* v_f_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_46_, 0, v_f_45_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_rangeRestrict___boxed(lean_object* v_G_47_, lean_object* v_inst_48_, lean_object* v_N_49_, lean_object* v_inst_50_, lean_object* v_f_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_AddMonoidHom_rangeRestrict(v_G_47_, v_inst_48_, v_N_49_, v_inst_50_, v_f_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_48_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__0(lean_object* v_g_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_apply_1(v_g_53_, v___y_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__1(lean_object* v_f_56_, lean_object* v___y_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(v_f_56_, v___y_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___redArg(lean_object* v_f_60_, lean_object* v_g_61_){
_start:
{
lean_object* v___f_62_; lean_object* v___f_63_; lean_object* v___f_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_62_, 0, v_g_61_);
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__1), 2, 1);
lean_closure_set(v___f_63_, 0, v_f_60_);
v___f_64_ = ((lean_object*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___closed__0));
v___x_65_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_65_, 0, lean_box(0));
lean_closure_set(v___x_65_, 1, lean_box(0));
lean_closure_set(v___x_65_, 2, lean_box(0));
lean_closure_set(v___x_65_, 3, v___f_62_);
lean_closure_set(v___x_65_, 4, v___f_64_);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___f_63_);
lean_ctor_set(v___x_66_, 1, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse(lean_object* v_G_67_, lean_object* v_inst_68_, lean_object* v_N_69_, lean_object* v_inst_70_, lean_object* v_f_71_, lean_object* v_g_72_, lean_object* v_h_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_MonoidHom_ofLeftInverse___redArg(v_f_71_, v_g_72_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ofLeftInverse___boxed(lean_object* v_G_75_, lean_object* v_inst_76_, lean_object* v_N_77_, lean_object* v_inst_78_, lean_object* v_f_79_, lean_object* v_g_80_, lean_object* v_h_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_MonoidHom_ofLeftInverse(v_G_75_, v_inst_76_, v_N_77_, v_inst_78_, v_f_79_, v_g_80_, v_h_81_);
lean_dec_ref(v_inst_78_);
lean_dec_ref(v_inst_76_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse___redArg(lean_object* v_f_83_, lean_object* v_g_84_){
_start:
{
lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___f_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_85_, 0, v_g_84_);
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___lam__1), 2, 1);
lean_closure_set(v___f_86_, 0, v_f_83_);
v___f_87_ = ((lean_object*)(lp_mathlib_MonoidHom_ofLeftInverse___redArg___closed__0));
v___x_88_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_88_, 0, lean_box(0));
lean_closure_set(v___x_88_, 1, lean_box(0));
lean_closure_set(v___x_88_, 2, lean_box(0));
lean_closure_set(v___x_88_, 3, v___f_85_);
lean_closure_set(v___x_88_, 4, v___f_87_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___f_86_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse(lean_object* v_G_90_, lean_object* v_inst_91_, lean_object* v_N_92_, lean_object* v_inst_93_, lean_object* v_f_94_, lean_object* v_g_95_, lean_object* v_h_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_mathlib_AddMonoidHom_ofLeftInverse___redArg(v_f_94_, v_g_95_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ofLeftInverse___boxed(lean_object* v_G_98_, lean_object* v_inst_99_, lean_object* v_N_100_, lean_object* v_inst_101_, lean_object* v_f_102_, lean_object* v_g_103_, lean_object* v_h_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_AddMonoidHom_ofLeftInverse(v_G_98_, v_inst_99_, v_N_100_, v_inst_101_, v_f_102_, v_g_103_, v_h_104_);
lean_dec_ref(v_inst_101_);
lean_dec_ref(v_inst_99_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ker(lean_object* v_G_106_, lean_object* v_inst_107_, lean_object* v_M_108_, lean_object* v_inst_109_, lean_object* v_f_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_box(0);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_ker___boxed(lean_object* v_G_112_, lean_object* v_inst_113_, lean_object* v_M_114_, lean_object* v_inst_115_, lean_object* v_f_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_MonoidHom_ker(v_G_112_, v_inst_113_, v_M_114_, v_inst_115_, v_f_116_);
lean_dec(v_f_116_);
lean_dec_ref(v_inst_115_);
lean_dec_ref(v_inst_113_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ker(lean_object* v_G_118_, lean_object* v_inst_119_, lean_object* v_M_120_, lean_object* v_inst_121_, lean_object* v_f_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lean_box(0);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ker___boxed(lean_object* v_G_124_, lean_object* v_inst_125_, lean_object* v_M_126_, lean_object* v_inst_127_, lean_object* v_f_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_AddMonoidHom_ker(v_G_124_, v_inst_125_, v_M_126_, v_inst_127_, v_f_128_);
lean_dec(v_f_128_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_125_);
return v_res_129_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemKer___redArg(lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_f_132_, lean_object* v_x_133_){
_start:
{
lean_object* v___x_134_; lean_object* v_toOne_135_; lean_object* v___x_136_; lean_object* v___x_137_; uint8_t v___x_138_; 
v___x_134_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_130_);
v_toOne_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_toOne_135_);
lean_dec_ref(v___x_134_);
v___x_136_ = lean_apply_1(v_f_132_, v_x_133_);
v___x_137_ = lean_apply_2(v_inst_131_, v___x_136_, v_toOne_135_);
v___x_138_ = lean_unbox(v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemKer___redArg___boxed(lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_f_141_, lean_object* v_x_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_MonoidHom_decidableMemKer___redArg(v_inst_139_, v_inst_140_, v_f_141_, v_x_142_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemKer(lean_object* v_G_145_, lean_object* v_inst_146_, lean_object* v_M_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_f_150_, lean_object* v_x_151_){
_start:
{
uint8_t v___x_152_; 
v___x_152_ = lp_mathlib_MonoidHom_decidableMemKer___redArg(v_inst_148_, v_inst_149_, v_f_150_, v_x_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemKer___boxed(lean_object* v_G_153_, lean_object* v_inst_154_, lean_object* v_M_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_, lean_object* v_x_159_){
_start:
{
uint8_t v_res_160_; lean_object* v_r_161_; 
v_res_160_ = lp_mathlib_MonoidHom_decidableMemKer(v_G_153_, v_inst_154_, v_M_155_, v_inst_156_, v_inst_157_, v_f_158_, v_x_159_);
lean_dec_ref(v_inst_154_);
v_r_161_ = lean_box(v_res_160_);
return v_r_161_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemKer___redArg(lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_x_165_){
_start:
{
lean_object* v___x_166_; lean_object* v_toZero_167_; lean_object* v___x_168_; lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_166_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_162_);
v_toZero_167_ = lean_ctor_get(v___x_166_, 0);
lean_inc(v_toZero_167_);
lean_dec_ref(v___x_166_);
v___x_168_ = lean_apply_1(v_f_164_, v_x_165_);
v___x_169_ = lean_apply_2(v_inst_163_, v___x_168_, v_toZero_167_);
v___x_170_ = lean_unbox(v___x_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemKer___redArg___boxed(lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_f_173_, lean_object* v_x_174_){
_start:
{
uint8_t v_res_175_; lean_object* v_r_176_; 
v_res_175_ = lp_mathlib_AddMonoidHom_decidableMemKer___redArg(v_inst_171_, v_inst_172_, v_f_173_, v_x_174_);
v_r_176_ = lean_box(v_res_175_);
return v_r_176_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemKer(lean_object* v_G_177_, lean_object* v_inst_178_, lean_object* v_M_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_f_182_, lean_object* v_x_183_){
_start:
{
uint8_t v___x_184_; 
v___x_184_ = lp_mathlib_AddMonoidHom_decidableMemKer___redArg(v_inst_180_, v_inst_181_, v_f_182_, v_x_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemKer___boxed(lean_object* v_G_185_, lean_object* v_inst_186_, lean_object* v_M_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_f_190_, lean_object* v_x_191_){
_start:
{
uint8_t v_res_192_; lean_object* v_r_193_; 
v_res_192_ = lp_mathlib_AddMonoidHom_decidableMemKer(v_G_185_, v_inst_186_, v_M_187_, v_inst_188_, v_inst_189_, v_f_190_, v_x_191_);
lean_dec_ref(v_inst_186_);
v_r_193_ = lean_box(v_res_192_);
return v_r_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocus(lean_object* v_G_194_, lean_object* v_inst_195_, lean_object* v_M_196_, lean_object* v_inst_197_, lean_object* v_f_198_, lean_object* v_g_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lean_box(0);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eqLocus___boxed(lean_object* v_G_201_, lean_object* v_inst_202_, lean_object* v_M_203_, lean_object* v_inst_204_, lean_object* v_f_205_, lean_object* v_g_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_MonoidHom_eqLocus(v_G_201_, v_inst_202_, v_M_203_, v_inst_204_, v_f_205_, v_g_206_);
lean_dec(v_g_206_);
lean_dec(v_f_205_);
lean_dec_ref(v_inst_204_);
lean_dec_ref(v_inst_202_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocus(lean_object* v_G_208_, lean_object* v_inst_209_, lean_object* v_M_210_, lean_object* v_inst_211_, lean_object* v_f_212_, lean_object* v_g_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_box(0);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eqLocus___boxed(lean_object* v_G_215_, lean_object* v_inst_216_, lean_object* v_M_217_, lean_object* v_inst_218_, lean_object* v_f_219_, lean_object* v_g_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_AddMonoidHom_eqLocus(v_G_215_, v_inst_216_, v_M_217_, v_inst_218_, v_f_219_, v_g_220_);
lean_dec(v_g_220_);
lean_dec(v_f_219_);
lean_dec_ref(v_inst_218_);
lean_dec_ref(v_inst_216_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___lam__0(lean_object* v_sH_x27_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_box(0);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___lam__1(lean_object* v_H_x27_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lean_box(0);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso(lean_object* v_G_231_, lean_object* v_inst_232_, lean_object* v_H_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = ((lean_object*)(lp_mathlib_Subgroup_MapSubtype_orderIso___closed__2));
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_MapSubtype_orderIso___boxed(lean_object* v_G_235_, lean_object* v_inst_236_, lean_object* v_H_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_Subgroup_MapSubtype_orderIso(v_G_235_, v_inst_236_, v_H_237_);
lean_dec_ref(v_inst_236_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__0(lean_object* v_sH_x27_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lean_box(0);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___lam__1(lean_object* v_H_x27_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lean_box(0);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso(lean_object* v_G_248_, lean_object* v_inst_249_, lean_object* v_H_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = ((lean_object*)(lp_mathlib_AddSubgroup_MapSubtype_orderIso___closed__2));
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_MapSubtype_orderIso___boxed(lean_object* v_G_252_, lean_object* v_inst_253_, lean_object* v_H_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_AddSubgroup_MapSubtype_orderIso(v_G_252_, v_inst_253_, v_H_254_);
lean_dec_ref(v_inst_253_);
return v_res_255_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Ker(builtin);
}
#ifdef __cplusplus
}
#endif
