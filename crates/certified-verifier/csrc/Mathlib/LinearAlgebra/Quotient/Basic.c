// Lean compiler output
// Module: Mathlib.LinearAlgebra.Quotient.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Basic public import Mathlib.GroupTheory.QuotientGroup.Basic public import Mathlib.LinearAlgebra.Pi public import Mathlib.LinearAlgebra.Quotient.Defs public import Mathlib.LinearAlgebra.Span.Basic
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
lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Fintype_ofSubsingleton___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofLinearMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quot_congrRight(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_restrictScalarsEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_factor___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_factor___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_factor___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_factor___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_factor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_comapMkQRelIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_comapMkQRelIso___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_comapMkQRelIso___closed__0 = (const lean_object*)&lp_mathlib_Submodule_comapMkQRelIso___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_comapMkQRelIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_comapMkQRelIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_comapMkQRelIso___closed__1 = (const lean_object*)&lp_mathlib_Submodule_comapMkQRelIso___closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_comapMkQRelIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_comapMkQRelIso___closed__1_value),((lean_object*)&lp_mathlib_Submodule_comapMkQRelIso___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_comapMkQRelIso___closed__2 = (const lean_object*)&lp_mathlib_Submodule_comapMkQRelIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___boxed(lean_object**);
static lean_once_cell_t lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEqBot___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEqBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Quot_congrRight(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_restrictScalarsEquiv(lean_object* v_R_2_, lean_object* v_M_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_S_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_P_12_){
_start:
{
lean_object* v___x_13_; lean_object* v_toFun_14_; lean_object* v_invFun_15_; lean_object* v___x_16_; 
v___x_13_ = lean_obj_once(&lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0, &lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0_once, _init_lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___closed__0);
v_toFun_14_ = lean_ctor_get(v___x_13_, 0);
v_invFun_15_ = lean_ctor_get(v___x_13_, 1);
lean_inc(v_invFun_15_);
lean_inc(v_toFun_14_);
v___x_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_16_, 0, v_toFun_14_);
lean_ctor_set(v___x_16_, 1, v_invFun_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_restrictScalarsEquiv___boxed(lean_object* v_R_17_, lean_object* v_M_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_S_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_P_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Submodule_Quotient_restrictScalarsEquiv(v_R_17_, v_M_18_, v_inst_19_, v_inst_20_, v_inst_21_, v_S_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_P_27_);
lean_dec(v_inst_25_);
lean_dec(v_inst_24_);
lean_dec_ref(v_inst_23_);
lean_dec(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec_ref(v_inst_19_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; lean_object* v_toZero_31_; 
v___x_30_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_29_);
v_toZero_31_ = lean_ctor_get(v___x_30_, 0);
lean_inc(v_toZero_31_);
lean_dec_ref(v___x_30_);
return v_toZero_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___redArg___boxed(lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Submodule_QuotientTop_unique___redArg(v_inst_32_);
lean_dec_ref(v_inst_32_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique(lean_object* v_R_34_, lean_object* v_M_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Submodule_QuotientTop_unique___redArg(v_inst_37_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_unique___boxed(lean_object* v_R_40_, lean_object* v_M_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Submodule_QuotientTop_unique(v_R_40_, v_M_41_, v_inst_42_, v_inst_43_, v_inst_44_);
lean_dec(v_inst_44_);
lean_dec_ref(v_inst_43_);
lean_dec_ref(v_inst_42_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; lean_object* v_toZero_48_; lean_object* v___x_49_; 
v___x_47_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_46_);
v_toZero_48_ = lean_ctor_get(v___x_47_, 0);
lean_inc(v_toZero_48_);
lean_dec_ref(v___x_47_);
v___x_49_ = lp_mathlib_Fintype_ofSubsingleton___redArg(v_toZero_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___redArg___boxed(lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Submodule_QuotientTop_fintype___redArg(v_inst_50_);
lean_dec_ref(v_inst_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype(lean_object* v_R_52_, lean_object* v_M_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_Submodule_QuotientTop_fintype___redArg(v_inst_55_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_QuotientTop_fintype___boxed(lean_object* v_R_58_, lean_object* v_M_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Submodule_QuotientTop_fintype(v_R_58_, v_M_59_, v_inst_60_, v_inst_61_, v_inst_62_);
lean_dec(v_inst_62_);
lean_dec_ref(v_inst_61_);
lean_dec_ref(v_inst_60_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ___redArg(lean_object* v_f_64_){
_start:
{
lean_object* v___f_65_; lean_object* v___f_66_; 
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_65_, 0, v_f_64_);
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_66_, 0, v___f_65_);
return v___f_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ(lean_object* v_R_67_, lean_object* v_M_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_p_72_, lean_object* v_R_u2082_73_, lean_object* v_M_u2082_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_00_u03c4_u2081_u2082_78_, lean_object* v_f_79_, lean_object* v_h_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_Submodule_liftQ___redArg(v_f_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQ___boxed(lean_object* v_R_82_, lean_object* v_M_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_p_87_, lean_object* v_R_u2082_88_, lean_object* v_M_u2082_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_00_u03c4_u2081_u2082_93_, lean_object* v_f_94_, lean_object* v_h_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_Submodule_liftQ(v_R_82_, v_M_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_p_87_, v_R_u2082_88_, v_M_u2082_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_00_u03c4_u2081_u2082_93_, v_f_94_, v_h_95_);
lean_dec(v_00_u03c4_u2081_u2082_93_);
lean_dec(v_inst_92_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
lean_dec(v_inst_86_);
lean_dec_ref(v_inst_85_);
lean_dec_ref(v_inst_84_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton___redArg(lean_object* v_f_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Submodule_liftQ___redArg(v_f_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton(lean_object* v_R_99_, lean_object* v_M_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_R_u2082_104_, lean_object* v_M_u2082_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_00_u03c4_u2081_u2082_109_, lean_object* v_x_110_, lean_object* v_f_111_, lean_object* v_h_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Submodule_liftQ___redArg(v_f_111_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_liftQSpanSingleton___boxed(lean_object* v_R_114_, lean_object* v_M_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_R_u2082_119_, lean_object* v_M_u2082_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_00_u03c4_u2081_u2082_124_, lean_object* v_x_125_, lean_object* v_f_126_, lean_object* v_h_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Submodule_liftQSpanSingleton(v_R_114_, v_M_115_, v_inst_116_, v_inst_117_, v_inst_118_, v_R_u2082_119_, v_M_u2082_120_, v_inst_121_, v_inst_122_, v_inst_123_, v_00_u03c4_u2081_u2082_124_, v_x_125_, v_f_126_, v_h_127_);
lean_dec(v_x_125_);
lean_dec(v_00_u03c4_u2081_u2082_124_);
lean_dec(v_inst_123_);
lean_dec_ref(v_inst_122_);
lean_dec_ref(v_inst_121_);
lean_dec(v_inst_118_);
lean_dec_ref(v_inst_117_);
lean_dec_ref(v_inst_116_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ___redArg(lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_q_132_, lean_object* v_f_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___f_135_; lean_object* v___x_136_; 
v___x_134_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_134_, 0, lean_box(0));
lean_closure_set(v___x_134_, 1, lean_box(0));
lean_closure_set(v___x_134_, 2, v_inst_129_);
lean_closure_set(v___x_134_, 3, v_inst_130_);
lean_closure_set(v___x_134_, 4, v_inst_131_);
lean_closure_set(v___x_134_, 5, v_q_132_);
v___f_135_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_135_, 0, v_f_133_);
lean_closure_set(v___f_135_, 1, v___x_134_);
v___x_136_ = lp_mathlib_Submodule_liftQ___redArg(v___f_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ(lean_object* v_R_137_, lean_object* v_M_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_p_142_, lean_object* v_R_u2082_143_, lean_object* v_M_u2082_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_00_u03c4_u2081_u2082_148_, lean_object* v_q_149_, lean_object* v_f_150_, lean_object* v_h_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_145_, v_inst_146_, v_inst_147_, v_q_149_, v_f_150_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQ___boxed(lean_object* v_R_153_, lean_object* v_M_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_p_158_, lean_object* v_R_u2082_159_, lean_object* v_M_u2082_160_, lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_00_u03c4_u2081_u2082_164_, lean_object* v_q_165_, lean_object* v_f_166_, lean_object* v_h_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Submodule_mapQ(v_R_153_, v_M_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_p_158_, v_R_u2082_159_, v_M_u2082_160_, v_inst_161_, v_inst_162_, v_inst_163_, v_00_u03c4_u2081_u2082_164_, v_q_165_, v_f_166_, v_h_167_);
lean_dec(v_00_u03c4_u2081_u2082_164_);
lean_dec(v_inst_157_);
lean_dec_ref(v_inst_156_);
lean_dec_ref(v_inst_155_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_factor___redArg(lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_p_x27_173_){
_start:
{
lean_object* v___f_174_; lean_object* v___x_175_; 
v___f_174_ = ((lean_object*)(lp_mathlib_Submodule_factor___redArg___closed__0));
v___x_175_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_170_, v_inst_171_, v_inst_172_, v_p_x27_173_, v___f_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_factor(lean_object* v_R_176_, lean_object* v_M_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_p_181_, lean_object* v_p_x27_182_, lean_object* v_H_183_){
_start:
{
lean_object* v___f_184_; lean_object* v___x_185_; 
v___f_184_ = ((lean_object*)(lp_mathlib_Submodule_factor___redArg___closed__0));
v___x_185_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_178_, v_inst_179_, v_inst_180_, v_p_x27_182_, v___f_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___lam__0(lean_object* v_q_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lean_box(0);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___lam__1(lean_object* v_p_x27_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lean_box(0);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso(lean_object* v_R_195_, lean_object* v_M_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_p_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_mathlib_Submodule_comapMkQRelIso___closed__2));
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQRelIso___boxed(lean_object* v_R_202_, lean_object* v_M_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_p_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Submodule_comapMkQRelIso(v_R_202_, v_M_203_, v_inst_204_, v_inst_205_, v_inst_206_, v_p_207_);
lean_dec(v_inst_206_);
lean_dec_ref(v_inst_205_);
lean_dec_ref(v_inst_204_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg(lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_p_213_){
_start:
{
lean_object* v___x_214_; lean_object* v___f_215_; lean_object* v___f_216_; lean_object* v___f_217_; 
v___x_214_ = lp_mathlib_Submodule_comapMkQRelIso(lean_box(0), lean_box(0), v_inst_210_, v_inst_211_, v_inst_212_, v_p_213_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_215_, 0, v___x_214_);
v___f_216_ = ((lean_object*)(lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___closed__0));
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_217_, 0, v___f_215_);
lean_closure_set(v___f_217_, 1, v___f_216_);
return v___f_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg___boxed(lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_p_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg(v_inst_218_, v_inst_219_, v_inst_220_, v_p_221_);
lean_dec(v_inst_220_);
lean_dec_ref(v_inst_219_);
lean_dec_ref(v_inst_218_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding(lean_object* v_R_223_, lean_object* v_M_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_p_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_Submodule_comapMkQOrderEmbedding___redArg(v_inst_225_, v_inst_226_, v_inst_227_, v_p_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_comapMkQOrderEmbedding___boxed(lean_object* v_R_230_, lean_object* v_M_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_p_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_Submodule_comapMkQOrderEmbedding(v_R_230_, v_M_231_, v_inst_232_, v_inst_233_, v_inst_234_, v_p_235_);
lean_dec(v_inst_234_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___redArg___lam__0(lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_P_240_, lean_object* v_toLinearMap_241_, lean_object* v___y_242_){
_start:
{
lean_object* v___x_57__overap_243_; lean_object* v___x_244_; 
v___x_57__overap_243_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_237_, v_inst_238_, v_inst_239_, v_P_240_, v_toLinearMap_241_);
v___x_244_ = lean_apply_1(v___x_57__overap_243_, v___y_242_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___redArg(lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_P_251_, lean_object* v_Q_252_, lean_object* v_f_253_){
_start:
{
lean_object* v_toLinearMap_254_; lean_object* v___x_255_; lean_object* v_toLinearMap_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_265_; 
v_toLinearMap_254_ = lean_ctor_get(v_f_253_, 0);
lean_inc(v_toLinearMap_254_);
v___x_255_ = lp_mathlib_LinearEquiv_symm___redArg(v_f_253_);
v_toLinearMap_256_ = lean_ctor_get(v___x_255_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_255_);
if (v_isSharedCheck_265_ == 0)
{
lean_object* v_unused_266_; 
v_unused_266_ = lean_ctor_get(v___x_255_, 1);
lean_dec(v_unused_266_);
v___x_258_ = v___x_255_;
v_isShared_259_ = v_isSharedCheck_265_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_toLinearMap_256_);
lean_dec(v___x_255_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_265_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_260_; lean_object* v___f_261_; lean_object* v___x_263_; 
v___x_260_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_246_, v_inst_249_, v_inst_250_, v_Q_252_, v_toLinearMap_254_);
v___f_261_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_equiv___redArg___lam__0), 6, 5);
lean_closure_set(v___f_261_, 0, v_inst_245_);
lean_closure_set(v___f_261_, 1, v_inst_247_);
lean_closure_set(v___f_261_, 2, v_inst_248_);
lean_closure_set(v___f_261_, 3, v_P_251_);
lean_closure_set(v___f_261_, 4, v_toLinearMap_256_);
if (v_isShared_259_ == 0)
{
lean_ctor_set(v___x_258_, 1, v___f_261_);
lean_ctor_set(v___x_258_, 0, v___x_260_);
v___x_263_ = v___x_258_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_264_, 1, v___f_261_);
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
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv(lean_object* v_R_267_, lean_object* v_inst_268_, lean_object* v_R_u2082_269_, lean_object* v_inst_270_, lean_object* v_00_u03c3_u2081_u2082_271_, lean_object* v_00_u03c3_u2082_u2081_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_M_275_, lean_object* v_N_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_P_281_, lean_object* v_Q_282_, lean_object* v_f_283_, lean_object* v_hf_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_mathlib_Submodule_Quotient_equiv___redArg(v_inst_268_, v_inst_270_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_P_281_, v_Q_282_, v_f_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_equiv___boxed(lean_object** _args){
lean_object* v_R_286_ = _args[0];
lean_object* v_inst_287_ = _args[1];
lean_object* v_R_u2082_288_ = _args[2];
lean_object* v_inst_289_ = _args[3];
lean_object* v_00_u03c3_u2081_u2082_290_ = _args[4];
lean_object* v_00_u03c3_u2082_u2081_291_ = _args[5];
lean_object* v_inst_292_ = _args[6];
lean_object* v_inst_293_ = _args[7];
lean_object* v_M_294_ = _args[8];
lean_object* v_N_295_ = _args[9];
lean_object* v_inst_296_ = _args[10];
lean_object* v_inst_297_ = _args[11];
lean_object* v_inst_298_ = _args[12];
lean_object* v_inst_299_ = _args[13];
lean_object* v_P_300_ = _args[14];
lean_object* v_Q_301_ = _args[15];
lean_object* v_f_302_ = _args[16];
lean_object* v_hf_303_ = _args[17];
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Submodule_Quotient_equiv(v_R_286_, v_inst_287_, v_R_u2082_288_, v_inst_289_, v_00_u03c3_u2081_u2082_290_, v_00_u03c3_u2082_u2081_291_, v_inst_292_, v_inst_293_, v_M_294_, v_N_295_, v_inst_296_, v_inst_297_, v_inst_298_, v_inst_299_, v_P_300_, v_Q_301_, v_f_302_, v_hf_303_);
lean_dec(v_00_u03c3_u2082_u2081_291_);
lean_dec(v_00_u03c3_u2081_u2082_290_);
return v_res_304_;
}
}
static lean_object* _init_lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0(void){
_start:
{
lean_object* v___f_305_; lean_object* v___x_306_; 
v___f_305_ = ((lean_object*)(lp_mathlib_Submodule_factor___redArg___closed__0));
v___x_306_ = lp_mathlib_Submodule_liftQ___redArg(v___f_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEqBot___redArg(lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_p_310_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_311_ = lean_obj_once(&lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0, &lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0_once, _init_lp_mathlib_Submodule_quotEquivOfEqBot___redArg___closed__0);
v___x_312_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_312_, 0, lean_box(0));
lean_closure_set(v___x_312_, 1, lean_box(0));
lean_closure_set(v___x_312_, 2, v_inst_307_);
lean_closure_set(v___x_312_, 3, v_inst_308_);
lean_closure_set(v___x_312_, 4, v_inst_309_);
lean_closure_set(v___x_312_, 5, v_p_310_);
v___x_313_ = lp_mathlib_LinearEquiv_ofLinearMap___redArg(v___x_311_, v___x_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEqBot(lean_object* v_R_314_, lean_object* v_M_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_p_319_, lean_object* v_hp_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_Submodule_quotEquivOfEqBot___redArg(v_inst_316_, v_inst_317_, v_inst_318_, v_p_319_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___redArg___lam__0(lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_q_325_, lean_object* v_f_326_, lean_object* v___y_327_){
_start:
{
lean_object* v___x_65__overap_328_; lean_object* v___x_329_; 
v___x_65__overap_328_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_322_, v_inst_323_, v_inst_324_, v_q_325_, v_f_326_);
v___x_329_ = lean_apply_1(v___x_65__overap_328_, v___y_327_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___redArg(lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_q_333_){
_start:
{
lean_object* v___f_334_; 
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_mapQLinear___redArg___lam__0), 6, 4);
lean_closure_set(v___f_334_, 0, v_inst_330_);
lean_closure_set(v___f_334_, 1, v_inst_331_);
lean_closure_set(v___f_334_, 2, v_inst_332_);
lean_closure_set(v___f_334_, 3, v_q_333_);
return v___f_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear(lean_object* v_R_335_, lean_object* v_M_336_, lean_object* v_M_u2082_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_p_343_, lean_object* v_q_344_){
_start:
{
lean_object* v___f_345_; 
v___f_345_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_mapQLinear___redArg___lam__0), 6, 4);
lean_closure_set(v___f_345_, 0, v_inst_338_);
lean_closure_set(v___f_345_, 1, v_inst_341_);
lean_closure_set(v___f_345_, 2, v_inst_342_);
lean_closure_set(v___f_345_, 3, v_q_344_);
return v___f_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapQLinear___boxed(lean_object* v_R_346_, lean_object* v_M_347_, lean_object* v_M_u2082_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_p_354_, lean_object* v_q_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_Submodule_mapQLinear(v_R_346_, v_M_347_, v_M_u2082_348_, v_inst_349_, v_inst_350_, v_inst_351_, v_inst_352_, v_inst_353_, v_p_354_, v_q_355_);
lean_dec(v_inst_351_);
lean_dec_ref(v_inst_350_);
return v_res_356_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
