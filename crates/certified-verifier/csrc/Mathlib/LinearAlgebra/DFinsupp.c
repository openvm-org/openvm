// Lean compiler output
// Module: Mathlib.LinearAlgebra.DFinsupp
// Imports: public import Init public meta import Init public import Mathlib.Data.DFinsupp.Submonoid public import Mathlib.Data.DFinsupp.Sigma public import Mathlib.Data.Finsupp.ToDFinsupp public import Mathlib.LinearAlgebra.Finsupp.SumProd public import Mathlib.LinearAlgebra.LinearIndependent.Lemmas
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_mapRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_sumAddHom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_id___lam__0(lean_object*);
lean_object* lp_mathlib_DFinsupp_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_coprodMap___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_coprodMap___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_coprodMap___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v_toZero_6_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_i_2_);
v___x_4_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_3_);
lean_dec_ref(v___x_3_);
v___x_5_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_4_);
v_toZero_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toZero_6_);
lean_dec_ref(v___x_5_);
return v_toZero_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___redArg(lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_s_9_){
_start:
{
lean_object* v___f_10_; lean_object* v___x_11_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_10_, 0, v_inst_7_);
v___x_11_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mk), 6, 5);
lean_closure_set(v___x_11_, 0, lean_box(0));
lean_closure_set(v___x_11_, 1, lean_box(0));
lean_closure_set(v___x_11_, 2, v___f_10_);
lean_closure_set(v___x_11_, 3, v_inst_8_);
lean_closure_set(v___x_11_, 4, v_s_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk(lean_object* v_00_u03b9_12_, lean_object* v_R_13_, lean_object* v_M_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_s_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_mathlib_DFinsupp_lmk___redArg(v_inst_16_, v_inst_18_, v_s_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lmk___boxed(lean_object* v_00_u03b9_21_, lean_object* v_R_22_, lean_object* v_M_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_s_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_DFinsupp_lmk(v_00_u03b9_21_, v_R_22_, v_M_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_inst_27_, v_s_28_);
lean_dec(v_inst_26_);
lean_dec_ref(v_inst_24_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle___redArg(lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_i_32_){
_start:
{
lean_object* v___f_33_; lean_object* v___x_34_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_33_, 0, v_inst_30_);
v___x_34_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_single), 6, 5);
lean_closure_set(v___x_34_, 0, lean_box(0));
lean_closure_set(v___x_34_, 1, lean_box(0));
lean_closure_set(v___x_34_, 2, v___f_33_);
lean_closure_set(v___x_34_, 3, v_inst_31_);
lean_closure_set(v___x_34_, 4, v_i_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle(lean_object* v_00_u03b9_35_, lean_object* v_R_36_, lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_i_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_DFinsupp_lsingle___redArg(v_inst_39_, v_inst_41_, v_i_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsingle___boxed(lean_object* v_00_u03b9_44_, lean_object* v_R_45_, lean_object* v_M_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_i_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_DFinsupp_lsingle(v_00_u03b9_44_, v_R_45_, v_M_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_i_51_);
lean_dec(v_inst_49_);
lean_dec_ref(v_inst_47_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___redArg___lam__0(lean_object* v_i_53_, lean_object* v_f_54_){
_start:
{
lean_object* v_toFun_55_; lean_object* v___x_56_; 
v_toFun_55_ = lean_ctor_get(v_f_54_, 0);
lean_inc(v_toFun_55_);
lean_dec_ref(v_f_54_);
v___x_56_ = lean_apply_1(v_toFun_55_, v_i_53_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___redArg(lean_object* v_i_57_){
_start:
{
lean_object* v___f_58_; 
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lapply___redArg___lam__0), 2, 1);
lean_closure_set(v___f_58_, 0, v_i_57_);
return v___f_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply(lean_object* v_00_u03b9_59_, lean_object* v_R_60_, lean_object* v_M_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_i_65_){
_start:
{
lean_object* v___f_66_; 
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lapply___redArg___lam__0), 2, 1);
lean_closure_set(v___f_66_, 0, v_i_65_);
return v___f_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lapply___boxed(lean_object* v_00_u03b9_67_, lean_object* v_R_68_, lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_i_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_DFinsupp_lapply(v_00_u03b9_67_, v_R_68_, v_M_69_, v_inst_70_, v_inst_71_, v_inst_72_, v_i_73_);
lean_dec(v_inst_72_);
lean_dec_ref(v_inst_71_);
lean_dec_ref(v_inst_70_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr___redArg(lean_object* v_inst_75_, lean_object* v_e_76_){
_start:
{
lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v_toFun_79_; lean_object* v_invFun_80_; lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_87_; 
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_77_, 0, v_inst_75_);
v___x_78_ = lp_mathlib_DFinsupp_equivCongrLeft___redArg(v___f_77_, v_e_76_);
v_toFun_79_ = lean_ctor_get(v___x_78_, 0);
v_invFun_80_ = lean_ctor_get(v___x_78_, 1);
v_isSharedCheck_87_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_87_ == 0)
{
v___x_82_ = v___x_78_;
v_isShared_83_ = v_isSharedCheck_87_;
goto v_resetjp_81_;
}
else
{
lean_inc(v_invFun_80_);
lean_inc(v_toFun_79_);
lean_dec(v___x_78_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_87_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v___x_85_; 
if (v_isShared_83_ == 0)
{
v___x_85_ = v___x_82_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v_toFun_79_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v_invFun_80_);
v___x_85_ = v_reuseFailAlloc_86_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
return v___x_85_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr(lean_object* v_00_u03b9_88_, lean_object* v_00_u03b9_x27_89_, lean_object* v_R_90_, lean_object* v_M_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_e_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_DFinsupp_domLCongr___redArg(v_inst_93_, v_e_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_domLCongr___boxed(lean_object* v_00_u03b9_97_, lean_object* v_00_u03b9_x27_98_, lean_object* v_R_99_, lean_object* v_M_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_e_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_DFinsupp_domLCongr(v_00_u03b9_97_, v_00_u03b9_x27_98_, v_R_99_, v_M_100_, v_inst_101_, v_inst_102_, v_inst_103_, v_e_104_);
lean_dec(v_inst_103_);
lean_dec_ref(v_inst_101_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg___lam__0(lean_object* v_inst_106_, lean_object* v_i_107_, lean_object* v_j_108_){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v_toZero_112_; 
v___x_109_ = lean_apply_2(v_inst_106_, v_i_107_, v_j_108_);
v___x_110_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_109_);
lean_dec_ref(v___x_109_);
v___x_111_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_110_);
v_toZero_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_toZero_112_);
lean_dec_ref(v___x_111_);
return v_toZero_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___x_116_; lean_object* v_toFun_117_; lean_object* v_invFun_118_; lean_object* v___x_120_; uint8_t v_isShared_121_; uint8_t v_isSharedCheck_125_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_115_, 0, v_inst_114_);
v___x_116_ = lp_mathlib_DFinsupp_sigmaCurryEquiv___redArg(v_inst_113_, v___f_115_);
v_toFun_117_ = lean_ctor_get(v___x_116_, 0);
v_invFun_118_ = lean_ctor_get(v___x_116_, 1);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_125_ == 0)
{
v___x_120_ = v___x_116_;
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
else
{
lean_inc(v_invFun_118_);
lean_inc(v_toFun_117_);
lean_dec(v___x_116_);
v___x_120_ = lean_box(0);
v_isShared_121_ = v_isSharedCheck_125_;
goto v_resetjp_119_;
}
v_resetjp_119_:
{
lean_object* v___x_123_; 
if (v_isShared_121_ == 0)
{
v___x_123_ = v___x_120_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v_toFun_117_);
lean_ctor_set(v_reuseFailAlloc_124_, 1, v_invFun_118_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv(lean_object* v_00_u03b9_126_, lean_object* v_R_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_00_u03b1_130_, lean_object* v_M_131_, lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_DFinsupp_sigmaCurryLEquiv___redArg(v_inst_129_, v_inst_132_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaCurryLEquiv___boxed(lean_object* v_00_u03b9_135_, lean_object* v_R_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_00_u03b1_139_, lean_object* v_M_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_DFinsupp_sigmaCurryLEquiv(v_00_u03b9_135_, v_R_136_, v_inst_137_, v_inst_138_, v_00_u03b1_139_, v_M_140_, v_inst_141_, v_inst_142_);
lean_dec(v_inst_142_);
lean_dec_ref(v_inst_137_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; lean_object* v_toFun_146_; lean_object* v_invFun_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
v___x_145_ = lp_mathlib_DFinsupp_equivFunOnFintype___redArg(v_inst_144_);
v_toFun_146_ = lean_ctor_get(v___x_145_, 0);
v_invFun_147_ = lean_ctor_get(v___x_145_, 1);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_145_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_145_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_invFun_147_);
lean_inc(v_toFun_146_);
lean_dec(v___x_145_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_toFun_146_);
lean_ctor_set(v_reuseFailAlloc_153_, 1, v_invFun_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype(lean_object* v_00_u03b9_155_, lean_object* v_R_156_, lean_object* v_M_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_DFinsupp_linearEquivFunOnFintype___redArg(v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_linearEquivFunOnFintype___boxed(lean_object* v_00_u03b9_163_, lean_object* v_R_164_, lean_object* v_M_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_DFinsupp_linearEquivFunOnFintype(v_00_u03b9_163_, v_R_164_, v_M_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_inst_169_);
lean_dec(v_inst_168_);
lean_dec_ref(v_inst_167_);
lean_dec_ref(v_inst_166_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__0(lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_F_173_, lean_object* v_i_174_, lean_object* v___y_175_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lp_mathlib_DFinsupp_lsingle___redArg(v_inst_171_, v_inst_172_, v_i_174_);
v___x_177_ = lp_mathlib_LinearMap_comp___redArg___lam__0(v___x_176_, v_F_173_, v___y_175_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__1(lean_object* v_F_178_, lean_object* v_i_179_, lean_object* v___y_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = lean_apply_1(v_F_178_, v_i_179_);
v___x_182_ = lp_mathlib_LinearMap_toAddMonoidHom___redArg___lam__0(v___x_181_, v___y_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg___lam__2(lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_F_185_, lean_object* v___y_186_){
_start:
{
lean_object* v___f_187_; lean_object* v___x_125__overap_188_; lean_object* v___x_189_; 
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lsum___redArg___lam__1), 3, 1);
lean_closure_set(v___f_187_, 0, v_F_185_);
v___x_125__overap_188_ = lp_mathlib_DFinsupp_sumAddHom___redArg(v_inst_183_, v_inst_184_, v___f_187_);
v___x_189_ = lean_apply_1(v___x_125__overap_188_, v___y_186_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___redArg(lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_195_; 
lean_inc_ref(v_inst_192_);
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lsum___redArg___lam__0), 5, 2);
lean_closure_set(v___f_193_, 0, v_inst_190_);
lean_closure_set(v___f_193_, 1, v_inst_192_);
v___f_194_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lsum___redArg___lam__2), 4, 2);
lean_closure_set(v___f_194_, 0, v_inst_192_);
lean_closure_set(v___f_194_, 1, v_inst_191_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___f_194_);
lean_ctor_set(v___x_195_, 1, v___f_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum(lean_object* v_00_u03b9_196_, lean_object* v_R_197_, lean_object* v_S_198_, lean_object* v_M_199_, lean_object* v_N_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_DFinsupp_lsum___redArg(v_inst_202_, v_inst_204_, v_inst_206_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lsum___boxed(lean_object* v_00_u03b9_211_, lean_object* v_R_212_, lean_object* v_S_213_, lean_object* v_M_214_, lean_object* v_N_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_DFinsupp_lsum(v_00_u03b9_211_, v_R_212_, v_S_213_, v_M_214_, v_N_215_, v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_inst_222_, v_inst_223_, v_inst_224_);
lean_dec(v_inst_223_);
lean_dec_ref(v_inst_222_);
lean_dec(v_inst_220_);
lean_dec(v_inst_218_);
lean_dec_ref(v_inst_216_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___redArg___lam__2(lean_object* v_f_226_, lean_object* v_i_227_, lean_object* v_x_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_apply_2(v_f_226_, v_i_227_, v_x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___redArg(lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_f_232_){
_start:
{
lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___f_235_; lean_object* v___x_236_; 
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_233_, 0, v_inst_230_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_234_, 0, v_inst_231_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_linearMap___redArg___lam__2), 3, 1);
lean_closure_set(v___f_235_, 0, v_f_232_);
v___x_236_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_236_, 0, lean_box(0));
lean_closure_set(v___x_236_, 1, lean_box(0));
lean_closure_set(v___x_236_, 2, lean_box(0));
lean_closure_set(v___x_236_, 3, v___f_233_);
lean_closure_set(v___x_236_, 4, v___f_234_);
lean_closure_set(v___x_236_, 5, v___f_235_);
lean_closure_set(v___x_236_, 6, lean_box(0));
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap(lean_object* v_00_u03b9_237_, lean_object* v_R_238_, lean_object* v_inst_239_, lean_object* v_00_u03b2_u2081_240_, lean_object* v_00_u03b2_u2082_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_f_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_DFinsupp_mapRange_linearMap___redArg(v_inst_242_, v_inst_243_, v_f_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearMap___boxed(lean_object* v_00_u03b9_248_, lean_object* v_R_249_, lean_object* v_inst_250_, lean_object* v_00_u03b2_u2081_251_, lean_object* v_00_u03b2_u2082_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_, lean_object* v_f_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_DFinsupp_mapRange_linearMap(v_00_u03b9_248_, v_R_249_, v_inst_250_, v_00_u03b2_u2081_251_, v_00_u03b2_u2082_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_f_257_);
lean_dec(v_inst_256_);
lean_dec(v_inst_255_);
lean_dec_ref(v_inst_250_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__0(lean_object* v_e_259_, lean_object* v_i_260_, lean_object* v_x_261_){
_start:
{
lean_object* v___x_262_; lean_object* v_toLinearMap_263_; lean_object* v___x_264_; 
v___x_262_ = lean_apply_1(v_e_259_, v_i_260_);
v_toLinearMap_263_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_toLinearMap_263_);
lean_dec_ref(v___x_262_);
v___x_264_ = lean_apply_1(v_toLinearMap_263_, v_x_261_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__3(lean_object* v_e_265_, lean_object* v_i_266_, lean_object* v_x_267_){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_toLinearMap_270_; lean_object* v___x_271_; 
v___x_268_ = lean_apply_1(v_e_265_, v_i_266_);
v___x_269_ = lp_mathlib_LinearEquiv_symm___redArg(v___x_268_);
v_toLinearMap_270_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_toLinearMap_270_);
lean_dec_ref(v___x_269_);
v___x_271_ = lean_apply_1(v_toLinearMap_270_, v_x_267_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg(lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_e_274_){
_start:
{
lean_object* v___f_275_; lean_object* v___f_276_; lean_object* v___f_277_; lean_object* v___f_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
lean_inc_ref(v_e_274_);
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_275_, 0, v_e_274_);
v___f_276_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_276_, 0, v_inst_273_);
v___f_277_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lmk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_277_, 0, v_inst_272_);
v___f_278_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg___lam__3), 3, 1);
lean_closure_set(v___f_278_, 0, v_e_274_);
lean_inc_ref(v___f_276_);
lean_inc_ref(v___f_277_);
v___x_279_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_279_, 0, lean_box(0));
lean_closure_set(v___x_279_, 1, lean_box(0));
lean_closure_set(v___x_279_, 2, lean_box(0));
lean_closure_set(v___x_279_, 3, v___f_277_);
lean_closure_set(v___x_279_, 4, v___f_276_);
lean_closure_set(v___x_279_, 5, v___f_275_);
lean_closure_set(v___x_279_, 6, lean_box(0));
v___x_280_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_280_, 0, lean_box(0));
lean_closure_set(v___x_280_, 1, lean_box(0));
lean_closure_set(v___x_280_, 2, lean_box(0));
lean_closure_set(v___x_280_, 3, v___f_276_);
lean_closure_set(v___x_280_, 4, v___f_277_);
lean_closure_set(v___x_280_, 5, v___f_278_);
lean_closure_set(v___x_280_, 6, lean_box(0));
v___x_281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_279_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv(lean_object* v_00_u03b9_282_, lean_object* v_R_283_, lean_object* v_inst_284_, lean_object* v_00_u03b2_u2081_285_, lean_object* v_00_u03b2_u2082_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_e_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_DFinsupp_mapRange_linearEquiv___redArg(v_inst_287_, v_inst_288_, v_e_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_linearEquiv___boxed(lean_object* v_00_u03b9_293_, lean_object* v_R_294_, lean_object* v_inst_295_, lean_object* v_00_u03b2_u2081_296_, lean_object* v_00_u03b2_u2082_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_e_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_DFinsupp_mapRange_linearEquiv(v_00_u03b9_293_, v_R_294_, v_inst_295_, v_00_u03b2_u2081_296_, v_00_u03b2_u2082_297_, v_inst_298_, v_inst_299_, v_inst_300_, v_inst_301_, v_e_302_);
lean_dec(v_inst_301_);
lean_dec(v_inst_300_);
lean_dec_ref(v_inst_295_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__0(lean_object* v_inst_304_, lean_object* v_i_305_){
_start:
{
lean_inc_ref(v_inst_304_);
return v_inst_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__0___boxed(lean_object* v_inst_306_, lean_object* v_i_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_DFinsupp_coprodMap___redArg___lam__0(v_inst_306_, v_i_307_);
lean_dec(v_i_307_);
lean_dec_ref(v_inst_306_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__1(lean_object* v_x_309_, lean_object* v___y_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_LinearMap_id___lam__0(v___y_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg___lam__1___boxed(lean_object* v_x_312_, lean_object* v___y_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_DFinsupp_coprodMap___redArg___lam__1(v_x_312_, v___y_313_);
lean_dec(v___y_313_);
lean_dec(v_x_312_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___redArg(lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_f_319_){
_start:
{
lean_object* v___f_320_; lean_object* v___x_321_; lean_object* v_toLinearMap_322_; lean_object* v___f_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___f_326_; 
lean_inc_ref(v_inst_317_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_coprodMap___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_320_, 0, v_inst_317_);
lean_inc_ref(v___f_320_);
v___x_321_ = lp_mathlib_DFinsupp_lsum___redArg(v___f_320_, v_inst_317_, v_inst_318_);
v_toLinearMap_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_toLinearMap_322_);
lean_dec_ref(v___x_321_);
v___f_323_ = ((lean_object*)(lp_mathlib_DFinsupp_coprodMap___redArg___closed__0));
v___x_324_ = lean_apply_1(v_toLinearMap_322_, v___f_323_);
v___x_325_ = lp_mathlib_DFinsupp_mapRange_linearMap___redArg(v_inst_316_, v___f_320_, v_f_319_);
v___f_326_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_326_, 0, v___x_325_);
lean_closure_set(v___f_326_, 1, v___x_324_);
return v___f_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap(lean_object* v_00_u03b9_327_, lean_object* v_R_328_, lean_object* v_M_329_, lean_object* v_N_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_f_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_mathlib_DFinsupp_coprodMap___redArg(v_inst_332_, v_inst_334_, v_inst_336_, v_f_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coprodMap___boxed(lean_object* v_00_u03b9_339_, lean_object* v_R_340_, lean_object* v_M_341_, lean_object* v_N_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_f_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_DFinsupp_coprodMap(v_00_u03b9_339_, v_R_340_, v_M_341_, v_N_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_inst_346_, v_inst_347_, v_inst_348_, v_f_349_);
lean_dec(v_inst_347_);
lean_dec(v_inst_345_);
lean_dec_ref(v_inst_343_);
return v_res_350_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Sigma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_DFinsupp(builtin);
}
#ifdef __cplusplus
}
#endif
