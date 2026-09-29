// Lean compiler output
// Module: Mathlib.RingTheory.Ideal.Maps
// Imports: public import Init public meta import Init public import Mathlib.Data.DFinsupp.Module public import Mathlib.Order.KrullDimension public import Mathlib.RingTheory.Ideal.Operations
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
lean_object* lp_mathlib_Ideal_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_restrict___redArg(lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_giMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_giMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_orderEmbeddingOfSurjective___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_orderEmbeddingOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Ideal_piOrderIso___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ideal_piOrderIso___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ideal_piOrderIso___redArg___closed__0 = (const lean_object*)&lp_mathlib_Ideal_piOrderIso___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfBijective___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfBijective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Ideal_relIsoOfSurjective___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ideal_relIsoOfSurjective___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ideal_relIsoOfSurjective___closed__0 = (const lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__0_value;
static const lean_closure_object lp_mathlib_Ideal_relIsoOfSurjective___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ideal_relIsoOfSurjective___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ideal_relIsoOfSurjective___closed__1 = (const lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__1_value;
static const lean_ctor_object lp_mathlib_Ideal_relIsoOfSurjective___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__1_value),((lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__0_value)}};
static const lean_object* lp_mathlib_Ideal_relIsoOfSurjective___closed__2 = (const lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_mapHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_mapHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_annihilator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Module_annihilator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_annihilator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_annihilator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_RingEquiv_idealComapOrderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__0_value),((lean_object*)&lp_mathlib_Ideal_relIsoOfSurjective___closed__0_value)}};
static const lean_object* lp_mathlib_RingEquiv_idealComapOrderIso___closed__0 = (const lean_object*)&lp_mathlib_RingEquiv_idealComapOrderIso___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_idealComapOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_idealComapOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_map(lean_object* v_R_1_, lean_object* v_S_2_, lean_object* v_F_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_f_7_, lean_object* v_I_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_box(0);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_map___boxed(lean_object* v_R_10_, lean_object* v_S_11_, lean_object* v_F_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v_I_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Ideal_map(v_R_10_, v_S_11_, v_F_12_, v_inst_13_, v_inst_14_, v_inst_15_, v_f_16_, v_I_17_);
lean_dec(v_f_16_);
lean_dec(v_inst_15_);
lean_dec_ref(v_inst_14_);
lean_dec_ref(v_inst_13_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_comap(lean_object* v_R_19_, lean_object* v_S_20_, lean_object* v_F_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_f_25_, lean_object* v_inst_26_, lean_object* v_I_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_comap___boxed(lean_object* v_R_29_, lean_object* v_S_30_, lean_object* v_F_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_f_35_, lean_object* v_inst_36_, lean_object* v_I_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Ideal_comap(v_R_29_, v_S_30_, v_F_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_f_35_, v_inst_36_, v_I_37_);
lean_dec(v_f_35_);
lean_dec(v_inst_34_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_32_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_giMapComap___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___f_44_; 
v___x_43_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_map___boxed), 8, 7);
lean_closure_set(v___x_43_, 0, lean_box(0));
lean_closure_set(v___x_43_, 1, lean_box(0));
lean_closure_set(v___x_43_, 2, lean_box(0));
lean_closure_set(v___x_43_, 3, v_inst_39_);
lean_closure_set(v___x_43_, 4, v_inst_40_);
lean_closure_set(v___x_43_, 5, v_inst_41_);
lean_closure_set(v___x_43_, 6, v_f_42_);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_44_, 0, v___x_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_giMapComap(lean_object* v_R_45_, lean_object* v_S_46_, lean_object* v_F_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_, lean_object* v_inst_52_, lean_object* v_hf_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Ideal_giMapComap___redArg(v_inst_48_, v_inst_49_, v_inst_50_, v_f_51_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_orderEmbeddingOfSurjective___redArg(lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_comap___boxed), 9, 8);
lean_closure_set(v___x_59_, 0, lean_box(0));
lean_closure_set(v___x_59_, 1, lean_box(0));
lean_closure_set(v___x_59_, 2, lean_box(0));
lean_closure_set(v___x_59_, 3, v_inst_55_);
lean_closure_set(v___x_59_, 4, v_inst_56_);
lean_closure_set(v___x_59_, 5, v_inst_57_);
lean_closure_set(v___x_59_, 6, v_f_58_);
lean_closure_set(v___x_59_, 7, lean_box(0));
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_orderEmbeddingOfSurjective(lean_object* v_R_60_, lean_object* v_S_61_, lean_object* v_F_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_f_66_, lean_object* v_inst_67_, lean_object* v_hf_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_comap___boxed), 9, 8);
lean_closure_set(v___x_69_, 0, lean_box(0));
lean_closure_set(v___x_69_, 1, lean_box(0));
lean_closure_set(v___x_69_, 2, lean_box(0));
lean_closure_set(v___x_69_, 3, v_inst_63_);
lean_closure_set(v___x_69_, 4, v_inst_64_);
lean_closure_set(v___x_69_, 5, v_inst_65_);
lean_closure_set(v___x_69_, 6, v_f_66_);
lean_closure_set(v___x_69_, 7, lean_box(0));
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg___lam__0(lean_object* v_I_70_, lean_object* v_i_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_box(0);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg___lam__0___boxed(lean_object* v_I_73_, lean_object* v_i_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Ideal_piOrderIso___redArg___lam__0(v_I_73_, v_i_74_);
lean_dec(v_i_74_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___f_78_ = ((lean_object*)(lp_mathlib_Ideal_piOrderIso___redArg___closed__0));
v___x_79_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_pi___boxed), 4, 3);
lean_closure_set(v___x_79_, 0, lean_box(0));
lean_closure_set(v___x_79_, 1, lean_box(0));
lean_closure_set(v___x_79_, 2, v_inst_77_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___f_78_);
v___x_81_ = lp_mathlib_Equiv_symm___redArg(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_piOrderIso(lean_object* v_00_u03b9_82_, lean_object* v_R_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Ideal_piOrderIso___redArg(v_inst_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfBijective___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_f_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
lean_inc(v_f_90_);
lean_inc(v_inst_89_);
lean_inc_ref(v_inst_88_);
lean_inc_ref(v_inst_87_);
v___x_91_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_comap___boxed), 9, 8);
lean_closure_set(v___x_91_, 0, lean_box(0));
lean_closure_set(v___x_91_, 1, lean_box(0));
lean_closure_set(v___x_91_, 2, lean_box(0));
lean_closure_set(v___x_91_, 3, v_inst_87_);
lean_closure_set(v___x_91_, 4, v_inst_88_);
lean_closure_set(v___x_91_, 5, v_inst_89_);
lean_closure_set(v___x_91_, 6, v_f_90_);
lean_closure_set(v___x_91_, 7, lean_box(0));
v___x_92_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_map___boxed), 8, 7);
lean_closure_set(v___x_92_, 0, lean_box(0));
lean_closure_set(v___x_92_, 1, lean_box(0));
lean_closure_set(v___x_92_, 2, lean_box(0));
lean_closure_set(v___x_92_, 3, v_inst_87_);
lean_closure_set(v___x_92_, 4, v_inst_88_);
lean_closure_set(v___x_92_, 5, v_inst_89_);
lean_closure_set(v___x_92_, 6, v_f_90_);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_91_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfBijective(lean_object* v_R_94_, lean_object* v_S_95_, lean_object* v_F_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_f_100_, lean_object* v_inst_101_, lean_object* v_hf_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_mathlib_Ideal_relIsoOfBijective___redArg(v_inst_97_, v_inst_98_, v_inst_99_, v_f_100_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___lam__0(lean_object* v_I_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_box(0);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___lam__1(lean_object* v_J_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_box(0);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective(lean_object* v_R_113_, lean_object* v_S_114_, lean_object* v_F_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_f_120_, lean_object* v_hf_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Ideal_relIsoOfSurjective___closed__2));
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_relIsoOfSurjective___boxed(lean_object* v_R_123_, lean_object* v_S_124_, lean_object* v_F_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_f_130_, lean_object* v_hf_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Ideal_relIsoOfSurjective(v_R_123_, v_S_124_, v_F_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_f_130_, v_hf_131_);
lean_dec(v_f_130_);
lean_dec(v_inst_128_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_126_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_mapHom___redArg(lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_f_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_map___boxed), 8, 7);
lean_closure_set(v___x_137_, 0, lean_box(0));
lean_closure_set(v___x_137_, 1, lean_box(0));
lean_closure_set(v___x_137_, 2, lean_box(0));
lean_closure_set(v___x_137_, 3, v_inst_133_);
lean_closure_set(v___x_137_, 4, v_inst_134_);
lean_closure_set(v___x_137_, 5, v_inst_135_);
lean_closure_set(v___x_137_, 6, v_f_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_mapHom(lean_object* v_R_138_, lean_object* v_S_139_, lean_object* v_F_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_rc_144_, lean_object* v_f_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_map___boxed), 8, 7);
lean_closure_set(v___x_146_, 0, lean_box(0));
lean_closure_set(v___x_146_, 1, lean_box(0));
lean_closure_set(v___x_146_, 2, lean_box(0));
lean_closure_set(v___x_146_, 3, v_inst_141_);
lean_closure_set(v___x_146_, 4, v_inst_142_);
lean_closure_set(v___x_146_, 5, v_inst_143_);
lean_closure_set(v___x_146_, 6, v_f_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ker(lean_object* v_R_147_, lean_object* v_S_148_, lean_object* v_F_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_rcf_153_, lean_object* v_f_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lean_box(0);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ker___boxed(lean_object* v_R_156_, lean_object* v_S_157_, lean_object* v_F_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_rcf_162_, lean_object* v_f_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_RingHom_ker(v_R_156_, v_S_157_, v_F_158_, v_inst_159_, v_inst_160_, v_inst_161_, v_rcf_162_, v_f_163_);
lean_dec(v_f_163_);
lean_dec(v_inst_161_);
lean_dec_ref(v_inst_160_);
lean_dec_ref(v_inst_159_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_annihilator(lean_object* v_R_165_, lean_object* v_M_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lean_box(0);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Module_annihilator___boxed(lean_object* v_R_171_, lean_object* v_M_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Module_annihilator(v_R_171_, v_M_172_, v_inst_173_, v_inst_174_, v_inst_175_);
lean_dec(v_inst_175_);
lean_dec_ref(v_inst_174_);
lean_dec_ref(v_inst_173_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_annihilator(lean_object* v_R_177_, lean_object* v_M_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_N_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_box(0);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_annihilator___boxed(lean_object* v_R_184_, lean_object* v_M_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_N_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Submodule_annihilator(v_R_184_, v_M_185_, v_inst_186_, v_inst_187_, v_inst_188_, v_N_189_);
lean_dec(v_inst_188_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___redArg___lam__0(lean_object* v_f__inv_191_, lean_object* v_g_192_, lean_object* v_b_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_194_ = lean_apply_1(v_f__inv_191_, v_b_193_);
v___x_195_ = lean_apply_1(v_g_192_, v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___redArg(lean_object* v_f__inv_196_, lean_object* v_g_197_){
_start:
{
lean_object* v___f_198_; 
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_198_, 0, v_f__inv_196_);
lean_closure_set(v___f_198_, 1, v_g_197_);
return v___f_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux(lean_object* v_A_199_, lean_object* v_B_200_, lean_object* v_C_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_f_205_, lean_object* v_f__inv_206_, lean_object* v_hf_207_, lean_object* v_g_208_, lean_object* v_hg_209_){
_start:
{
lean_object* v___f_210_; 
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_liftOfRightInverseAux___redArg___lam__0), 3, 2);
lean_closure_set(v___f_210_, 0, v_f__inv_206_);
lean_closure_set(v___f_210_, 1, v_g_208_);
return v___f_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverseAux___boxed(lean_object* v_A_211_, lean_object* v_B_212_, lean_object* v_C_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_, lean_object* v_f__inv_218_, lean_object* v_hf_219_, lean_object* v_g_220_, lean_object* v_hg_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_RingHom_liftOfRightInverseAux(v_A_211_, v_B_212_, v_C_213_, v_inst_214_, v_inst_215_, v_inst_216_, v_f_217_, v_f__inv_218_, v_hf_219_, v_g_220_, v_hg_221_);
lean_dec(v_f_217_);
lean_dec_ref(v_inst_216_);
lean_dec_ref(v_inst_215_);
lean_dec_ref(v_inst_214_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__0(lean_object* v_f__inv_223_, lean_object* v_g_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_RingHom_liftOfRightInverseAux___redArg___lam__0(v_f__inv_223_, v_g_224_, v___y_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__1(lean_object* v_f_227_, lean_object* v_00_u03c6_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_RingHom_comp___redArg___lam__0(v_f_227_, v_00_u03c6_228_, v___y_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___redArg(lean_object* v_f_231_, lean_object* v_f__inv_232_){
_start:
{
lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___x_235_; 
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__0), 3, 1);
lean_closure_set(v___f_233_, 0, v_f__inv_232_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_liftOfRightInverse___redArg___lam__1), 3, 1);
lean_closure_set(v___f_234_, 0, v_f_231_);
v___x_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_235_, 0, v___f_233_);
lean_ctor_set(v___x_235_, 1, v___f_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse(lean_object* v_A_236_, lean_object* v_B_237_, lean_object* v_C_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_f_242_, lean_object* v_f__inv_243_, lean_object* v_hf_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lp_mathlib_RingHom_liftOfRightInverse___redArg(v_f_242_, v_f__inv_243_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_liftOfRightInverse___boxed(lean_object* v_A_246_, lean_object* v_B_247_, lean_object* v_C_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_f_252_, lean_object* v_f__inv_253_, lean_object* v_hf_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_RingHom_liftOfRightInverse(v_A_246_, v_B_247_, v_C_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_f_252_, v_f__inv_253_, v_hf_254_);
lean_dec_ref(v_inst_251_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_249_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_idealComapOrderIso(lean_object* v_R_258_, lean_object* v_S_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_e_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = ((lean_object*)(lp_mathlib_RingEquiv_idealComapOrderIso___closed__0));
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingEquiv_idealComapOrderIso___boxed(lean_object* v_R_264_, lean_object* v_S_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_e_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_RingEquiv_idealComapOrderIso(v_R_264_, v_S_265_, v_inst_266_, v_inst_267_, v_e_268_);
lean_dec_ref(v_e_268_);
lean_dec_ref(v_inst_267_);
lean_dec_ref(v_inst_266_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap___redArg(lean_object* v_inst_270_){
_start:
{
lean_object* v_algebraMap_271_; lean_object* v___x_272_; 
v_algebraMap_271_ = lean_ctor_get(v_inst_270_, 1);
lean_inc(v_algebraMap_271_);
lean_dec_ref(v_inst_270_);
v___x_272_ = lp_mathlib_LinearMap_restrict___redArg(v_algebraMap_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap(lean_object* v_R_273_, lean_object* v_inst_274_, lean_object* v_S_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_I_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_Algebra_idealMap___redArg(v_inst_277_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_idealMap___boxed(lean_object* v_R_280_, lean_object* v_inst_281_, lean_object* v_S_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_I_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_Algebra_idealMap(v_R_280_, v_inst_281_, v_S_282_, v_inst_283_, v_inst_284_, v_I_285_);
lean_dec_ref(v_inst_283_);
lean_dec_ref(v_inst_281_);
return v_res_286_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_KrullDimension(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_KrullDimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_KrullDimension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_KrullDimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(builtin);
}
#ifdef __cplusplus
}
#endif
