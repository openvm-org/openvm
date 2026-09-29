// Lean compiler output
// Module: Mathlib.Data.Finsupp.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.BigOperators.Finsupp.Basic public import Mathlib.Algebra.BigOperators.Group.Finset.Preimage public import Mathlib.Algebra.Group.Indicator public import Mathlib.Data.Rat.BigOperators
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
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finsupp_instInhabited___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_sectR___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_disjiUnion___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_disjSum___redArg(lean_object*, lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivCongrLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filterAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filterAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finsupp_piecewise___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_piecewise___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_piecewise___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___redArg___lam__0(lean_object* v_toFun_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
lean_inc(v_a_2_);
v___x_3_ = lean_apply_1(v_toFun_1_, v_a_2_);
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v_a_2_);
lean_ctor_set(v___x_4_, 1, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___redArg(lean_object* v_f_5_){
_start:
{
lean_object* v_support_6_; lean_object* v_toFun_7_; lean_object* v___f_8_; lean_object* v___x_9_; 
v_support_6_ = lean_ctor_get(v_f_5_, 0);
lean_inc(v_support_6_);
v_toFun_7_ = lean_ctor_get(v_f_5_, 1);
lean_inc(v_toFun_7_);
lean_dec_ref(v_f_5_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_graph___redArg___lam__0), 2, 1);
lean_closure_set(v___f_8_, 0, v_toFun_7_);
v___x_9_ = lp_mathlib_Finset_map___redArg(v___f_8_, v_support_6_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph(lean_object* v_00_u03b1_10_, lean_object* v_M_11_, lean_object* v_inst_12_, lean_object* v_f_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Finsupp_graph___redArg(v_f_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_graph___boxed(lean_object* v_00_u03b1_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_f_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Finsupp_graph(v_00_u03b1_15_, v_M_16_, v_inst_17_, v_f_18_);
lean_dec(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___redArg___lam__0(lean_object* v_f_20_, lean_object* v_toFun_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_23_; lean_object* v_toFun_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_23_ = lp_mathlib_Equiv_symm___redArg(v_f_20_);
v_toFun_24_ = lean_ctor_get(v___x_23_, 0);
lean_inc(v_toFun_24_);
lean_dec_ref(v___x_23_);
v___x_25_ = lean_apply_1(v_toFun_24_, v_a_22_);
v___x_26_ = lean_apply_1(v_toFun_21_, v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___redArg(lean_object* v_f_27_, lean_object* v_l_28_){
_start:
{
lean_object* v_support_29_; lean_object* v_toFun_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_40_; 
v_support_29_ = lean_ctor_get(v_l_28_, 0);
v_toFun_30_ = lean_ctor_get(v_l_28_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v_l_28_);
if (v_isSharedCheck_40_ == 0)
{
v___x_32_ = v_l_28_;
v_isShared_33_ = v_isSharedCheck_40_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_toFun_30_);
lean_inc(v_support_29_);
lean_dec(v_l_28_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_40_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___f_34_; lean_object* v___f_35_; lean_object* v___x_36_; lean_object* v___x_38_; 
lean_inc_ref(v_f_27_);
v___f_34_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_equivMapDomain___redArg___lam__0), 3, 2);
lean_closure_set(v___f_34_, 0, v_f_27_);
lean_closure_set(v___f_34_, 1, v_toFun_30_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_35_, 0, v_f_27_);
v___x_36_ = lp_mathlib_Finset_map___redArg(v___f_35_, v_support_29_);
if (v_isShared_33_ == 0)
{
lean_ctor_set(v___x_32_, 1, v___f_34_);
lean_ctor_set(v___x_32_, 0, v___x_36_);
v___x_38_ = v___x_32_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v___x_36_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v___f_34_);
v___x_38_ = v_reuseFailAlloc_39_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain(lean_object* v_00_u03b1_41_, lean_object* v_00_u03b2_42_, lean_object* v_M_43_, lean_object* v_inst_44_, lean_object* v_f_45_, lean_object* v_l_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_Finsupp_equivMapDomain___redArg(v_f_45_, v_l_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivMapDomain___boxed(lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_M_50_, lean_object* v_inst_51_, lean_object* v_f_52_, lean_object* v_l_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Finsupp_equivMapDomain(v_00_u03b1_48_, v_00_u03b2_49_, v_M_50_, v_inst_51_, v_f_52_, v_l_53_);
lean_dec(v_inst_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivCongrLeft___redArg(lean_object* v_inst_55_, lean_object* v_f_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
lean_inc_ref(v_f_56_);
lean_inc(v_inst_55_);
v___x_57_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_equivMapDomain___boxed), 6, 5);
lean_closure_set(v___x_57_, 0, lean_box(0));
lean_closure_set(v___x_57_, 1, lean_box(0));
lean_closure_set(v___x_57_, 2, lean_box(0));
lean_closure_set(v___x_57_, 3, v_inst_55_);
lean_closure_set(v___x_57_, 4, v_f_56_);
v___x_58_ = lp_mathlib_Equiv_symm___redArg(v_f_56_);
v___x_59_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_equivMapDomain___boxed), 6, 5);
lean_closure_set(v___x_59_, 0, lean_box(0));
lean_closure_set(v___x_59_, 1, lean_box(0));
lean_closure_set(v___x_59_, 2, lean_box(0));
lean_closure_set(v___x_59_, 3, v_inst_55_);
lean_closure_set(v___x_59_, 4, v___x_58_);
v___x_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_57_);
lean_ctor_set(v___x_60_, 1, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_equivCongrLeft(lean_object* v_00_u03b1_61_, lean_object* v_00_u03b2_62_, lean_object* v_M_63_, lean_object* v_inst_64_, lean_object* v_f_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Finsupp_equivCongrLeft___redArg(v_inst_64_, v_f_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg___lam__0(lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_toFun_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
lean_inc(v_a_70_);
v___x_71_ = lean_apply_1(v_inst_67_, v_a_70_);
v___x_72_ = lean_unbox(v___x_71_);
if (v___x_72_ == 0)
{
lean_dec(v_a_70_);
lean_dec(v_toFun_69_);
lean_inc(v_inst_68_);
return v_inst_68_;
}
else
{
lean_object* v___x_73_; 
v___x_73_ = lean_apply_1(v_toFun_69_, v_a_70_);
return v___x_73_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg___lam__0___boxed(lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_toFun_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Finsupp_filter___redArg___lam__0(v_inst_74_, v_inst_75_, v_toFun_76_, v_a_77_);
lean_dec(v_inst_75_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_f_81_){
_start:
{
lean_object* v_support_82_; lean_object* v_toFun_83_; lean_object* v___x_85_; uint8_t v_isShared_86_; uint8_t v_isSharedCheck_92_; 
v_support_82_ = lean_ctor_get(v_f_81_, 0);
v_toFun_83_ = lean_ctor_get(v_f_81_, 1);
v_isSharedCheck_92_ = !lean_is_exclusive(v_f_81_);
if (v_isSharedCheck_92_ == 0)
{
v___x_85_ = v_f_81_;
v_isShared_86_ = v_isSharedCheck_92_;
goto v_resetjp_84_;
}
else
{
lean_inc(v_toFun_83_);
lean_inc(v_support_82_);
lean_dec(v_f_81_);
v___x_85_ = lean_box(0);
v_isShared_86_ = v_isSharedCheck_92_;
goto v_resetjp_84_;
}
v_resetjp_84_:
{
lean_object* v___f_87_; lean_object* v___x_88_; lean_object* v___x_90_; 
lean_inc_ref(v_inst_80_);
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_filter___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_87_, 0, v_inst_80_);
lean_closure_set(v___f_87_, 1, v_inst_79_);
lean_closure_set(v___f_87_, 2, v_toFun_83_);
v___x_88_ = lp_mathlib_Multiset_filter___redArg(v_inst_80_, v_support_82_);
if (v_isShared_86_ == 0)
{
lean_ctor_set(v___x_85_, 1, v___f_87_);
lean_ctor_set(v___x_85_, 0, v___x_88_);
v___x_90_ = v___x_85_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_91_; 
v_reuseFailAlloc_91_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_91_, 0, v___x_88_);
lean_ctor_set(v_reuseFailAlloc_91_, 1, v___f_87_);
v___x_90_ = v_reuseFailAlloc_91_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
return v___x_90_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filter(lean_object* v_00_u03b1_93_, lean_object* v_M_94_, lean_object* v_inst_95_, lean_object* v_p_96_, lean_object* v_inst_97_, lean_object* v_f_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_mathlib_Finsupp_filter___redArg(v_inst_95_, v_inst_97_, v_f_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filterAddHom___redArg(lean_object* v_inst_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; lean_object* v_toZero_103_; lean_object* v___x_104_; 
v___x_102_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_100_);
v_toZero_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_toZero_103_);
lean_dec_ref(v___x_102_);
v___x_104_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_filter), 6, 5);
lean_closure_set(v___x_104_, 0, lean_box(0));
lean_closure_set(v___x_104_, 1, lean_box(0));
lean_closure_set(v___x_104_, 2, v_toZero_103_);
lean_closure_set(v___x_104_, 3, lean_box(0));
lean_closure_set(v___x_104_, 4, v_inst_101_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_filterAddHom(lean_object* v_00_u03b1_105_, lean_object* v_M_106_, lean_object* v_inst_107_, lean_object* v_p_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Finsupp_filterAddHom___redArg(v_inst_107_, v_inst_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg___lam__0(lean_object* v_toFun_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___f_113_; lean_object* v___x_114_; lean_object* v_support_115_; lean_object* v___x_116_; 
lean_inc(v_a_112_);
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sectR___redArg___lam__0), 2, 1);
lean_closure_set(v___f_113_, 0, v_a_112_);
v___x_114_ = lean_apply_1(v_toFun_111_, v_a_112_);
v_support_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_support_115_);
lean_dec_ref(v___x_114_);
v___x_116_ = lp_mathlib_Finset_map___redArg(v___f_113_, v_support_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg___lam__1(lean_object* v_toFun_117_, lean_object* v_x_118_){
_start:
{
lean_object* v_fst_119_; lean_object* v_snd_120_; lean_object* v___x_121_; lean_object* v_toFun_122_; lean_object* v___x_123_; 
v_fst_119_ = lean_ctor_get(v_x_118_, 0);
lean_inc(v_fst_119_);
v_snd_120_ = lean_ctor_get(v_x_118_, 1);
lean_inc(v_snd_120_);
lean_dec_ref(v_x_118_);
v___x_121_ = lean_apply_1(v_toFun_117_, v_fst_119_);
v_toFun_122_ = lean_ctor_get(v___x_121_, 1);
lean_inc(v_toFun_122_);
lean_dec_ref(v___x_121_);
v___x_123_ = lean_apply_1(v_toFun_122_, v_snd_120_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___redArg(lean_object* v_f_124_){
_start:
{
lean_object* v_support_125_; lean_object* v_toFun_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_136_; 
v_support_125_ = lean_ctor_get(v_f_124_, 0);
v_toFun_126_ = lean_ctor_get(v_f_124_, 1);
v_isSharedCheck_136_ = !lean_is_exclusive(v_f_124_);
if (v_isSharedCheck_136_ == 0)
{
v___x_128_ = v_f_124_;
v_isShared_129_ = v_isSharedCheck_136_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_toFun_126_);
lean_inc(v_support_125_);
lean_dec(v_f_124_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_136_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___f_130_; lean_object* v___f_131_; lean_object* v___x_132_; lean_object* v___x_134_; 
lean_inc(v_toFun_126_);
v___f_130_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_uncurry___redArg___lam__0), 2, 1);
lean_closure_set(v___f_130_, 0, v_toFun_126_);
v___f_131_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_uncurry___redArg___lam__1), 2, 1);
lean_closure_set(v___f_131_, 0, v_toFun_126_);
v___x_132_ = lp_mathlib_Finset_disjiUnion___redArg(v_support_125_, v___f_130_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 1, v___f_131_);
lean_ctor_set(v___x_128_, 0, v___x_132_);
v___x_134_ = v___x_128_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___x_132_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v___f_131_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry(lean_object* v_00_u03b1_137_, lean_object* v_00_u03b2_138_, lean_object* v_M_139_, lean_object* v_inst_140_, lean_object* v_f_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_Finsupp_uncurry___redArg(v_f_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uncurry___boxed(lean_object* v_00_u03b1_143_, lean_object* v_00_u03b2_144_, lean_object* v_M_145_, lean_object* v_inst_146_, lean_object* v_f_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Finsupp_uncurry(v_00_u03b1_143_, v_00_u03b2_144_, v_M_145_, v_inst_146_, v_f_147_);
lean_dec(v_inst_146_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___redArg___lam__0(lean_object* v_toFun_149_, lean_object* v___y_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lean_apply_1(v_toFun_149_, v___y_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___redArg(lean_object* v_f_152_, lean_object* v_g_153_){
_start:
{
lean_object* v_support_154_; lean_object* v_toFun_155_; lean_object* v_support_156_; lean_object* v_toFun_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_168_; 
v_support_154_ = lean_ctor_get(v_f_152_, 0);
lean_inc(v_support_154_);
v_toFun_155_ = lean_ctor_get(v_f_152_, 1);
lean_inc(v_toFun_155_);
lean_dec_ref(v_f_152_);
v_support_156_ = lean_ctor_get(v_g_153_, 0);
v_toFun_157_ = lean_ctor_get(v_g_153_, 1);
v_isSharedCheck_168_ = !lean_is_exclusive(v_g_153_);
if (v_isSharedCheck_168_ == 0)
{
v___x_159_ = v_g_153_;
v_isShared_160_ = v_isSharedCheck_168_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_toFun_157_);
lean_inc(v_support_156_);
lean_dec(v_g_153_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_168_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_166_; 
v___f_161_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_sumElim___redArg___lam__0), 2, 1);
lean_closure_set(v___f_161_, 0, v_toFun_155_);
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_sumElim___redArg___lam__0), 2, 1);
lean_closure_set(v___f_162_, 0, v_toFun_157_);
v___x_163_ = lp_mathlib_Multiset_disjSum___redArg(v_support_154_, v_support_156_);
v___x_164_ = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(v___x_164_, 0, lean_box(0));
lean_closure_set(v___x_164_, 1, lean_box(0));
lean_closure_set(v___x_164_, 2, lean_box(0));
lean_closure_set(v___x_164_, 3, v___f_161_);
lean_closure_set(v___x_164_, 4, v___f_162_);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 1, v___x_164_);
lean_ctor_set(v___x_159_, 0, v___x_163_);
v___x_166_ = v___x_159_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v___x_163_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v___x_164_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_00_u03b3_171_, lean_object* v_inst_172_, lean_object* v_f_173_, lean_object* v_g_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_Finsupp_sumElim___redArg(v_f_173_, v_g_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sumElim___boxed(lean_object* v_00_u03b1_176_, lean_object* v_00_u03b2_177_, lean_object* v_00_u03b3_178_, lean_object* v_inst_179_, lean_object* v_f_180_, lean_object* v_g_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Finsupp_sumElim(v_00_u03b1_176_, v_00_u03b2_177_, v_00_u03b3_178_, v_inst_179_, v_f_180_, v_g_181_);
lean_dec(v_inst_179_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfRight___redArg(lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_Finsupp_instInhabited___redArg(v_inst_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfRight(lean_object* v_00_u03b1_185_, lean_object* v_R_186_, lean_object* v_inst_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Finsupp_instInhabited___redArg(v_inst_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfLeft___redArg(lean_object* v_inst_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_Finsupp_instInhabited___redArg(v_inst_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_uniqueOfLeft(lean_object* v_00_u03b1_192_, lean_object* v_R_193_, lean_object* v_inst_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_Finsupp_instInhabited___redArg(v_inst_194_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___redArg___lam__0(lean_object* v_inst_197_, lean_object* v_toFun_198_, lean_object* v_toFun_199_, lean_object* v_a_200_){
_start:
{
lean_object* v___x_201_; uint8_t v___x_202_; 
lean_inc(v_a_200_);
v___x_201_ = lean_apply_1(v_inst_197_, v_a_200_);
v___x_202_ = lean_unbox(v___x_201_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; 
lean_dec(v_toFun_199_);
v___x_203_ = lean_apply_1(v_toFun_198_, v_a_200_);
return v___x_203_;
}
else
{
lean_object* v___x_204_; 
lean_dec(v_toFun_198_);
v___x_204_ = lean_apply_1(v_toFun_199_, v_a_200_);
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___redArg(lean_object* v_inst_206_, lean_object* v_f_207_, lean_object* v_g_208_){
_start:
{
lean_object* v_support_209_; lean_object* v_toFun_210_; lean_object* v_support_211_; lean_object* v_toFun_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_224_; 
v_support_209_ = lean_ctor_get(v_f_207_, 0);
lean_inc(v_support_209_);
v_toFun_210_ = lean_ctor_get(v_f_207_, 1);
lean_inc(v_toFun_210_);
lean_dec_ref(v_f_207_);
v_support_211_ = lean_ctor_get(v_g_208_, 0);
v_toFun_212_ = lean_ctor_get(v_g_208_, 1);
v_isSharedCheck_224_ = !lean_is_exclusive(v_g_208_);
if (v_isSharedCheck_224_ == 0)
{
v___x_214_ = v_g_208_;
v_isShared_215_ = v_isSharedCheck_224_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_toFun_212_);
lean_inc(v_support_211_);
lean_dec(v_g_208_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_224_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___f_216_; lean_object* v___f_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_222_; 
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_piecewise___redArg___lam__0), 4, 3);
lean_closure_set(v___f_216_, 0, v_inst_206_);
lean_closure_set(v___f_216_, 1, v_toFun_212_);
lean_closure_set(v___f_216_, 2, v_toFun_210_);
v___f_217_ = ((lean_object*)(lp_mathlib_Finsupp_piecewise___redArg___closed__0));
v___x_218_ = lp_mathlib_Finset_map___redArg(v___f_217_, v_support_209_);
v___x_219_ = lp_mathlib_Finset_map___redArg(v___f_217_, v_support_211_);
v___x_220_ = l_List_appendTR___redArg(v___x_218_, v___x_219_);
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v___f_216_);
lean_ctor_set(v___x_214_, 0, v___x_220_);
v___x_222_ = v___x_214_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_220_);
lean_ctor_set(v_reuseFailAlloc_223_, 1, v___f_216_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise(lean_object* v_00_u03b1_225_, lean_object* v_M_226_, lean_object* v_inst_227_, lean_object* v_P_228_, lean_object* v_inst_229_, lean_object* v_f_230_, lean_object* v_g_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_Finsupp_piecewise___redArg(v_inst_229_, v_f_230_, v_g_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_piecewise___boxed(lean_object* v_00_u03b1_233_, lean_object* v_M_234_, lean_object* v_inst_235_, lean_object* v_P_236_, lean_object* v_inst_237_, lean_object* v_f_238_, lean_object* v_g_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_Finsupp_piecewise(v_00_u03b1_233_, v_M_234_, v_inst_235_, v_P_236_, v_inst_237_, v_f_238_, v_g_239_);
lean_dec(v_inst_235_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg___lam__0(lean_object* v_inst_241_, lean_object* v_x_242_){
_start:
{
lean_inc(v_inst_241_);
return v_inst_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg___lam__0___boxed(lean_object* v_inst_243_, lean_object* v_x_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_Finsupp_extendDomain___redArg___lam__0(v_inst_243_, v_x_244_);
lean_dec(v_x_244_);
lean_dec(v_inst_243_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain___redArg(lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_f_248_){
_start:
{
lean_object* v___f_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_extendDomain___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_249_, 0, v_inst_246_);
v___x_250_ = lean_box(0);
v___x_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, v___f_249_);
v___x_252_ = lp_mathlib_Finsupp_piecewise___redArg(v_inst_247_, v_f_248_, v___x_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_extendDomain(lean_object* v_00_u03b1_253_, lean_object* v_M_254_, lean_object* v_inst_255_, lean_object* v_P_256_, lean_object* v_inst_257_, lean_object* v_f_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Finsupp_extendDomain___redArg(v_inst_255_, v_inst_257_, v_f_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___redArg(lean_object* v_inst_260_, lean_object* v_e_261_){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v_toZero_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_274_; 
v___x_262_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_260_);
v___x_263_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_262_);
v_toZero_264_ = lean_ctor_get(v___x_263_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_263_);
if (v_isSharedCheck_274_ == 0)
{
lean_object* v_unused_275_; 
v_unused_275_ = lean_ctor_get(v___x_263_, 1);
lean_dec(v_unused_275_);
v___x_266_ = v___x_263_;
v_isShared_267_ = v_isSharedCheck_274_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_toZero_264_);
lean_dec(v___x_263_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_274_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_272_; 
lean_inc_ref(v_e_261_);
lean_inc(v_toZero_264_);
v___x_268_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_equivMapDomain___boxed), 6, 5);
lean_closure_set(v___x_268_, 0, lean_box(0));
lean_closure_set(v___x_268_, 1, lean_box(0));
lean_closure_set(v___x_268_, 2, lean_box(0));
lean_closure_set(v___x_268_, 3, v_toZero_264_);
lean_closure_set(v___x_268_, 4, v_e_261_);
v___x_269_ = lp_mathlib_Equiv_symm___redArg(v_e_261_);
v___x_270_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_equivMapDomain___boxed), 6, 5);
lean_closure_set(v___x_270_, 0, lean_box(0));
lean_closure_set(v___x_270_, 1, lean_box(0));
lean_closure_set(v___x_270_, 2, lean_box(0));
lean_closure_set(v___x_270_, 3, v_toZero_264_);
lean_closure_set(v___x_270_, 4, v___x_269_);
if (v_isShared_267_ == 0)
{
lean_ctor_set(v___x_266_, 1, v___x_270_);
lean_ctor_set(v___x_266_, 0, v___x_268_);
v___x_272_ = v___x_266_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v___x_268_);
lean_ctor_set(v_reuseFailAlloc_273_, 1, v___x_270_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___redArg___boxed(lean_object* v_inst_276_, lean_object* v_e_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_Finsupp_domCongr___redArg(v_inst_276_, v_e_277_);
lean_dec_ref(v_inst_276_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_M_281_, lean_object* v_inst_282_, lean_object* v_e_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_Finsupp_domCongr___redArg(v_inst_282_, v_e_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_domCongr___boxed(lean_object* v_00_u03b1_285_, lean_object* v_00_u03b2_286_, lean_object* v_M_287_, lean_object* v_inst_288_, lean_object* v_e_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_Finsupp_domCongr(v_00_u03b1_285_, v_00_u03b2_286_, v_M_287_, v_inst_288_, v_e_289_);
lean_dec_ref(v_inst_288_);
return v_res_290_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Preimage(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Indicator(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_BigOperators(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Preimage(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Indicator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_BigOperators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Indicator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
