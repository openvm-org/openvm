// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Order
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Module.Defs public import Mathlib.Algebra.Order.Pi public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.DFinsupp.Module
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
lean_object* lp_mathlib_DFinsupp_zipWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_orderEmbeddingToFun___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_orderEmbeddingToFun___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_DFinsupp_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_DFinsupp_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE(lean_object* v_00_u03b9_1_, lean_object* v_00_u03b1_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instLE___boxed(lean_object* v_00_u03b9_6_, lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_DFinsupp_instLE(v_00_u03b9_6_, v_00_u03b1_7_, v_inst_8_, v_inst_9_);
lean_dec_ref(v_inst_9_);
lean_dec(v_inst_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___lam__0(lean_object* v_f_11_, lean_object* v___y_12_){
_start:
{
lean_object* v_toFun_13_; lean_object* v___x_14_; 
v_toFun_13_ = lean_ctor_get(v_f_11_, 0);
lean_inc(v_toFun_13_);
lean_dec_ref(v_f_11_);
v___x_14_ = lean_apply_1(v_toFun_13_, v___y_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun(lean_object* v_00_u03b9_16_, lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = ((lean_object*)(lp_mathlib_DFinsupp_orderEmbeddingToFun___closed__0));
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_orderEmbeddingToFun___boxed(lean_object* v_00_u03b9_21_, lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_DFinsupp_orderEmbeddingToFun(v_00_u03b9_21_, v_00_u03b1_22_, v_inst_23_, v_inst_24_);
lean_dec_ref(v_inst_24_);
lean_dec(v_inst_23_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder(lean_object* v_00_u03b9_29_, lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_DFinsupp_instPreorder___closed__0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPreorder___boxed(lean_object* v_00_u03b9_34_, lean_object* v_00_u03b1_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_DFinsupp_instPreorder(v_00_u03b9_34_, v_00_u03b1_35_, v_inst_36_, v_inst_37_);
lean_dec_ref(v_inst_37_);
lean_dec(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0(lean_object* v_inst_39_, lean_object* v_i_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_apply_1(v_inst_39_, v_i_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg(lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; lean_object* v___x_45_; 
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instPartialOrder___redArg___lam__0), 2, 1);
lean_closure_set(v___f_44_, 0, v_inst_43_);
v___x_45_ = lp_mathlib_DFinsupp_instPreorder(lean_box(0), lean_box(0), v_inst_42_, v___f_44_);
lean_dec_ref(v___f_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___redArg___boxed(lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_DFinsupp_instPartialOrder___redArg(v_inst_46_, v_inst_47_);
lean_dec(v_inst_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder(lean_object* v_00_u03b9_49_, lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_DFinsupp_instPartialOrder___redArg(v_inst_51_, v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instPartialOrder___boxed(lean_object* v_00_u03b9_54_, lean_object* v_00_u03b1_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_DFinsupp_instPartialOrder(v_00_u03b9_54_, v_00_u03b1_55_, v_inst_56_, v_inst_57_);
lean_dec(v_inst_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0(lean_object* v_inst_59_, lean_object* v_i_60_){
_start:
{
lean_object* v___x_61_; lean_object* v_toPartialOrder_62_; 
v___x_61_ = lean_apply_1(v_inst_59_, v_i_60_);
v_toPartialOrder_62_ = lean_ctor_get(v___x_61_, 0);
lean_inc_ref(v_toPartialOrder_62_);
lean_dec_ref(v___x_61_);
return v_toPartialOrder_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1(lean_object* v_inst_63_, lean_object* v_x_64_, lean_object* v_x1_65_, lean_object* v_x2_66_){
_start:
{
lean_object* v___x_67_; lean_object* v_inf_68_; lean_object* v___x_69_; 
v___x_67_ = lean_apply_1(v_inst_63_, v_x_64_);
v_inf_68_ = lean_ctor_get(v___x_67_, 1);
lean_inc(v_inf_68_);
lean_dec_ref(v___x_67_);
v___x_69_ = lean_apply_2(v_inf_68_, v_x1_65_, v_x2_66_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf___redArg(lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v___f_72_; lean_object* v___f_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
lean_inc_ref(v_inst_71_);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__0), 2, 1);
lean_closure_set(v___f_72_, 0, v_inst_71_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeInf___redArg___lam__1), 4, 1);
lean_closure_set(v___f_73_, 0, v_inst_71_);
v___x_74_ = lp_mathlib_DFinsupp_instPartialOrder___redArg(v_inst_70_, v___f_72_);
lean_inc_n(v_inst_70_, 2);
v___x_75_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_75_, 0, lean_box(0));
lean_closure_set(v___x_75_, 1, lean_box(0));
lean_closure_set(v___x_75_, 2, lean_box(0));
lean_closure_set(v___x_75_, 3, lean_box(0));
lean_closure_set(v___x_75_, 4, v_inst_70_);
lean_closure_set(v___x_75_, 5, v_inst_70_);
lean_closure_set(v___x_75_, 6, v_inst_70_);
lean_closure_set(v___x_75_, 7, v___f_73_);
lean_closure_set(v___x_75_, 8, lean_box(0));
v___x_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_74_);
lean_ctor_set(v___x_76_, 1, v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeInf(lean_object* v_00_u03b9_77_, lean_object* v_00_u03b1_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_DFinsupp_instSemilatticeInf___redArg(v_inst_79_, v_inst_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0(lean_object* v_inst_82_, lean_object* v_i_83_){
_start:
{
lean_object* v___x_84_; lean_object* v_toPartialOrder_85_; 
v___x_84_ = lean_apply_1(v_inst_82_, v_i_83_);
v_toPartialOrder_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc_ref(v_toPartialOrder_85_);
lean_dec_ref(v___x_84_);
return v_toPartialOrder_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1(lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_x1_88_, lean_object* v_x2_89_){
_start:
{
lean_object* v___x_90_; lean_object* v_sup_91_; lean_object* v___x_92_; 
v___x_90_ = lean_apply_1(v_inst_86_, v_x_87_);
v_sup_91_ = lean_ctor_get(v___x_90_, 1);
lean_inc(v_sup_91_);
lean_dec_ref(v___x_90_);
v___x_92_ = lean_apply_2(v_sup_91_, v_x1_88_, v_x2_89_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup___redArg(lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; lean_object* v___f_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
lean_inc_ref(v_inst_94_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_95_, 0, v_inst_94_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSemilatticeSup___redArg___lam__1), 4, 1);
lean_closure_set(v___f_96_, 0, v_inst_94_);
v___x_97_ = lp_mathlib_DFinsupp_instPartialOrder___redArg(v_inst_93_, v___f_95_);
lean_inc_n(v_inst_93_, 2);
v___x_98_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_98_, 0, lean_box(0));
lean_closure_set(v___x_98_, 1, lean_box(0));
lean_closure_set(v___x_98_, 2, lean_box(0));
lean_closure_set(v___x_98_, 3, lean_box(0));
lean_closure_set(v___x_98_, 4, v_inst_93_);
lean_closure_set(v___x_98_, 5, v_inst_93_);
lean_closure_set(v___x_98_, 6, v_inst_93_);
lean_closure_set(v___x_98_, 7, v___f_96_);
lean_closure_set(v___x_98_, 8, lean_box(0));
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_97_);
lean_ctor_set(v___x_99_, 1, v___x_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSemilatticeSup(lean_object* v_00_u03b9_100_, lean_object* v_00_u03b1_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_DFinsupp_instSemilatticeSup___redArg(v_inst_102_, v_inst_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__0(lean_object* v_inst_105_, lean_object* v_i_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = lean_apply_1(v_inst_105_, v_i_106_);
v___x_108_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__1(lean_object* v_inst_109_, lean_object* v_x_110_, lean_object* v_x1_111_, lean_object* v_x2_112_){
_start:
{
lean_object* v___x_113_; lean_object* v_toSemilatticeSup_114_; lean_object* v_sup_115_; lean_object* v___x_116_; 
v___x_113_ = lean_apply_1(v_inst_109_, v_x_110_);
v_toSemilatticeSup_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc_ref(v_toSemilatticeSup_114_);
lean_dec_ref(v___x_113_);
v_sup_115_ = lean_ctor_get(v_toSemilatticeSup_114_, 1);
lean_inc(v_sup_115_);
lean_dec_ref(v_toSemilatticeSup_114_);
v___x_116_ = lean_apply_2(v_sup_115_, v_x1_111_, v_x2_112_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg___lam__2(lean_object* v___f_117_, lean_object* v_x_118_, lean_object* v_x1_119_, lean_object* v_x2_120_){
_start:
{
lean_object* v___x_121_; lean_object* v_inf_122_; lean_object* v___x_123_; 
v___x_121_ = lean_apply_1(v___f_117_, v_x_118_);
v_inf_122_ = lean_ctor_get(v___x_121_, 1);
lean_inc(v_inf_122_);
lean_dec_ref(v___x_121_);
v___x_123_ = lean_apply_2(v_inf_122_, v_x1_119_, v_x2_120_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice___redArg(lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___x_127_; lean_object* v_toPartialOrder_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_140_; 
lean_inc_ref(v_inst_125_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_126_, 0, v_inst_125_);
lean_inc_ref(v___f_126_);
lean_inc(v_inst_124_);
v___x_127_ = lp_mathlib_DFinsupp_instSemilatticeInf___redArg(v_inst_124_, v___f_126_);
v_toPartialOrder_128_ = lean_ctor_get(v___x_127_, 0);
v_isSharedCheck_140_ = !lean_is_exclusive(v___x_127_);
if (v_isSharedCheck_140_ == 0)
{
lean_object* v_unused_141_; 
v_unused_141_ = lean_ctor_get(v___x_127_, 1);
lean_dec(v_unused_141_);
v___x_130_ = v___x_127_;
v_isShared_131_ = v_isSharedCheck_140_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_toPartialOrder_128_);
lean_dec(v___x_127_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_140_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___f_132_; lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_136_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__1), 4, 1);
lean_closure_set(v___f_132_, 0, v_inst_125_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lattice___redArg___lam__2), 4, 1);
lean_closure_set(v___f_133_, 0, v___f_126_);
lean_inc_n(v_inst_124_, 3);
v___x_134_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_134_, 0, lean_box(0));
lean_closure_set(v___x_134_, 1, lean_box(0));
lean_closure_set(v___x_134_, 2, lean_box(0));
lean_closure_set(v___x_134_, 3, lean_box(0));
lean_closure_set(v___x_134_, 4, v_inst_124_);
lean_closure_set(v___x_134_, 5, v_inst_124_);
lean_closure_set(v___x_134_, 6, v_inst_124_);
lean_closure_set(v___x_134_, 7, v___f_132_);
lean_closure_set(v___x_134_, 8, lean_box(0));
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 1, v___x_134_);
v___x_136_ = v___x_130_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v_toPartialOrder_128_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v___x_134_);
v___x_136_ = v_reuseFailAlloc_139_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
lean_inc_n(v_inst_124_, 2);
v___x_137_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_137_, 0, lean_box(0));
lean_closure_set(v___x_137_, 1, lean_box(0));
lean_closure_set(v___x_137_, 2, lean_box(0));
lean_closure_set(v___x_137_, 3, lean_box(0));
lean_closure_set(v___x_137_, 4, v_inst_124_);
lean_closure_set(v___x_137_, 5, v_inst_124_);
lean_closure_set(v___x_137_, 6, v_inst_124_);
lean_closure_set(v___x_137_, 7, v___f_133_);
lean_closure_set(v___x_137_, 8, lean_box(0));
v___x_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_136_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
return v___x_138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_lattice(lean_object* v_00_u03b9_142_, lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_DFinsupp_lattice___redArg(v_inst_144_, v_inst_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg___lam__0(lean_object* v_inst_147_, lean_object* v_x_148_){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v_toZero_152_; 
v___x_149_ = lean_apply_1(v_inst_147_, v_x_148_);
v___x_150_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_149_);
lean_dec_ref(v___x_149_);
v___x_151_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_150_);
v_toZero_152_ = lean_ctor_get(v___x_151_, 0);
lean_inc(v_toZero_152_);
lean_dec_ref(v___x_151_);
return v_toZero_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v___f_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_154_, 0, v_inst_153_);
v___x_155_ = lean_box(0);
v___x_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_156_, 0, v___f_154_);
lean_ctor_set(v___x_156_, 1, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass(lean_object* v_00_u03b9_157_, lean_object* v_00_u03b1_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___redArg(v_inst_159_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass___boxed(lean_object* v_00_u03b9_163_, lean_object* v_00_u03b1_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_DFinsupp_instOrderBotOfIsBotZeroClass(v_00_u03b9_163_, v_00_u03b1_164_, v_inst_165_, v_inst_166_, v_inst_167_);
lean_dec_ref(v_inst_166_);
return v_res_168_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg___lam__1(lean_object* v___f_169_, lean_object* v_x_170_, lean_object* v_x_171_, lean_object* v_inst_172_, lean_object* v_a_173_, lean_object* v_h_174_){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; uint8_t v___x_178_; 
lean_inc(v___f_169_);
lean_inc_n(v_a_173_, 2);
v___x_175_ = lean_apply_2(v___f_169_, v_x_170_, v_a_173_);
v___x_176_ = lean_apply_2(v___f_169_, v_x_171_, v_a_173_);
v___x_177_ = lean_apply_3(v_inst_172_, v_a_173_, v___x_175_, v___x_176_);
v___x_178_ = lean_unbox(v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___lam__1___boxed(lean_object* v___f_179_, lean_object* v_x_180_, lean_object* v_x_181_, lean_object* v_inst_182_, lean_object* v_a_183_, lean_object* v_h_184_){
_start:
{
uint8_t v_res_185_; lean_object* v_r_186_; 
v_res_185_ = lp_mathlib_DFinsupp_decidableLE___redArg___lam__1(v___f_179_, v_x_180_, v_x_181_, v_inst_182_, v_a_183_, v_h_184_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE___redArg(lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_x_190_, lean_object* v_x_191_){
_start:
{
lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___x_194_; uint8_t v___x_195_; 
v___f_192_ = ((lean_object*)(lp_mathlib_DFinsupp_orderEmbeddingToFun___closed__0));
lean_inc_ref(v_x_190_);
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_decidableLE___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_193_, 0, v___f_192_);
lean_closure_set(v___f_193_, 1, v_x_190_);
lean_closure_set(v___f_193_, 2, v_x_191_);
lean_closure_set(v___f_193_, 3, v_inst_189_);
v___x_194_ = lp_mathlib_DFinsupp_support___redArg(v_inst_187_, v_inst_188_, v_x_190_);
v___x_195_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v___x_194_, v___f_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___redArg___boxed(lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_x_199_, lean_object* v_x_200_){
_start:
{
uint8_t v_res_201_; lean_object* v_r_202_; 
v_res_201_ = lp_mathlib_DFinsupp_decidableLE___redArg(v_inst_196_, v_inst_197_, v_inst_198_, v_x_199_, v_x_200_);
v_r_202_ = lean_box(v_res_201_);
return v_r_202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableLE(lean_object* v_00_u03b9_203_, lean_object* v_00_u03b1_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_x_211_, lean_object* v_x_212_){
_start:
{
uint8_t v___x_213_; 
v___x_213_ = lp_mathlib_DFinsupp_decidableLE___redArg(v_inst_208_, v_inst_209_, v_inst_210_, v_x_211_, v_x_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableLE___boxed(lean_object* v_00_u03b9_214_, lean_object* v_00_u03b1_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_x_222_, lean_object* v_x_223_){
_start:
{
uint8_t v_res_224_; lean_object* v_r_225_; 
v_res_224_ = lp_mathlib_DFinsupp_decidableLE(v_00_u03b9_214_, v_00_u03b1_215_, v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_x_222_, v_x_223_);
lean_dec_ref(v_inst_217_);
lean_dec_ref(v_inst_216_);
v_r_225_ = lean_box(v_res_224_);
return v_r_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__0(lean_object* v_inst_226_, lean_object* v_i_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v_toZero_231_; 
v___x_228_ = lean_apply_1(v_inst_226_, v_i_227_);
v___x_229_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_228_);
lean_dec_ref(v___x_228_);
v___x_230_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_229_);
v_toZero_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_toZero_231_);
lean_dec_ref(v___x_230_);
return v_toZero_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg___lam__1(lean_object* v_inst_232_, lean_object* v_x_233_, lean_object* v_m_234_, lean_object* v_n_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_apply_3(v_inst_232_, v_x_233_, v_m_234_, v_n_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___redArg(lean_object* v_inst_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___f_239_; lean_object* v___f_240_; lean_object* v___x_241_; 
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_tsub___redArg___lam__0), 2, 1);
lean_closure_set(v___f_239_, 0, v_inst_237_);
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_tsub___redArg___lam__1), 4, 1);
lean_closure_set(v___f_240_, 0, v_inst_238_);
lean_inc_ref_n(v___f_239_, 2);
v___x_241_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_241_, 0, lean_box(0));
lean_closure_set(v___x_241_, 1, lean_box(0));
lean_closure_set(v___x_241_, 2, lean_box(0));
lean_closure_set(v___x_241_, 3, lean_box(0));
lean_closure_set(v___x_241_, 4, v___f_239_);
lean_closure_set(v___x_241_, 5, v___f_239_);
lean_closure_set(v___x_241_, 6, v___f_239_);
lean_closure_set(v___x_241_, 7, v___f_240_);
lean_closure_set(v___x_241_, 8, lean_box(0));
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub(lean_object* v_00_u03b9_242_, lean_object* v_00_u03b1_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_DFinsupp_tsub___redArg(v_inst_244_, v_inst_247_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_tsub___boxed(lean_object* v_00_u03b9_250_, lean_object* v_00_u03b1_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib_DFinsupp_tsub(v_00_u03b9_250_, v_00_u03b1_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_);
lean_dec_ref(v_inst_253_);
return v_res_257_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_Order(builtin);
}
#ifdef __cplusplus
}
#endif
