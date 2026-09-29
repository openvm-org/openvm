// Lean compiler output
// Module: Mathlib.Order.CompleteLattice.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.NAry public import Mathlib.Data.ULift public import Mathlib.Order.Bounds.Image public import Mathlib.Order.CompleteLattice.Defs public import Mathlib.Order.Hom.Set
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
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instBoundedOrder___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instBoundedOrder___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
extern lean_object* lp_mathlib_Prop_instBoundedOrder;
extern lean_object* lp_mathlib_Prop_instDistribLattice;
static lean_once_cell_t lp_mathlib_Prop_instCompleteLattice___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instCompleteLattice___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instCompleteLattice;
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__6(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_completeLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_completeLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_completeLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_completeLattice___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Prop_instCompleteLattice___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lp_mathlib_Prop_instBoundedOrder;
v___x_2_ = lp_mathlib_Prop_instDistribLattice;
v___x_3_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
lean_ctor_set(v___x_3_, 1, lean_box(0));
lean_ctor_set(v___x_3_, 2, lean_box(0));
lean_ctor_set(v___x_3_, 3, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompleteLattice(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Prop_instCompleteLattice___closed__0, &lp_mathlib_Prop_instCompleteLattice___closed__0_once, _init_lp_mathlib_Prop_instCompleteLattice___closed__0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg___lam__0(lean_object* v_inst_5_, lean_object* v_s_6_, lean_object* v_i_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_apply_2(v_inst_5_, v_i_7_, lean_box(0));
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
return v___f_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_16_, 0, v_inst_15_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet(lean_object* v_00_u03b1_17_, lean_object* v_00_u03b2_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__0(lean_object* v_inst_21_, lean_object* v_i_22_){
_start:
{
lean_object* v___x_23_; lean_object* v_toBoundedOrder_24_; 
v___x_23_ = lean_apply_1(v_inst_21_, v_i_22_);
v_toBoundedOrder_24_ = lean_ctor_get(v___x_23_, 3);
lean_inc_ref(v_toBoundedOrder_24_);
lean_dec_ref(v___x_23_);
return v_toBoundedOrder_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__1(lean_object* v_inst_25_, lean_object* v_i_26_){
_start:
{
lean_object* v___x_27_; lean_object* v_toLattice_28_; 
v___x_27_ = lean_apply_1(v_inst_25_, v_i_26_);
v_toLattice_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc_ref(v_toLattice_28_);
lean_dec_ref(v___x_27_);
return v_toLattice_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__2(lean_object* v_inst_29_, lean_object* v_i_30_, lean_object* v___y_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v_toSupSet_34_; lean_object* v___x_35_; 
v___x_32_ = lean_apply_1(v_inst_29_, v_i_30_);
v___x_33_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v___x_32_);
v_toSupSet_34_ = lean_ctor_get(v___x_33_, 1);
lean_inc(v_toSupSet_34_);
lean_dec_ref(v___x_33_);
v___x_35_ = lean_apply_1(v_toSupSet_34_, lean_box(0));
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__3(lean_object* v_inst_36_, lean_object* v_i_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v_toInfSet_41_; lean_object* v___x_42_; 
v___x_39_ = lean_apply_1(v_inst_36_, v_i_37_);
v___x_40_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v___x_39_);
v_toInfSet_41_ = lean_ctor_get(v___x_40_, 1);
lean_inc(v_toInfSet_41_);
lean_dec_ref(v___x_40_);
v___x_42_ = lean_apply_1(v_toInfSet_41_, lean_box(0));
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v___f_44_; lean_object* v___f_45_; lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___f_50_; lean_object* v___f_51_; lean_object* v___x_52_; 
lean_inc_ref_n(v_inst_43_, 3);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(v___f_44_, 0, v_inst_43_);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__1), 2, 1);
lean_closure_set(v___f_45_, 0, v_inst_43_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__2), 3, 1);
lean_closure_set(v___f_46_, 0, v_inst_43_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__3), 3, 1);
lean_closure_set(v___f_47_, 0, v_inst_43_);
v___x_48_ = lp_mathlib_Pi_instBoundedOrder___redArg(v___f_44_);
v___x_49_ = lp_mathlib_Pi_instLattice___redArg(v___f_45_);
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_50_, 0, v___f_46_);
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(v___f_51_, 0, v___f_47_);
v___x_52_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_52_, 0, v___x_49_);
lean_ctor_set(v___x_52_, 1, v___f_50_);
lean_ctor_set(v___x_52_, 2, v___f_51_);
lean_ctor_set(v___x_52_, 3, v___x_48_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice(lean_object* v_00_u03b1_53_, lean_object* v_00_u03b2_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Pi_instCompleteLattice___redArg(v_inst_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg___lam__0(lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_s_59_){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_apply_1(v_inst_57_, lean_box(0));
v___x_61_ = lean_apply_1(v_inst_58_, lean_box(0));
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_60_);
lean_ctor_set(v___x_62_, 1, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg(lean_object* v_inst_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v___f_65_; 
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_65_, 0, v_inst_63_);
lean_closure_set(v___f_65_, 1, v_inst_64_);
return v___f_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet(lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___f_70_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_70_, 0, v_inst_68_);
lean_closure_set(v___f_70_, 1, v_inst_69_);
return v___f_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet___redArg(lean_object* v_inst_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___f_73_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_73_, 0, v_inst_71_);
lean_closure_set(v___f_73_, 1, v_inst_72_);
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet(lean_object* v_00_u03b1_74_, lean_object* v_00_u03b2_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_78_, 0, v_inst_76_);
lean_closure_set(v___f_78_, 1, v_inst_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v_toLattice_83_; lean_object* v_toBoundedOrder_84_; lean_object* v_toLattice_85_; lean_object* v_toBoundedOrder_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v_toSupSet_90_; lean_object* v___x_91_; lean_object* v_toSupSet_92_; lean_object* v_toInfSet_93_; lean_object* v_toInfSet_94_; lean_object* v___f_95_; lean_object* v___f_96_; lean_object* v___x_97_; 
lean_inc_ref(v_inst_79_);
v___x_81_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_79_);
lean_inc_ref(v_inst_80_);
v___x_82_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_80_);
v_toLattice_83_ = lean_ctor_get(v_inst_79_, 0);
v_toBoundedOrder_84_ = lean_ctor_get(v_inst_79_, 3);
v_toLattice_85_ = lean_ctor_get(v_inst_80_, 0);
v_toBoundedOrder_86_ = lean_ctor_get(v_inst_80_, 3);
lean_inc_ref(v_toBoundedOrder_86_);
lean_inc_ref(v_toBoundedOrder_84_);
v___x_87_ = lp_mathlib_Prod_instBoundedOrder___redArg(v_toBoundedOrder_84_, v_toBoundedOrder_86_);
lean_inc_ref(v_toLattice_85_);
lean_inc_ref(v_toLattice_83_);
v___x_88_ = lp_mathlib_Prod_instLattice___redArg(v_toLattice_83_, v_toLattice_85_);
v___x_89_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_79_);
v_toSupSet_90_ = lean_ctor_get(v___x_89_, 1);
lean_inc(v_toSupSet_90_);
lean_dec_ref(v___x_89_);
v___x_91_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_80_);
v_toSupSet_92_ = lean_ctor_get(v___x_91_, 1);
lean_inc(v_toSupSet_92_);
lean_dec_ref(v___x_91_);
v_toInfSet_93_ = lean_ctor_get(v___x_81_, 1);
lean_inc(v_toInfSet_93_);
lean_dec_ref(v___x_81_);
v_toInfSet_94_ = lean_ctor_get(v___x_82_, 1);
lean_inc(v_toInfSet_94_);
lean_dec_ref(v___x_82_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_95_, 0, v_toSupSet_90_);
lean_closure_set(v___f_95_, 1, v_toSupSet_92_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(v___f_96_, 0, v_toInfSet_93_);
lean_closure_set(v___f_96_, 1, v_toInfSet_94_);
v___x_97_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_97_, 0, v___x_88_);
lean_ctor_set(v___x_97_, 1, v___f_95_);
lean_ctor_set(v___x_97_, 2, v___f_96_);
lean_ctor_set(v___x_97_, 3, v___x_87_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_99_, lean_object* v_inst_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Prod_instCompleteLattice___redArg(v_inst_100_, v_inst_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg___lam__0(lean_object* v_inst_103_, lean_object* v_a_104_, lean_object* v_b_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_apply_2(v_inst_103_, v_a_104_, v_b_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg(lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___f_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_115_, 0, v_inst_107_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_116_, 0, v_inst_108_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v_inst_109_);
lean_ctor_set(v___x_117_, 1, v_inst_110_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v___f_115_);
v___x_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v___f_116_);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v_inst_113_);
lean_ctor_set(v___x_120_, 1, v_inst_114_);
v___x_121_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_121_, 0, v___x_119_);
lean_ctor_set(v___x_121_, 1, v_inst_111_);
lean_ctor_set(v___x_121_, 2, v_inst_112_);
lean_ctor_set(v___x_121_, 3, v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice(lean_object* v_00_u03b1_122_, lean_object* v_00_u03b2_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_f_133_, lean_object* v_hf_134_, lean_object* v_le_135_, lean_object* v_lt_136_, lean_object* v_map__sup_137_, lean_object* v_map__inf_138_, lean_object* v_map__sSup_139_, lean_object* v_map__sInf_140_, lean_object* v_map__top_141_, lean_object* v_map__bot_142_){
_start:
{
lean_object* v___f_143_; lean_object* v___f_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___f_143_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_143_, 0, v_inst_124_);
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(v___f_144_, 0, v_inst_125_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v_inst_126_);
lean_ctor_set(v___x_145_, 1, v_inst_127_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v___f_143_);
v___x_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___f_144_);
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, v_inst_130_);
lean_ctor_set(v___x_148_, 1, v_inst_131_);
v___x_149_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_149_, 0, v___x_147_);
lean_ctor_set(v___x_149_, 1, v_inst_128_);
lean_ctor_set(v___x_149_, 2, v_inst_129_);
lean_ctor_set(v___x_149_, 3, v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___boxed(lean_object** _args){
lean_object* v_00_u03b1_150_ = _args[0];
lean_object* v_00_u03b2_151_ = _args[1];
lean_object* v_inst_152_ = _args[2];
lean_object* v_inst_153_ = _args[3];
lean_object* v_inst_154_ = _args[4];
lean_object* v_inst_155_ = _args[5];
lean_object* v_inst_156_ = _args[6];
lean_object* v_inst_157_ = _args[7];
lean_object* v_inst_158_ = _args[8];
lean_object* v_inst_159_ = _args[9];
lean_object* v_inst_160_ = _args[10];
lean_object* v_f_161_ = _args[11];
lean_object* v_hf_162_ = _args[12];
lean_object* v_le_163_ = _args[13];
lean_object* v_lt_164_ = _args[14];
lean_object* v_map__sup_165_ = _args[15];
lean_object* v_map__inf_166_ = _args[16];
lean_object* v_map__sSup_167_ = _args[17];
lean_object* v_map__sInf_168_ = _args[18];
lean_object* v_map__top_169_ = _args[19];
lean_object* v_map__bot_170_ = _args[20];
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Function_Injective_completeLattice(v_00_u03b1_150_, v_00_u03b2_151_, v_inst_152_, v_inst_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_inst_158_, v_inst_159_, v_inst_160_, v_f_161_, v_hf_162_, v_le_163_, v_lt_164_, v_map__sup_165_, v_map__inf_166_, v_map__sSup_167_, v_map__sInf_168_, v_map__top_169_, v_map__bot_170_);
lean_dec(v_f_161_);
lean_dec_ref(v_inst_160_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__0(lean_object* v_self_172_, lean_object* v___y_173_){
_start:
{
lean_object* v_toFun_174_; lean_object* v___x_175_; 
v_toFun_174_ = lean_ctor_get(v_self_172_, 0);
lean_inc(v_toFun_174_);
lean_dec_ref(v_self_172_);
v___x_175_ = lean_apply_1(v_toFun_174_, v___y_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__1(lean_object* v___f_176_, lean_object* v_e_177_, lean_object* v_inf_178_, lean_object* v_toFun_179_, lean_object* v_a_180_, lean_object* v_b_181_){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
lean_inc(v___f_176_);
lean_inc_ref(v_e_177_);
v___x_182_ = lean_apply_2(v___f_176_, v_e_177_, v_a_180_);
v___x_183_ = lean_apply_2(v___f_176_, v_e_177_, v_b_181_);
v___x_184_ = lean_apply_2(v_inf_178_, v___x_182_, v___x_183_);
v___x_185_ = lean_apply_1(v_toFun_179_, v___x_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__3(lean_object* v_min_186_, lean_object* v_a_187_, lean_object* v_b_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lean_apply_2(v_min_186_, v_a_187_, v_b_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__2(lean_object* v_toSemilatticeSup_190_, lean_object* v___f_191_, lean_object* v_e_192_, lean_object* v_toFun_193_, lean_object* v_a_194_, lean_object* v_b_195_){
_start:
{
lean_object* v_sup_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v_sup_196_ = lean_ctor_get(v_toSemilatticeSup_190_, 1);
lean_inc(v_sup_196_);
lean_dec_ref(v_toSemilatticeSup_190_);
lean_inc(v___f_191_);
lean_inc_ref(v_e_192_);
v___x_197_ = lean_apply_2(v___f_191_, v_e_192_, v_a_194_);
v___x_198_ = lean_apply_2(v___f_191_, v_e_192_, v_b_195_);
v___x_199_ = lean_apply_2(v_sup_196_, v___x_197_, v___x_198_);
v___x_200_ = lean_apply_1(v_toFun_193_, v___x_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__5(lean_object* v_toSupSet_201_, lean_object* v_toFun_202_, lean_object* v_s_203_){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = lean_apply_1(v_toSupSet_201_, lean_box(0));
v___x_205_ = lean_apply_1(v_toFun_202_, v___x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__4(lean_object* v_toInfSet_206_, lean_object* v_toFun_207_, lean_object* v_s_208_){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = lean_apply_1(v_toInfSet_206_, lean_box(0));
v___x_210_ = lean_apply_1(v_toFun_207_, v___x_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg___lam__6(lean_object* v___f_211_, lean_object* v_a_212_, lean_object* v_b_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_apply_2(v___f_211_, v_a_212_, v_b_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice___redArg(lean_object* v_e_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v_toBoundedOrder_218_; lean_object* v_toLattice_219_; lean_object* v_toOrderTop_220_; lean_object* v_toOrderBot_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_301_; 
v_toBoundedOrder_218_ = lean_ctor_get(v_inst_217_, 3);
lean_inc_ref(v_toBoundedOrder_218_);
v_toLattice_219_ = lean_ctor_get(v_inst_217_, 0);
lean_inc_ref(v_toLattice_219_);
v_toOrderTop_220_ = lean_ctor_get(v_toBoundedOrder_218_, 0);
v_toOrderBot_221_ = lean_ctor_get(v_toBoundedOrder_218_, 1);
v_isSharedCheck_301_ = !lean_is_exclusive(v_toBoundedOrder_218_);
if (v_isSharedCheck_301_ == 0)
{
v___x_223_ = v_toBoundedOrder_218_;
v_isShared_224_ = v_isSharedCheck_301_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_toOrderBot_221_);
lean_inc(v_toOrderTop_220_);
lean_dec(v_toBoundedOrder_218_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_301_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v_toFun_226_; lean_object* v___x_227_; lean_object* v_toSupSet_228_; lean_object* v___x_229_; lean_object* v_toInfSet_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_299_; 
lean_inc_ref(v_e_216_);
v___x_225_ = lp_mathlib_Equiv_symm___redArg(v_e_216_);
v_toFun_226_ = lean_ctor_get(v___x_225_, 0);
lean_inc(v_toFun_226_);
lean_dec_ref(v___x_225_);
lean_inc_ref(v_inst_217_);
v___x_227_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_217_);
v_toSupSet_228_ = lean_ctor_get(v___x_227_, 1);
lean_inc(v_toSupSet_228_);
lean_dec_ref(v___x_227_);
v___x_229_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_217_);
v_toInfSet_230_ = lean_ctor_get(v___x_229_, 1);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_229_);
if (v_isSharedCheck_299_ == 0)
{
lean_object* v_unused_300_; 
v_unused_300_ = lean_ctor_get(v___x_229_, 0);
lean_dec(v_unused_300_);
v___x_232_ = v___x_229_;
v_isShared_233_ = v_isSharedCheck_299_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_toInfSet_230_);
lean_dec(v___x_229_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_299_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v_toSemilatticeSup_234_; lean_object* v_inf_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_298_; 
v_toSemilatticeSup_234_ = lean_ctor_get(v_toLattice_219_, 0);
v_inf_235_ = lean_ctor_get(v_toLattice_219_, 1);
v_isSharedCheck_298_ = !lean_is_exclusive(v_toLattice_219_);
if (v_isSharedCheck_298_ == 0)
{
v___x_237_ = v_toLattice_219_;
v_isShared_238_ = v_isSharedCheck_298_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_inf_235_);
lean_inc(v_toSemilatticeSup_234_);
lean_dec(v_toLattice_219_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_298_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___f_239_; lean_object* v_min_240_; lean_object* v_le_241_; lean_object* v_lt_242_; lean_object* v_semilatticeInf_243_; lean_object* v_toPartialOrder_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_296_; 
v___f_239_ = ((lean_object*)(lp_mathlib_Equiv_completeLattice___redArg___closed__0));
lean_inc(v_toFun_226_);
lean_inc_ref(v_e_216_);
v_min_240_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__1), 6, 4);
lean_closure_set(v_min_240_, 0, v___f_239_);
lean_closure_set(v_min_240_, 1, v_e_216_);
lean_closure_set(v_min_240_, 2, v_inf_235_);
lean_closure_set(v_min_240_, 3, v_toFun_226_);
v_le_241_ = lean_box(0);
v_lt_242_ = lean_box(0);
lean_inc_ref(v_min_240_);
v_semilatticeInf_243_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_240_, v_le_241_, v_lt_242_);
v_toPartialOrder_244_ = lean_ctor_get(v_semilatticeInf_243_, 0);
v_isSharedCheck_296_ = !lean_is_exclusive(v_semilatticeInf_243_);
if (v_isSharedCheck_296_ == 0)
{
lean_object* v_unused_297_; 
v_unused_297_ = lean_ctor_get(v_semilatticeInf_243_, 1);
lean_dec(v_unused_297_);
v___x_246_ = v_semilatticeInf_243_;
v_isShared_247_ = v_isSharedCheck_296_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_toPartialOrder_244_);
lean_dec(v_semilatticeInf_243_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_296_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v_toLE_248_; lean_object* v_toLT_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_295_; 
v_toLE_248_ = lean_ctor_get(v_toPartialOrder_244_, 0);
v_toLT_249_ = lean_ctor_get(v_toPartialOrder_244_, 1);
v_isSharedCheck_295_ = !lean_is_exclusive(v_toPartialOrder_244_);
if (v_isSharedCheck_295_ == 0)
{
v___x_251_ = v_toPartialOrder_244_;
v_isShared_252_ = v_isSharedCheck_295_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_toLT_249_);
lean_inc(v_toLE_248_);
lean_dec(v_toPartialOrder_244_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_295_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___f_253_; lean_object* v___f_254_; lean_object* v___x_256_; 
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__3), 3, 1);
lean_closure_set(v___f_253_, 0, v_min_240_);
lean_inc(v_toFun_226_);
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__2), 6, 4);
lean_closure_set(v___f_254_, 0, v_toSemilatticeSup_234_);
lean_closure_set(v___f_254_, 1, v___f_239_);
lean_closure_set(v___f_254_, 2, v_e_216_);
lean_closure_set(v___f_254_, 3, v_toFun_226_);
if (v_isShared_252_ == 0)
{
v___x_256_ = v___x_251_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v_toLE_248_);
lean_ctor_set(v_reuseFailAlloc_294_, 1, v_toLT_249_);
v___x_256_ = v_reuseFailAlloc_294_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
lean_object* v___x_258_; 
lean_inc_ref(v___f_254_);
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 1, v___f_254_);
lean_ctor_set(v___x_246_, 0, v___x_256_);
v___x_258_ = v___x_246_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_256_);
lean_ctor_set(v_reuseFailAlloc_293_, 1, v___f_254_);
v___x_258_ = v_reuseFailAlloc_293_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
lean_object* v_lattice_260_; 
lean_inc_ref(v___f_253_);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 1, v___f_253_);
lean_ctor_set(v___x_237_, 0, v___x_258_);
v_lattice_260_ = v___x_237_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___x_258_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v___f_253_);
v_lattice_260_ = v_reuseFailAlloc_292_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
lean_object* v___x_261_; lean_object* v_toPartialOrder_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_290_; 
v___x_261_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_260_);
v_toPartialOrder_262_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_290_ == 0)
{
lean_object* v_unused_291_; 
v_unused_291_ = lean_ctor_get(v___x_261_, 1);
lean_dec(v_unused_291_);
v___x_264_ = v___x_261_;
v_isShared_265_ = v_isSharedCheck_290_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_toPartialOrder_262_);
lean_dec(v___x_261_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_290_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v_toLE_266_; lean_object* v_toLT_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_289_; 
v_toLE_266_ = lean_ctor_get(v_toPartialOrder_262_, 0);
v_toLT_267_ = lean_ctor_get(v_toPartialOrder_262_, 1);
v_isSharedCheck_289_ = !lean_is_exclusive(v_toPartialOrder_262_);
if (v_isSharedCheck_289_ == 0)
{
v___x_269_ = v_toPartialOrder_262_;
v_isShared_270_ = v_isSharedCheck_289_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_toLT_267_);
lean_inc(v_toLE_266_);
lean_dec(v_toPartialOrder_262_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_289_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v_top_271_; lean_object* v_bot_272_; lean_object* v_supSet_273_; lean_object* v_infSet_274_; lean_object* v___f_275_; lean_object* v___x_277_; 
lean_inc_n(v_toFun_226_, 3);
v_top_271_ = lean_apply_1(v_toFun_226_, v_toOrderTop_220_);
v_bot_272_ = lean_apply_1(v_toFun_226_, v_toOrderBot_221_);
v_supSet_273_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_273_, 0, v_toSupSet_228_);
lean_closure_set(v_supSet_273_, 1, v_toFun_226_);
v_infSet_274_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_274_, 0, v_toInfSet_230_);
lean_closure_set(v_infSet_274_, 1, v_toFun_226_);
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__6), 3, 1);
lean_closure_set(v___f_275_, 0, v___f_254_);
if (v_isShared_270_ == 0)
{
v___x_277_ = v___x_269_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v_toLE_266_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v_toLT_267_);
v___x_277_ = v_reuseFailAlloc_288_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
lean_object* v___x_279_; 
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 1, v___f_275_);
lean_ctor_set(v___x_264_, 0, v___x_277_);
v___x_279_ = v___x_264_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_287_; 
v_reuseFailAlloc_287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_287_, 0, v___x_277_);
lean_ctor_set(v_reuseFailAlloc_287_, 1, v___f_275_);
v___x_279_ = v_reuseFailAlloc_287_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
lean_object* v___x_281_; 
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 1, v___f_253_);
lean_ctor_set(v___x_232_, 0, v___x_279_);
v___x_281_ = v___x_232_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v___x_279_);
lean_ctor_set(v_reuseFailAlloc_286_, 1, v___f_253_);
v___x_281_ = v_reuseFailAlloc_286_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
lean_object* v___x_283_; 
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 1, v_bot_272_);
lean_ctor_set(v___x_223_, 0, v_top_271_);
v___x_283_ = v___x_223_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_top_271_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v_bot_272_);
v___x_283_ = v_reuseFailAlloc_285_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
lean_object* v___x_284_; 
v___x_284_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_284_, 0, v___x_281_);
lean_ctor_set(v___x_284_, 1, v_supSet_273_);
lean_ctor_set(v___x_284_, 2, v_infSet_274_);
lean_ctor_set(v___x_284_, 3, v___x_283_);
return v___x_284_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_completeLattice(lean_object* v_00_u03b1_302_, lean_object* v_00_u03b2_303_, lean_object* v_e_304_, lean_object* v_inst_305_){
_start:
{
lean_object* v_toBoundedOrder_306_; lean_object* v_toLattice_307_; lean_object* v_toOrderTop_308_; lean_object* v_toOrderBot_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_389_; 
v_toBoundedOrder_306_ = lean_ctor_get(v_inst_305_, 3);
lean_inc_ref(v_toBoundedOrder_306_);
v_toLattice_307_ = lean_ctor_get(v_inst_305_, 0);
lean_inc_ref(v_toLattice_307_);
v_toOrderTop_308_ = lean_ctor_get(v_toBoundedOrder_306_, 0);
v_toOrderBot_309_ = lean_ctor_get(v_toBoundedOrder_306_, 1);
v_isSharedCheck_389_ = !lean_is_exclusive(v_toBoundedOrder_306_);
if (v_isSharedCheck_389_ == 0)
{
v___x_311_ = v_toBoundedOrder_306_;
v_isShared_312_ = v_isSharedCheck_389_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_toOrderBot_309_);
lean_inc(v_toOrderTop_308_);
lean_dec(v_toBoundedOrder_306_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_389_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; lean_object* v_toFun_314_; lean_object* v___x_315_; lean_object* v_toSupSet_316_; lean_object* v___x_317_; lean_object* v_toInfSet_318_; lean_object* v___x_320_; uint8_t v_isShared_321_; uint8_t v_isSharedCheck_387_; 
lean_inc_ref(v_e_304_);
v___x_313_ = lp_mathlib_Equiv_symm___redArg(v_e_304_);
v_toFun_314_ = lean_ctor_get(v___x_313_, 0);
lean_inc(v_toFun_314_);
lean_dec_ref(v___x_313_);
lean_inc_ref(v_inst_305_);
v___x_315_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_305_);
v_toSupSet_316_ = lean_ctor_get(v___x_315_, 1);
lean_inc(v_toSupSet_316_);
lean_dec_ref(v___x_315_);
v___x_317_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(v_inst_305_);
v_toInfSet_318_ = lean_ctor_get(v___x_317_, 1);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_317_);
if (v_isSharedCheck_387_ == 0)
{
lean_object* v_unused_388_; 
v_unused_388_ = lean_ctor_get(v___x_317_, 0);
lean_dec(v_unused_388_);
v___x_320_ = v___x_317_;
v_isShared_321_ = v_isSharedCheck_387_;
goto v_resetjp_319_;
}
else
{
lean_inc(v_toInfSet_318_);
lean_dec(v___x_317_);
v___x_320_ = lean_box(0);
v_isShared_321_ = v_isSharedCheck_387_;
goto v_resetjp_319_;
}
v_resetjp_319_:
{
lean_object* v_toSemilatticeSup_322_; lean_object* v_inf_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_386_; 
v_toSemilatticeSup_322_ = lean_ctor_get(v_toLattice_307_, 0);
v_inf_323_ = lean_ctor_get(v_toLattice_307_, 1);
v_isSharedCheck_386_ = !lean_is_exclusive(v_toLattice_307_);
if (v_isSharedCheck_386_ == 0)
{
v___x_325_ = v_toLattice_307_;
v_isShared_326_ = v_isSharedCheck_386_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_inf_323_);
lean_inc(v_toSemilatticeSup_322_);
lean_dec(v_toLattice_307_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_386_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___f_327_; lean_object* v_min_328_; lean_object* v_le_329_; lean_object* v_lt_330_; lean_object* v_semilatticeInf_331_; lean_object* v_toPartialOrder_332_; lean_object* v___x_334_; uint8_t v_isShared_335_; uint8_t v_isSharedCheck_384_; 
v___f_327_ = ((lean_object*)(lp_mathlib_Equiv_completeLattice___redArg___closed__0));
lean_inc(v_toFun_314_);
lean_inc_ref(v_e_304_);
v_min_328_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__1), 6, 4);
lean_closure_set(v_min_328_, 0, v___f_327_);
lean_closure_set(v_min_328_, 1, v_e_304_);
lean_closure_set(v_min_328_, 2, v_inf_323_);
lean_closure_set(v_min_328_, 3, v_toFun_314_);
v_le_329_ = lean_box(0);
v_lt_330_ = lean_box(0);
lean_inc_ref(v_min_328_);
v_semilatticeInf_331_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v_min_328_, v_le_329_, v_lt_330_);
v_toPartialOrder_332_ = lean_ctor_get(v_semilatticeInf_331_, 0);
v_isSharedCheck_384_ = !lean_is_exclusive(v_semilatticeInf_331_);
if (v_isSharedCheck_384_ == 0)
{
lean_object* v_unused_385_; 
v_unused_385_ = lean_ctor_get(v_semilatticeInf_331_, 1);
lean_dec(v_unused_385_);
v___x_334_ = v_semilatticeInf_331_;
v_isShared_335_ = v_isSharedCheck_384_;
goto v_resetjp_333_;
}
else
{
lean_inc(v_toPartialOrder_332_);
lean_dec(v_semilatticeInf_331_);
v___x_334_ = lean_box(0);
v_isShared_335_ = v_isSharedCheck_384_;
goto v_resetjp_333_;
}
v_resetjp_333_:
{
lean_object* v_toLE_336_; lean_object* v_toLT_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_383_; 
v_toLE_336_ = lean_ctor_get(v_toPartialOrder_332_, 0);
v_toLT_337_ = lean_ctor_get(v_toPartialOrder_332_, 1);
v_isSharedCheck_383_ = !lean_is_exclusive(v_toPartialOrder_332_);
if (v_isSharedCheck_383_ == 0)
{
v___x_339_ = v_toPartialOrder_332_;
v_isShared_340_ = v_isSharedCheck_383_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_toLT_337_);
lean_inc(v_toLE_336_);
lean_dec(v_toPartialOrder_332_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_383_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___f_341_; lean_object* v___f_342_; lean_object* v___x_344_; 
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__3), 3, 1);
lean_closure_set(v___f_341_, 0, v_min_328_);
lean_inc(v_toFun_314_);
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__2), 6, 4);
lean_closure_set(v___f_342_, 0, v_toSemilatticeSup_322_);
lean_closure_set(v___f_342_, 1, v___f_327_);
lean_closure_set(v___f_342_, 2, v_e_304_);
lean_closure_set(v___f_342_, 3, v_toFun_314_);
if (v_isShared_340_ == 0)
{
v___x_344_ = v___x_339_;
goto v_reusejp_343_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_toLE_336_);
lean_ctor_set(v_reuseFailAlloc_382_, 1, v_toLT_337_);
v___x_344_ = v_reuseFailAlloc_382_;
goto v_reusejp_343_;
}
v_reusejp_343_:
{
lean_object* v___x_346_; 
lean_inc_ref(v___f_342_);
if (v_isShared_335_ == 0)
{
lean_ctor_set(v___x_334_, 1, v___f_342_);
lean_ctor_set(v___x_334_, 0, v___x_344_);
v___x_346_ = v___x_334_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v___x_344_);
lean_ctor_set(v_reuseFailAlloc_381_, 1, v___f_342_);
v___x_346_ = v_reuseFailAlloc_381_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
lean_object* v_lattice_348_; 
lean_inc_ref(v___f_341_);
if (v_isShared_326_ == 0)
{
lean_ctor_set(v___x_325_, 1, v___f_341_);
lean_ctor_set(v___x_325_, 0, v___x_346_);
v_lattice_348_ = v___x_325_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v___f_341_);
v_lattice_348_ = v_reuseFailAlloc_380_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
lean_object* v___x_349_; lean_object* v_toPartialOrder_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_378_; 
v___x_349_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_lattice_348_);
v_toPartialOrder_350_ = lean_ctor_get(v___x_349_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_349_);
if (v_isSharedCheck_378_ == 0)
{
lean_object* v_unused_379_; 
v_unused_379_ = lean_ctor_get(v___x_349_, 1);
lean_dec(v_unused_379_);
v___x_352_ = v___x_349_;
v_isShared_353_ = v_isSharedCheck_378_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_toPartialOrder_350_);
lean_dec(v___x_349_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_378_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v_toLE_354_; lean_object* v_toLT_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_377_; 
v_toLE_354_ = lean_ctor_get(v_toPartialOrder_350_, 0);
v_toLT_355_ = lean_ctor_get(v_toPartialOrder_350_, 1);
v_isSharedCheck_377_ = !lean_is_exclusive(v_toPartialOrder_350_);
if (v_isSharedCheck_377_ == 0)
{
v___x_357_ = v_toPartialOrder_350_;
v_isShared_358_ = v_isSharedCheck_377_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_toLT_355_);
lean_inc(v_toLE_354_);
lean_dec(v_toPartialOrder_350_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_377_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v_top_359_; lean_object* v_bot_360_; lean_object* v_supSet_361_; lean_object* v_infSet_362_; lean_object* v___f_363_; lean_object* v___x_365_; 
lean_inc_n(v_toFun_314_, 3);
v_top_359_ = lean_apply_1(v_toFun_314_, v_toOrderTop_308_);
v_bot_360_ = lean_apply_1(v_toFun_314_, v_toOrderBot_309_);
v_supSet_361_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__5), 3, 2);
lean_closure_set(v_supSet_361_, 0, v_toSupSet_316_);
lean_closure_set(v_supSet_361_, 1, v_toFun_314_);
v_infSet_362_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__4), 3, 2);
lean_closure_set(v_infSet_362_, 0, v_toInfSet_318_);
lean_closure_set(v_infSet_362_, 1, v_toFun_314_);
v___f_363_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_completeLattice___redArg___lam__6), 3, 1);
lean_closure_set(v___f_363_, 0, v___f_342_);
if (v_isShared_358_ == 0)
{
v___x_365_ = v___x_357_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_toLE_354_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v_toLT_355_);
v___x_365_ = v_reuseFailAlloc_376_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
lean_object* v___x_367_; 
if (v_isShared_353_ == 0)
{
lean_ctor_set(v___x_352_, 1, v___f_363_);
lean_ctor_set(v___x_352_, 0, v___x_365_);
v___x_367_ = v___x_352_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_365_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v___f_363_);
v___x_367_ = v_reuseFailAlloc_375_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
lean_object* v___x_369_; 
if (v_isShared_321_ == 0)
{
lean_ctor_set(v___x_320_, 1, v___f_341_);
lean_ctor_set(v___x_320_, 0, v___x_367_);
v___x_369_ = v___x_320_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v___x_367_);
lean_ctor_set(v_reuseFailAlloc_374_, 1, v___f_341_);
v___x_369_ = v_reuseFailAlloc_374_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
lean_object* v___x_371_; 
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 1, v_bot_360_);
lean_ctor_set(v___x_311_, 0, v_top_359_);
v___x_371_ = v___x_311_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v_top_359_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v_bot_360_);
v___x_371_ = v_reuseFailAlloc_373_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___x_372_; 
v___x_372_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_372_, 0, v___x_369_);
lean_ctor_set(v___x_372_, 1, v_supSet_361_);
lean_ctor_set(v___x_372_, 2, v_infSet_362_);
lean_ctor_set(v___x_372_, 3, v___x_371_);
return v___x_372_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instCompleteLattice = _init_lp_mathlib_Prop_instCompleteLattice();
lean_mark_persistent(lp_mathlib_Prop_instCompleteLattice);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
