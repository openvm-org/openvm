// Lean compiler output
// Module: Mathlib.Order.Sublattice
// Imports: public import Init public meta import Init public import Mathlib.Order.SupClosed
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
lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Sublattice_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sublattice_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_ofIsSublattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_ofIsSublattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Sublattice_instLatticeCoe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sublattice_instLatticeCoe___redArg___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_instLatticeCoe___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instDistribLatticeCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instDistribLatticeCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Sublattice_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sublattice_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sublattice_subtype___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sublattice_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_inclusion___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Sublattice_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_inclusion___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sublattice_instInf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sublattice_instInf___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sublattice_instInf___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_instInf___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Sublattice_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sublattice_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sublattice_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInhabited___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Sublattice_topEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sublattice_topEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sublattice_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instUniqueOfIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instUniqueOfIsEmpty___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Sublattice_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sublattice_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_LatticeHom_range(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_LatticeHom_range___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSetLike(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSetLike___boxed(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Sublattice_instSetLike(v_00_u03b1_4_, v_inst_5_);
lean_dec_ref(v_inst_5_);
return v_res_6_;
}
}
static lean_object* _init_lp_mathlib_Sublattice_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_box(0);
v___x_8_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instPartialOrder(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_mathlib_Sublattice_instPartialOrder___closed__0, &lp_mathlib_Sublattice_instPartialOrder___closed__0_once, _init_lp_mathlib_Sublattice_instPartialOrder___closed__0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instPartialOrder___boxed(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Sublattice_instPartialOrder(v_00_u03b1_12_, v_inst_13_);
lean_dec_ref(v_inst_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_ofIsSublattice(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_s_17_, lean_object* v_hs_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_box(0);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_ofIsSublattice___boxed(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_, lean_object* v_s_22_, lean_object* v_hs_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Sublattice_ofIsSublattice(v_00_u03b1_20_, v_inst_21_, v_s_22_, v_hs_23_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_copy(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_L_27_, lean_object* v_s_28_, lean_object* v_hs_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_box(0);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_copy___boxed(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_, lean_object* v_L_33_, lean_object* v_s_34_, lean_object* v_hs_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Sublattice_copy(v_00_u03b1_31_, v_inst_32_, v_L_33_, v_s_34_, v_hs_35_);
lean_dec_ref(v_inst_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe___redArg___lam__0(lean_object* v_toSemilatticeSup_37_, lean_object* v_a_38_, lean_object* v_b_39_){
_start:
{
lean_object* v_sup_40_; lean_object* v___x_41_; 
v_sup_40_ = lean_ctor_get(v_toSemilatticeSup_37_, 1);
lean_inc(v_sup_40_);
lean_dec_ref(v_toSemilatticeSup_37_);
v___x_41_ = lean_apply_2(v_sup_40_, v_a_38_, v_b_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v_toSemilatticeSup_43_; lean_object* v___f_44_; 
v_toSemilatticeSup_43_ = lean_ctor_get(v_inst_42_, 0);
lean_inc_ref(v_toSemilatticeSup_43_);
lean_dec_ref(v_inst_42_);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instSupCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_44_, 0, v_toSemilatticeSup_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instSupCoe(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_L_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Sublattice_instSupCoe___redArg(v_inst_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe___redArg___lam__0(lean_object* v_inst_49_, lean_object* v_a_50_, lean_object* v_b_51_){
_start:
{
lean_object* v_inf_52_; lean_object* v___x_53_; 
v_inf_52_ = lean_ctor_get(v_inst_49_, 1);
lean_inc(v_inf_52_);
lean_dec_ref(v_inst_49_);
v___x_53_ = lean_apply_2(v_inf_52_, v_a_50_, v_b_51_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe___redArg(lean_object* v_inst_54_){
_start:
{
lean_object* v___f_55_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instInfCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_55_, 0, v_inst_54_);
return v___f_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfCoe(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_L_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instInfCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_59_, 0, v_inst_57_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe___redArg___lam__0(lean_object* v_inst_60_, lean_object* v_a_61_, lean_object* v_b_62_){
_start:
{
lean_object* v_toSemilatticeSup_63_; lean_object* v_sup_64_; lean_object* v___x_65_; 
v_toSemilatticeSup_63_ = lean_ctor_get(v_inst_60_, 0);
lean_inc_ref(v_toSemilatticeSup_63_);
lean_dec_ref(v_inst_60_);
v_sup_64_ = lean_ctor_get(v_toSemilatticeSup_63_, 1);
lean_inc(v_sup_64_);
lean_dec_ref(v_toSemilatticeSup_63_);
v___x_65_ = lean_apply_2(v_sup_64_, v_a_61_, v_b_62_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe___redArg(lean_object* v_inst_69_){
_start:
{
lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
lean_inc_ref(v_inst_69_);
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instLatticeCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_70_, 0, v_inst_69_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instInfCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_71_, 0, v_inst_69_);
v___x_72_ = ((lean_object*)(lp_mathlib_Sublattice_instLatticeCoe___redArg___closed__0));
v___x_73_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___f_70_);
v___x_74_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___f_71_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instLatticeCoe(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_L_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Sublattice_instLatticeCoe___redArg(v_inst_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instDistribLatticeCoe___redArg(lean_object* v_inst_79_){
_start:
{
lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
lean_inc_ref(v_inst_79_);
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instInfCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_80_, 0, v_inst_79_);
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Sublattice_instLatticeCoe___redArg___lam__0), 3, 1);
lean_closure_set(v___f_81_, 0, v_inst_79_);
v___x_82_ = ((lean_object*)(lp_mathlib_Sublattice_instLatticeCoe___redArg___closed__0));
v___x_83_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___f_81_);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v___f_80_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instDistribLatticeCoe(lean_object* v_00_u03b1_85_, lean_object* v_inst_86_, lean_object* v_L_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_Sublattice_instDistribLatticeCoe___redArg(v_inst_86_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___lam__0(lean_object* v_self_89_){
_start:
{
lean_inc(v_self_89_);
return v_self_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___lam__0___boxed(lean_object* v_self_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Sublattice_subtype___lam__0(v_self_90_);
lean_dec(v_self_90_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_, lean_object* v_L_95_){
_start:
{
lean_object* v___f_96_; 
v___f_96_ = ((lean_object*)(lp_mathlib_Sublattice_subtype___closed__0));
return v___f_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_subtype___boxed(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_L_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Sublattice_subtype(v_00_u03b1_97_, v_inst_98_, v_L_99_);
lean_dec_ref(v_inst_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_inclusion(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_, lean_object* v_L_104_, lean_object* v_M_105_, lean_object* v_h_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = ((lean_object*)(lp_mathlib_Sublattice_inclusion___closed__0));
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_inclusion___boxed(lean_object* v_00_u03b1_108_, lean_object* v_inst_109_, lean_object* v_L_110_, lean_object* v_M_111_, lean_object* v_h_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Sublattice_inclusion(v_00_u03b1_108_, v_inst_109_, v_L_110_, v_M_111_, v_h_112_);
lean_dec_ref(v_inst_109_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instTop(lean_object* v_00_u03b1_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instTop___boxed(lean_object* v_00_u03b1_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Sublattice_instTop(v_00_u03b1_117_, v_inst_118_);
lean_dec_ref(v_inst_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instBot(lean_object* v_00_u03b1_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_box(0);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instBot___boxed(lean_object* v_00_u03b1_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Sublattice_instBot(v_00_u03b1_123_, v_inst_124_);
lean_dec_ref(v_inst_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf___lam__0(lean_object* v_L_126_, lean_object* v_M_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_box(0);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = ((lean_object*)(lp_mathlib_Sublattice_instInf___closed__0));
return v___f_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInf___boxed(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Sublattice_instInf(v_00_u03b1_133_, v_inst_134_);
lean_dec_ref(v_inst_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet___lam__0(lean_object* v_S_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_box(0);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v___f_141_; 
v___f_141_ = ((lean_object*)(lp_mathlib_Sublattice_instInfSet___closed__0));
return v___f_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInfSet___boxed(lean_object* v_00_u03b1_142_, lean_object* v_inst_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Sublattice_instInfSet(v_00_u03b1_142_, v_inst_143_);
lean_dec_ref(v_inst_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInhabited(lean_object* v_00_u03b1_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lean_box(0);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instInhabited___boxed(lean_object* v_00_u03b1_148_, lean_object* v_inst_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Sublattice_instInhabited(v_00_u03b1_148_, v_inst_149_);
lean_dec_ref(v_inst_149_);
return v_res_150_;
}
}
static lean_object* _init_lp_mathlib_Sublattice_topEquiv___closed__0(void){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib_Equiv_Set_univ(lean_box(0));
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_topEquiv(lean_object* v_00_u03b1_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_obj_once(&lp_mathlib_Sublattice_topEquiv___closed__0, &lp_mathlib_Sublattice_topEquiv___closed__0_once, _init_lp_mathlib_Sublattice_topEquiv___closed__0);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_topEquiv___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Sublattice_topEquiv(v_00_u03b1_155_, v_inst_156_);
lean_dec_ref(v_inst_156_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___lam__0(lean_object* v_x1_158_, lean_object* v_x2_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_box(0);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg(lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___f_166_; lean_object* v___x_167_; lean_object* v_toLattice_168_; lean_object* v_toSupSet_169_; lean_object* v_toInfSet_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_188_; 
v___x_165_ = lp_mathlib_Sublattice_instPartialOrder(lean_box(0), v_inst_164_);
v___f_166_ = ((lean_object*)(lp_mathlib_Sublattice_instInfSet___closed__0));
v___x_167_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_165_, v___f_166_);
v_toLattice_168_ = lean_ctor_get(v___x_167_, 0);
v_toSupSet_169_ = lean_ctor_get(v___x_167_, 1);
v_toInfSet_170_ = lean_ctor_get(v___x_167_, 2);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_188_ == 0)
{
lean_object* v_unused_189_; 
v_unused_189_ = lean_ctor_get(v___x_167_, 3);
lean_dec(v_unused_189_);
v___x_172_ = v___x_167_;
v_isShared_173_ = v_isSharedCheck_188_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_toInfSet_170_);
lean_inc(v_toSupSet_169_);
lean_inc(v_toLattice_168_);
lean_dec(v___x_167_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_188_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v_toSemilatticeSup_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_186_; 
v_toSemilatticeSup_174_ = lean_ctor_get(v_toLattice_168_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v_toLattice_168_);
if (v_isSharedCheck_186_ == 0)
{
lean_object* v_unused_187_; 
v_unused_187_ = lean_ctor_get(v_toLattice_168_, 1);
lean_dec(v_unused_187_);
v___x_176_ = v_toLattice_168_;
v_isShared_177_ = v_isSharedCheck_186_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_toSemilatticeSup_174_);
lean_dec(v_toLattice_168_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_186_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___f_178_; lean_object* v___x_180_; 
v___f_178_ = ((lean_object*)(lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__0));
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 1, v___f_178_);
v___x_180_ = v___x_176_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v_toSemilatticeSup_174_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v___f_178_);
v___x_180_ = v_reuseFailAlloc_185_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
lean_object* v___x_181_; lean_object* v___x_183_; 
v___x_181_ = ((lean_object*)(lp_mathlib_Sublattice_instCompleteLattice___redArg___closed__1));
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 3, v___x_181_);
lean_ctor_set(v___x_172_, 0, v___x_180_);
v___x_183_ = v___x_172_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v_toSupSet_169_);
lean_ctor_set(v_reuseFailAlloc_184_, 2, v_toInfSet_170_);
lean_ctor_set(v_reuseFailAlloc_184_, 3, v___x_181_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___redArg___boxed(lean_object* v_inst_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Sublattice_instCompleteLattice___redArg(v_inst_190_);
lean_dec_ref(v_inst_190_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_Sublattice_instCompleteLattice___redArg(v_inst_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instCompleteLattice___boxed(lean_object* v_00_u03b1_195_, lean_object* v_inst_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Sublattice_instCompleteLattice(v_00_u03b1_195_, v_inst_196_);
lean_dec_ref(v_inst_196_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instUniqueOfIsEmpty(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lean_box(0);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_instUniqueOfIsEmpty___boxed(lean_object* v_00_u03b1_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Sublattice_instUniqueOfIsEmpty(v_00_u03b1_202_, v_inst_203_, v_inst_204_);
lean_dec_ref(v_inst_203_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_comap(lean_object* v_00_u03b1_206_, lean_object* v_00_u03b2_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_f_210_, lean_object* v_L_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lean_box(0);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_comap___boxed(lean_object* v_00_u03b1_213_, lean_object* v_00_u03b2_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_, lean_object* v_L_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Sublattice_comap(v_00_u03b1_213_, v_00_u03b2_214_, v_inst_215_, v_inst_216_, v_f_217_, v_L_218_);
lean_dec(v_f_217_);
lean_dec_ref(v_inst_216_);
lean_dec_ref(v_inst_215_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_map(lean_object* v_00_u03b1_220_, lean_object* v_00_u03b2_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_f_224_, lean_object* v_L_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lean_box(0);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_map___boxed(lean_object* v_00_u03b1_227_, lean_object* v_00_u03b2_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_f_231_, lean_object* v_L_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Sublattice_map(v_00_u03b1_227_, v_00_u03b2_228_, v_inst_229_, v_inst_230_, v_f_231_, v_L_232_);
lean_dec(v_f_231_);
lean_dec_ref(v_inst_230_);
lean_dec_ref(v_inst_229_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prod(lean_object* v_00_u03b1_234_, lean_object* v_00_u03b2_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_L_238_, lean_object* v_M_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lean_box(0);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prod___boxed(lean_object* v_00_u03b1_241_, lean_object* v_00_u03b2_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_L_245_, lean_object* v_M_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Sublattice_prod(v_00_u03b1_241_, v_00_u03b2_242_, v_inst_243_, v_inst_244_, v_L_245_, v_M_246_);
lean_dec_ref(v_inst_244_);
lean_dec_ref(v_inst_243_);
return v_res_247_;
}
}
static lean_object* _init_lp_mathlib_Sublattice_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prodEquiv(lean_object* v_00_u03b1_249_, lean_object* v_00_u03b2_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_L_253_, lean_object* v_M_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lean_obj_once(&lp_mathlib_Sublattice_prodEquiv___closed__0, &lp_mathlib_Sublattice_prodEquiv___closed__0_once, _init_lp_mathlib_Sublattice_prodEquiv___closed__0);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_prodEquiv___boxed(lean_object* v_00_u03b1_256_, lean_object* v_00_u03b2_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_L_260_, lean_object* v_M_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_Sublattice_prodEquiv(v_00_u03b1_256_, v_00_u03b2_257_, v_inst_258_, v_inst_259_, v_L_260_, v_M_261_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_pi(lean_object* v_00_u03ba_263_, lean_object* v_00_u03c0_264_, lean_object* v_inst_265_, lean_object* v_s_266_, lean_object* v_L_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lean_box(0);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_pi___boxed(lean_object* v_00_u03ba_269_, lean_object* v_00_u03c0_270_, lean_object* v_inst_271_, lean_object* v_s_272_, lean_object* v_L_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_Sublattice_pi(v_00_u03ba_269_, v_00_u03c0_270_, v_inst_271_, v_s_272_, v_L_273_);
lean_dec_ref(v_L_273_);
lean_dec_ref(v_inst_271_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_LatticeHom_range(lean_object* v_00_u03b1_275_, lean_object* v_00_u03b2_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_f_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lean_box(0);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sublattice_LatticeHom_range___boxed(lean_object* v_00_u03b1_281_, lean_object* v_00_u03b2_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_f_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_Sublattice_LatticeHom_range(v_00_u03b1_281_, v_00_u03b2_282_, v_inst_283_, v_inst_284_, v_f_285_);
lean_dec(v_f_285_);
lean_dec_ref(v_inst_284_);
lean_dec_ref(v_inst_283_);
return v_res_286_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Sublattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SupClosed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Sublattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Sublattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SupClosed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Sublattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Sublattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Sublattice(builtin);
}
#ifdef __cplusplus
}
#endif
