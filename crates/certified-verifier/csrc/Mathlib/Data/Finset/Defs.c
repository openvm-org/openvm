// Lean compiler output
// Module: Mathlib.Data.Finset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Defs public import Mathlib.Data.Set.Pairwise.Basic public import Mathlib.Data.SetLike.Basic public import Mathlib.Order.Hom.Basic
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
uint8_t lp_mathlib_Multiset_decidableDexistsMultiset___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableEqPiMultiset___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSetLike(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Finset_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_instPartialOrder(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeEmb(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDforallFinset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDforallFinset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDforallFinset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDforallFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSSubset___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSSubset___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSSubset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSSubset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDExistsFinset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDExistsFinset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDExistsFinset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDExistsFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinsetCoe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinsetCoe___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinsetCoe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinsetCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEqPiFinset___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEqPiFinset___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEqPiFinset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEqPiFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEq___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_List_decidablePerm___redArg(v_inst_1_, v_x_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEq___redArg___boxed(lean_object* v_inst_5_, lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_mathlib_Finset_decidableEq___redArg(v_inst_5_, v_x_6_, v_x_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEq(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = l_List_decidablePerm___redArg(v_inst_11_, v_x_12_, v_x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEq___boxed(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_mathlib_Finset_decidableEq(v_00_u03b1_15_, v_inst_16_, v_x_17_, v_x_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instSetLike(lean_object* v_00_u03b1_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_box(0);
return v___x_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem___redArg(lean_object* v___h_23_, lean_object* v_a_24_, lean_object* v_s_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v___h_23_, v_a_24_, v_s_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem___redArg___boxed(lean_object* v___h_27_, lean_object* v_a_28_, lean_object* v_s_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Finset_decidableMem___redArg(v___h_27_, v_a_28_, v_s_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem(lean_object* v_00_u03b1_32_, lean_object* v___h_33_, lean_object* v_a_34_, lean_object* v_s_35_){
_start:
{
uint8_t v___x_36_; 
v___x_36_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v___h_33_, v_a_34_, v_s_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem___boxed(lean_object* v_00_u03b1_37_, lean_object* v___h_38_, lean_object* v_a_39_, lean_object* v_s_40_){
_start:
{
uint8_t v_res_41_; lean_object* v_r_42_; 
v_res_41_ = lp_mathlib_Finset_decidableMem(v_00_u03b1_37_, v___h_38_, v_a_39_, v_s_40_);
v_r_42_ = lean_box(v_res_41_);
return v_r_42_;
}
}
static lean_object* _init_lp_mathlib_Finset_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_box(0);
v___x_44_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instPartialOrder(lean_object* v_00_u03b1_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_obj_once(&lp_mathlib_Finset_instPartialOrder___closed__0, &lp_mathlib_Finset_instPartialOrder___closed__0_once, _init_lp_mathlib_Finset_instPartialOrder___closed__0);
return v___x_46_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem_x27___redArg(lean_object* v_inst_47_, lean_object* v_a_48_, lean_object* v_s_49_){
_start:
{
uint8_t v___x_50_; 
v___x_50_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_47_, v_a_48_, v_s_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem_x27___redArg___boxed(lean_object* v_inst_51_, lean_object* v_a_52_, lean_object* v_s_53_){
_start:
{
uint8_t v_res_54_; lean_object* v_r_55_; 
v_res_54_ = lp_mathlib_Finset_decidableMem_x27___redArg(v_inst_51_, v_a_52_, v_s_53_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableMem_x27(lean_object* v_00_u03b1_56_, lean_object* v_inst_57_, lean_object* v_a_58_, lean_object* v_s_59_){
_start:
{
uint8_t v___x_60_; 
v___x_60_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_57_, v_a_58_, v_s_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableMem_x27___boxed(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_a_63_, lean_object* v_s_64_){
_start:
{
uint8_t v_res_65_; lean_object* v_r_66_; 
v_res_65_ = lp_mathlib_Finset_decidableMem_x27(v_00_u03b1_61_, v_inst_62_, v_a_63_, v_s_64_);
v_r_66_ = lean_box(v_res_65_);
return v_r_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_coeEmb(lean_object* v_00_u03b1_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_box(0);
return v___x_68_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDforallFinset___redArg(lean_object* v_s_69_, lean_object* v___hp_70_){
_start:
{
uint8_t v___x_71_; 
v___x_71_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_s_69_, v___hp_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDforallFinset___redArg___boxed(lean_object* v_s_72_, lean_object* v___hp_73_){
_start:
{
uint8_t v_res_74_; lean_object* v_r_75_; 
v_res_74_ = lp_mathlib_Finset_decidableDforallFinset___redArg(v_s_72_, v___hp_73_);
v_r_75_ = lean_box(v_res_74_);
return v_r_75_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDforallFinset(lean_object* v_00_u03b1_76_, lean_object* v_s_77_, lean_object* v_p_78_, lean_object* v___hp_79_){
_start:
{
uint8_t v___x_80_; 
v___x_80_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_s_77_, v___hp_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDforallFinset___boxed(lean_object* v_00_u03b1_81_, lean_object* v_s_82_, lean_object* v_p_83_, lean_object* v___hp_84_){
_start:
{
uint8_t v_res_85_; lean_object* v_r_86_; 
v_res_85_ = lp_mathlib_Finset_decidableDforallFinset(v_00_u03b1_81_, v_s_82_, v_p_83_, v___hp_84_);
v_r_86_ = lean_box(v_res_85_);
return v_r_86_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0(lean_object* v_inst_87_, lean_object* v_x_88_, lean_object* v_a_89_, lean_object* v_h_90_){
_start:
{
uint8_t v___x_91_; 
v___x_91_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_87_, v_a_89_, v_x_88_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0___boxed(lean_object* v_inst_92_, lean_object* v_x_93_, lean_object* v_a_94_, lean_object* v_h_95_){
_start:
{
uint8_t v_res_96_; lean_object* v_r_97_; 
v_res_96_ = lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0(v_inst_92_, v_x_93_, v_a_94_, v_h_95_);
v_r_97_ = lean_box(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset___redArg(lean_object* v_inst_98_, lean_object* v_x_99_, lean_object* v_x_100_){
_start:
{
lean_object* v___f_101_; uint8_t v___x_102_; 
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instDecidableRelSubset___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_101_, 0, v_inst_98_);
lean_closure_set(v___f_101_, 1, v_x_100_);
v___x_102_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_x_99_, v___f_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___redArg___boxed(lean_object* v_inst_103_, lean_object* v_x_104_, lean_object* v_x_105_){
_start:
{
uint8_t v_res_106_; lean_object* v_r_107_; 
v_res_106_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_103_, v_x_104_, v_x_105_);
v_r_107_ = lean_box(v_res_106_);
return v_r_107_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSubset(lean_object* v_00_u03b1_108_, lean_object* v_inst_109_, lean_object* v_x_110_, lean_object* v_x_111_){
_start:
{
uint8_t v___x_112_; 
v___x_112_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_109_, v_x_110_, v_x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSubset___boxed(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_x_115_, lean_object* v_x_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_mathlib_Finset_instDecidableRelSubset(v_00_u03b1_113_, v_inst_114_, v_x_115_, v_x_116_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSSubset___redArg(lean_object* v_inst_119_, lean_object* v_x_120_, lean_object* v_x_121_){
_start:
{
uint8_t v___x_122_; uint8_t v___x_123_; 
lean_inc(v_x_120_);
lean_inc(v_x_121_);
lean_inc_ref(v_inst_119_);
v___x_122_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_119_, v_x_121_, v_x_120_);
v___x_123_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_119_, v_x_120_, v_x_121_);
if (v___x_123_ == 0)
{
return v___x_123_;
}
else
{
if (v___x_122_ == 0)
{
return v___x_123_;
}
else
{
uint8_t v___x_124_; 
v___x_124_ = 0;
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSSubset___redArg___boxed(lean_object* v_inst_125_, lean_object* v_x_126_, lean_object* v_x_127_){
_start:
{
uint8_t v_res_128_; lean_object* v_r_129_; 
v_res_128_ = lp_mathlib_Finset_instDecidableRelSSubset___redArg(v_inst_125_, v_x_126_, v_x_127_);
v_r_129_ = lean_box(v_res_128_);
return v_r_129_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableRelSSubset(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_, lean_object* v_x_132_, lean_object* v_x_133_){
_start:
{
uint8_t v___x_134_; 
v___x_134_ = lp_mathlib_Finset_instDecidableRelSSubset___redArg(v_inst_131_, v_x_132_, v_x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableRelSSubset___boxed(lean_object* v_00_u03b1_135_, lean_object* v_inst_136_, lean_object* v_x_137_, lean_object* v_x_138_){
_start:
{
uint8_t v_res_139_; lean_object* v_r_140_; 
v_res_139_ = lp_mathlib_Finset_instDecidableRelSSubset(v_00_u03b1_135_, v_inst_136_, v_x_137_, v_x_138_);
v_r_140_ = lean_box(v_res_139_);
return v_r_140_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLE___redArg(lean_object* v_inst_141_, lean_object* v_a_142_, lean_object* v_b_143_){
_start:
{
uint8_t v___x_144_; 
v___x_144_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_141_, v_a_142_, v_b_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLE___redArg___boxed(lean_object* v_inst_145_, lean_object* v_a_146_, lean_object* v_b_147_){
_start:
{
uint8_t v_res_148_; lean_object* v_r_149_; 
v_res_148_ = lp_mathlib_Finset_instDecidableLE___redArg(v_inst_145_, v_a_146_, v_b_147_);
v_r_149_ = lean_box(v_res_148_);
return v_r_149_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLE(lean_object* v_00_u03b1_150_, lean_object* v_inst_151_, lean_object* v_a_152_, lean_object* v_b_153_){
_start:
{
uint8_t v___x_154_; 
v___x_154_ = lp_mathlib_Finset_instDecidableRelSubset___redArg(v_inst_151_, v_a_152_, v_b_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLE___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_a_157_, lean_object* v_b_158_){
_start:
{
uint8_t v_res_159_; lean_object* v_r_160_; 
v_res_159_ = lp_mathlib_Finset_instDecidableLE(v_00_u03b1_155_, v_inst_156_, v_a_157_, v_b_158_);
v_r_160_ = lean_box(v_res_159_);
return v_r_160_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLT___redArg(lean_object* v_inst_161_, lean_object* v_a_162_, lean_object* v_b_163_){
_start:
{
uint8_t v___x_164_; 
v___x_164_ = lp_mathlib_Finset_instDecidableRelSSubset___redArg(v_inst_161_, v_a_162_, v_b_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLT___redArg___boxed(lean_object* v_inst_165_, lean_object* v_a_166_, lean_object* v_b_167_){
_start:
{
uint8_t v_res_168_; lean_object* v_r_169_; 
v_res_168_ = lp_mathlib_Finset_instDecidableLT___redArg(v_inst_165_, v_a_166_, v_b_167_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_instDecidableLT(lean_object* v_00_u03b1_170_, lean_object* v_inst_171_, lean_object* v_a_172_, lean_object* v_b_173_){
_start:
{
uint8_t v___x_174_; 
v___x_174_ = lp_mathlib_Finset_instDecidableRelSSubset___redArg(v_inst_171_, v_a_172_, v_b_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instDecidableLT___boxed(lean_object* v_00_u03b1_175_, lean_object* v_inst_176_, lean_object* v_a_177_, lean_object* v_b_178_){
_start:
{
uint8_t v_res_179_; lean_object* v_r_180_; 
v_res_179_ = lp_mathlib_Finset_instDecidableLT(v_00_u03b1_175_, v_inst_176_, v_a_177_, v_b_178_);
v_r_180_ = lean_box(v_res_179_);
return v_r_180_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDExistsFinset___redArg(lean_object* v_s_181_, lean_object* v___hp_182_){
_start:
{
uint8_t v___x_183_; 
v___x_183_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v_s_181_, v___hp_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDExistsFinset___redArg___boxed(lean_object* v_s_184_, lean_object* v___hp_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib_Finset_decidableDExistsFinset___redArg(v_s_184_, v___hp_185_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableDExistsFinset(lean_object* v_00_u03b1_188_, lean_object* v_s_189_, lean_object* v_p_190_, lean_object* v___hp_191_){
_start:
{
uint8_t v___x_192_; 
v___x_192_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v_s_189_, v___hp_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableDExistsFinset___boxed(lean_object* v_00_u03b1_193_, lean_object* v_s_194_, lean_object* v_p_195_, lean_object* v___hp_196_){
_start:
{
uint8_t v_res_197_; lean_object* v_r_198_; 
v_res_197_ = lp_mathlib_Finset_decidableDExistsFinset(v_00_u03b1_193_, v_s_194_, v_p_195_, v___hp_196_);
v_r_198_ = lean_box(v_res_197_);
return v_r_198_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0(lean_object* v___hp_199_, lean_object* v_a_200_, lean_object* v_h_201_){
_start:
{
lean_object* v___x_202_; uint8_t v___x_203_; 
v___x_202_ = lean_apply_1(v___hp_199_, v_a_200_);
v___x_203_ = lean_unbox(v___x_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0___boxed(lean_object* v___hp_204_, lean_object* v_a_205_, lean_object* v_h_206_){
_start:
{
uint8_t v_res_207_; lean_object* v_r_208_; 
v_res_207_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0(v___hp_204_, v_a_205_, v_h_206_);
v_r_208_ = lean_box(v_res_207_);
return v_r_208_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object* v_s_209_, lean_object* v___hp_210_){
_start:
{
lean_object* v___f_211_; uint8_t v___x_212_; 
v___f_211_ = lean_alloc_closure((void*)(lp_mathlib_Finset_decidableExistsAndFinset___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_211_, 0, v___hp_210_);
v___x_212_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v_s_209_, v___f_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___redArg___boxed(lean_object* v_s_213_, lean_object* v___hp_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_213_, v___hp_214_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinset(lean_object* v_00_u03b1_217_, lean_object* v_s_218_, lean_object* v_p_219_, lean_object* v___hp_220_){
_start:
{
uint8_t v___x_221_; 
v___x_221_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_218_, v___hp_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinset___boxed(lean_object* v_00_u03b1_222_, lean_object* v_s_223_, lean_object* v_p_224_, lean_object* v___hp_225_){
_start:
{
uint8_t v_res_226_; lean_object* v_r_227_; 
v_res_226_ = lp_mathlib_Finset_decidableExistsAndFinset(v_00_u03b1_222_, v_s_223_, v_p_224_, v___hp_225_);
v_r_227_ = lean_box(v_res_226_);
return v_r_227_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinsetCoe___redArg(lean_object* v_s_228_, lean_object* v_inst_229_){
_start:
{
uint8_t v___x_230_; 
v___x_230_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_228_, v_inst_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinsetCoe___redArg___boxed(lean_object* v_s_231_, lean_object* v_inst_232_){
_start:
{
uint8_t v_res_233_; lean_object* v_r_234_; 
v_res_233_ = lp_mathlib_Finset_decidableExistsAndFinsetCoe___redArg(v_s_231_, v_inst_232_);
v_r_234_ = lean_box(v_res_233_);
return v_r_234_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableExistsAndFinsetCoe(lean_object* v_00_u03b1_235_, lean_object* v_s_236_, lean_object* v_p_237_, lean_object* v_inst_238_){
_start:
{
uint8_t v___x_239_; 
v___x_239_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_236_, v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableExistsAndFinsetCoe___boxed(lean_object* v_00_u03b1_240_, lean_object* v_s_241_, lean_object* v_p_242_, lean_object* v_inst_243_){
_start:
{
uint8_t v_res_244_; lean_object* v_r_245_; 
v_res_244_ = lp_mathlib_Finset_decidableExistsAndFinsetCoe(v_00_u03b1_240_, v_s_241_, v_p_242_, v_inst_243_);
v_r_245_ = lean_box(v_res_244_);
return v_r_245_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEqPiFinset___redArg(lean_object* v_s_246_, lean_object* v___h_247_, lean_object* v_a_248_, lean_object* v_b_249_){
_start:
{
uint8_t v___x_250_; 
v___x_250_ = lp_mathlib_Multiset_decidableEqPiMultiset___redArg(v_s_246_, v___h_247_, v_a_248_, v_b_249_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEqPiFinset___redArg___boxed(lean_object* v_s_251_, lean_object* v___h_252_, lean_object* v_a_253_, lean_object* v_b_254_){
_start:
{
uint8_t v_res_255_; lean_object* v_r_256_; 
v_res_255_ = lp_mathlib_Finset_decidableEqPiFinset___redArg(v_s_251_, v___h_252_, v_a_253_, v_b_254_);
v_r_256_ = lean_box(v_res_255_);
return v_r_256_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_decidableEqPiFinset(lean_object* v_00_u03b1_257_, lean_object* v_s_258_, lean_object* v_00_u03b2_259_, lean_object* v___h_260_, lean_object* v_a_261_, lean_object* v_b_262_){
_start:
{
uint8_t v___x_263_; 
v___x_263_ = lp_mathlib_Multiset_decidableEqPiMultiset___redArg(v_s_258_, v___h_260_, v_a_261_, v_b_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_decidableEqPiFinset___boxed(lean_object* v_00_u03b1_264_, lean_object* v_s_265_, lean_object* v_00_u03b2_266_, lean_object* v___h_267_, lean_object* v_a_268_, lean_object* v_b_269_){
_start:
{
uint8_t v_res_270_; lean_object* v_r_271_; 
v_res_270_ = lp_mathlib_Finset_decidableEqPiFinset(v_00_u03b1_264_, v_s_265_, v_00_u03b2_266_, v___h_267_, v_a_268_, v_b_269_);
v_r_271_ = lean_box(v_res_270_);
return v_r_271_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0(lean_object* v_f_272_, lean_object* v_inst_273_, lean_object* v_a_274_, lean_object* v_h_275_){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; uint8_t v___x_278_; 
v___x_276_ = lean_apply_1(v_f_272_, v_a_274_);
v___x_277_ = lean_apply_1(v_inst_273_, v___x_276_);
v___x_278_ = lean_unbox(v___x_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0___boxed(lean_object* v_f_279_, lean_object* v_inst_280_, lean_object* v_a_281_, lean_object* v_h_282_){
_start:
{
uint8_t v_res_283_; lean_object* v_r_284_; 
v_res_283_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0(v_f_279_, v_inst_280_, v_a_281_, v_h_282_);
v_r_284_ = lean_box(v_res_283_);
return v_r_284_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(lean_object* v_f_285_, lean_object* v_s_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___f_288_; uint8_t v___x_289_; 
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_288_, 0, v_f_285_);
lean_closure_set(v___f_288_, 1, v_inst_287_);
v___x_289_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_s_286_, v___f_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg___boxed(lean_object* v_f_290_, lean_object* v_s_291_, lean_object* v_inst_292_){
_start:
{
uint8_t v_res_293_; lean_object* v_r_294_; 
v_res_293_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(v_f_290_, v_s_291_, v_inst_292_);
v_r_294_ = lean_box(v_res_293_);
return v_r_294_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1(lean_object* v_00_u03b1_295_, lean_object* v_00_u03b2_296_, lean_object* v_f_297_, lean_object* v_s_298_, lean_object* v_t_299_, lean_object* v_inst_300_){
_start:
{
uint8_t v___x_301_; 
v___x_301_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(v_f_297_, v_s_298_, v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___boxed(lean_object* v_00_u03b1_302_, lean_object* v_00_u03b2_303_, lean_object* v_f_304_, lean_object* v_s_305_, lean_object* v_t_306_, lean_object* v_inst_307_){
_start:
{
uint8_t v_res_308_; lean_object* v_r_309_; 
v_res_308_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1(v_00_u03b1_302_, v_00_u03b2_303_, v_f_304_, v_s_305_, v_t_306_, v_inst_307_);
v_r_309_ = lean_box(v_res_308_);
return v_r_309_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___redArg(lean_object* v_f_310_, lean_object* v_s_311_, lean_object* v_inst_312_){
_start:
{
uint8_t v___x_313_; 
v___x_313_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(v_f_310_, v_s_311_, v_inst_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___redArg___boxed(lean_object* v_f_314_, lean_object* v_s_315_, lean_object* v_inst_316_){
_start:
{
uint8_t v_res_317_; lean_object* v_r_318_; 
v_res_317_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___redArg(v_f_314_, v_s_315_, v_inst_316_);
v_r_318_ = lean_box(v_res_317_);
return v_r_318_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_f_321_, lean_object* v_s_322_, lean_object* v_t_323_, lean_object* v_inst_324_){
_start:
{
uint8_t v___x_325_; 
v___x_325_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(v_f_321_, v_s_322_, v_inst_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___boxed(lean_object* v_00_u03b1_326_, lean_object* v_00_u03b2_327_, lean_object* v_f_328_, lean_object* v_s_329_, lean_object* v_t_330_, lean_object* v_inst_331_){
_start:
{
uint8_t v_res_332_; lean_object* v_r_333_; 
v_res_332_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet(v_00_u03b1_326_, v_00_u03b2_327_, v_f_328_, v_s_329_, v_t_330_, v_inst_331_);
v_r_333_ = lean_box(v_res_332_);
return v_r_333_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(lean_object* v_f_334_, lean_object* v_inst_335_, lean_object* v_a_336_, lean_object* v_a_337_){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_338_ = lean_apply_1(v_f_334_, v_a_337_);
v___x_339_ = lean_apply_2(v_inst_335_, v___x_338_, v_a_336_);
v___x_340_ = lean_unbox(v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object* v_f_341_, lean_object* v_inst_342_, lean_object* v_a_343_, lean_object* v_a_344_){
_start:
{
uint8_t v_res_345_; lean_object* v_r_346_; 
v_res_345_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(v_f_341_, v_inst_342_, v_a_343_, v_a_344_);
v_r_346_ = lean_box(v_res_345_);
return v_r_346_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1(lean_object* v_f_347_, lean_object* v_inst_348_, lean_object* v_s_349_, lean_object* v_a_350_, lean_object* v_h_351_){
_start:
{
lean_object* v___f_352_; uint8_t v___x_353_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_352_, 0, v_f_347_);
lean_closure_set(v___f_352_, 1, v_inst_348_);
lean_closure_set(v___f_352_, 2, v_a_350_);
v___x_353_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_s_349_, v___f_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1___boxed(lean_object* v_f_354_, lean_object* v_inst_355_, lean_object* v_s_356_, lean_object* v_a_357_, lean_object* v_h_358_){
_start:
{
uint8_t v_res_359_; lean_object* v_r_360_; 
v_res_359_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1(v_f_354_, v_inst_355_, v_s_356_, v_a_357_, v_h_358_);
v_r_360_ = lean_box(v_res_359_);
return v_r_360_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(lean_object* v_f_361_, lean_object* v_s_362_, lean_object* v_t_x27_363_, lean_object* v_inst_364_){
_start:
{
lean_object* v___f_365_; uint8_t v___x_366_; 
v___f_365_ = lean_alloc_closure((void*)(lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_365_, 0, v_f_361_);
lean_closure_set(v___f_365_, 1, v_inst_364_);
lean_closure_set(v___f_365_, 2, v_s_362_);
v___x_366_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_t_x27_363_, v___f_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg___boxed(lean_object* v_f_367_, lean_object* v_s_368_, lean_object* v_t_x27_369_, lean_object* v_inst_370_){
_start:
{
uint8_t v_res_371_; lean_object* v_r_372_; 
v_res_371_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_367_, v_s_368_, v_t_x27_369_, v_inst_370_);
v_r_372_ = lean_box(v_res_371_);
return v_r_372_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1(lean_object* v_00_u03b1_373_, lean_object* v_00_u03b2_374_, lean_object* v_f_375_, lean_object* v_s_376_, lean_object* v_t_x27_377_, lean_object* v_inst_378_){
_start:
{
uint8_t v___x_379_; 
v___x_379_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_375_, v_s_376_, v_t_x27_377_, v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___boxed(lean_object* v_00_u03b1_380_, lean_object* v_00_u03b2_381_, lean_object* v_f_382_, lean_object* v_s_383_, lean_object* v_t_x27_384_, lean_object* v_inst_385_){
_start:
{
uint8_t v_res_386_; lean_object* v_r_387_; 
v_res_386_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1(v_00_u03b1_380_, v_00_u03b2_381_, v_f_382_, v_s_383_, v_t_x27_384_, v_inst_385_);
v_r_387_ = lean_box(v_res_386_);
return v_r_387_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___redArg(lean_object* v_f_388_, lean_object* v_s_389_, lean_object* v_t_x27_390_, lean_object* v_inst_391_){
_start:
{
uint8_t v___x_392_; 
v___x_392_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_388_, v_s_389_, v_t_x27_390_, v_inst_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object* v_f_393_, lean_object* v_s_394_, lean_object* v_t_x27_395_, lean_object* v_inst_396_){
_start:
{
uint8_t v_res_397_; lean_object* v_r_398_; 
v_res_397_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___redArg(v_f_393_, v_s_394_, v_t_x27_395_, v_inst_396_);
v_r_398_ = lean_box(v_res_397_);
return v_r_398_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_f_401_, lean_object* v_s_402_, lean_object* v_t_x27_403_, lean_object* v_inst_404_){
_start:
{
uint8_t v___x_405_; 
v___x_405_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_401_, v_s_402_, v_t_x27_403_, v_inst_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___boxed(lean_object* v_00_u03b1_406_, lean_object* v_00_u03b2_407_, lean_object* v_f_408_, lean_object* v_s_409_, lean_object* v_t_x27_410_, lean_object* v_inst_411_){
_start:
{
uint8_t v_res_412_; lean_object* v_r_413_; 
v_res_412_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq(v_00_u03b1_406_, v_00_u03b2_407_, v_f_408_, v_s_409_, v_t_x27_410_, v_inst_411_);
v_r_413_ = lean_box(v_res_412_);
return v_r_413_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
