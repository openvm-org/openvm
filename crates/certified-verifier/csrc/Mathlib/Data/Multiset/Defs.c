// Lean compiler output
// Module: Mathlib.Data.Multiset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Perm.Subperm public import Mathlib.Data.Nat.Basic public import Mathlib.Data.Quot public import Mathlib.Order.Monotone.Defs public import Mathlib.Order.RelClasses public import Mathlib.Tactic.Monotonicity.Attr public import Mathlib.Util.CompileInductive
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_List_decidableBEx___redArg(lean_object*, lean_object*);
uint8_t l_List_nodupDecidable___redArg(lean_object*, lean_object*);
uint8_t lp_batteries_List_isSubperm___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_instCoeList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_ofList___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_instCoeList___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instCoeList___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instCoeList(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableRListOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableRListOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableRListOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableRListOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instMembership(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instHasSubset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instHasSSubset(lean_object*);
static const lean_ctor_object lp_mathlib_Multiset_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_Multiset_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instPartialOrder(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pmap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_attach___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_attach___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_attach___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_attach___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableForallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableForallMultiset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableForallMultiset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableForallMultiset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableExistsMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableExistsMultiset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableExistsMultiset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableExistsMultiset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDexistsMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDexistsMultiset___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDexistsMultiset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDexistsMultiset___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_nodupDecidable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_nodupDecidable___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_nodupDecidable(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_nodupDecidable___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sizeOf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sizeOf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSizeOf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSizeOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___redArg(lean_object* v_a_1_){
_start:
{
lean_inc(v_a_1_);
return v_a_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___redArg___boxed(lean_object* v_a_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Multiset_ofList___redArg(v_a_2_);
lean_dec(v_a_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList(lean_object* v_00_u03b1_4_, lean_object* v_a_5_){
_start:
{
lean_inc(v_a_5_);
return v_a_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_ofList___boxed(lean_object* v_00_u03b1_6_, lean_object* v_a_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Multiset_ofList(v_00_u03b1_6_, v_a_7_);
lean_dec(v_a_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instCoeList(lean_object* v_00_u03b1_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = ((lean_object*)(lp_mathlib_Multiset_instCoeList___closed__0));
return v___x_11_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___redArg(lean_object* v_inst_12_, lean_object* v_l_u2081_13_, lean_object* v_l_u2082_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = l_List_decidablePerm___redArg(v_inst_12_, v_l_u2081_13_, v_l_u2082_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___redArg___boxed(lean_object* v_inst_16_, lean_object* v_l_u2081_17_, lean_object* v_l_u2082_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___redArg(v_inst_16_, v_l_u2081_17_, v_l_u2082_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_l_u2081_23_, lean_object* v_l_u2082_24_){
_start:
{
uint8_t v___x_25_; 
v___x_25_ = l_List_decidablePerm___redArg(v_inst_22_, v_l_u2081_23_, v_l_u2082_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq___boxed(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_l_u2081_28_, lean_object* v_l_u2082_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Multiset_instDecidableEquivListOfDecidableEq(v_00_u03b1_26_, v_inst_27_, v_l_u2081_28_, v_l_u2082_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableRListOfDecidableEq___redArg(lean_object* v_inst_32_, lean_object* v_l_u2081_33_, lean_object* v_l_u2082_34_){
_start:
{
uint8_t v___x_35_; 
v___x_35_ = l_List_decidablePerm___redArg(v_inst_32_, v_l_u2081_33_, v_l_u2082_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableRListOfDecidableEq___redArg___boxed(lean_object* v_inst_36_, lean_object* v_l_u2081_37_, lean_object* v_l_u2082_38_){
_start:
{
uint8_t v_res_39_; lean_object* v_r_40_; 
v_res_39_ = lp_mathlib_Multiset_instDecidableRListOfDecidableEq___redArg(v_inst_36_, v_l_u2081_37_, v_l_u2082_38_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_instDecidableRListOfDecidableEq(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_l_u2081_43_, lean_object* v_l_u2082_44_){
_start:
{
uint8_t v___x_45_; 
v___x_45_ = l_List_decidablePerm___redArg(v_inst_42_, v_l_u2081_43_, v_l_u2082_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instDecidableRListOfDecidableEq___boxed(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_l_u2081_48_, lean_object* v_l_u2082_49_){
_start:
{
uint8_t v_res_50_; lean_object* v_r_51_; 
v_res_50_ = lp_mathlib_Multiset_instDecidableRListOfDecidableEq(v_00_u03b1_46_, v_inst_47_, v_l_u2081_48_, v_l_u2082_49_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEq___redArg(lean_object* v_inst_52_, lean_object* v_x_53_, lean_object* v_x_54_){
_start:
{
uint8_t v___x_55_; 
v___x_55_ = l_List_decidablePerm___redArg(v_inst_52_, v_x_53_, v_x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEq___redArg___boxed(lean_object* v_inst_56_, lean_object* v_x_57_, lean_object* v_x_58_){
_start:
{
uint8_t v_res_59_; lean_object* v_r_60_; 
v_res_59_ = lp_mathlib_Multiset_decidableEq___redArg(v_inst_56_, v_x_57_, v_x_58_);
v_r_60_ = lean_box(v_res_59_);
return v_r_60_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEq(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
uint8_t v___x_65_; 
v___x_65_ = l_List_decidablePerm___redArg(v_inst_62_, v_x_63_, v_x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEq___boxed(lean_object* v_00_u03b1_66_, lean_object* v_inst_67_, lean_object* v_x_68_, lean_object* v_x_69_){
_start:
{
uint8_t v_res_70_; lean_object* v_r_71_; 
v_res_70_ = lp_mathlib_Multiset_decidableEq(v_00_u03b1_66_, v_inst_67_, v_x_68_, v_x_69_);
v_r_71_ = lean_box(v_res_70_);
return v_r_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instMembership(lean_object* v_00_u03b1_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_box(0);
return v___x_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object* v_inst_74_, lean_object* v_a_75_, lean_object* v_l_76_){
_start:
{
lean_object* v___f_77_; uint8_t v___x_78_; 
v___f_77_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_77_, 0, v_inst_74_);
v___x_78_ = l_List_elem___redArg(v___f_77_, v_a_75_, v_l_76_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___aux__1___redArg___boxed(lean_object* v_inst_79_, lean_object* v_a_80_, lean_object* v_l_81_){
_start:
{
uint8_t v_res_82_; lean_object* v_r_83_; 
v_res_82_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_79_, v_a_80_, v_l_81_);
v_r_83_ = lean_box(v_res_82_);
return v_r_83_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___aux__1(lean_object* v_00_u03b1_84_, lean_object* v_inst_85_, lean_object* v_a_86_, lean_object* v_l_87_){
_start:
{
uint8_t v___x_88_; 
v___x_88_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_85_, v_a_86_, v_l_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___aux__1___boxed(lean_object* v_00_u03b1_89_, lean_object* v_inst_90_, lean_object* v_a_91_, lean_object* v_l_92_){
_start:
{
uint8_t v_res_93_; lean_object* v_r_94_; 
v_res_93_ = lp_mathlib_Multiset_decidableMem___aux__1(v_00_u03b1_89_, v_inst_90_, v_a_91_, v_l_92_);
v_r_94_ = lean_box(v_res_93_);
return v_r_94_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object* v_inst_95_, lean_object* v_a_96_, lean_object* v_s_97_){
_start:
{
uint8_t v___x_98_; 
v___x_98_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_95_, v_a_96_, v_s_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___redArg___boxed(lean_object* v_inst_99_, lean_object* v_a_100_, lean_object* v_s_101_){
_start:
{
uint8_t v_res_102_; lean_object* v_r_103_; 
v_res_102_ = lp_mathlib_Multiset_decidableMem___redArg(v_inst_99_, v_a_100_, v_s_101_);
v_r_103_ = lean_box(v_res_102_);
return v_r_103_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableMem(lean_object* v_00_u03b1_104_, lean_object* v_inst_105_, lean_object* v_a_106_, lean_object* v_s_107_){
_start:
{
uint8_t v___x_108_; 
v___x_108_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_105_, v_a_106_, v_s_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableMem___boxed(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_a_111_, lean_object* v_s_112_){
_start:
{
uint8_t v_res_113_; lean_object* v_r_114_; 
v_res_113_ = lp_mathlib_Multiset_decidableMem(v_00_u03b1_109_, v_inst_110_, v_a_111_, v_s_112_);
v_r_114_ = lean_box(v_res_113_);
return v_r_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instHasSubset(lean_object* v_00_u03b1_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instHasSSubset(lean_object* v_00_u03b1_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_box(0);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instPartialOrder(lean_object* v_00_u03b1_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = ((lean_object*)(lp_mathlib_Multiset_instPartialOrder___closed__0));
return v___x_123_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableLE___redArg(lean_object* v_inst_124_, lean_object* v_s_125_, lean_object* v_t_126_){
_start:
{
lean_object* v___f_127_; uint8_t v___x_128_; 
v___f_127_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_127_, 0, v_inst_124_);
v___x_128_ = lp_batteries_List_isSubperm___redArg(v___f_127_, v_s_125_, v_t_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableLE___redArg___boxed(lean_object* v_inst_129_, lean_object* v_s_130_, lean_object* v_t_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_mathlib_Multiset_decidableLE___redArg(v_inst_129_, v_s_130_, v_t_131_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableLE(lean_object* v_00_u03b1_134_, lean_object* v_inst_135_, lean_object* v_s_136_, lean_object* v_t_137_){
_start:
{
uint8_t v___x_138_; 
v___x_138_ = lp_mathlib_Multiset_decidableLE___redArg(v_inst_135_, v_s_136_, v_t_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableLE___boxed(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_, lean_object* v_s_141_, lean_object* v_t_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_Multiset_decidableLE(v_00_u03b1_139_, v_inst_140_, v_s_141_, v_t_142_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___redArg(lean_object* v_a_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = l_List_lengthTR___redArg(v_a_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___redArg___boxed(lean_object* v_a_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Multiset_card___redArg(v_a_147_);
lean_dec(v_a_147_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card(lean_object* v_00_u03b1_149_, lean_object* v_a_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = l_List_lengthTR___redArg(v_a_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_card___boxed(lean_object* v_00_u03b1_152_, lean_object* v_a_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Multiset_card(v_00_u03b1_152_, v_a_153_);
lean_dec(v_a_153_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0___redArg(lean_object* v_f_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
if (lean_obj_tag(v_a_156_) == 0)
{
lean_object* v___x_158_; 
lean_dec(v_f_155_);
v___x_158_ = l_List_reverse___redArg(v_a_157_);
return v___x_158_;
}
else
{
lean_object* v_head_159_; lean_object* v_tail_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_169_; 
v_head_159_ = lean_ctor_get(v_a_156_, 0);
v_tail_160_ = lean_ctor_get(v_a_156_, 1);
v_isSharedCheck_169_ = !lean_is_exclusive(v_a_156_);
if (v_isSharedCheck_169_ == 0)
{
v___x_162_ = v_a_156_;
v_isShared_163_ = v_isSharedCheck_169_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_tail_160_);
lean_inc(v_head_159_);
lean_dec(v_a_156_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_169_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_164_; lean_object* v___x_166_; 
lean_inc(v_f_155_);
v___x_164_ = lean_apply_2(v_f_155_, v_head_159_, lean_box(0));
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 1, v_a_157_);
lean_ctor_set(v___x_162_, 0, v___x_164_);
v___x_166_ = v___x_162_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_168_; 
v_reuseFailAlloc_168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_168_, 0, v___x_164_);
lean_ctor_set(v_reuseFailAlloc_168_, 1, v_a_157_);
v___x_166_ = v_reuseFailAlloc_168_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
v_a_156_ = v_tail_160_;
v_a_157_ = v___x_166_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object* v_f_170_, lean_object* v_s_171_){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = lean_box(0);
v___x_173_ = lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0___redArg(v_f_170_, v_s_171_, v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pmap(lean_object* v_00_u03b1_174_, lean_object* v_00_u03b2_175_, lean_object* v_p_176_, lean_object* v_f_177_, lean_object* v_s_178_, lean_object* v_a_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lp_mathlib_Multiset_pmap___redArg(v_f_177_, v_s_178_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0(lean_object* v_00_u03b1_181_, lean_object* v_00_u03b2_182_, lean_object* v_f_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_List_mapTR_loop___at___00Multiset_pmap_spec__0___redArg(v_f_183_, v_a_184_, v_a_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg___lam__0(lean_object* v_val_187_, lean_object* v_property_188_){
_start:
{
lean_inc(v_val_187_);
return v_val_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg___lam__0___boxed(lean_object* v_val_189_, lean_object* v_property_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Multiset_attach___redArg___lam__0(v_val_189_, v_property_190_);
lean_dec(v_val_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach___redArg(lean_object* v_s_193_){
_start:
{
lean_object* v___f_194_; lean_object* v___x_195_; 
v___f_194_ = ((lean_object*)(lp_mathlib_Multiset_attach___redArg___closed__0));
v___x_195_ = lp_mathlib_Multiset_pmap___redArg(v___f_194_, v_s_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_attach(lean_object* v_00_u03b1_196_, lean_object* v_s_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Multiset_attach___redArg(v_s_197_);
return v___x_198_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableForallMultiset___redArg(lean_object* v_m_199_, lean_object* v_inst_200_){
_start:
{
uint8_t v___x_201_; 
v___x_201_ = l_List_decidableBAll___redArg(v_inst_200_, v_m_199_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableForallMultiset___redArg___boxed(lean_object* v_m_202_, lean_object* v_inst_203_){
_start:
{
uint8_t v_res_204_; lean_object* v_r_205_; 
v_res_204_ = lp_mathlib_Multiset_decidableForallMultiset___redArg(v_m_202_, v_inst_203_);
v_r_205_ = lean_box(v_res_204_);
return v_r_205_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableForallMultiset(lean_object* v_00_u03b1_206_, lean_object* v_m_207_, lean_object* v_p_208_, lean_object* v_inst_209_){
_start:
{
uint8_t v___x_210_; 
v___x_210_ = l_List_decidableBAll___redArg(v_inst_209_, v_m_207_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableForallMultiset___boxed(lean_object* v_00_u03b1_211_, lean_object* v_m_212_, lean_object* v_p_213_, lean_object* v_inst_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_mathlib_Multiset_decidableForallMultiset(v_00_u03b1_211_, v_m_212_, v_p_213_, v_inst_214_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0(lean_object* v___hp_217_, lean_object* v_a_218_){
_start:
{
lean_object* v___x_219_; uint8_t v___x_220_; 
v___x_219_ = lean_apply_2(v___hp_217_, v_a_218_, lean_box(0));
v___x_220_ = lean_unbox(v___x_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0___boxed(lean_object* v___hp_221_, lean_object* v_a_222_){
_start:
{
uint8_t v_res_223_; lean_object* v_r_224_; 
v_res_223_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0(v___hp_221_, v_a_222_);
v_r_224_ = lean_box(v_res_223_);
return v_r_224_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object* v_m_225_, lean_object* v___hp_226_){
_start:
{
lean_object* v___f_227_; lean_object* v___x_228_; uint8_t v___x_229_; 
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_227_, 0, v___hp_226_);
v___x_228_ = lp_mathlib_Multiset_attach___redArg(v_m_225_);
v___x_229_ = l_List_decidableBAll___redArg(v___f_227_, v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___redArg___boxed(lean_object* v_m_230_, lean_object* v___hp_231_){
_start:
{
uint8_t v_res_232_; lean_object* v_r_233_; 
v_res_232_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_m_230_, v___hp_231_);
v_r_233_ = lean_box(v_res_232_);
return v_r_233_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDforallMultiset(lean_object* v_00_u03b1_234_, lean_object* v_m_235_, lean_object* v_p_236_, lean_object* v___hp_237_){
_start:
{
uint8_t v___x_238_; 
v___x_238_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_m_235_, v___hp_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDforallMultiset___boxed(lean_object* v_00_u03b1_239_, lean_object* v_m_240_, lean_object* v_p_241_, lean_object* v___hp_242_){
_start:
{
uint8_t v_res_243_; lean_object* v_r_244_; 
v_res_243_ = lp_mathlib_Multiset_decidableDforallMultiset(v_00_u03b1_239_, v_m_240_, v_p_241_, v___hp_242_);
v_r_244_ = lean_box(v_res_243_);
return v_r_244_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0(lean_object* v_f_245_, lean_object* v_g_246_, lean_object* v_inst_247_, lean_object* v_a_248_, lean_object* v_h_249_){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
lean_inc_n(v_a_248_, 2);
v___x_250_ = lean_apply_2(v_f_245_, v_a_248_, lean_box(0));
v___x_251_ = lean_apply_2(v_g_246_, v_a_248_, lean_box(0));
v___x_252_ = lean_apply_3(v_inst_247_, v_a_248_, v___x_250_, v___x_251_);
v___x_253_ = lean_unbox(v___x_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0___boxed(lean_object* v_f_254_, lean_object* v_g_255_, lean_object* v_inst_256_, lean_object* v_a_257_, lean_object* v_h_258_){
_start:
{
uint8_t v_res_259_; lean_object* v_r_260_; 
v_res_259_ = lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0(v_f_254_, v_g_255_, v_inst_256_, v_a_257_, v_h_258_);
v_r_260_ = lean_box(v_res_259_);
return v_r_260_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset___redArg(lean_object* v_m_261_, lean_object* v_inst_262_, lean_object* v_f_263_, lean_object* v_g_264_){
_start:
{
lean_object* v___f_265_; uint8_t v___x_266_; 
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_decidableEqPiMultiset___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_265_, 0, v_f_263_);
lean_closure_set(v___f_265_, 1, v_g_264_);
lean_closure_set(v___f_265_, 2, v_inst_262_);
v___x_266_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_m_261_, v___f_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___redArg___boxed(lean_object* v_m_267_, lean_object* v_inst_268_, lean_object* v_f_269_, lean_object* v_g_270_){
_start:
{
uint8_t v_res_271_; lean_object* v_r_272_; 
v_res_271_ = lp_mathlib_Multiset_decidableEqPiMultiset___redArg(v_m_267_, v_inst_268_, v_f_269_, v_g_270_);
v_r_272_ = lean_box(v_res_271_);
return v_r_272_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableEqPiMultiset(lean_object* v_00_u03b1_273_, lean_object* v_m_274_, lean_object* v_00_u03b2_275_, lean_object* v_inst_276_, lean_object* v_f_277_, lean_object* v_g_278_){
_start:
{
uint8_t v___x_279_; 
v___x_279_ = lp_mathlib_Multiset_decidableEqPiMultiset___redArg(v_m_274_, v_inst_276_, v_f_277_, v_g_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableEqPiMultiset___boxed(lean_object* v_00_u03b1_280_, lean_object* v_m_281_, lean_object* v_00_u03b2_282_, lean_object* v_inst_283_, lean_object* v_f_284_, lean_object* v_g_285_){
_start:
{
uint8_t v_res_286_; lean_object* v_r_287_; 
v_res_286_ = lp_mathlib_Multiset_decidableEqPiMultiset(v_00_u03b1_280_, v_m_281_, v_00_u03b2_282_, v_inst_283_, v_f_284_, v_g_285_);
v_r_287_ = lean_box(v_res_286_);
return v_r_287_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableExistsMultiset___redArg(lean_object* v_m_288_, lean_object* v_inst_289_){
_start:
{
uint8_t v___x_290_; 
v___x_290_ = l_List_decidableBEx___redArg(v_inst_289_, v_m_288_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableExistsMultiset___redArg___boxed(lean_object* v_m_291_, lean_object* v_inst_292_){
_start:
{
uint8_t v_res_293_; lean_object* v_r_294_; 
v_res_293_ = lp_mathlib_Multiset_decidableExistsMultiset___redArg(v_m_291_, v_inst_292_);
v_r_294_ = lean_box(v_res_293_);
return v_r_294_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableExistsMultiset(lean_object* v_00_u03b1_295_, lean_object* v_m_296_, lean_object* v_p_297_, lean_object* v_inst_298_){
_start:
{
uint8_t v___x_299_; 
v___x_299_ = l_List_decidableBEx___redArg(v_inst_298_, v_m_296_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableExistsMultiset___boxed(lean_object* v_00_u03b1_300_, lean_object* v_m_301_, lean_object* v_p_302_, lean_object* v_inst_303_){
_start:
{
uint8_t v_res_304_; lean_object* v_r_305_; 
v_res_304_ = lp_mathlib_Multiset_decidableExistsMultiset(v_00_u03b1_300_, v_m_301_, v_p_302_, v_inst_303_);
v_r_305_ = lean_box(v_res_304_);
return v_r_305_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDexistsMultiset___redArg(lean_object* v_m_306_, lean_object* v___hp_307_){
_start:
{
lean_object* v___f_308_; lean_object* v___x_309_; uint8_t v___x_310_; 
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_decidableDforallMultiset___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_308_, 0, v___hp_307_);
v___x_309_ = lp_mathlib_Multiset_attach___redArg(v_m_306_);
v___x_310_ = l_List_decidableBEx___redArg(v___f_308_, v___x_309_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDexistsMultiset___redArg___boxed(lean_object* v_m_311_, lean_object* v___hp_312_){
_start:
{
uint8_t v_res_313_; lean_object* v_r_314_; 
v_res_313_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v_m_311_, v___hp_312_);
v_r_314_ = lean_box(v_res_313_);
return v_r_314_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_decidableDexistsMultiset(lean_object* v_00_u03b1_315_, lean_object* v_m_316_, lean_object* v_p_317_, lean_object* v___hp_318_){
_start:
{
uint8_t v___x_319_; 
v___x_319_ = lp_mathlib_Multiset_decidableDexistsMultiset___redArg(v_m_316_, v___hp_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_decidableDexistsMultiset___boxed(lean_object* v_00_u03b1_320_, lean_object* v_m_321_, lean_object* v_p_322_, lean_object* v___hp_323_){
_start:
{
uint8_t v_res_324_; lean_object* v_r_325_; 
v_res_324_ = lp_mathlib_Multiset_decidableDexistsMultiset(v_00_u03b1_320_, v_m_321_, v_p_322_, v___hp_323_);
v_r_325_ = lean_box(v_res_324_);
return v_r_325_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_nodupDecidable___redArg(lean_object* v_inst_326_, lean_object* v_s_327_){
_start:
{
uint8_t v___x_328_; 
v___x_328_ = l_List_nodupDecidable___redArg(v_inst_326_, v_s_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_nodupDecidable___redArg___boxed(lean_object* v_inst_329_, lean_object* v_s_330_){
_start:
{
uint8_t v_res_331_; lean_object* v_r_332_; 
v_res_331_ = lp_mathlib_Multiset_nodupDecidable___redArg(v_inst_329_, v_s_330_);
v_r_332_ = lean_box(v_res_331_);
return v_r_332_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Multiset_nodupDecidable(lean_object* v_00_u03b1_333_, lean_object* v_inst_334_, lean_object* v_s_335_){
_start:
{
uint8_t v___x_336_; 
v___x_336_ = l_List_nodupDecidable___redArg(v_inst_334_, v_s_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_nodupDecidable___boxed(lean_object* v_00_u03b1_337_, lean_object* v_inst_338_, lean_object* v_s_339_){
_start:
{
uint8_t v_res_340_; lean_object* v_r_341_; 
v_res_340_ = lp_mathlib_Multiset_nodupDecidable(v_00_u03b1_337_, v_inst_338_, v_s_339_);
v_r_341_ = lean_box(v_res_340_);
return v_r_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sizeOf___redArg(lean_object* v_inst_342_, lean_object* v_s_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_342_, v_s_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sizeOf(lean_object* v_00_u03b1_345_, lean_object* v_inst_346_, lean_object* v_s_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_List___sizeOf__1___redArg_00___x40_Mathlib_Util_CompileInductive_1590845460____hygCtx___hyg_11_(v_inst_346_, v_s_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSizeOf___redArg(lean_object* v_inst_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sizeOf), 3, 2);
lean_closure_set(v___x_350_, 0, lean_box(0));
lean_closure_set(v___x_350_, 1, v_inst_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_instSizeOf(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_sizeOf), 3, 2);
lean_closure_set(v___x_353_, 0, lean_box(0));
lean_closure_set(v___x_353_, 1, v_inst_352_);
return v___x_353_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Perm_Subperm(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Quot(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Perm_Subperm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Quot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Monotone_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Perm_Subperm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Quot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Monotone_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Perm_Subperm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Quot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Monotone_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
