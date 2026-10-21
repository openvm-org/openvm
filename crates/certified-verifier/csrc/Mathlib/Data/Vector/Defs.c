// Lean compiler output
// Module: Mathlib.Data.Vector.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Defs public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_List_mapAccumr___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_mapAccumr_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_nil(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_Vector_instHAppendHAddNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_appendTR___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_Vector_instHAppendHAddNat___closed__0 = (const lean_object*)&lp_mathlib_List_Vector_instHAppendHAddNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instHAppendHAddNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instHAppendHAddNat___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_Vector_map_u2082___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_Vector_map_u2082___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_Vector_map_u2082___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_replicate___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_replicate(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftRightFill___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftRightFill(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_Vector_instGetElemNatLt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_Vector_instGetElemNatLt___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_Vector_instGetElemNatLt___closed__0 = (const lean_object*)&lp_mathlib_List_Vector_instGetElemNatLt___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___aux__1___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_instDecidableEqList___redArg(v_inst_1_, v_a_2_, v_b_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___aux__1___redArg___boxed(lean_object* v_inst_5_, lean_object* v_a_6_, lean_object* v_b_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_mathlib_List_Vector_instDecidableEq___aux__1___redArg(v_inst_5_, v_a_6_, v_b_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___aux__1(lean_object* v_00_u03b1_10_, lean_object* v_n_11_, lean_object* v_inst_12_, lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = l_instDecidableEqList___redArg(v_inst_12_, v_a_13_, v_b_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___aux__1___boxed(lean_object* v_00_u03b1_16_, lean_object* v_n_17_, lean_object* v_inst_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_mathlib_List_Vector_instDecidableEq___aux__1(v_00_u03b1_16_, v_n_17_, v_inst_18_, v_a_19_, v_b_20_);
lean_dec(v_n_17_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq___redArg(lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = l_instDecidableEqList___redArg(v_inst_23_, v_a_24_, v_b_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___redArg___boxed(lean_object* v_inst_27_, lean_object* v_a_28_, lean_object* v_b_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_List_Vector_instDecidableEq___redArg(v_inst_27_, v_a_28_, v_b_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_Vector_instDecidableEq(lean_object* v_00_u03b1_32_, lean_object* v_n_33_, lean_object* v_inst_34_, lean_object* v_a_35_, lean_object* v_b_36_){
_start:
{
uint8_t v___x_37_; 
v___x_37_ = l_instDecidableEqList___redArg(v_inst_34_, v_a_35_, v_b_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instDecidableEq___boxed(lean_object* v_00_u03b1_38_, lean_object* v_n_39_, lean_object* v_inst_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_mathlib_List_Vector_instDecidableEq(v_00_u03b1_38_, v_n_39_, v_inst_40_, v_a_41_, v_b_42_);
lean_dec(v_n_39_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_nil(lean_object* v_00_u03b1_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons___redArg(lean_object* v_x_47_, lean_object* v_x_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_49_, 0, v_x_47_);
lean_ctor_set(v___x_49_, 1, v_x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons(lean_object* v_00_u03b1_50_, lean_object* v_n_51_, lean_object* v_x_52_, lean_object* v_x_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_54_, 0, v_x_52_);
lean_ctor_set(v___x_54_, 1, v_x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_cons___boxed(lean_object* v_00_u03b1_55_, lean_object* v_n_56_, lean_object* v_x_57_, lean_object* v_x_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_List_Vector_cons(v_00_u03b1_55_, v_n_56_, v_x_57_, v_x_58_);
lean_dec(v_n_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___redArg(lean_object* v_n_60_){
_start:
{
lean_inc(v_n_60_);
return v_n_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___redArg___boxed(lean_object* v_n_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_List_Vector_length___redArg(v_n_61_);
lean_dec(v_n_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length(lean_object* v_00_u03b1_63_, lean_object* v_n_64_, lean_object* v_x_65_){
_start:
{
lean_inc(v_n_64_);
return v_n_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_length___boxed(lean_object* v_00_u03b1_66_, lean_object* v_n_67_, lean_object* v_x_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_List_Vector_length(v_00_u03b1_66_, v_n_67_, v_x_68_);
lean_dec(v_x_68_);
lean_dec(v_n_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___redArg(lean_object* v_x_70_){
_start:
{
lean_object* v_head_71_; 
v_head_71_ = lean_ctor_get(v_x_70_, 0);
lean_inc(v_head_71_);
return v_head_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___redArg___boxed(lean_object* v_x_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_List_Vector_head___redArg(v_x_72_);
lean_dec(v_x_72_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head(lean_object* v_00_u03b1_74_, lean_object* v_n_75_, lean_object* v_x_76_){
_start:
{
lean_object* v_head_77_; 
v_head_77_ = lean_ctor_get(v_x_76_, 0);
lean_inc(v_head_77_);
return v_head_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_head___boxed(lean_object* v_00_u03b1_78_, lean_object* v_n_79_, lean_object* v_x_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_List_Vector_head(v_00_u03b1_78_, v_n_79_, v_x_80_);
lean_dec(v_x_80_);
lean_dec(v_n_79_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___redArg(lean_object* v_x_82_){
_start:
{
if (lean_obj_tag(v_x_82_) == 0)
{
return v_x_82_;
}
else
{
lean_object* v_tail_83_; 
v_tail_83_ = lean_ctor_get(v_x_82_, 1);
lean_inc(v_tail_83_);
return v_tail_83_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___redArg___boxed(lean_object* v_x_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_List_Vector_tail___redArg(v_x_84_);
lean_dec(v_x_84_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail(lean_object* v_00_u03b1_86_, lean_object* v_n_87_, lean_object* v_x_88_){
_start:
{
if (lean_obj_tag(v_x_88_) == 0)
{
return v_x_88_;
}
else
{
lean_object* v_tail_89_; 
v_tail_89_ = lean_ctor_get(v_x_88_, 1);
lean_inc(v_tail_89_);
return v_tail_89_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_tail___boxed(lean_object* v_00_u03b1_90_, lean_object* v_n_91_, lean_object* v_x_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_List_Vector_tail(v_00_u03b1_90_, v_n_91_, v_x_92_);
lean_dec(v_x_92_);
lean_dec(v_n_91_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___redArg(lean_object* v_v_94_){
_start:
{
lean_inc(v_v_94_);
return v_v_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___redArg___boxed(lean_object* v_v_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_List_Vector_toList___redArg(v_v_95_);
lean_dec(v_v_95_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList(lean_object* v_00_u03b1_97_, lean_object* v_n_98_, lean_object* v_v_99_){
_start:
{
lean_inc(v_v_99_);
return v_v_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_toList___boxed(lean_object* v_00_u03b1_100_, lean_object* v_n_101_, lean_object* v_v_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_List_Vector_toList(v_00_u03b1_100_, v_n_101_, v_v_102_);
lean_dec(v_v_102_);
lean_dec(v_n_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___redArg(lean_object* v_l_104_, lean_object* v_i_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = l_List_get___redArg(v_l_104_, v_i_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___redArg___boxed(lean_object* v_l_107_, lean_object* v_i_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_List_Vector_get___redArg(v_l_107_, v_i_108_);
lean_dec(v_l_107_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get(lean_object* v_00_u03b1_110_, lean_object* v_n_111_, lean_object* v_l_112_, lean_object* v_i_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = l_List_get___redArg(v_l_112_, v_i_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_get___boxed(lean_object* v_00_u03b1_115_, lean_object* v_n_116_, lean_object* v_l_117_, lean_object* v_i_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_List_Vector_get(v_00_u03b1_115_, v_n_116_, v_l_117_, v_i_118_);
lean_dec(v_l_117_);
lean_dec(v_n_116_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instHAppendHAddNat(lean_object* v_00_u03b1_121_, lean_object* v_n_122_, lean_object* v_m_123_){
_start:
{
lean_object* v___f_124_; 
v___f_124_ = ((lean_object*)(lp_mathlib_List_Vector_instHAppendHAddNat___closed__0));
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instHAppendHAddNat___boxed(lean_object* v_00_u03b1_125_, lean_object* v_n_126_, lean_object* v_m_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_List_Vector_instHAppendHAddNat(v_00_u03b1_125_, v_n_126_, v_m_127_);
lean_dec(v_m_127_);
lean_dec(v_n_126_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim___redArg(lean_object* v_H_129_, lean_object* v_x_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lean_apply_1(v_H_129_, v_x_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim(lean_object* v_00_u03b1_132_, lean_object* v_C_133_, lean_object* v_H_134_, lean_object* v_n_135_, lean_object* v_x_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_apply_1(v_H_134_, v_x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_elim___boxed(lean_object* v_00_u03b1_138_, lean_object* v_C_139_, lean_object* v_H_140_, lean_object* v_n_141_, lean_object* v_x_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_List_Vector_elim(v_00_u03b1_138_, v_C_139_, v_H_140_, v_n_141_, v_x_142_);
lean_dec(v_n_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0___redArg(lean_object* v_f_144_, lean_object* v_a_145_, lean_object* v_a_146_){
_start:
{
if (lean_obj_tag(v_a_145_) == 0)
{
lean_object* v___x_147_; 
lean_dec(v_f_144_);
v___x_147_ = l_List_reverse___redArg(v_a_146_);
return v___x_147_;
}
else
{
lean_object* v_head_148_; lean_object* v_tail_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_158_; 
v_head_148_ = lean_ctor_get(v_a_145_, 0);
v_tail_149_ = lean_ctor_get(v_a_145_, 1);
v_isSharedCheck_158_ = !lean_is_exclusive(v_a_145_);
if (v_isSharedCheck_158_ == 0)
{
v___x_151_ = v_a_145_;
v_isShared_152_ = v_isSharedCheck_158_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_tail_149_);
lean_inc(v_head_148_);
lean_dec(v_a_145_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_158_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v___x_153_; lean_object* v___x_155_; 
lean_inc(v_f_144_);
v___x_153_ = lean_apply_1(v_f_144_, v_head_148_);
if (v_isShared_152_ == 0)
{
lean_ctor_set(v___x_151_, 1, v_a_146_);
lean_ctor_set(v___x_151_, 0, v___x_153_);
v___x_155_ = v___x_151_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v___x_153_);
lean_ctor_set(v_reuseFailAlloc_157_, 1, v_a_146_);
v___x_155_ = v_reuseFailAlloc_157_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
v_a_145_ = v_tail_149_;
v_a_146_ = v___x_155_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map___redArg(lean_object* v_f_159_, lean_object* v_x_160_){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = lean_box(0);
v___x_162_ = lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0___redArg(v_f_159_, v_x_160_, v___x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map(lean_object* v_00_u03b1_163_, lean_object* v_00_u03b2_164_, lean_object* v_n_165_, lean_object* v_f_166_, lean_object* v_x_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_List_Vector_map___redArg(v_f_166_, v_x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map___boxed(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_n_171_, lean_object* v_f_172_, lean_object* v_x_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_List_Vector_map(v_00_u03b1_169_, v_00_u03b2_170_, v_n_171_, v_f_172_, v_x_173_);
lean_dec(v_n_171_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0(lean_object* v_00_u03b1_175_, lean_object* v_00_u03b2_176_, lean_object* v_f_177_, lean_object* v_a_178_, lean_object* v_a_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lp_mathlib_List_mapTR_loop___at___00List_Vector_map_spec__0___redArg(v_f_177_, v_a_178_, v_a_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0___redArg(lean_object* v_f_181_, lean_object* v_a_182_, lean_object* v_a_183_){
_start:
{
if (lean_obj_tag(v_a_182_) == 0)
{
lean_object* v___x_184_; 
lean_dec(v_f_181_);
v___x_184_ = l_List_reverse___redArg(v_a_183_);
return v___x_184_;
}
else
{
lean_object* v_head_185_; lean_object* v_tail_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_195_; 
v_head_185_ = lean_ctor_get(v_a_182_, 0);
v_tail_186_ = lean_ctor_get(v_a_182_, 1);
v_isSharedCheck_195_ = !lean_is_exclusive(v_a_182_);
if (v_isSharedCheck_195_ == 0)
{
v___x_188_ = v_a_182_;
v_isShared_189_ = v_isSharedCheck_195_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_tail_186_);
lean_inc(v_head_185_);
lean_dec(v_a_182_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_195_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_190_; lean_object* v___x_192_; 
lean_inc(v_f_181_);
v___x_190_ = lean_apply_2(v_f_181_, v_head_185_, lean_box(0));
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 1, v_a_183_);
lean_ctor_set(v___x_188_, 0, v___x_190_);
v___x_192_ = v___x_188_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_194_, 1, v_a_183_);
v___x_192_ = v_reuseFailAlloc_194_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
v_a_182_ = v_tail_186_;
v_a_183_ = v___x_192_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap___redArg(lean_object* v_f_196_, lean_object* v_x_197_){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = lean_box(0);
v___x_199_ = lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0___redArg(v_f_196_, v_x_197_, v___x_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap(lean_object* v_00_u03b1_200_, lean_object* v_00_u03b2_201_, lean_object* v_n_202_, lean_object* v_p_203_, lean_object* v_f_204_, lean_object* v_x_205_, lean_object* v_x_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_List_Vector_pmap___redArg(v_f_204_, v_x_205_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_pmap___boxed(lean_object* v_00_u03b1_208_, lean_object* v_00_u03b2_209_, lean_object* v_n_210_, lean_object* v_p_211_, lean_object* v_f_212_, lean_object* v_x_213_, lean_object* v_x_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_List_Vector_pmap(v_00_u03b1_208_, v_00_u03b2_209_, v_n_210_, v_p_211_, v_f_212_, v_x_213_, v_x_214_);
lean_dec(v_n_210_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0(lean_object* v_00_u03b1_216_, lean_object* v_00_u03b2_217_, lean_object* v_f_218_, lean_object* v_a_219_, lean_object* v_a_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lp_mathlib_List_mapTR_loop___at___00List_Vector_pmap_spec__0___redArg(v_f_218_, v_a_219_, v_a_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082___redArg(lean_object* v_f_224_, lean_object* v_x_225_, lean_object* v_x_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = ((lean_object*)(lp_mathlib_List_Vector_map_u2082___redArg___closed__0));
v___x_228_ = l___private_Init_Data_List_Impl_0__List_zipWithTR_go(lean_box(0), lean_box(0), lean_box(0), v_f_224_, v_x_225_, v_x_226_, v___x_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082(lean_object* v_00_u03b1_229_, lean_object* v_00_u03b2_230_, lean_object* v_00_u03c6_231_, lean_object* v_n_232_, lean_object* v_f_233_, lean_object* v_x_234_, lean_object* v_x_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_List_Vector_map_u2082___redArg(v_f_233_, v_x_234_, v_x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_map_u2082___boxed(lean_object* v_00_u03b1_237_, lean_object* v_00_u03b2_238_, lean_object* v_00_u03c6_239_, lean_object* v_n_240_, lean_object* v_f_241_, lean_object* v_x_242_, lean_object* v_x_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_List_Vector_map_u2082(v_00_u03b1_237_, v_00_u03b2_238_, v_00_u03c6_239_, v_n_240_, v_f_241_, v_x_242_, v_x_243_);
lean_dec(v_n_240_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_replicate___redArg(lean_object* v_n_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = l_List_replicateTR___redArg(v_n_245_, v_a_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_replicate(lean_object* v_00_u03b1_248_, lean_object* v_n_249_, lean_object* v_a_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = l_List_replicateTR___redArg(v_n_249_, v_a_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___redArg(lean_object* v_i_252_, lean_object* v_x_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = l_List_drop___redArg(v_i_252_, v_x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___redArg___boxed(lean_object* v_i_255_, lean_object* v_x_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib_List_Vector_drop___redArg(v_i_255_, v_x_256_);
lean_dec(v_x_256_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop(lean_object* v_00_u03b1_258_, lean_object* v_n_259_, lean_object* v_i_260_, lean_object* v_x_261_){
_start:
{
lean_object* v___x_262_; 
v___x_262_ = l_List_drop___redArg(v_i_260_, v_x_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_drop___boxed(lean_object* v_00_u03b1_263_, lean_object* v_n_264_, lean_object* v_i_265_, lean_object* v_x_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_List_Vector_drop(v_00_u03b1_263_, v_n_264_, v_i_265_, v_x_266_);
lean_dec(v_x_266_);
lean_dec(v_n_264_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take___redArg(lean_object* v_i_268_, lean_object* v_x_269_){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = ((lean_object*)(lp_mathlib_List_Vector_map_u2082___redArg___closed__0));
lean_inc(v_x_269_);
v___x_271_ = l___private_Init_Data_List_Impl_0__List_takeTR_go(lean_box(0), v_x_269_, v_x_269_, v_i_268_, v___x_270_);
lean_dec(v_x_269_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take(lean_object* v_00_u03b1_272_, lean_object* v_n_273_, lean_object* v_i_274_, lean_object* v_x_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_List_Vector_take___redArg(v_i_274_, v_x_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_take___boxed(lean_object* v_00_u03b1_277_, lean_object* v_n_278_, lean_object* v_i_279_, lean_object* v_x_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_List_Vector_take(v_00_u03b1_277_, v_n_278_, v_i_279_, v_x_280_);
lean_dec(v_n_278_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx___redArg(lean_object* v_i_282_, lean_object* v_x_283_){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_284_ = ((lean_object*)(lp_mathlib_List_Vector_map_u2082___redArg___closed__0));
lean_inc(v_x_283_);
v___x_285_ = l___private_Init_Data_List_Impl_0__List_eraseIdxTR_go(lean_box(0), v_x_283_, v_x_283_, v_i_282_, v___x_284_);
lean_dec(v_x_283_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx(lean_object* v_00_u03b1_286_, lean_object* v_n_287_, lean_object* v_i_288_, lean_object* v_x_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_mathlib_List_Vector_eraseIdx___redArg(v_i_288_, v_x_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_eraseIdx___boxed(lean_object* v_00_u03b1_291_, lean_object* v_n_292_, lean_object* v_i_293_, lean_object* v_x_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib_List_Vector_eraseIdx(v_00_u03b1_291_, v_n_292_, v_i_293_, v_x_294_);
lean_dec(v_n_292_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___lam__0(lean_object* v_x_296_, lean_object* v_i_297_){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = l_Fin_succ___redArg(v_i_297_);
v___x_299_ = lean_apply_1(v_x_296_, v___x_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___lam__0___boxed(lean_object* v_x_300_, lean_object* v_i_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_List_Vector_ofFn___redArg___lam__0(v_x_300_, v_i_301_);
lean_dec(v_i_301_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg(lean_object* v_x_303_, lean_object* v_x_304_){
_start:
{
lean_object* v_zero_305_; uint8_t v_isZero_306_; 
v_zero_305_ = lean_unsigned_to_nat(0u);
v_isZero_306_ = lean_nat_dec_eq(v_x_303_, v_zero_305_);
if (v_isZero_306_ == 1)
{
lean_object* v___x_307_; 
lean_dec(v_x_304_);
v___x_307_ = lean_box(0);
return v___x_307_;
}
else
{
lean_object* v___f_308_; lean_object* v_one_309_; lean_object* v_n_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
lean_inc(v_x_304_);
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_ofFn___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_308_, 0, v_x_304_);
v_one_309_ = lean_unsigned_to_nat(1u);
v_n_310_ = lean_nat_sub(v_x_303_, v_one_309_);
v___x_311_ = lean_nat_add(v_n_310_, v_one_309_);
v___x_312_ = lean_nat_mod(v_zero_305_, v___x_311_);
lean_dec(v___x_311_);
v___x_313_ = lean_apply_1(v_x_304_, v___x_312_);
v___x_314_ = lp_mathlib_List_Vector_ofFn___redArg(v_n_310_, v___f_308_);
lean_dec(v_n_310_);
v___x_315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_313_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
return v___x_315_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___redArg___boxed(lean_object* v_x_316_, lean_object* v_x_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_List_Vector_ofFn___redArg(v_x_316_, v_x_317_);
lean_dec(v_x_316_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn(lean_object* v_00_u03b1_319_, lean_object* v_x_320_, lean_object* v_x_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_List_Vector_ofFn___redArg(v_x_320_, v_x_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_ofFn___boxed(lean_object* v_00_u03b1_323_, lean_object* v_x_324_, lean_object* v_x_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_List_Vector_ofFn(v_00_u03b1_323_, v_x_324_, v_x_325_);
lean_dec(v_x_324_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___redArg(lean_object* v_x_327_){
_start:
{
lean_inc(v_x_327_);
return v_x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___redArg___boxed(lean_object* v_x_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_List_Vector_congr___redArg(v_x_328_);
lean_dec(v_x_328_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr(lean_object* v_00_u03b1_330_, lean_object* v_n_331_, lean_object* v_m_332_, lean_object* v_h_333_, lean_object* v_x_334_){
_start:
{
lean_inc(v_x_334_);
return v_x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_congr___boxed(lean_object* v_00_u03b1_335_, lean_object* v_n_336_, lean_object* v_m_337_, lean_object* v_h_338_, lean_object* v_x_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_List_Vector_congr(v_00_u03b1_335_, v_n_336_, v_m_337_, v_h_338_, v_x_339_);
lean_dec(v_x_339_);
lean_dec(v_m_337_);
lean_dec(v_n_336_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr___redArg(lean_object* v_f_341_, lean_object* v_x_342_, lean_object* v_x_343_){
_start:
{
lean_object* v_res_344_; lean_object* v_fst_345_; lean_object* v_snd_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_353_; 
v_res_344_ = lp_mathlib_List_mapAccumr___redArg(v_f_341_, v_x_342_, v_x_343_);
v_fst_345_ = lean_ctor_get(v_res_344_, 0);
v_snd_346_ = lean_ctor_get(v_res_344_, 1);
v_isSharedCheck_353_ = !lean_is_exclusive(v_res_344_);
if (v_isSharedCheck_353_ == 0)
{
v___x_348_ = v_res_344_;
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_snd_346_);
lean_inc(v_fst_345_);
lean_dec(v_res_344_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_351_; 
if (v_isShared_349_ == 0)
{
v___x_351_ = v___x_348_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_fst_345_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v_snd_346_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr(lean_object* v_00_u03b1_354_, lean_object* v_00_u03b2_355_, lean_object* v_00_u03c3_356_, lean_object* v_n_357_, lean_object* v_f_358_, lean_object* v_x_359_, lean_object* v_x_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_List_Vector_mapAccumr___redArg(v_f_358_, v_x_359_, v_x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr___boxed(lean_object* v_00_u03b1_362_, lean_object* v_00_u03b2_363_, lean_object* v_00_u03c3_364_, lean_object* v_n_365_, lean_object* v_f_366_, lean_object* v_x_367_, lean_object* v_x_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_List_Vector_mapAccumr(v_00_u03b1_362_, v_00_u03b2_363_, v_00_u03c3_364_, v_n_365_, v_f_366_, v_x_367_, v_x_368_);
lean_dec(v_n_365_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082___redArg(lean_object* v_f_370_, lean_object* v_x_371_, lean_object* v_x_372_, lean_object* v_x_373_){
_start:
{
lean_object* v_res_374_; lean_object* v_fst_375_; lean_object* v_snd_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
v_res_374_ = lp_mathlib_List_mapAccumr_u2082___redArg(v_f_370_, v_x_371_, v_x_372_, v_x_373_);
v_fst_375_ = lean_ctor_get(v_res_374_, 0);
v_snd_376_ = lean_ctor_get(v_res_374_, 1);
v_isSharedCheck_383_ = !lean_is_exclusive(v_res_374_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v_res_374_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_snd_376_);
lean_inc(v_fst_375_);
lean_dec(v_res_374_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_fst_375_);
lean_ctor_set(v_reuseFailAlloc_382_, 1, v_snd_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082(lean_object* v_00_u03b1_384_, lean_object* v_00_u03b2_385_, lean_object* v_00_u03c3_386_, lean_object* v_00_u03c6_387_, lean_object* v_n_388_, lean_object* v_f_389_, lean_object* v_x_390_, lean_object* v_x_391_, lean_object* v_x_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_List_Vector_mapAccumr_u2082___redArg(v_f_389_, v_x_390_, v_x_391_, v_x_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_mapAccumr_u2082___boxed(lean_object* v_00_u03b1_394_, lean_object* v_00_u03b2_395_, lean_object* v_00_u03c3_396_, lean_object* v_00_u03c6_397_, lean_object* v_n_398_, lean_object* v_f_399_, lean_object* v_x_400_, lean_object* v_x_401_, lean_object* v_x_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_List_Vector_mapAccumr_u2082(v_00_u03b1_394_, v_00_u03b2_395_, v_00_u03c3_396_, v_00_u03c6_397_, v_n_398_, v_f_399_, v_x_400_, v_x_401_, v_x_402_);
lean_dec(v_n_398_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___redArg(lean_object* v_n_404_, lean_object* v_v_405_, lean_object* v_i_406_, lean_object* v_fill_407_){
_start:
{
lean_object* v___y_409_; uint8_t v___x_413_; 
v___x_413_ = lean_nat_dec_le(v_n_404_, v_i_406_);
if (v___x_413_ == 0)
{
lean_dec(v_n_404_);
lean_inc(v_i_406_);
v___y_409_ = v_i_406_;
goto v___jp_408_;
}
else
{
v___y_409_ = v_n_404_;
goto v___jp_408_;
}
v___jp_408_:
{
lean_object* v_val_410_; lean_object* v_val_411_; lean_object* v___x_412_; 
v_val_410_ = l_List_drop___redArg(v_i_406_, v_v_405_);
v_val_411_ = l_List_replicateTR___redArg(v___y_409_, v_fill_407_);
v___x_412_ = l_List_appendTR___redArg(v_val_410_, v_val_411_);
return v___x_412_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___redArg___boxed(lean_object* v_n_414_, lean_object* v_v_415_, lean_object* v_i_416_, lean_object* v_fill_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_List_Vector_shiftLeftFill___redArg(v_n_414_, v_v_415_, v_i_416_, v_fill_417_);
lean_dec(v_v_415_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill(lean_object* v_00_u03b1_419_, lean_object* v_n_420_, lean_object* v_v_421_, lean_object* v_i_422_, lean_object* v_fill_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib_List_Vector_shiftLeftFill___redArg(v_n_420_, v_v_421_, v_i_422_, v_fill_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftLeftFill___boxed(lean_object* v_00_u03b1_425_, lean_object* v_n_426_, lean_object* v_v_427_, lean_object* v_i_428_, lean_object* v_fill_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_mathlib_List_Vector_shiftLeftFill(v_00_u03b1_425_, v_n_426_, v_v_427_, v_i_428_, v_fill_429_);
lean_dec(v_v_427_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftRightFill___redArg(lean_object* v_n_431_, lean_object* v_v_432_, lean_object* v_i_433_, lean_object* v_fill_434_){
_start:
{
lean_object* v___y_436_; uint8_t v___x_441_; 
v___x_441_ = lean_nat_dec_le(v_n_431_, v_i_433_);
if (v___x_441_ == 0)
{
lean_inc(v_i_433_);
v___y_436_ = v_i_433_;
goto v___jp_435_;
}
else
{
lean_inc(v_n_431_);
v___y_436_ = v_n_431_;
goto v___jp_435_;
}
v___jp_435_:
{
lean_object* v___x_437_; lean_object* v_val_438_; lean_object* v_val_439_; lean_object* v___x_440_; 
v___x_437_ = lean_nat_sub(v_n_431_, v_i_433_);
lean_dec(v_i_433_);
lean_dec(v_n_431_);
v_val_438_ = l_List_replicateTR___redArg(v___y_436_, v_fill_434_);
v_val_439_ = lp_mathlib_List_Vector_take___redArg(v___x_437_, v_v_432_);
v___x_440_ = l_List_appendTR___redArg(v_val_438_, v_val_439_);
return v___x_440_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_shiftRightFill(lean_object* v_00_u03b1_442_, lean_object* v_n_443_, lean_object* v_v_444_, lean_object* v_i_445_, lean_object* v_fill_446_){
_start:
{
lean_object* v___x_447_; 
v___x_447_ = lp_mathlib_List_Vector_shiftRightFill___redArg(v_n_443_, v_v_444_, v_i_445_, v_fill_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___lam__0(lean_object* v_x_448_, lean_object* v_i_449_, lean_object* v_h_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = l_List_get___redArg(v_x_448_, v_i_449_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___lam__0___boxed(lean_object* v_x_452_, lean_object* v_i_453_, lean_object* v_h_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_List_Vector_instGetElemNatLt___lam__0(v_x_452_, v_i_453_, v_h_454_);
lean_dec(v_x_452_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt(lean_object* v_00_u03b1_457_, lean_object* v_n_458_){
_start:
{
lean_object* v___f_459_; 
v___f_459_ = ((lean_object*)(lp_mathlib_List_Vector_instGetElemNatLt___closed__0));
return v___f_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instGetElemNatLt___boxed(lean_object* v_00_u03b1_460_, lean_object* v_n_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_List_Vector_instGetElemNatLt(v_00_u03b1_460_, v_n_461_);
lean_dec(v_n_461_);
return v_res_462_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
