// Lean compiler output
// Module: Mathlib.Data.List.Cycle
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.List public import Mathlib.Data.Fintype.OfMap public import Mathlib.Data.Fin.Basic
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
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_getLast___redArg(lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
lean_object* l_repr(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_cyclicPermutations___redArg(lean_object*);
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t l_List_nodupDecidable___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_List_isRotatedDecidable___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_permutations___redArg(lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Finset_powerset___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_fintypeNodupList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_nextOr_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_nextOr_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_next___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_next(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_prev___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_prev(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_prev_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_prev_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cycle_instCoeList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cycle_ofList___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Cycle_instCoeList___closed__0 = (const lean_object*)&lp_mathlib_Cycle_instCoeList___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instCoeList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_nil(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instEmptyCollection(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instMembership(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableMemOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableMemOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_reverse(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Cycle_toMultiset_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Cycle_toMultiset_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_lists___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_lists(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_decidableNontrivialCoe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_decidableNontrivialCoe___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_decidableNontrivialCoe(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_decidableNontrivialCoe___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__Cycle_decidableNontrivialCoe_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__Cycle_decidableNontrivialCoe_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNontrivial___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNontrivial___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNontrivial(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNontrivial___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNodup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNodup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNodup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNodup___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Cycle_fintypeNodupCycle___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___closed__0 = (const lean_object*)&lp_mathlib_Cycle_fintypeNodupCycle___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_permutations___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__0 = (const lean_object*)&lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__1 = (const lean_object*)&lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__1_value;
static const lean_closure_object lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__2 = (const lean_object*)&lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toFinset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toFinset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_next___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_next(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_prev___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_prev(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "c["};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__2_value)}};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__4_value)}};
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cycle_instRepr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cycle_instRepr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Cycle_instRepr___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_dec(v_x_3_);
lean_dec_ref(v_inst_1_);
lean_inc(v_x_4_);
return v_x_4_;
}
else
{
lean_object* v_tail_5_; 
v_tail_5_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_tail_5_);
if (lean_obj_tag(v_tail_5_) == 0)
{
lean_dec_ref_known(v_x_2_, 2);
lean_dec(v_x_3_);
lean_dec_ref(v_inst_1_);
lean_inc(v_x_4_);
return v_x_4_;
}
else
{
lean_object* v_head_6_; lean_object* v_head_7_; lean_object* v___x_8_; uint8_t v___x_9_; 
v_head_6_ = lean_ctor_get(v_x_2_, 0);
lean_inc(v_head_6_);
lean_dec_ref_known(v_x_2_, 2);
v_head_7_ = lean_ctor_get(v_tail_5_, 0);
lean_inc_ref(v_inst_1_);
lean_inc(v_x_3_);
v___x_8_ = lean_apply_2(v_inst_1_, v_x_3_, v_head_6_);
v___x_9_ = lean_unbox(v___x_8_);
if (v___x_9_ == 0)
{
v_x_2_ = v_tail_5_;
goto _start;
}
else
{
lean_inc(v_head_7_);
lean_dec_ref_known(v_tail_5_, 2);
lean_dec(v_x_3_);
lean_dec_ref(v_inst_1_);
return v_head_7_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___redArg___boxed(lean_object* v_inst_11_, lean_object* v_x_12_, lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_List_nextOr___redArg(v_inst_11_, v_x_12_, v_x_13_, v_x_14_);
lean_dec(v_x_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_x_18_, lean_object* v_x_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_List_nextOr___redArg(v_inst_17_, v_x_18_, v_x_19_, v_x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_nextOr___boxed(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_x_24_, lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_List_nextOr(v_00_u03b1_22_, v_inst_23_, v_x_24_, v_x_25_, v_x_26_);
lean_dec(v_x_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_nextOr_match__1_splitter___redArg(lean_object* v_x_28_, lean_object* v_x_29_, lean_object* v_x_30_, lean_object* v_h__1_31_, lean_object* v_h__2_32_, lean_object* v_h__3_33_){
_start:
{
if (lean_obj_tag(v_x_28_) == 0)
{
lean_object* v___x_34_; 
lean_dec(v_h__3_33_);
lean_dec(v_h__2_32_);
v___x_34_ = lean_apply_2(v_h__1_31_, v_x_29_, v_x_30_);
return v___x_34_;
}
else
{
lean_object* v_tail_35_; 
lean_dec(v_h__1_31_);
v_tail_35_ = lean_ctor_get(v_x_28_, 1);
if (lean_obj_tag(v_tail_35_) == 0)
{
lean_object* v_head_36_; lean_object* v___x_37_; 
lean_dec(v_h__3_33_);
v_head_36_ = lean_ctor_get(v_x_28_, 0);
lean_inc(v_head_36_);
lean_dec_ref_known(v_x_28_, 2);
v___x_37_ = lean_apply_3(v_h__2_32_, v_head_36_, v_x_29_, v_x_30_);
return v___x_37_;
}
else
{
lean_object* v_head_38_; lean_object* v_head_39_; lean_object* v_tail_40_; lean_object* v___x_41_; 
lean_inc_ref(v_tail_35_);
lean_dec(v_h__2_32_);
v_head_38_ = lean_ctor_get(v_x_28_, 0);
lean_inc(v_head_38_);
lean_dec_ref_known(v_x_28_, 2);
v_head_39_ = lean_ctor_get(v_tail_35_, 0);
lean_inc(v_head_39_);
v_tail_40_ = lean_ctor_get(v_tail_35_, 1);
lean_inc(v_tail_40_);
lean_dec_ref_known(v_tail_35_, 2);
v___x_41_ = lean_apply_5(v_h__3_33_, v_head_38_, v_head_39_, v_tail_40_, v_x_29_, v_x_30_);
return v___x_41_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_nextOr_match__1_splitter(lean_object* v_00_u03b1_42_, lean_object* v_motive_43_, lean_object* v_x_44_, lean_object* v_x_45_, lean_object* v_x_46_, lean_object* v_h__1_47_, lean_object* v_h__2_48_, lean_object* v_h__3_49_){
_start:
{
if (lean_obj_tag(v_x_44_) == 0)
{
lean_object* v___x_50_; 
lean_dec(v_h__3_49_);
lean_dec(v_h__2_48_);
v___x_50_ = lean_apply_2(v_h__1_47_, v_x_45_, v_x_46_);
return v___x_50_;
}
else
{
lean_object* v_tail_51_; 
lean_dec(v_h__1_47_);
v_tail_51_ = lean_ctor_get(v_x_44_, 1);
if (lean_obj_tag(v_tail_51_) == 0)
{
lean_object* v_head_52_; lean_object* v___x_53_; 
lean_dec(v_h__3_49_);
v_head_52_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_head_52_);
lean_dec_ref_known(v_x_44_, 2);
v___x_53_ = lean_apply_3(v_h__2_48_, v_head_52_, v_x_45_, v_x_46_);
return v___x_53_;
}
else
{
lean_object* v_head_54_; lean_object* v_head_55_; lean_object* v_tail_56_; lean_object* v___x_57_; 
lean_inc_ref(v_tail_51_);
lean_dec(v_h__2_48_);
v_head_54_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_head_54_);
lean_dec_ref_known(v_x_44_, 2);
v_head_55_ = lean_ctor_get(v_tail_51_, 0);
lean_inc(v_head_55_);
v_tail_56_ = lean_ctor_get(v_tail_51_, 1);
lean_inc(v_tail_56_);
lean_dec_ref_known(v_tail_51_, 2);
v___x_57_ = lean_apply_5(v_h__3_49_, v_head_54_, v_head_55_, v_tail_56_, v_x_45_, v_x_46_);
return v___x_57_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_next___redArg(lean_object* v_inst_58_, lean_object* v_l_59_, lean_object* v_x_60_){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_unsigned_to_nat(0u);
v___x_62_ = l_List_get___redArg(v_l_59_, v___x_61_);
v___x_63_ = lp_mathlib_List_nextOr___redArg(v_inst_58_, v_l_59_, v_x_60_, v___x_62_);
lean_dec(v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_next(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_, lean_object* v_l_66_, lean_object* v_x_67_, lean_object* v_h_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_List_next___redArg(v_inst_65_, v_l_66_, v_x_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_prev___redArg(lean_object* v_inst_70_, lean_object* v_x_71_, lean_object* v_x_72_){
_start:
{
lean_object* v_tail_73_; 
v_tail_73_ = lean_ctor_get(v_x_71_, 1);
if (lean_obj_tag(v_tail_73_) == 0)
{
lean_object* v_head_74_; 
lean_dec(v_x_72_);
lean_dec_ref(v_inst_70_);
v_head_74_ = lean_ctor_get(v_x_71_, 0);
lean_inc(v_head_74_);
lean_dec(v_x_71_);
return v_head_74_;
}
else
{
lean_object* v_head_75_; lean_object* v_head_76_; lean_object* v___x_77_; uint8_t v___x_78_; 
lean_inc_ref(v_tail_73_);
v_head_75_ = lean_ctor_get(v_x_71_, 0);
lean_inc_n(v_head_75_, 2);
lean_dec(v_x_71_);
v_head_76_ = lean_ctor_get(v_tail_73_, 0);
lean_inc_ref(v_inst_70_);
lean_inc(v_x_72_);
v___x_77_ = lean_apply_2(v_inst_70_, v_x_72_, v_head_75_);
v___x_78_ = lean_unbox(v___x_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; uint8_t v___x_80_; 
lean_inc_ref(v_inst_70_);
lean_inc(v_head_76_);
lean_inc(v_x_72_);
v___x_79_ = lean_apply_2(v_inst_70_, v_x_72_, v_head_76_);
v___x_80_ = lean_unbox(v___x_79_);
if (v___x_80_ == 0)
{
lean_dec(v_head_75_);
v_x_71_ = v_tail_73_;
goto _start;
}
else
{
lean_dec_ref_known(v_tail_73_, 2);
lean_dec(v_x_72_);
lean_dec_ref(v_inst_70_);
return v_head_75_;
}
}
else
{
lean_object* v___x_82_; 
lean_dec(v_head_75_);
lean_dec(v_x_72_);
lean_dec_ref(v_inst_70_);
v___x_82_ = l_List_getLast___redArg(v_tail_73_);
lean_dec_ref_known(v_tail_73_, 2);
return v___x_82_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_prev(lean_object* v_00_u03b1_83_, lean_object* v_inst_84_, lean_object* v_x_85_, lean_object* v_x_86_, lean_object* v_x_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_List_prev___redArg(v_inst_84_, v_x_85_, v_x_86_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_prev_match__1_splitter___redArg(lean_object* v_x_89_, lean_object* v_x_90_, lean_object* v_h__1_91_, lean_object* v_h__2_92_, lean_object* v_h__3_93_){
_start:
{
if (lean_obj_tag(v_x_89_) == 0)
{
lean_object* v___x_94_; 
lean_dec(v_h__3_93_);
lean_dec(v_h__2_92_);
v___x_94_ = lean_apply_2(v_h__1_91_, v_x_90_, lean_box(0));
return v___x_94_;
}
else
{
lean_object* v_tail_95_; 
lean_dec(v_h__1_91_);
v_tail_95_ = lean_ctor_get(v_x_89_, 1);
if (lean_obj_tag(v_tail_95_) == 0)
{
lean_object* v_head_96_; lean_object* v___x_97_; 
lean_dec(v_h__3_93_);
v_head_96_ = lean_ctor_get(v_x_89_, 0);
lean_inc(v_head_96_);
lean_dec_ref_known(v_x_89_, 2);
v___x_97_ = lean_apply_3(v_h__2_92_, v_head_96_, v_x_90_, lean_box(0));
return v___x_97_;
}
else
{
lean_object* v_head_98_; lean_object* v_head_99_; lean_object* v_tail_100_; lean_object* v___x_101_; 
lean_inc_ref(v_tail_95_);
lean_dec(v_h__2_92_);
v_head_98_ = lean_ctor_get(v_x_89_, 0);
lean_inc(v_head_98_);
lean_dec_ref_known(v_x_89_, 2);
v_head_99_ = lean_ctor_get(v_tail_95_, 0);
lean_inc(v_head_99_);
v_tail_100_ = lean_ctor_get(v_tail_95_, 1);
lean_inc(v_tail_100_);
lean_dec_ref_known(v_tail_95_, 2);
v___x_101_ = lean_apply_5(v_h__3_93_, v_head_98_, v_head_99_, v_tail_100_, v_x_90_, lean_box(0));
return v___x_101_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__List_prev_match__1_splitter(lean_object* v_00_u03b1_102_, lean_object* v_motive_103_, lean_object* v_x_104_, lean_object* v_x_105_, lean_object* v_x_106_, lean_object* v_h__1_107_, lean_object* v_h__2_108_, lean_object* v_h__3_109_){
_start:
{
if (lean_obj_tag(v_x_104_) == 0)
{
lean_object* v___x_110_; 
lean_dec(v_h__3_109_);
lean_dec(v_h__2_108_);
v___x_110_ = lean_apply_2(v_h__1_107_, v_x_105_, lean_box(0));
return v___x_110_;
}
else
{
lean_object* v_tail_111_; 
lean_dec(v_h__1_107_);
v_tail_111_ = lean_ctor_get(v_x_104_, 1);
if (lean_obj_tag(v_tail_111_) == 0)
{
lean_object* v_head_112_; lean_object* v___x_113_; 
lean_dec(v_h__3_109_);
v_head_112_ = lean_ctor_get(v_x_104_, 0);
lean_inc(v_head_112_);
lean_dec_ref_known(v_x_104_, 2);
v___x_113_ = lean_apply_3(v_h__2_108_, v_head_112_, v_x_105_, lean_box(0));
return v___x_113_;
}
else
{
lean_object* v_head_114_; lean_object* v_head_115_; lean_object* v_tail_116_; lean_object* v___x_117_; 
lean_inc_ref(v_tail_111_);
lean_dec(v_h__2_108_);
v_head_114_ = lean_ctor_get(v_x_104_, 0);
lean_inc(v_head_114_);
lean_dec_ref_known(v_x_104_, 2);
v_head_115_ = lean_ctor_get(v_tail_111_, 0);
lean_inc(v_head_115_);
v_tail_116_ = lean_ctor_get(v_tail_111_, 1);
lean_inc(v_tail_116_);
lean_dec_ref_known(v_tail_111_, 2);
v___x_117_ = lean_apply_5(v_h__3_109_, v_head_114_, v_head_115_, v_tail_116_, v_x_105_, lean_box(0));
return v___x_117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___redArg(lean_object* v_a_118_){
_start:
{
lean_inc(v_a_118_);
return v_a_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___redArg___boxed(lean_object* v_a_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_Cycle_ofList___redArg(v_a_119_);
lean_dec(v_a_119_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList(lean_object* v_00_u03b1_121_, lean_object* v_a_122_){
_start:
{
lean_inc(v_a_122_);
return v_a_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_ofList___boxed(lean_object* v_00_u03b1_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Cycle_ofList(v_00_u03b1_123_, v_a_124_);
lean_dec(v_a_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instCoeList(lean_object* v_00_u03b1_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = ((lean_object*)(lp_mathlib_Cycle_instCoeList___closed__0));
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_nil(lean_object* v_00_u03b1_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_box(0);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instEmptyCollection(lean_object* v_00_u03b1_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lean_box(0);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instInhabited(lean_object* v_00_u03b1_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_box(0);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instMembership(lean_object* v_00_u03b1_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_box(0);
return v___x_136_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableEq___redArg(lean_object* v_inst_137_, lean_object* v_s_u2081_138_, lean_object* v_s_u2082_139_){
_start:
{
uint8_t v___x_140_; 
v___x_140_ = lp_mathlib_List_isRotatedDecidable___redArg(v_inst_137_, v_s_u2081_138_, v_s_u2082_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableEq___redArg___boxed(lean_object* v_inst_141_, lean_object* v_s_u2081_142_, lean_object* v_s_u2082_143_){
_start:
{
uint8_t v_res_144_; lean_object* v_r_145_; 
v_res_144_ = lp_mathlib_Cycle_instDecidableEq___redArg(v_inst_141_, v_s_u2081_142_, v_s_u2082_143_);
v_r_145_ = lean_box(v_res_144_);
return v_r_145_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableEq(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_s_u2081_148_, lean_object* v_s_u2082_149_){
_start:
{
uint8_t v___x_150_; 
v___x_150_ = lp_mathlib_List_isRotatedDecidable___redArg(v_inst_147_, v_s_u2081_148_, v_s_u2082_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableEq___boxed(lean_object* v_00_u03b1_151_, lean_object* v_inst_152_, lean_object* v_s_u2081_153_, lean_object* v_s_u2082_154_){
_start:
{
uint8_t v_res_155_; lean_object* v_r_156_; 
v_res_155_ = lp_mathlib_Cycle_instDecidableEq(v_00_u03b1_151_, v_inst_152_, v_s_u2081_153_, v_s_u2082_154_);
v_r_156_ = lean_box(v_res_155_);
return v_r_156_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg(lean_object* v_inst_157_, lean_object* v_x_158_, lean_object* v_s_159_){
_start:
{
lean_object* v___f_160_; uint8_t v___x_161_; 
v___f_160_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_160_, 0, v_inst_157_);
v___x_161_ = l_List_elem___redArg(v___f_160_, v_x_158_, v_s_159_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg___boxed(lean_object* v_inst_162_, lean_object* v_x_163_, lean_object* v_s_164_){
_start:
{
uint8_t v_res_165_; lean_object* v_r_166_; 
v_res_165_ = lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg(v_inst_162_, v_x_163_, v_s_164_);
v_r_166_ = lean_box(v_res_165_);
return v_r_166_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableMemOfDecidableEq(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_x_169_, lean_object* v_s_170_){
_start:
{
uint8_t v___x_171_; 
v___x_171_ = lp_mathlib_Cycle_instDecidableMemOfDecidableEq___redArg(v_inst_168_, v_x_169_, v_s_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableMemOfDecidableEq___boxed(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_, lean_object* v_x_174_, lean_object* v_s_175_){
_start:
{
uint8_t v_res_176_; lean_object* v_r_177_; 
v_res_176_ = lp_mathlib_Cycle_instDecidableMemOfDecidableEq(v_00_u03b1_172_, v_inst_173_, v_x_174_, v_s_175_);
v_r_177_ = lean_box(v_res_176_);
return v_r_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_reverse___redArg(lean_object* v_s_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = l_List_reverse___redArg(v_s_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_reverse(lean_object* v_00_u03b1_180_, lean_object* v_s_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = l_List_reverse___redArg(v_s_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___redArg(lean_object* v_s_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = l_List_lengthTR___redArg(v_s_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___redArg___boxed(lean_object* v_s_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Cycle_length___redArg(v_s_185_);
lean_dec(v_s_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length(lean_object* v_00_u03b1_187_, lean_object* v_s_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = l_List_lengthTR___redArg(v_s_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_length___boxed(lean_object* v_00_u03b1_190_, lean_object* v_s_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Cycle_length(v_00_u03b1_190_, v_s_191_);
lean_dec(v_s_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Cycle_toMultiset_spec__0___redArg(lean_object* v_q_193_, lean_object* v_f_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_apply_1(v_f_194_, v_q_193_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Cycle_toMultiset_spec__0(lean_object* v_00_u03b1_196_, lean_object* v_00_u03c6_197_, lean_object* v_q_198_, lean_object* v_f_199_, lean_object* v_h_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lean_apply_1(v_f_199_, v_q_198_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___redArg(lean_object* v_s_202_){
_start:
{
lean_inc(v_s_202_);
return v_s_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___redArg___boxed(lean_object* v_s_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Cycle_toMultiset___redArg(v_s_203_);
lean_dec(v_s_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset(lean_object* v_00_u03b1_205_, lean_object* v_s_206_){
_start:
{
lean_inc(v_s_206_);
return v_s_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toMultiset___boxed(lean_object* v_00_u03b1_207_, lean_object* v_s_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_Cycle_toMultiset(v_00_u03b1_207_, v_s_208_);
lean_dec(v_s_208_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0___redArg(lean_object* v_f_210_, lean_object* v_a_211_, lean_object* v_a_212_){
_start:
{
if (lean_obj_tag(v_a_211_) == 0)
{
lean_object* v___x_213_; 
lean_dec(v_f_210_);
v___x_213_ = l_List_reverse___redArg(v_a_212_);
return v___x_213_;
}
else
{
lean_object* v_head_214_; lean_object* v_tail_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_224_; 
v_head_214_ = lean_ctor_get(v_a_211_, 0);
v_tail_215_ = lean_ctor_get(v_a_211_, 1);
v_isSharedCheck_224_ = !lean_is_exclusive(v_a_211_);
if (v_isSharedCheck_224_ == 0)
{
v___x_217_ = v_a_211_;
v_isShared_218_ = v_isSharedCheck_224_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_tail_215_);
lean_inc(v_head_214_);
lean_dec(v_a_211_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_224_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v___x_219_; lean_object* v___x_221_; 
lean_inc(v_f_210_);
v___x_219_ = lean_apply_1(v_f_210_, v_head_214_);
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 1, v_a_212_);
lean_ctor_set(v___x_217_, 0, v___x_219_);
v___x_221_ = v___x_217_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_219_);
lean_ctor_set(v_reuseFailAlloc_223_, 1, v_a_212_);
v___x_221_ = v_reuseFailAlloc_223_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
v_a_211_ = v_tail_215_;
v_a_212_ = v___x_221_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_map___redArg(lean_object* v_f_225_, lean_object* v_a_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = lean_box(0);
v___x_228_ = lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0___redArg(v_f_225_, v_a_226_, v___x_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_map(lean_object* v_00_u03b1_229_, lean_object* v_00_u03b2_230_, lean_object* v_f_231_, lean_object* v_a_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_Cycle_map___redArg(v_f_231_, v_a_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0(lean_object* v_00_u03b1_234_, lean_object* v_00_u03b2_235_, lean_object* v_f_236_, lean_object* v_a_237_, lean_object* v_a_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_List_mapTR_loop___at___00Cycle_map_spec__0___redArg(v_f_236_, v_a_237_, v_a_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_lists___redArg(lean_object* v_s_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_List_cyclicPermutations___redArg(v_s_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_lists(lean_object* v_00_u03b1_242_, lean_object* v_s_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_mathlib_List_cyclicPermutations___redArg(v_s_243_);
return v___x_244_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_decidableNontrivialCoe___redArg(lean_object* v_inst_245_, lean_object* v_x_246_){
_start:
{
if (lean_obj_tag(v_x_246_) == 0)
{
uint8_t v___x_247_; 
lean_dec_ref(v_inst_245_);
v___x_247_ = 0;
return v___x_247_;
}
else
{
lean_object* v_head_248_; lean_object* v_tail_249_; uint8_t v___x_250_; 
v_head_248_ = lean_ctor_get(v_x_246_, 0);
lean_inc(v_head_248_);
v_tail_249_ = lean_ctor_get(v_x_246_, 1);
lean_inc(v_tail_249_);
lean_dec_ref_known(v_x_246_, 2);
v___x_250_ = 0;
if (lean_obj_tag(v_tail_249_) == 0)
{
lean_dec(v_head_248_);
lean_dec_ref(v_inst_245_);
return v___x_250_;
}
else
{
lean_object* v_head_251_; lean_object* v_tail_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_263_; 
v_head_251_ = lean_ctor_get(v_tail_249_, 0);
v_tail_252_ = lean_ctor_get(v_tail_249_, 1);
v_isSharedCheck_263_ = !lean_is_exclusive(v_tail_249_);
if (v_isSharedCheck_263_ == 0)
{
v___x_254_ = v_tail_249_;
v_isShared_255_ = v_isSharedCheck_263_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_tail_252_);
lean_inc(v_head_251_);
lean_dec(v_tail_249_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_263_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; uint8_t v___x_257_; 
lean_inc_ref(v_inst_245_);
lean_inc(v_head_248_);
v___x_256_ = lean_apply_2(v_inst_245_, v_head_248_, v_head_251_);
v___x_257_ = lean_unbox(v___x_256_);
if (v___x_257_ == 0)
{
uint8_t v___x_258_; 
lean_del_object(v___x_254_);
lean_dec(v_tail_252_);
lean_dec(v_head_248_);
lean_dec_ref(v_inst_245_);
v___x_258_ = 1;
return v___x_258_;
}
else
{
lean_object* v___x_260_; 
if (v_isShared_255_ == 0)
{
lean_ctor_set(v___x_254_, 0, v_head_248_);
v___x_260_ = v___x_254_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v_head_248_);
lean_ctor_set(v_reuseFailAlloc_262_, 1, v_tail_252_);
v___x_260_ = v_reuseFailAlloc_262_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
uint8_t v___x_261_; 
v___x_261_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_245_, v___x_260_);
if (v___x_261_ == 0)
{
return v___x_250_;
}
else
{
return v___x_261_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_decidableNontrivialCoe___redArg___boxed(lean_object* v_inst_264_, lean_object* v_x_265_){
_start:
{
uint8_t v_res_266_; lean_object* v_r_267_; 
v_res_266_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_264_, v_x_265_);
v_r_267_ = lean_box(v_res_266_);
return v_r_267_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_decidableNontrivialCoe(lean_object* v_00_u03b1_268_, lean_object* v_inst_269_, lean_object* v_x_270_){
_start:
{
uint8_t v___x_271_; 
v___x_271_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_269_, v_x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_decidableNontrivialCoe___boxed(lean_object* v_00_u03b1_272_, lean_object* v_inst_273_, lean_object* v_x_274_){
_start:
{
uint8_t v_res_275_; lean_object* v_r_276_; 
v_res_275_ = lp_mathlib_Cycle_decidableNontrivialCoe(v_00_u03b1_272_, v_inst_273_, v_x_274_);
v_r_276_ = lean_box(v_res_275_);
return v_r_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__Cycle_decidableNontrivialCoe_match__1_splitter___redArg(lean_object* v_x_277_, lean_object* v_h__1_278_, lean_object* v_h__2_279_, lean_object* v_h__3_280_){
_start:
{
if (lean_obj_tag(v_x_277_) == 0)
{
lean_object* v___x_281_; lean_object* v___x_282_; 
lean_dec(v_h__3_280_);
lean_dec(v_h__2_279_);
v___x_281_ = lean_box(0);
v___x_282_ = lean_apply_1(v_h__1_278_, v___x_281_);
return v___x_282_;
}
else
{
lean_object* v_tail_283_; 
lean_dec(v_h__1_278_);
v_tail_283_ = lean_ctor_get(v_x_277_, 1);
if (lean_obj_tag(v_tail_283_) == 0)
{
lean_object* v_head_284_; lean_object* v___x_285_; 
lean_dec(v_h__3_280_);
v_head_284_ = lean_ctor_get(v_x_277_, 0);
lean_inc(v_head_284_);
lean_dec_ref_known(v_x_277_, 2);
v___x_285_ = lean_apply_1(v_h__2_279_, v_head_284_);
return v___x_285_;
}
else
{
lean_object* v_head_286_; lean_object* v_head_287_; lean_object* v_tail_288_; lean_object* v___x_289_; 
lean_inc_ref(v_tail_283_);
lean_dec(v_h__2_279_);
v_head_286_ = lean_ctor_get(v_x_277_, 0);
lean_inc(v_head_286_);
lean_dec_ref_known(v_x_277_, 2);
v_head_287_ = lean_ctor_get(v_tail_283_, 0);
lean_inc(v_head_287_);
v_tail_288_ = lean_ctor_get(v_tail_283_, 1);
lean_inc(v_tail_288_);
lean_dec_ref_known(v_tail_283_, 2);
v___x_289_ = lean_apply_3(v_h__3_280_, v_head_286_, v_head_287_, v_tail_288_);
return v___x_289_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Cycle_0__Cycle_decidableNontrivialCoe_match__1_splitter(lean_object* v_00_u03b1_290_, lean_object* v_motive_291_, lean_object* v_x_292_, lean_object* v_h__1_293_, lean_object* v_h__2_294_, lean_object* v_h__3_295_){
_start:
{
if (lean_obj_tag(v_x_292_) == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; 
lean_dec(v_h__3_295_);
lean_dec(v_h__2_294_);
v___x_296_ = lean_box(0);
v___x_297_ = lean_apply_1(v_h__1_293_, v___x_296_);
return v___x_297_;
}
else
{
lean_object* v_tail_298_; 
lean_dec(v_h__1_293_);
v_tail_298_ = lean_ctor_get(v_x_292_, 1);
if (lean_obj_tag(v_tail_298_) == 0)
{
lean_object* v_head_299_; lean_object* v___x_300_; 
lean_dec(v_h__3_295_);
v_head_299_ = lean_ctor_get(v_x_292_, 0);
lean_inc(v_head_299_);
lean_dec_ref_known(v_x_292_, 2);
v___x_300_ = lean_apply_1(v_h__2_294_, v_head_299_);
return v___x_300_;
}
else
{
lean_object* v_head_301_; lean_object* v_head_302_; lean_object* v_tail_303_; lean_object* v___x_304_; 
lean_inc_ref(v_tail_298_);
lean_dec(v_h__2_294_);
v_head_301_ = lean_ctor_get(v_x_292_, 0);
lean_inc(v_head_301_);
lean_dec_ref_known(v_x_292_, 2);
v_head_302_ = lean_ctor_get(v_tail_298_, 0);
lean_inc(v_head_302_);
v_tail_303_ = lean_ctor_get(v_tail_298_, 1);
lean_inc(v_tail_303_);
lean_dec_ref_known(v_tail_298_, 2);
v___x_304_ = lean_apply_3(v_h__3_295_, v_head_301_, v_head_302_, v_tail_303_);
return v___x_304_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNontrivial___redArg(lean_object* v_inst_305_, lean_object* v_s_306_){
_start:
{
uint8_t v___x_307_; 
v___x_307_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_305_, v_s_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNontrivial___redArg___boxed(lean_object* v_inst_308_, lean_object* v_s_309_){
_start:
{
uint8_t v_res_310_; lean_object* v_r_311_; 
v_res_310_ = lp_mathlib_Cycle_instDecidableNontrivial___redArg(v_inst_308_, v_s_309_);
v_r_311_ = lean_box(v_res_310_);
return v_r_311_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNontrivial(lean_object* v_00_u03b1_312_, lean_object* v_inst_313_, lean_object* v_s_314_){
_start:
{
uint8_t v___x_315_; 
v___x_315_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_313_, v_s_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNontrivial___boxed(lean_object* v_00_u03b1_316_, lean_object* v_inst_317_, lean_object* v_s_318_){
_start:
{
uint8_t v_res_319_; lean_object* v_r_320_; 
v_res_319_ = lp_mathlib_Cycle_instDecidableNontrivial(v_00_u03b1_316_, v_inst_317_, v_s_318_);
v_r_320_ = lean_box(v_res_319_);
return v_r_320_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNodup___redArg(lean_object* v_inst_321_, lean_object* v_s_322_){
_start:
{
uint8_t v___x_323_; 
v___x_323_ = l_List_nodupDecidable___redArg(v_inst_321_, v_s_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNodup___redArg___boxed(lean_object* v_inst_324_, lean_object* v_s_325_){
_start:
{
uint8_t v_res_326_; lean_object* v_r_327_; 
v_res_326_ = lp_mathlib_Cycle_instDecidableNodup___redArg(v_inst_324_, v_s_325_);
v_r_327_ = lean_box(v_res_326_);
return v_r_327_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_instDecidableNodup(lean_object* v_00_u03b1_328_, lean_object* v_inst_329_, lean_object* v_s_330_){
_start:
{
uint8_t v___x_331_; 
v___x_331_ = l_List_nodupDecidable___redArg(v_inst_329_, v_s_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instDecidableNodup___boxed(lean_object* v_00_u03b1_332_, lean_object* v_inst_333_, lean_object* v_s_334_){
_start:
{
uint8_t v_res_335_; lean_object* v_r_336_; 
v_res_335_ = lp_mathlib_Cycle_instDecidableNodup(v_00_u03b1_332_, v_inst_333_, v_s_334_);
v_r_336_ = lean_box(v_res_335_);
return v_r_336_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0(lean_object* v_inst_337_, lean_object* v_a_338_, lean_object* v_b_339_){
_start:
{
uint8_t v___x_340_; 
v___x_340_ = lp_mathlib_List_isRotatedDecidable___redArg(v_inst_337_, v_a_338_, v_b_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0___boxed(lean_object* v_inst_341_, lean_object* v_a_342_, lean_object* v_b_343_){
_start:
{
uint8_t v_res_344_; lean_object* v_r_345_; 
v_res_344_ = lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0(v_inst_341_, v_a_342_, v_b_343_);
v_r_345_ = lean_box(v_res_344_);
return v_r_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1(lean_object* v_l_346_){
_start:
{
lean_inc(v_l_346_);
return v_l_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1___boxed(lean_object* v_l_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__1(v_l_347_);
lean_dec(v_l_347_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle___redArg(lean_object* v_inst_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v___f_352_; lean_object* v___f_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_352_, 0, v_inst_350_);
v___f_353_ = ((lean_object*)(lp_mathlib_Cycle_fintypeNodupCycle___redArg___closed__0));
v___x_354_ = lp_mathlib_fintypeNodupList___redArg(v_inst_351_);
v___x_355_ = lp_mathlib_Finset_image___redArg(v___f_352_, v___f_353_, v___x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupCycle(lean_object* v_00_u03b1_356_, lean_object* v_inst_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_Cycle_fintypeNodupCycle___redArg(v_inst_357_, v_inst_358_);
return v___x_359_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0(lean_object* v_inst_360_, lean_object* v_a_361_){
_start:
{
uint8_t v___x_362_; 
v___x_362_ = lp_mathlib_Cycle_decidableNontrivialCoe___redArg(v_inst_360_, v_a_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0___boxed(lean_object* v_inst_363_, lean_object* v_a_364_){
_start:
{
uint8_t v_res_365_; lean_object* v_r_366_; 
v_res_365_ = lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0(v_inst_363_, v_a_364_);
v_r_366_ = lean_box(v_res_365_);
return v_r_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3(lean_object* v_val_367_, lean_object* v_property_368_){
_start:
{
lean_inc(v_val_367_);
return v_val_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3___boxed(lean_object* v_val_369_, lean_object* v_property_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__3(v_val_369_, v_property_370_);
lean_dec(v_val_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg(lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___f_377_; lean_object* v___f_378_; lean_object* v___f_379_; lean_object* v___f_380_; lean_object* v___f_381_; lean_object* v___f_382_; lean_object* v_univSubsets_383_; lean_object* v_allPerms_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; 
lean_inc_ref(v_inst_375_);
v___f_377_ = lean_alloc_closure((void*)(lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_377_, 0, v_inst_375_);
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_Cycle_fintypeNodupCycle___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_378_, 0, v_inst_375_);
v___f_379_ = ((lean_object*)(lp_mathlib_Cycle_fintypeNodupCycle___redArg___closed__0));
v___f_380_ = ((lean_object*)(lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__0));
v___f_381_ = ((lean_object*)(lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__1));
v___f_382_ = ((lean_object*)(lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg___closed__2));
v_univSubsets_383_ = lp_mathlib_Finset_powerset___redArg(v_inst_376_);
v_allPerms_384_ = lp_mathlib_Multiset_bind___redArg(v_univSubsets_383_, v___f_380_);
v___x_385_ = lp_mathlib_Multiset_pmap___redArg(v___f_381_, v_allPerms_384_);
v___x_386_ = lp_mathlib_Finset_image___redArg(v___f_378_, v___f_379_, v___x_385_);
v___x_387_ = lp_mathlib_Finset_map___redArg(v___f_382_, v___x_386_);
v___x_388_ = lp_mathlib_Multiset_filter___redArg(v___f_377_, v___x_387_);
v___x_389_ = lp_mathlib_Fintype_subtype___redArg(v___x_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_fintypeNodupNontrivialCycle(lean_object* v_00_u03b1_390_, lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Cycle_fintypeNodupNontrivialCycle___redArg(v_inst_391_, v_inst_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toFinset___redArg(lean_object* v_inst_394_, lean_object* v_s_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_List_dedup___redArg(v_inst_394_, v_s_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_toFinset(lean_object* v_00_u03b1_397_, lean_object* v_inst_398_, lean_object* v_s_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_mathlib_List_dedup___redArg(v_inst_398_, v_s_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_next___redArg(lean_object* v_inst_401_, lean_object* v_s_402_, lean_object* v_x_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_List_next___redArg(v_inst_401_, v_s_402_, v_x_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_next(lean_object* v_00_u03b1_405_, lean_object* v_inst_406_, lean_object* v_s_407_, lean_object* v___hs_408_, lean_object* v_x_409_, lean_object* v___hx_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lp_mathlib_List_next___redArg(v_inst_406_, v_s_407_, v_x_409_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_prev___redArg(lean_object* v_inst_412_, lean_object* v_s_413_, lean_object* v_x_414_){
_start:
{
lean_object* v___x_415_; 
v___x_415_ = lp_mathlib_List_prev___redArg(v_inst_412_, v_s_413_, v_x_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_prev(lean_object* v_00_u03b1_416_, lean_object* v_inst_417_, lean_object* v_s_418_, lean_object* v___hs_419_, lean_object* v_x_420_, lean_object* v___hx_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_List_prev___redArg(v_inst_417_, v_s_418_, v_x_420_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0(lean_object* v_inst_432_, lean_object* v___x_433_, lean_object* v___f_434_, lean_object* v_s_435_, lean_object* v_x_436_){
_start:
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_437_ = ((lean_object*)(lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__1));
v___x_438_ = lean_alloc_closure((void*)(l_repr), 3, 2);
lean_closure_set(v___x_438_, 0, lean_box(0));
lean_closure_set(v___x_438_, 1, v_inst_432_);
v___x_439_ = lp_mathlib_Cycle_map___redArg(v___x_438_, v_s_435_);
v___x_440_ = lp_mathlib_List_cyclicPermutations___redArg(v___x_439_);
v___x_441_ = l_List_head_x21___redArg(v___x_433_, v___x_440_);
lean_dec(v___x_440_);
v___x_442_ = ((lean_object*)(lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__3));
v___x_443_ = l_Std_Format_joinSep___redArg(v___f_434_, v___x_441_, v___x_442_);
v___x_444_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_437_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = ((lean_object*)(lp_mathlib_Cycle_instRepr___redArg___lam__0___closed__5));
v___x_446_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_444_);
lean_ctor_set(v___x_446_, 1, v___x_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg___lam__0___boxed(lean_object* v_inst_447_, lean_object* v___x_448_, lean_object* v___f_449_, lean_object* v_s_450_, lean_object* v_x_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_Cycle_instRepr___redArg___lam__0(v_inst_447_, v___x_448_, v___f_449_, v_s_450_, v_x_451_);
lean_dec(v_x_451_);
lean_dec(v___x_448_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr___redArg(lean_object* v_inst_454_){
_start:
{
lean_object* v___f_455_; lean_object* v___x_456_; lean_object* v___f_457_; 
v___f_455_ = ((lean_object*)(lp_mathlib_Cycle_instRepr___redArg___closed__0));
v___x_456_ = lean_box(0);
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib_Cycle_instRepr___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_457_, 0, v_inst_454_);
lean_closure_set(v___f_457_, 1, v___x_456_);
lean_closure_set(v___f_457_, 2, v___f_455_);
return v___f_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cycle_instRepr(lean_object* v_00_u03b1_458_, lean_object* v_inst_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_Cycle_instRepr___redArg(v_inst_459_);
return v___x_460_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_List(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Cycle(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Cycle(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_List(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Cycle(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Cycle(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Cycle(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Cycle(builtin);
}
#ifdef __cplusplus
}
#endif
