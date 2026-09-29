// Lean compiler output
// Module: Mathlib.Data.Part
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Subsingleton public import Mathlib.Logic.Equiv.Defs public import Mathlib.Tactic.ToAdditive
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
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___redArg(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMembership(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_none___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Part_none___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_none___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_none___closed__0 = (const lean_object*)&lp_mathlib_Part_none___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Part_none(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_some(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Part_noneDecidable(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_noneDecidable___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Part_someDecidable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_someDecidable___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Part_0__Part_ofOption_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Part_0__Part_ofOption_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Part_instCoeOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_ofOption, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Part_instCoeOption___closed__0 = (const lean_object*)&lp_mathlib_Part_instCoeOption___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Part_instCoeOption(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Part_ofOptionDecidable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOptionDecidable___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Part_ofOptionDecidable(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOptionDecidable___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Part_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Part_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_Part_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Part_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instOrderBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_assert___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_assert___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_assert(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_bind___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_bind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_bind(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Part_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_instMonad___lam__0, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__0 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_instMonad___lam__2, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__1 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_instMonad___lam__5, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__2 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_instMonad___lam__7, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__3 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__4 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__4_value;
static const lean_ctor_object lp_mathlib_Part_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Part_instMonad___closed__4_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__0_value)}};
static const lean_object* lp_mathlib_Part_instMonad___closed__5 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__5_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_some, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__6 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__6_value;
static const lean_ctor_object lp_mathlib_Part_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Part_instMonad___closed__5_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__6_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__1_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__2_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__3_value)}};
static const lean_object* lp_mathlib_Part_instMonad___closed__7 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__7_value;
static const lean_closure_object lp_mathlib_Part_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Part_bind, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Part_instMonad___closed__8 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_Part_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Part_instMonad___closed__7_value),((lean_object*)&lp_mathlib_Part_instMonad___closed__8_value)}};
static const lean_object* lp_mathlib_Part_instMonad___closed__9 = (const lean_object*)&lp_mathlib_Part_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Part_instMonad = (const lean_object*)&lp_mathlib_Part_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_unwrap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_unwrap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instAdd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instDiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMod___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instMod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instAppend___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instAppend(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instInter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instInter(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instUnion___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instUnion(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instSDiff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_instSDiff(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___redArg(lean_object* v_o_1_, uint8_t v_inst_2_){
_start:
{
if (v_inst_2_ == 0)
{
lean_object* v___x_3_; 
lean_dec(v_o_1_);
v___x_3_ = lean_box(0);
return v___x_3_;
}
else
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_apply_1(v_o_1_, lean_box(0));
v___x_5_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5_, 0, v___x_4_);
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___redArg___boxed(lean_object* v_o_6_, lean_object* v_inst_7_){
_start:
{
uint8_t v_inst_10__boxed_8_; lean_object* v_res_9_; 
v_inst_10__boxed_8_ = lean_unbox(v_inst_7_);
v_res_9_ = lp_mathlib_Part_toOption___redArg(v_o_6_, v_inst_10__boxed_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption(lean_object* v_00_u03b1_10_, lean_object* v_o_11_, uint8_t v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_Part_toOption___redArg(v_o_11_, v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_toOption___boxed(lean_object* v_00_u03b1_14_, lean_object* v_o_15_, lean_object* v_inst_16_){
_start:
{
uint8_t v_inst_19__boxed_17_; lean_object* v_res_18_; 
v_inst_19__boxed_17_ = lean_unbox(v_inst_16_);
v_res_18_ = lp_mathlib_Part_toOption(v_00_u03b1_14_, v_o_15_, v_inst_19__boxed_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMembership(lean_object* v_00_u03b1_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_none___lam__0(lean_object* v_t_21_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_none(lean_object* v_00_u03b1_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = ((lean_object*)(lp_mathlib_Part_none___closed__0));
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instInhabited(lean_object* v_00_u03b1_25_){
_start:
{
lean_object* v___f_26_; 
v___f_26_ = ((lean_object*)(lp_mathlib_Part_none___closed__0));
return v___f_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg___lam__0(lean_object* v_a_27_, lean_object* v_x_28_){
_start:
{
lean_inc(v_a_27_);
return v_a_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg___lam__0___boxed(lean_object* v_a_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Part_some___redArg___lam__0(v_a_29_, v_x_30_);
lean_dec(v_a_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_some___redArg(lean_object* v_a_32_){
_start:
{
lean_object* v___f_33_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_33_, 0, v_a_32_);
return v___f_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_some(lean_object* v_00_u03b1_34_, lean_object* v_a_35_){
_start:
{
lean_object* v___f_36_; 
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_36_, 0, v_a_35_);
return v___f_36_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Part_noneDecidable(lean_object* v_00_u03b1_37_){
_start:
{
uint8_t v___x_38_; 
v___x_38_ = 0;
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_noneDecidable___boxed(lean_object* v_00_u03b1_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_mathlib_Part_noneDecidable(v_00_u03b1_39_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Part_someDecidable(lean_object* v_00_u03b1_42_, lean_object* v_a_43_){
_start:
{
uint8_t v___x_44_; 
v___x_44_ = 1;
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_someDecidable___boxed(lean_object* v_00_u03b1_45_, lean_object* v_a_46_){
_start:
{
uint8_t v_res_47_; lean_object* v_r_48_; 
v_res_47_ = lp_mathlib_Part_someDecidable(v_00_u03b1_45_, v_a_46_);
lean_dec(v_a_46_);
v_r_48_ = lean_box(v_res_47_);
return v_r_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___redArg(lean_object* v_a_49_, uint8_t v_inst_50_, lean_object* v_d_51_){
_start:
{
if (v_inst_50_ == 0)
{
lean_dec(v_a_49_);
lean_inc(v_d_51_);
return v_d_51_;
}
else
{
lean_object* v___x_52_; 
v___x_52_ = lean_apply_1(v_a_49_, lean_box(0));
return v___x_52_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___redArg___boxed(lean_object* v_a_53_, lean_object* v_inst_54_, lean_object* v_d_55_){
_start:
{
uint8_t v_inst_8__boxed_56_; lean_object* v_res_57_; 
v_inst_8__boxed_56_ = lean_unbox(v_inst_54_);
v_res_57_ = lp_mathlib_Part_getOrElse___redArg(v_a_53_, v_inst_8__boxed_56_, v_d_55_);
lean_dec(v_d_55_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse(lean_object* v_00_u03b1_58_, lean_object* v_a_59_, uint8_t v_inst_60_, lean_object* v_d_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_Part_getOrElse___redArg(v_a_59_, v_inst_60_, v_d_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_getOrElse___boxed(lean_object* v_00_u03b1_63_, lean_object* v_a_64_, lean_object* v_inst_65_, lean_object* v_d_66_){
_start:
{
uint8_t v_inst_13__boxed_67_; lean_object* v_res_68_; 
v_inst_13__boxed_67_ = lean_unbox(v_inst_65_);
v_res_68_ = lp_mathlib_Part_getOrElse(v_00_u03b1_63_, v_a_64_, v_inst_13__boxed_67_, v_d_66_);
lean_dec(v_d_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOption___redArg(lean_object* v_x_69_){
_start:
{
if (lean_obj_tag(v_x_69_) == 0)
{
lean_object* v___f_70_; 
v___f_70_ = ((lean_object*)(lp_mathlib_Part_none___closed__0));
return v___f_70_;
}
else
{
lean_object* v_val_71_; lean_object* v___f_72_; 
v_val_71_ = lean_ctor_get(v_x_69_, 0);
lean_inc(v_val_71_);
lean_dec_ref_known(v_x_69_, 1);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_72_, 0, v_val_71_);
return v___f_72_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOption(lean_object* v_00_u03b1_73_, lean_object* v_x_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Part_ofOption___redArg(v_x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Part_0__Part_ofOption_match__1_splitter___redArg(lean_object* v_x_76_, lean_object* v_h__1_77_, lean_object* v_h__2_78_){
_start:
{
if (lean_obj_tag(v_x_76_) == 0)
{
lean_object* v___x_79_; lean_object* v___x_80_; 
lean_dec(v_h__2_78_);
v___x_79_ = lean_box(0);
v___x_80_ = lean_apply_1(v_h__1_77_, v___x_79_);
return v___x_80_;
}
else
{
lean_object* v_val_81_; lean_object* v___x_82_; 
lean_dec(v_h__1_77_);
v_val_81_ = lean_ctor_get(v_x_76_, 0);
lean_inc(v_val_81_);
lean_dec_ref_known(v_x_76_, 1);
v___x_82_ = lean_apply_1(v_h__2_78_, v_val_81_);
return v___x_82_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Part_0__Part_ofOption_match__1_splitter(lean_object* v_00_u03b1_83_, lean_object* v_motive_84_, lean_object* v_x_85_, lean_object* v_h__1_86_, lean_object* v_h__2_87_){
_start:
{
if (lean_obj_tag(v_x_85_) == 0)
{
lean_object* v___x_88_; lean_object* v___x_89_; 
lean_dec(v_h__2_87_);
v___x_88_ = lean_box(0);
v___x_89_ = lean_apply_1(v_h__1_86_, v___x_88_);
return v___x_89_;
}
else
{
lean_object* v_val_90_; lean_object* v___x_91_; 
lean_dec(v_h__1_86_);
v_val_90_ = lean_ctor_get(v_x_85_, 0);
lean_inc(v_val_90_);
lean_dec_ref_known(v_x_85_, 1);
v___x_91_ = lean_apply_1(v_h__2_87_, v_val_90_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instCoeOption(lean_object* v_00_u03b1_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = ((lean_object*)(lp_mathlib_Part_instCoeOption___closed__0));
return v___x_94_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Part_ofOptionDecidable___redArg(lean_object* v_x_95_){
_start:
{
if (lean_obj_tag(v_x_95_) == 0)
{
uint8_t v___x_96_; 
v___x_96_ = 0;
return v___x_96_;
}
else
{
uint8_t v___x_97_; 
v___x_97_ = 1;
return v___x_97_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOptionDecidable___redArg___boxed(lean_object* v_x_98_){
_start:
{
uint8_t v_res_99_; lean_object* v_r_100_; 
v_res_99_ = lp_mathlib_Part_ofOptionDecidable___redArg(v_x_98_);
lean_dec(v_x_98_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Part_ofOptionDecidable(lean_object* v_00_u03b1_101_, lean_object* v_x_102_){
_start:
{
uint8_t v___x_103_; 
v___x_103_ = lp_mathlib_Part_ofOptionDecidable___redArg(v_x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_ofOptionDecidable___boxed(lean_object* v_00_u03b1_104_, lean_object* v_x_105_){
_start:
{
uint8_t v_res_106_; lean_object* v_r_107_; 
v_res_106_ = lp_mathlib_Part_ofOptionDecidable(v_00_u03b1_104_, v_x_105_);
lean_dec(v_x_105_);
v_r_107_ = lean_box(v_res_106_);
return v_r_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instPartialOrder(lean_object* v_00_u03b1_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = ((lean_object*)(lp_mathlib_Part_instPartialOrder___closed__0));
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instOrderBot(lean_object* v_00_u03b1_113_){
_start:
{
lean_object* v___f_114_; 
v___f_114_ = ((lean_object*)(lp_mathlib_Part_none___closed__0));
return v___f_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_assert___redArg___lam__0(lean_object* v_f_115_, lean_object* v_ha_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lean_apply_2(v_f_115_, lean_box(0), lean_box(0));
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_assert___redArg(lean_object* v_f_118_){
_start:
{
lean_object* v___f_119_; 
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_Part_assert___redArg___lam__0), 2, 1);
lean_closure_set(v___f_119_, 0, v_f_118_);
return v___f_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_assert(lean_object* v_00_u03b1_120_, lean_object* v_p_121_, lean_object* v_f_122_){
_start:
{
lean_object* v___f_123_; 
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_Part_assert___redArg___lam__0), 2, 1);
lean_closure_set(v___f_123_, 0, v_f_122_);
return v___f_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_bind___redArg___lam__0(lean_object* v_f_124_, lean_object* v_g_125_, lean_object* v_b_126_, lean_object* v___y_127_){
_start:
{
lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_128_ = lean_apply_1(v_f_124_, lean_box(0));
v___x_129_ = lean_apply_2(v_g_125_, v___x_128_, lean_box(0));
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_bind___redArg(lean_object* v_f_130_, lean_object* v_g_131_){
_start:
{
lean_object* v___f_132_; lean_object* v___f_133_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_Part_bind___redArg___lam__0), 4, 2);
lean_closure_set(v___f_132_, 0, v_f_130_);
lean_closure_set(v___f_132_, 1, v_g_131_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_Part_assert___redArg___lam__0), 2, 1);
lean_closure_set(v___f_133_, 0, v___f_132_);
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_bind(lean_object* v_00_u03b1_134_, lean_object* v_00_u03b2_135_, lean_object* v_f_136_, lean_object* v_g_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Part_bind___redArg(v_f_136_, v_g_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_map___redArg___lam__0(lean_object* v_o_139_, lean_object* v_f_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = lean_apply_1(v_o_139_, lean_box(0));
v___x_143_ = lean_apply_1(v_f_140_, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_map___redArg(lean_object* v_f_144_, lean_object* v_o_145_){
_start:
{
lean_object* v___f_146_; 
v___f_146_ = lean_alloc_closure((void*)(lp_mathlib_Part_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_146_, 0, v_o_145_);
lean_closure_set(v___f_146_, 1, v_f_144_);
return v___f_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_map(lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_f_149_, lean_object* v_o_150_){
_start:
{
lean_object* v___f_151_; 
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Part_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_151_, 0, v_o_150_);
lean_closure_set(v___f_151_, 1, v_f_149_);
return v___f_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__0(lean_object* v_00_u03b1_152_, lean_object* v_00_u03b2_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_157_, 0, lean_box(0));
lean_closure_set(v___x_157_, 1, lean_box(0));
lean_closure_set(v___x_157_, 2, v___y_154_);
v___x_158_ = lp_mathlib_Part_map___redArg___lam__0(v___y_155_, v___x_157_, lean_box(0));
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__1(lean_object* v_x_159_, lean_object* v_y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_162_ = lean_box(0);
v___x_163_ = lean_apply_1(v_x_159_, v___x_162_);
v___x_164_ = lp_mathlib_Part_map___redArg___lam__0(v___x_163_, v_y_160_, lean_box(0));
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__2(lean_object* v_00_u03b1_165_, lean_object* v_00_u03b2_166_, lean_object* v_f_167_, lean_object* v_x_168_, lean_object* v___y_169_){
_start:
{
lean_object* v___f_170_; lean_object* v___x_88__overap_171_; lean_object* v___x_172_; 
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMonad___lam__1), 3, 1);
lean_closure_set(v___f_170_, 0, v_x_168_);
v___x_88__overap_171_ = lp_mathlib_Part_bind___redArg(v_f_167_, v___f_170_);
v___x_172_ = lean_apply_1(v___x_88__overap_171_, lean_box(0));
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__3(lean_object* v_a_173_, lean_object* v_x_174_, lean_object* v___y_175_){
_start:
{
lean_inc(v_a_173_);
return v_a_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__3___boxed(lean_object* v_a_176_, lean_object* v_x_177_, lean_object* v___y_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Part_instMonad___lam__3(v_a_176_, v_x_177_, v___y_178_);
lean_dec(v_x_177_);
lean_dec(v_a_176_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__4(lean_object* v_y_180_, lean_object* v_a_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___f_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_95__overap_186_; lean_object* v___x_187_; 
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMonad___lam__3___boxed), 3, 1);
lean_closure_set(v___f_183_, 0, v_a_181_);
v___x_184_ = lean_box(0);
v___x_185_ = lean_apply_1(v_y_180_, v___x_184_);
v___x_95__overap_186_ = lp_mathlib_Part_bind___redArg(v___x_185_, v___f_183_);
v___x_187_ = lean_apply_1(v___x_95__overap_186_, lean_box(0));
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__5(lean_object* v_00_u03b1_188_, lean_object* v_00_u03b2_189_, lean_object* v_x_190_, lean_object* v_y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v___f_193_; lean_object* v___x_100__overap_194_; lean_object* v___x_195_; 
v___f_193_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMonad___lam__4), 3, 1);
lean_closure_set(v___f_193_, 0, v_y_191_);
v___x_100__overap_194_ = lp_mathlib_Part_bind___redArg(v_x_190_, v___f_193_);
v___x_195_ = lean_apply_1(v___x_100__overap_194_, lean_box(0));
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__6(lean_object* v_y_196_, lean_object* v_x_197_, lean_object* v___y_198_){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = lean_box(0);
v___x_200_ = lean_apply_2(v_y_196_, v___x_199_, lean_box(0));
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__6___boxed(lean_object* v_y_201_, lean_object* v_x_202_, lean_object* v___y_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Part_instMonad___lam__6(v_y_201_, v_x_202_, v___y_203_);
lean_dec(v_x_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMonad___lam__7(lean_object* v_00_u03b1_205_, lean_object* v_00_u03b2_206_, lean_object* v_x_207_, lean_object* v_y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_109__overap_211_; lean_object* v___x_212_; 
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMonad___lam__6___boxed), 3, 1);
lean_closure_set(v___f_210_, 0, v_y_208_);
v___x_109__overap_211_ = lp_mathlib_Part_bind___redArg(v_x_207_, v___f_210_);
v___x_212_ = lean_apply_1(v___x_109__overap_211_, lean_box(0));
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict___redArg___lam__0(lean_object* v_o_233_, lean_object* v_h_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lean_apply_1(v_o_233_, lean_box(0));
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict___redArg(lean_object* v_o_236_){
_start:
{
lean_object* v___f_237_; 
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_Part_restrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_237_, 0, v_o_236_);
return v___f_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_restrict(lean_object* v_00_u03b1_238_, lean_object* v_p_239_, lean_object* v_o_240_, lean_object* v_H_241_){
_start:
{
lean_object* v___f_242_; 
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_Part_restrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_242_, 0, v_o_240_);
return v___f_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_unwrap___redArg(lean_object* v_o_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_apply_1(v_o_243_, lean_box(0));
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_unwrap(lean_object* v_00_u03b1_245_, lean_object* v_o_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lean_apply_1(v_o_246_, lean_box(0));
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instOne___redArg(lean_object* v_inst_248_){
_start:
{
lean_object* v___f_249_; 
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_249_, 0, v_inst_248_);
return v___f_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instOne(lean_object* v_00_u03b1_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v___f_252_; 
v___f_252_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_252_, 0, v_inst_251_);
return v___f_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instZero___redArg(lean_object* v_inst_253_){
_start:
{
lean_object* v___f_254_; 
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_254_, 0, v_inst_253_);
return v___f_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instZero(lean_object* v_00_u03b1_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v___f_257_; 
v___f_257_ = lean_alloc_closure((void*)(lp_mathlib_Part_some___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_257_, 0, v_inst_256_);
return v___f_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__0(lean_object* v_inst_258_, lean_object* v_x1_259_, lean_object* v_x2_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_apply_2(v_inst_258_, v_x1_259_, v_x2_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__1(lean_object* v_b_262_, lean_object* v_y_263_, lean_object* v___y_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_Part_map___redArg___lam__0(v_b_262_, v_y_263_, lean_box(0));
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg___lam__2(lean_object* v___f_266_, lean_object* v_a_267_, lean_object* v_b_268_, lean_object* v___y_269_){
_start:
{
lean_object* v___f_270_; lean_object* v___f_271_; lean_object* v___x_140__overap_272_; lean_object* v___x_273_; 
v___f_270_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__1), 3, 1);
lean_closure_set(v___f_270_, 0, v_b_268_);
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_Part_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_271_, 0, v_a_267_);
lean_closure_set(v___f_271_, 1, v___f_266_);
v___x_140__overap_272_ = lp_mathlib_Part_bind___redArg(v___f_271_, v___f_270_);
v___x_273_ = lean_apply_1(v___x_140__overap_272_, lean_box(0));
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul___redArg(lean_object* v_inst_274_){
_start:
{
lean_object* v___f_275_; lean_object* v___f_276_; 
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_275_, 0, v_inst_274_);
v___f_276_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_276_, 0, v___f_275_);
return v___f_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMul(lean_object* v_00_u03b1_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_Part_instMul___redArg(v_inst_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instAdd___redArg(lean_object* v_inst_280_){
_start:
{
lean_object* v___f_281_; lean_object* v___f_282_; 
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_281_, 0, v_inst_280_);
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_282_, 0, v___f_281_);
return v___f_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instAdd(lean_object* v_00_u03b1_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_mathlib_Part_instAdd___redArg(v_inst_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instInv___redArg(lean_object* v_inst_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lean_alloc_closure((void*)(lp_mathlib_Part_map), 4, 3);
lean_closure_set(v___x_287_, 0, lean_box(0));
lean_closure_set(v___x_287_, 1, lean_box(0));
lean_closure_set(v___x_287_, 2, v_inst_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instInv(lean_object* v_00_u03b1_288_, lean_object* v_inst_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lean_alloc_closure((void*)(lp_mathlib_Part_map), 4, 3);
lean_closure_set(v___x_290_, 0, lean_box(0));
lean_closure_set(v___x_290_, 1, lean_box(0));
lean_closure_set(v___x_290_, 2, v_inst_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instNeg___redArg(lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lean_alloc_closure((void*)(lp_mathlib_Part_map), 4, 3);
lean_closure_set(v___x_292_, 0, lean_box(0));
lean_closure_set(v___x_292_, 1, lean_box(0));
lean_closure_set(v___x_292_, 2, v_inst_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instNeg(lean_object* v_00_u03b1_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lean_alloc_closure((void*)(lp_mathlib_Part_map), 4, 3);
lean_closure_set(v___x_295_, 0, lean_box(0));
lean_closure_set(v___x_295_, 1, lean_box(0));
lean_closure_set(v___x_295_, 2, v_inst_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instDiv___redArg(lean_object* v_inst_296_){
_start:
{
lean_object* v___f_297_; lean_object* v___f_298_; 
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_297_, 0, v_inst_296_);
v___f_298_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_298_, 0, v___f_297_);
return v___f_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instDiv(lean_object* v_00_u03b1_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Part_instDiv___redArg(v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instSub___redArg(lean_object* v_inst_302_){
_start:
{
lean_object* v___f_303_; lean_object* v___f_304_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_303_, 0, v_inst_302_);
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_304_, 0, v___f_303_);
return v___f_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instSub(lean_object* v_00_u03b1_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_Part_instSub___redArg(v_inst_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMod___redArg(lean_object* v_inst_308_){
_start:
{
lean_object* v___f_309_; lean_object* v___f_310_; 
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_309_, 0, v_inst_308_);
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_310_, 0, v___f_309_);
return v___f_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instMod(lean_object* v_00_u03b1_311_, lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_Part_instMod___redArg(v_inst_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instAppend___redArg(lean_object* v_inst_314_){
_start:
{
lean_object* v___f_315_; lean_object* v___f_316_; 
v___f_315_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_315_, 0, v_inst_314_);
v___f_316_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_316_, 0, v___f_315_);
return v___f_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instAppend(lean_object* v_00_u03b1_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_mathlib_Part_instAppend___redArg(v_inst_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instInter___redArg(lean_object* v_inst_320_){
_start:
{
lean_object* v___f_321_; lean_object* v___f_322_; 
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_321_, 0, v_inst_320_);
v___f_322_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_322_, 0, v___f_321_);
return v___f_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instInter(lean_object* v_00_u03b1_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lp_mathlib_Part_instInter___redArg(v_inst_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instUnion___redArg(lean_object* v_inst_326_){
_start:
{
lean_object* v___f_327_; lean_object* v___f_328_; 
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_327_, 0, v_inst_326_);
v___f_328_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_328_, 0, v___f_327_);
return v___f_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instUnion(lean_object* v_00_u03b1_329_, lean_object* v_inst_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_mathlib_Part_instUnion___redArg(v_inst_330_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instSDiff___redArg(lean_object* v_inst_332_){
_start:
{
lean_object* v___f_333_; lean_object* v___f_334_; 
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_333_, 0, v_inst_332_);
v___f_334_ = lean_alloc_closure((void*)(lp_mathlib_Part_instMul___redArg___lam__2), 4, 1);
lean_closure_set(v___f_334_, 0, v___f_333_);
return v___f_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Part_instSDiff(lean_object* v_00_u03b1_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Part_instSDiff___redArg(v_inst_336_);
return v___x_337_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Part(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Part(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Part(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Part(builtin);
}
#ifdef __cplusplus
}
#endif
