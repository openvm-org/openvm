// Lean compiler output
// Module: Mathlib.Data.Rat.Init
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Notation public import Batteries.Classes.RatCast
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
static const lean_string_object lp_mathlib_term_u211a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = "termℚ"};
static const lean_object* lp_mathlib_term_u211a___closed__0 = (const lean_object*)&lp_mathlib_term_u211a___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u211a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u211a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 107, 235, 123, 89, 23, 135, 218)}};
static const lean_object* lp_mathlib_term_u211a___closed__1 = (const lean_object*)&lp_mathlib_term_u211a___closed__1_value;
static const lean_string_object lp_mathlib_term_u211a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ℚ"};
static const lean_object* lp_mathlib_term_u211a___closed__2 = (const lean_object*)&lp_mathlib_term_u211a___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u211a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u211a___closed__2_value)}};
static const lean_object* lp_mathlib_term_u211a___closed__3 = (const lean_object*)&lp_mathlib_term_u211a___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u211a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u211a___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u211a___closed__3_value)}};
static const lean_object* lp_mathlib_term_u211a___closed__4 = (const lean_object*)&lp_mathlib_term_u211a___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u211a = (const lean_object*)&lp_mathlib_term_u211a___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Rat"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 55, 105, 214, 206, 30, 120, 51)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u211a_u22650___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "termℚ≥0"};
static const lean_object* lp_mathlib_term_u211a_u22650___closed__0 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u211a_u22650___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u211a_u22650___closed__0_value),LEAN_SCALAR_PTR_LITERAL(188, 79, 250, 169, 68, 133, 190, 98)}};
static const lean_object* lp_mathlib_term_u211a_u22650___closed__1 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__1_value;
static const lean_string_object lp_mathlib_term_u211a_u22650___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "ℚ≥0"};
static const lean_object* lp_mathlib_term_u211a_u22650___closed__2 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u211a_u22650___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u211a_u22650___closed__2_value)}};
static const lean_object* lp_mathlib_term_u211a_u22650___closed__3 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u211a_u22650___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u211a_u22650___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u211a_u22650___closed__3_value)}};
static const lean_object* lp_mathlib_term_u211a_u22650___closed__4 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u211a_u22650 = (const lean_object*)&lp_mathlib_term_u211a_u22650___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "NNRat"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 217, 98, 171, 152, 255, 249, 89)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__NNRat__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__NNRat__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NNRat_instNNRatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instNNRatCast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instNNRatCast___closed__0 = (const lean_object*)&lp_mathlib_NNRat_instNNRatCast___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_instNNRatCast = (const lean_object*)&lp_mathlib_NNRat_instNNRatCast___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeTail___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeTail(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeHTCT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeHTCT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instNNRatCast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instNNRatCast___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Rat_instNNRatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_instNNRatCast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_instNNRatCast___closed__0 = (const lean_object*)&lp_mathlib_Rat_instNNRatCast___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Rat_instNNRatCast = (const lean_object*)&lp_mathlib_Rat_instNNRatCast___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_num_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_num_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_num(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_num___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_den(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_den___boxed(lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__0));
v___x_14_ = l_String_toRawSubstring_x27(v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1(lean_object* v_x_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v___x_31_; uint8_t v___x_32_; 
v___x_31_ = ((lean_object*)(lp_mathlib_term_u211a___closed__1));
v___x_32_ = l_Lean_Syntax_isOfKind(v_x_28_, v___x_31_);
if (v___x_32_ == 0)
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_box(1);
v___x_34_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v_a_30_);
return v___x_34_;
}
else
{
lean_object* v_quotContext_35_; lean_object* v_currMacroScope_36_; lean_object* v_ref_37_; uint8_t v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v_quotContext_35_ = lean_ctor_get(v_a_29_, 1);
v_currMacroScope_36_ = lean_ctor_get(v_a_29_, 2);
v_ref_37_ = lean_ctor_get(v_a_29_, 5);
v___x_38_ = 0;
v___x_39_ = l_Lean_SourceInfo_fromRef(v_ref_37_, v___x_38_);
v___x_40_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1, &lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__1);
v___x_41_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__2));
lean_inc(v_currMacroScope_36_);
lean_inc(v_quotContext_35_);
v___x_42_ = l_Lean_addMacroScope(v_quotContext_35_, v___x_41_, v_currMacroScope_36_);
v___x_43_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___closed__6));
v___x_44_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_44_, 0, v___x_39_);
lean_ctor_set(v___x_44_, 1, v___x_40_);
lean_ctor_set(v___x_44_, 2, v___x_42_);
lean_ctor_set(v___x_44_, 3, v___x_43_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v_a_30_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1___boxed(lean_object* v_x_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a__1(v_x_46_, v_a_47_, v_a_48_);
lean_dec_ref(v_a_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__1));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(0);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_ref_60_; uint8_t v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v_ref_60_ = l_Lean_replaceRef(v_x_53_, v_a_54_);
lean_dec(v_x_53_);
v___x_61_ = 0;
v___x_62_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_61_);
lean_dec(v_ref_60_);
v___x_63_ = ((lean_object*)(lp_mathlib_term_u211a___closed__1));
v___x_64_ = ((lean_object*)(lp_mathlib_term_u211a___closed__2));
lean_inc(v___x_62_);
v___x_65_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_62_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
v___x_66_ = l_Lean_Syntax_node1(v___x_62_, v___x_63_, v___x_65_);
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v_a_55_);
return v___x_67_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___boxed(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1(v_x_68_, v_a_69_, v_a_70_);
lean_dec(v_a_69_);
return v_res_71_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_84_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__0));
v___x_85_ = l_String_toRawSubstring_x27(v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1(lean_object* v_x_94_, lean_object* v_a_95_, lean_object* v_a_96_){
_start:
{
lean_object* v___x_97_; uint8_t v___x_98_; 
v___x_97_ = ((lean_object*)(lp_mathlib_term_u211a_u22650___closed__1));
v___x_98_ = l_Lean_Syntax_isOfKind(v_x_94_, v___x_97_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_box(1);
v___x_100_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v_a_96_);
return v___x_100_;
}
else
{
lean_object* v_quotContext_101_; lean_object* v_currMacroScope_102_; lean_object* v_ref_103_; uint8_t v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_quotContext_101_ = lean_ctor_get(v_a_95_, 1);
v_currMacroScope_102_ = lean_ctor_get(v_a_95_, 2);
v_ref_103_ = lean_ctor_get(v_a_95_, 5);
v___x_104_ = 0;
v___x_105_ = l_Lean_SourceInfo_fromRef(v_ref_103_, v___x_104_);
v___x_106_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1, &lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__1);
v___x_107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__2));
lean_inc(v_currMacroScope_102_);
lean_inc(v_quotContext_101_);
v___x_108_ = l_Lean_addMacroScope(v_quotContext_101_, v___x_107_, v_currMacroScope_102_);
v___x_109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___closed__4));
v___x_110_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_110_, 0, v___x_105_);
lean_ctor_set(v___x_110_, 1, v___x_106_);
lean_ctor_set(v___x_110_, 2, v___x_108_);
lean_ctor_set(v___x_110_, 3, v___x_109_);
v___x_111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v_a_96_);
return v___x_111_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1___boxed(lean_object* v_x_112_, lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib___aux__Mathlib__Data__Rat__Init______macroRules__term_u211a_u22650__1(v_x_112_, v_a_113_, v_a_114_);
lean_dec_ref(v_a_113_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__NNRat__1(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__Rat__1___closed__1));
lean_inc(v_x_116_);
v___x_120_ = l_Lean_Syntax_isOfKind(v_x_116_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v_x_116_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_118_);
return v___x_122_;
}
else
{
lean_object* v_ref_123_; uint8_t v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_ref_123_ = l_Lean_replaceRef(v_x_116_, v_a_117_);
lean_dec(v_x_116_);
v___x_124_ = 0;
v___x_125_ = l_Lean_SourceInfo_fromRef(v_ref_123_, v___x_124_);
lean_dec(v_ref_123_);
v___x_126_ = ((lean_object*)(lp_mathlib_term_u211a_u22650___closed__1));
v___x_127_ = ((lean_object*)(lp_mathlib_term_u211a_u22650___closed__2));
lean_inc(v___x_125_);
v___x_128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_125_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
v___x_129_ = l_Lean_Syntax_node1(v___x_125_, v___x_126_, v___x_128_);
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v_a_118_);
return v___x_130_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__NNRat__1___boxed(lean_object* v_x_131_, lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib___aux__Mathlib__Data__Rat__Init______unexpand__NNRat__1(v_x_131_, v_a_132_, v_a_133_);
lean_dec(v_a_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0(lean_object* v_q_135_){
_start:
{
lean_inc_ref(v_q_135_);
return v_q_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0___boxed(lean_object* v_q_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_NNRat_instNNRatCast___lam__0(v_q_136_);
lean_dec_ref(v_q_136_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___redArg(lean_object* v_inst_140_, lean_object* v_a_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lean_apply_1(v_inst_140_, v_a_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast(lean_object* v_K_143_, lean_object* v_inst_144_, lean_object* v_a_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lean_apply_1(v_inst_144_, v_a_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeTail___redArg(lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_cast), 3, 2);
lean_closure_set(v___x_148_, 0, lean_box(0));
lean_closure_set(v___x_148_, 1, v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeTail(lean_object* v_K_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_cast), 3, 2);
lean_closure_set(v___x_151_, 0, lean_box(0));
lean_closure_set(v___x_151_, 1, v_inst_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeHTCT___redArg(lean_object* v_inst_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_cast), 3, 2);
lean_closure_set(v___x_153_, 0, lean_box(0));
lean_closure_set(v___x_153_, 1, v_inst_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRatCast_toCoeHTCT(lean_object* v_K_154_, lean_object* v_inst_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_cast), 3, 2);
lean_closure_set(v___x_156_, 0, lean_box(0));
lean_closure_set(v___x_156_, 1, v_inst_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instNNRatCast___lam__0(lean_object* v_self_157_){
_start:
{
lean_inc_ref(v_self_157_);
return v_self_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instNNRatCast___lam__0___boxed(lean_object* v_self_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_self_158_);
lean_dec_ref(v_self_158_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_num_spec__0(lean_object* v_a_162_){
_start:
{
lean_inc_ref(v_a_162_);
return v_a_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_num_spec__0___boxed(lean_object* v_a_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_NNRat_cast___at___00NNRat_num_spec__0(v_a_163_);
lean_dec_ref(v_a_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_num(lean_object* v_q_165_){
_start:
{
lean_object* v_num_166_; lean_object* v___x_167_; 
v_num_166_ = lean_ctor_get(v_q_165_, 0);
v___x_167_ = lean_nat_abs(v_num_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_num___boxed(lean_object* v_q_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_NNRat_num(v_q_168_);
lean_dec_ref(v_q_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_den(lean_object* v_q_170_){
_start:
{
lean_object* v_den_171_; 
v_den_171_ = lean_ctor_get(v_q_170_, 1);
lean_inc(v_den_171_);
return v_den_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_den___boxed(lean_object* v_q_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_NNRat_den(v_q_172_);
lean_dec_ref(v_q_172_);
return v_res_173_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Classes_RatCast(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Classes_RatCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Classes_RatCast(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Classes_RatCast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
}
#ifdef __cplusplus
}
#endif
