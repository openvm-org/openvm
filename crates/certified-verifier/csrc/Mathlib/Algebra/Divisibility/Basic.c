// Lean compiler output
// Module: Mathlib.Algebra.Divisibility.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Opposite public import Mathlib.Tactic.Common public import Batteries.Tactic.SeqFocus public import Mathlib.Tactic.Attr.Core
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_semigroupDvd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_semigroupDvd___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2223_u1d63___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_∣ᵣ_"};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 189, 119, 164, 78, 147, 208, 112)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2223_u1d63___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2223_u1d63___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ∣ᵣ "};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2223_u1d63___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__7_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2223_u1d63___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(51) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2223_u1d63___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2223_u1d63__ = (const lean_object*)&lp_mathlib_term___u2223_u1d63___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "RightDvd"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(42, 248, 153, 191, 5, 2, 73, 88)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_semigroupDvd(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_semigroupDvd___boxed(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_semigroupDvd(v_00_u03b1_4_, v_inst_5_);
lean_dec(v_inst_5_);
return v_res_6_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__5));
v___x_43_ = l_String_toRawSubstring_x27(v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1(lean_object* v_x_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; uint8_t v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib_term___u2223_u1d63___00__closed__1));
lean_inc(v_x_55_);
v___x_59_ = l_Lean_Syntax_isOfKind(v_x_55_, v___x_58_);
if (v___x_59_ == 0)
{
lean_object* v___x_60_; lean_object* v___x_61_; 
lean_dec(v_x_55_);
v___x_60_ = lean_box(1);
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_a_57_);
return v___x_61_;
}
else
{
lean_object* v_quotContext_62_; lean_object* v_currMacroScope_63_; lean_object* v_ref_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; uint8_t v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_quotContext_62_ = lean_ctor_get(v_a_56_, 1);
v_currMacroScope_63_ = lean_ctor_get(v_a_56_, 2);
v_ref_64_ = lean_ctor_get(v_a_56_, 5);
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_55_, v___x_65_);
v___x_67_ = lean_unsigned_to_nat(2u);
v___x_68_ = l_Lean_Syntax_getArg(v_x_55_, v___x_67_);
lean_dec(v_x_55_);
v___x_69_ = 0;
v___x_70_ = l_Lean_SourceInfo_fromRef(v_ref_64_, v___x_69_);
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4));
v___x_72_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__6);
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__7));
lean_inc(v_currMacroScope_63_);
lean_inc(v_quotContext_62_);
v___x_74_ = l_Lean_addMacroScope(v_quotContext_62_, v___x_73_, v_currMacroScope_63_);
v___x_75_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__9));
lean_inc_n(v___x_70_, 2);
v___x_76_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_76_, 0, v___x_70_);
lean_ctor_set(v___x_76_, 1, v___x_72_);
lean_ctor_set(v___x_76_, 2, v___x_74_);
lean_ctor_set(v___x_76_, 3, v___x_75_);
v___x_77_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__11));
v___x_78_ = l_Lean_Syntax_node2(v___x_70_, v___x_77_, v___x_66_, v___x_68_);
v___x_79_ = l_Lean_Syntax_node2(v___x_70_, v___x_71_, v___x_76_, v___x_78_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v_a_57_);
return v___x_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___boxed(lean_object* v_x_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1(v_x_81_, v_a_82_, v_a_83_);
lean_dec_ref(v_a_82_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1(lean_object* v_x_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v___x_91_; uint8_t v___x_92_; 
v___x_91_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______macroRules__term___u2223_u1d63____1___closed__4));
lean_inc(v_x_88_);
v___x_92_ = l_Lean_Syntax_isOfKind(v_x_88_, v___x_91_);
if (v___x_92_ == 0)
{
lean_object* v___x_93_; lean_object* v___x_94_; 
lean_dec(v_x_88_);
v___x_93_ = lean_box(0);
v___x_94_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v_a_90_);
return v___x_94_;
}
else
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; uint8_t v___x_98_; 
v___x_95_ = lean_unsigned_to_nat(0u);
v___x_96_ = l_Lean_Syntax_getArg(v_x_88_, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___closed__1));
lean_inc(v___x_96_);
v___x_98_ = l_Lean_Syntax_isOfKind(v___x_96_, v___x_97_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; lean_object* v___x_100_; 
lean_dec(v___x_96_);
lean_dec(v_x_88_);
v___x_99_ = lean_box(0);
v___x_100_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v_a_90_);
return v___x_100_;
}
else
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; uint8_t v___x_104_; 
v___x_101_ = lean_unsigned_to_nat(1u);
v___x_102_ = l_Lean_Syntax_getArg(v_x_88_, v___x_101_);
lean_dec(v_x_88_);
v___x_103_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_102_);
v___x_104_ = l_Lean_Syntax_matchesNull(v___x_102_, v___x_103_);
if (v___x_104_ == 0)
{
lean_object* v___x_105_; lean_object* v___x_106_; 
lean_dec(v___x_102_);
lean_dec(v___x_96_);
v___x_105_ = lean_box(0);
v___x_106_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_a_90_);
return v___x_106_;
}
else
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v_ref_109_; uint8_t v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_107_ = l_Lean_Syntax_getArg(v___x_102_, v___x_95_);
v___x_108_ = l_Lean_Syntax_getArg(v___x_102_, v___x_101_);
lean_dec(v___x_102_);
v_ref_109_ = l_Lean_replaceRef(v___x_96_, v_a_89_);
lean_dec(v___x_96_);
v___x_110_ = 0;
v___x_111_ = l_Lean_SourceInfo_fromRef(v_ref_109_, v___x_110_);
lean_dec(v_ref_109_);
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2223_u1d63___00__closed__1));
v___x_113_ = ((lean_object*)(lp_mathlib_term___u2223_u1d63___00__closed__4));
lean_inc(v___x_111_);
v___x_114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_111_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = l_Lean_Syntax_node3(v___x_111_, v___x_112_, v___x_107_, v___x_114_, v___x_108_);
v___x_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_90_);
return v___x_116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1___boxed(lean_object* v_x_117_, lean_object* v_a_118_, lean_object* v_a_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib___aux__Mathlib__Algebra__Divisibility__Basic______unexpand__RightDvd__1(v_x_117_, v_a_118_, v_a_119_);
lean_dec(v_a_118_);
return v_res_120_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_SeqFocus(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_SeqFocus(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Divisibility_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
