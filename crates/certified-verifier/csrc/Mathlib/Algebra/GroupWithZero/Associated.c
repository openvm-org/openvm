// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Associated
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.IsBotOne public import Mathlib.Algebra.Prime.Lemmas public import Mathlib.Order.BoundedOrder.Basic
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instInhabited___redArg(lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "GroupWithZero"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(58, 121, 55, 130, 17, 219, 245, 76)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Associated"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(108, 136, 167, 27, 196, 11, 122, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(237, 166, 91, 195, 232, 30, 3, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_~ᵤ_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(31, 139, 213, 43, 36, 12, 174, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ~ᵤ "};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__18_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__21_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64__ = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(129, 148, 232, 81, 178, 191, 175, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associated_setoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associated_setoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Associates_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Associates_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_Associates_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mkMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_mkMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBoundedOrder(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Associates_instDecidableRelDvd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instDecidableRelDvd___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Associates_instDecidableRelDvd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instDecidableRelDvd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__8));
v___x_59_ = l_String_toRawSubstring_x27(v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1(lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12));
lean_inc(v_x_71_);
v___x_76_ = l_Lean_Syntax_isOfKind(v_x_71_, v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; lean_object* v___x_78_; 
lean_dec(v_x_71_);
v___x_77_ = lean_box(1);
v___x_78_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_73_);
return v___x_78_;
}
else
{
lean_object* v_quotContext_79_; lean_object* v_currMacroScope_80_; lean_object* v_ref_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_quotContext_79_ = lean_ctor_get(v_a_72_, 1);
v_currMacroScope_80_ = lean_ctor_get(v_a_72_, 2);
v_ref_81_ = lean_ctor_get(v_a_72_, 5);
v___x_82_ = l_Lean_Syntax_getArg(v_x_71_, v___x_74_);
v___x_83_ = lean_unsigned_to_nat(2u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_71_, v___x_83_);
lean_dec(v_x_71_);
v___x_85_ = 0;
v___x_86_ = l_Lean_SourceInfo_fromRef(v_ref_81_, v___x_85_);
v___x_87_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4));
v___x_88_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5, &lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5_once, _init_lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__5);
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__6));
lean_inc(v_currMacroScope_80_);
lean_inc(v_quotContext_79_);
v___x_90_ = l_Lean_addMacroScope(v_quotContext_79_, v___x_89_, v_currMacroScope_80_);
v___x_91_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__8));
lean_inc_n(v___x_86_, 2);
v___x_92_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_92_, 0, v___x_86_);
lean_ctor_set(v___x_92_, 1, v___x_88_);
lean_ctor_set(v___x_92_, 2, v___x_90_);
lean_ctor_set(v___x_92_, 3, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__10));
v___x_94_ = l_Lean_Syntax_node2(v___x_86_, v___x_93_, v___x_82_, v___x_84_);
v___x_95_ = l_Lean_Syntax_node2(v___x_86_, v___x_87_, v___x_92_, v___x_94_);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_73_);
return v___x_96_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___boxed(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1(v_x_97_, v_a_98_, v_a_99_);
lean_dec_ref(v_a_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1(lean_object* v_x_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______macroRules____private__Mathlib__Algebra__GroupWithZero__Associated__0__term___x7e_u1d64____1___closed__4));
lean_inc(v_x_104_);
v___x_108_ = l_Lean_Syntax_isOfKind(v_x_104_, v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v_x_104_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_106_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_111_ = lean_unsigned_to_nat(0u);
v___x_112_ = l_Lean_Syntax_getArg(v_x_104_, v___x_111_);
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___closed__1));
lean_inc(v___x_112_);
v___x_114_ = l_Lean_Syntax_isOfKind(v___x_112_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v___x_115_; lean_object* v___x_116_; 
lean_dec(v___x_112_);
lean_dec(v_x_104_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v_a_106_);
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_117_ = lean_unsigned_to_nat(1u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_104_, v___x_117_);
lean_dec(v_x_104_);
v___x_119_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_118_);
v___x_120_ = l_Lean_Syntax_matchesNull(v___x_118_, v___x_119_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; 
lean_dec(v___x_118_);
lean_dec(v___x_112_);
v___x_121_ = lean_box(0);
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v_a_106_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v_ref_125_; uint8_t v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_123_ = l_Lean_Syntax_getArg(v___x_118_, v___x_111_);
v___x_124_ = l_Lean_Syntax_getArg(v___x_118_, v___x_117_);
lean_dec(v___x_118_);
v_ref_125_ = l_Lean_replaceRef(v___x_112_, v_a_105_);
lean_dec(v___x_112_);
v___x_126_ = 0;
v___x_127_ = l_Lean_SourceInfo_fromRef(v_ref_125_, v___x_126_);
lean_dec(v_ref_125_);
v___x_128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__12));
v___x_129_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0__term___x7e_u1d64___00__closed__15));
lean_inc(v___x_127_);
v___x_130_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_127_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
v___x_131_ = l_Lean_Syntax_node3(v___x_127_, v___x_128_, v___x_123_, v___x_130_, v___x_124_);
v___x_132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_a_106_);
return v___x_132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1___boxed(lean_object* v_x_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib___private_Mathlib_Algebra_GroupWithZero_Associated_0____aux__Mathlib__Algebra__GroupWithZero__Associated______unexpand__Associated__1(v_x_133_, v_a_134_, v_a_135_);
lean_dec(v_a_134_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associated_setoid(lean_object* v_M_137_, lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lean_box(0);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associated_setoid___boxed(lean_object* v_M_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_Associated_setoid(v_M_140_, v_inst_141_);
lean_dec_ref(v_inst_141_);
return v_res_142_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg(lean_object* v_inst_143_, lean_object* v_x_144_, lean_object* v_x_145_){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; uint8_t v___x_148_; 
lean_inc_ref(v_inst_143_);
lean_inc(v_x_144_);
lean_inc(v_x_145_);
v___x_146_ = lean_apply_2(v_inst_143_, v_x_145_, v_x_144_);
v___x_147_ = lean_apply_2(v_inst_143_, v_x_144_, v_x_145_);
v___x_148_ = lean_unbox(v___x_147_);
if (v___x_148_ == 0)
{
uint8_t v___x_149_; 
v___x_149_ = lean_unbox(v___x_147_);
return v___x_149_;
}
else
{
uint8_t v___x_150_; 
v___x_150_ = lean_unbox(v___x_146_);
return v___x_150_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg___boxed(lean_object* v_inst_151_, lean_object* v_x_152_, lean_object* v_x_153_){
_start:
{
uint8_t v_res_154_; lean_object* v_r_155_; 
v_res_154_ = lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg(v_inst_151_, v_x_152_, v_x_153_);
v_r_155_ = lean_box(v_res_154_);
return v_r_155_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd(lean_object* v_M_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_x_160_, lean_object* v_x_161_){
_start:
{
uint8_t v___x_162_; 
v___x_162_ = lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___redArg(v_inst_159_, v_x_160_, v_x_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd___boxed(lean_object* v_M_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_x_167_, lean_object* v_x_168_){
_start:
{
uint8_t v_res_169_; lean_object* v_r_170_; 
v_res_169_ = lp_mathlib_instDecidableRelAssociatedOfIsLeftCancelMulZeroOfDvd(v_M_163_, v_inst_164_, v_inst_165_, v_inst_166_, v_x_167_, v_x_168_);
lean_dec_ref(v_inst_164_);
v_r_170_ = lean_box(v_res_169_);
return v_r_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___redArg(lean_object* v_a_171_){
_start:
{
lean_inc(v_a_171_);
return v_a_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___redArg___boxed(lean_object* v_a_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_Associates_mk___redArg(v_a_172_);
lean_dec(v_a_172_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk(lean_object* v_M_174_, lean_object* v_inst_175_, lean_object* v_a_176_){
_start:
{
lean_inc(v_a_176_);
return v_a_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mk___boxed(lean_object* v_M_177_, lean_object* v_inst_178_, lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Associates_mk(v_M_177_, v_inst_178_, v_a_179_);
lean_dec(v_a_179_);
lean_dec_ref(v_inst_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___redArg(lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v_toOne_184_; 
v___x_182_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_181_);
v___x_183_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_182_);
v_toOne_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc(v_toOne_184_);
lean_dec_ref(v___x_183_);
return v_toOne_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___redArg___boxed(lean_object* v_inst_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Associates_instInhabited___redArg(v_inst_185_);
lean_dec_ref(v_inst_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited(lean_object* v_M_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Associates_instInhabited___redArg(v_inst_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instInhabited___boxed(lean_object* v_M_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Associates_instInhabited(v_M_190_, v_inst_191_);
lean_dec_ref(v_inst_191_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___redArg(lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v_toOne_196_; 
v___x_194_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_193_);
v___x_195_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_194_);
v_toOne_196_ = lean_ctor_get(v___x_195_, 0);
lean_inc(v_toOne_196_);
lean_dec_ref(v___x_195_);
return v_toOne_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___redArg___boxed(lean_object* v_inst_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_Associates_instOne___redArg(v_inst_197_);
lean_dec_ref(v_inst_197_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne(lean_object* v_M_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_Associates_instOne___redArg(v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOne___boxed(lean_object* v_M_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Associates_instOne(v_M_202_, v_inst_203_);
lean_dec_ref(v_inst_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___redArg(lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v_toOne_208_; 
v___x_206_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_205_);
v___x_207_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_206_);
v_toOne_208_ = lean_ctor_get(v___x_207_, 0);
lean_inc(v_toOne_208_);
lean_dec_ref(v___x_207_);
return v_toOne_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___redArg___boxed(lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_Associates_instBot___redArg(v_inst_209_);
lean_dec_ref(v_inst_209_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot(lean_object* v_M_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_Associates_instBot___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBot___boxed(lean_object* v_M_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Associates_instBot(v_M_214_, v_inst_215_);
lean_dec_ref(v_inst_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___redArg(lean_object* v_inst_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v_toOne_220_; 
v___x_218_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_217_);
v___x_219_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_218_);
v_toOne_220_ = lean_ctor_get(v___x_219_, 0);
lean_inc(v_toOne_220_);
lean_dec_ref(v___x_219_);
return v_toOne_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___redArg___boxed(lean_object* v_inst_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_Associates_instUniqueOfSubsingleton___redArg(v_inst_221_);
lean_dec_ref(v_inst_221_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton(lean_object* v_M_223_, lean_object* v_inst_224_, lean_object* v_inst_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_Associates_instUniqueOfSubsingleton___redArg(v_inst_224_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instUniqueOfSubsingleton___boxed(lean_object* v_M_227_, lean_object* v_inst_228_, lean_object* v_inst_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib_Associates_instUniqueOfSubsingleton(v_M_227_, v_inst_228_, v_inst_229_);
lean_dec_ref(v_inst_228_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg___lam__0(lean_object* v_toMul_231_, lean_object* v_x1_232_, lean_object* v_x2_233_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lean_apply_2(v_toMul_231_, v_x1_232_, v_x2_233_);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg(lean_object* v_inst_235_){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v_toMul_239_; lean_object* v___f_240_; lean_object* v___x_241_; 
v___x_236_ = lean_box(0);
v___x_237_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_235_);
v___x_238_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_237_);
v_toMul_239_ = lean_ctor_get(v___x_238_, 1);
lean_inc(v_toMul_239_);
lean_dec_ref(v___x_238_);
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_Associates_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_240_, 0, v_toMul_239_);
v___x_241_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_u2082), 10, 8);
lean_closure_set(v___x_241_, 0, lean_box(0));
lean_closure_set(v___x_241_, 1, lean_box(0));
lean_closure_set(v___x_241_, 2, v___x_236_);
lean_closure_set(v___x_241_, 3, v___x_236_);
lean_closure_set(v___x_241_, 4, lean_box(0));
lean_closure_set(v___x_241_, 5, v___x_236_);
lean_closure_set(v___x_241_, 6, v___f_240_);
lean_closure_set(v___x_241_, 7, lean_box(0));
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___redArg___boxed(lean_object* v_inst_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Associates_instMul___redArg(v_inst_242_);
lean_dec_ref(v_inst_242_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul(lean_object* v_M_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_mathlib_Associates_instMul___redArg(v_inst_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instMul___boxed(lean_object* v_M_247_, lean_object* v_inst_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Associates_instMul(v_M_247_, v_inst_248_);
lean_dec_ref(v_inst_248_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___redArg(lean_object* v_inst_250_){
_start:
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
v___x_251_ = lp_mathlib_Associates_instOne___redArg(v_inst_250_);
v___x_252_ = lp_mathlib_Associates_instMul___redArg(v_inst_250_);
lean_inc(v___x_251_);
lean_inc(v___x_252_);
v___x_253_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_253_, 0, lean_box(0));
lean_closure_set(v___x_253_, 1, v___x_252_);
lean_closure_set(v___x_253_, 2, v___x_251_);
v___x_254_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_254_, 0, v___x_251_);
lean_ctor_set(v___x_254_, 1, v___x_252_);
lean_ctor_set(v___x_254_, 2, v___x_253_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___redArg___boxed(lean_object* v_inst_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Associates_instCommMonoid___redArg(v_inst_255_);
lean_dec_ref(v_inst_255_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid(lean_object* v_M_257_, lean_object* v_inst_258_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Associates_instCommMonoid___redArg(v_inst_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoid___boxed(lean_object* v_M_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_Associates_instCommMonoid(v_M_260_, v_inst_261_);
lean_dec_ref(v_inst_261_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPreorder(lean_object* v_M_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = ((lean_object*)(lp_mathlib_Associates_instPreorder___closed__0));
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPreorder___boxed(lean_object* v_M_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_Associates_instPreorder(v_M_269_, v_inst_270_);
lean_dec_ref(v_inst_270_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mkMonoidHom___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lean_alloc_closure((void*)(lp_mathlib_Associates_mk___boxed), 3, 2);
lean_closure_set(v___x_273_, 0, lean_box(0));
lean_closure_set(v___x_273_, 1, v_inst_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_mkMonoidHom(lean_object* v_M_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lean_alloc_closure((void*)(lp_mathlib_Associates_mk___boxed), 3, 2);
lean_closure_set(v___x_276_, 0, lean_box(0));
lean_closure_set(v___x_276_, 1, v_inst_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = lp_mathlib_Associates_instCommMonoid___redArg(v_inst_277_);
v___x_279_ = lp_mathlib_Units_instInhabited___redArg(v___x_278_);
lean_dec_ref(v___x_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___redArg___boxed(lean_object* v_inst_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_Associates_uniqueUnits___redArg(v_inst_280_);
lean_dec_ref(v_inst_280_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits(lean_object* v_M_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_Associates_uniqueUnits___redArg(v_inst_283_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_uniqueUnits___boxed(lean_object* v_M_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_Associates_uniqueUnits(v_M_285_, v_inst_286_);
lean_dec_ref(v_inst_286_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___redArg(lean_object* v_inst_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_mathlib_Associates_instBot___redArg(v_inst_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___redArg___boxed(lean_object* v_inst_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Associates_instOrderBot___redArg(v_inst_290_);
lean_dec_ref(v_inst_290_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot(lean_object* v_M_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_mathlib_Associates_instBot___redArg(v_inst_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderBot___boxed(lean_object* v_M_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_Associates_instOrderBot(v_M_295_, v_inst_296_);
lean_dec_ref(v_inst_296_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___redArg(lean_object* v_inst_298_){
_start:
{
lean_inc(v_inst_298_);
return v_inst_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___redArg___boxed(lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_Associates_instZero___redArg(v_inst_299_);
lean_dec(v_inst_299_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero(lean_object* v_M_301_, lean_object* v_inst_302_, lean_object* v_inst_303_){
_start:
{
lean_inc(v_inst_302_);
return v_inst_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instZero___boxed(lean_object* v_M_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_Associates_instZero(v_M_304_, v_inst_305_, v_inst_306_);
lean_dec_ref(v_inst_306_);
lean_dec(v_inst_305_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___redArg(lean_object* v_inst_308_){
_start:
{
lean_inc(v_inst_308_);
return v_inst_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___redArg___boxed(lean_object* v_inst_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_Associates_instTopOfZero___redArg(v_inst_309_);
lean_dec(v_inst_309_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero(lean_object* v_M_311_, lean_object* v_inst_312_, lean_object* v_inst_313_){
_start:
{
lean_inc(v_inst_312_);
return v_inst_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instTopOfZero___boxed(lean_object* v_M_314_, lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Associates_instTopOfZero(v_M_314_, v_inst_315_, v_inst_316_);
lean_dec_ref(v_inst_316_);
lean_dec(v_inst_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoidWithZero___redArg(lean_object* v_inst_318_){
_start:
{
lean_object* v_toCommMonoid_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v_toZero_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_331_; 
v_toCommMonoid_319_ = lean_ctor_get(v_inst_318_, 0);
v___x_320_ = lp_mathlib_Associates_instCommMonoid___redArg(v_toCommMonoid_319_);
v___x_321_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_318_);
v___x_322_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_321_);
v___x_323_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_322_);
v_toZero_324_ = lean_ctor_get(v___x_323_, 1);
v_isSharedCheck_331_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_331_ == 0)
{
lean_object* v_unused_332_; 
v_unused_332_ = lean_ctor_get(v___x_323_, 0);
lean_dec(v_unused_332_);
v___x_326_ = v___x_323_;
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_toZero_324_);
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_329_; 
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_320_);
v___x_329_ = v___x_326_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_320_);
lean_ctor_set(v_reuseFailAlloc_330_, 1, v_toZero_324_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instCommMonoidWithZero(lean_object* v_M_333_, lean_object* v_inst_334_){
_start:
{
lean_object* v___x_335_; 
v___x_335_ = lp_mathlib_Associates_instCommMonoidWithZero___redArg(v_inst_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderTop___redArg(lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_toZero_340_; 
v___x_337_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_336_);
v___x_338_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v___x_337_);
v___x_339_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v___x_338_);
v_toZero_340_ = lean_ctor_get(v___x_339_, 1);
lean_inc(v_toZero_340_);
lean_dec_ref(v___x_339_);
return v_toZero_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instOrderTop(lean_object* v_M_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_Associates_instOrderTop___redArg(v_inst_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBoundedOrder___redArg(lean_object* v_inst_344_){
_start:
{
lean_object* v_toCommMonoid_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v_toCommMonoid_345_ = lean_ctor_get(v_inst_344_, 0);
lean_inc_ref(v_toCommMonoid_345_);
v___x_346_ = lp_mathlib_Associates_instOrderTop___redArg(v_inst_344_);
v___x_347_ = lp_mathlib_Associates_instBot___redArg(v_toCommMonoid_345_);
lean_dec_ref(v_toCommMonoid_345_);
v___x_348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instBoundedOrder(lean_object* v_M_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_mathlib_Associates_instBoundedOrder___redArg(v_inst_350_);
return v___x_351_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Associates_instDecidableRelDvd___redArg(lean_object* v_inst_352_, lean_object* v_a_353_, lean_object* v_b_354_){
_start:
{
lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_355_ = lean_apply_2(v_inst_352_, v_a_353_, v_b_354_);
v___x_356_ = lean_unbox(v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instDecidableRelDvd___redArg___boxed(lean_object* v_inst_357_, lean_object* v_a_358_, lean_object* v_b_359_){
_start:
{
uint8_t v_res_360_; lean_object* v_r_361_; 
v_res_360_ = lp_mathlib_Associates_instDecidableRelDvd___redArg(v_inst_357_, v_a_358_, v_b_359_);
v_r_361_ = lean_box(v_res_360_);
return v_r_361_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Associates_instDecidableRelDvd(lean_object* v_M_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_a_365_, lean_object* v_b_366_){
_start:
{
lean_object* v___x_367_; uint8_t v___x_368_; 
v___x_367_ = lean_apply_2(v_inst_364_, v_a_365_, v_b_366_);
v___x_368_ = lean_unbox(v___x_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instDecidableRelDvd___boxed(lean_object* v_M_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_a_372_, lean_object* v_b_373_){
_start:
{
uint8_t v_res_374_; lean_object* v_r_375_; 
v_res_374_ = lp_mathlib_Associates_instDecidableRelDvd(v_M_369_, v_inst_370_, v_inst_371_, v_a_372_, v_b_373_);
lean_dec_ref(v_inst_370_);
v_r_375_ = lean_box(v_res_374_);
return v_r_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___redArg(lean_object* v_inst_376_){
_start:
{
lean_object* v_toCommMonoid_377_; lean_object* v___x_378_; 
v_toCommMonoid_377_ = lean_ctor_get(v_inst_376_, 0);
v___x_378_ = lp_mathlib_Associates_instPreorder(lean_box(0), v_toCommMonoid_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___redArg___boxed(lean_object* v_inst_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_mathlib_Associates_instPartialOrder___redArg(v_inst_379_);
lean_dec_ref(v_inst_379_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder(lean_object* v_M_381_, lean_object* v_inst_382_, lean_object* v_inst_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_Associates_instPartialOrder___redArg(v_inst_382_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_instPartialOrder___boxed(lean_object* v_M_385_, lean_object* v_inst_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_Associates_instPartialOrder(v_M_385_, v_inst_386_, v_inst_387_);
lean_dec_ref(v_inst_386_);
return v_res_388_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Prime_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Prime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Prime_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Prime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Associated(builtin);
}
#ifdef __cplusplus
}
#endif
