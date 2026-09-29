// Lean compiler output
// Module: Mathlib.Order.Directed
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Image public import Mathlib.Util.Delaborators
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(11, 240, 1, 62, 92, 163, 173, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Directed"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(157, 195, 61, 233, 245, 218, 211, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(168, 190, 61, 62, 246, 213, 71, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≼_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(77, 159, 187, 187, 193, 247, 60, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≼ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__16_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__10_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(100, 2, 144, 119, 127, 225, 14, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(85, 59, 115, 197, 18, 0, 59, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 117, 91, 242, 111, 51, 163, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(234, 196, 118, 247, 86, 47, 74, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__13_value),((lean_object*)(((size_t)(164667150) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(60, 82, 181, 206, 119, 16, 178, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(147, 25, 62, 115, 172, 22, 238, 152)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(211, 10, 104, 78, 234, 185, 162, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__18_value),((lean_object*)(((size_t)(26) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(206, 209, 206, 198, 125, 111, 23, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "DirectedOn"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 25, 46, 230, 15, 218, 93, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_≼₁_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 55, 129, 11, 165, 228, 235, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ≼₁ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__3_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2081____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2081____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_≼₂_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(114, 146, 15, 161, 108, 12, 213, 220)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ≼₂ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "r₂"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 155, 120, 71, 1, 67, 223, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 155, 120, 71, 1, 67, 223, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(46, 175, 126, 77, 10, 121, 212, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(231, 86, 202, 198, 91, 33, 252, 238)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(122, 5, 170, 119, 242, 250, 26, 59)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(120, 194, 68, 186, 153, 138, 41, 166)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__7_value),((lean_object*)(((size_t)(1790937067) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(203, 129, 51, 215, 89, 70, 166, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(48, 76, 179, 202, 216, 66, 122, 52)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(52, 143, 133, 87, 207, 21, 149, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__10_value),((lean_object*)(((size_t)(27) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(177, 163, 142, 119, 208, 131, 158, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__5));
v___x_56_ = l_String_toRawSubstring_x27(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1(lean_object* v_x_98_, lean_object* v_a_99_, lean_object* v_a_100_){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_101_ = lean_unsigned_to_nat(0u);
v___x_102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__term___u227c___00__closed__10));
lean_inc(v_x_98_);
v___x_103_ = l_Lean_Syntax_isOfKind(v_x_98_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v_x_98_);
v___x_104_ = lean_box(1);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_100_);
return v___x_105_;
}
else
{
lean_object* v_quotContext_106_; lean_object* v_currMacroScope_107_; lean_object* v_ref_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_quotContext_106_ = lean_ctor_get(v_a_99_, 1);
v_currMacroScope_107_ = lean_ctor_get(v_a_99_, 2);
v_ref_108_ = lean_ctor_get(v_a_99_, 5);
v___x_109_ = l_Lean_Syntax_getArg(v_x_98_, v___x_101_);
v___x_110_ = lean_unsigned_to_nat(2u);
v___x_111_ = l_Lean_Syntax_getArg(v_x_98_, v___x_110_);
lean_dec(v_x_98_);
v___x_112_ = 0;
v___x_113_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_112_);
v___x_114_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4));
v___x_115_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6, &lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6);
v___x_116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__7));
lean_inc(v_currMacroScope_107_);
lean_inc(v_quotContext_106_);
v___x_117_ = l_Lean_addMacroScope(v_quotContext_106_, v___x_116_, v_currMacroScope_107_);
v___x_118_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__21));
lean_inc_n(v___x_113_, 2);
v___x_119_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_119_, 0, v___x_113_);
lean_ctor_set(v___x_119_, 1, v___x_115_);
lean_ctor_set(v___x_119_, 2, v___x_117_);
lean_ctor_set(v___x_119_, 3, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23));
v___x_121_ = l_Lean_Syntax_node2(v___x_113_, v___x_120_, v___x_109_, v___x_111_);
v___x_122_ = l_Lean_Syntax_node2(v___x_113_, v___x_114_, v___x_119_, v___x_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_100_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___boxed(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1(v_x_124_, v_a_125_, v_a_126_);
lean_dec_ref(v_a_125_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2081____1(lean_object* v_x_148_, lean_object* v_a_149_, lean_object* v_a_150_){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; uint8_t v___x_153_; 
v___x_151_ = lean_unsigned_to_nat(0u);
v___x_152_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2081___00__closed__3));
lean_inc(v_x_148_);
v___x_153_ = l_Lean_Syntax_isOfKind(v_x_148_, v___x_152_);
if (v___x_153_ == 0)
{
lean_object* v___x_154_; lean_object* v___x_155_; 
lean_dec(v_x_148_);
v___x_154_ = lean_box(1);
v___x_155_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
lean_ctor_set(v___x_155_, 1, v_a_150_);
return v___x_155_;
}
else
{
lean_object* v_quotContext_156_; lean_object* v_currMacroScope_157_; lean_object* v_ref_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; uint8_t v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v_quotContext_156_ = lean_ctor_get(v_a_149_, 1);
v_currMacroScope_157_ = lean_ctor_get(v_a_149_, 2);
v_ref_158_ = lean_ctor_get(v_a_149_, 5);
v___x_159_ = l_Lean_Syntax_getArg(v_x_148_, v___x_151_);
v___x_160_ = lean_unsigned_to_nat(2u);
v___x_161_ = l_Lean_Syntax_getArg(v_x_148_, v___x_160_);
lean_dec(v_x_148_);
v___x_162_ = 0;
v___x_163_ = l_Lean_SourceInfo_fromRef(v_ref_158_, v___x_162_);
v___x_164_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4));
v___x_165_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6, &lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__6);
v___x_166_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__7));
lean_inc(v_currMacroScope_157_);
lean_inc(v_quotContext_156_);
v___x_167_ = l_Lean_addMacroScope(v_quotContext_156_, v___x_166_, v_currMacroScope_157_);
v___x_168_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__21));
lean_inc_n(v___x_163_, 2);
v___x_169_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_169_, 0, v___x_163_);
lean_ctor_set(v___x_169_, 1, v___x_165_);
lean_ctor_set(v___x_169_, 2, v___x_167_);
lean_ctor_set(v___x_169_, 3, v___x_168_);
v___x_170_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23));
v___x_171_ = l_Lean_Syntax_node2(v___x_163_, v___x_170_, v___x_159_, v___x_161_);
v___x_172_ = l_Lean_Syntax_node2(v___x_163_, v___x_164_, v___x_169_, v___x_171_);
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v_a_150_);
return v___x_173_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2081____1___boxed(lean_object* v_x_174_, lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2081____1(v_x_174_, v_a_175_, v_a_176_);
lean_dec_ref(v_a_175_);
return v_res_177_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1(void){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_195_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__0));
v___x_196_ = l_String_toRawSubstring_x27(v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; uint8_t v___x_237_; 
v___x_235_ = lean_unsigned_to_nat(0u);
v___x_236_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn_term___u227c_u2082___00__closed__1));
lean_inc(v_x_232_);
v___x_237_ = l_Lean_Syntax_isOfKind(v_x_232_, v___x_236_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; lean_object* v___x_239_; 
lean_dec(v_x_232_);
v___x_238_ = lean_box(1);
v___x_239_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
lean_ctor_set(v___x_239_, 1, v_a_234_);
return v___x_239_;
}
else
{
lean_object* v_quotContext_240_; lean_object* v_currMacroScope_241_; lean_object* v_ref_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v_quotContext_240_ = lean_ctor_get(v_a_233_, 1);
v_currMacroScope_241_ = lean_ctor_get(v_a_233_, 2);
v_ref_242_ = lean_ctor_get(v_a_233_, 5);
v___x_243_ = l_Lean_Syntax_getArg(v_x_232_, v___x_235_);
v___x_244_ = lean_unsigned_to_nat(2u);
v___x_245_ = l_Lean_Syntax_getArg(v_x_232_, v___x_244_);
lean_dec(v_x_232_);
v___x_246_ = 0;
v___x_247_ = l_Lean_SourceInfo_fromRef(v_ref_242_, v___x_246_);
v___x_248_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__4));
v___x_249_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1, &lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__1);
v___x_250_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__2));
lean_inc(v_currMacroScope_241_);
lean_inc(v_quotContext_240_);
v___x_251_ = l_Lean_addMacroScope(v_quotContext_240_, v___x_250_, v_currMacroScope_241_);
v___x_252_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___closed__13));
lean_inc_n(v___x_247_, 2);
v___x_253_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_253_, 0, v___x_247_);
lean_ctor_set(v___x_253_, 1, v___x_249_);
lean_ctor_set(v___x_253_, 2, v___x_251_);
lean_ctor_set(v___x_253_, 3, v___x_252_);
v___x_254_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Directed_0____aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__term___u227c____1___closed__23));
v___x_255_ = l_Lean_Syntax_node2(v___x_247_, v___x_254_, v___x_243_, v___x_245_);
v___x_256_ = l_Lean_Syntax_node2(v___x_247_, v___x_248_, v___x_253_, v___x_255_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v_a_234_);
return v___x_257_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1___boxed(lean_object* v_x_258_, lean_object* v_a_259_, lean_object* v_a_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib___private_Mathlib_Order_Directed_0__DirectedOn___aux__Mathlib__Order__Directed______macroRules____private__Mathlib__Order__Directed__0__DirectedOn__term___u227c_u2082____1(v_x_258_, v_a_259_, v_a_260_);
lean_dec_ref(v_a_259_);
return v_res_261_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Directed(builtin);
}
#ifdef __cplusplus
}
#endif
