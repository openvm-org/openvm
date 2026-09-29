// Lean compiler output
// Module: Mathlib.Order.Zorn
// Imports: public import Init public meta import Init public import Mathlib.Order.CompleteLattice.Chain public import Mathlib.Order.Minimal
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(11, 240, 1, 62, 92, 163, 173, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zorn"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(20, 6, 209, 203, 36, 210, 247, 103)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(21, 207, 216, 13, 225, 41, 47, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(45, 11, 253, 210, 167, 228, 85, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__16_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__10_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(100, 2, 144, 119, 127, 225, 14, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(85, 59, 115, 197, 18, 0, 59, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 117, 91, 242, 111, 51, 163, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(27, 171, 207, 248, 127, 182, 7, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__5));
v___x_56_ = l_String_toRawSubstring_x27(v___x_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = lean_unsigned_to_nat(4124226829u);
v___x_76_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__13));
v___x_77_ = l_Lean_Name_num___override(v___x_76_, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__15));
v___x_80_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__14);
v___x_81_ = l_Lean_Name_str___override(v___x_80_, v___x_79_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__17));
v___x_84_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__16);
v___x_85_ = l_Lean_Name_str___override(v___x_84_, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_86_ = lean_unsigned_to_nat(12u);
v___x_87_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__18);
v___x_88_ = l_Lean_Name_num___override(v___x_87_, v___x_86_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_box(0);
v___x_90_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__19);
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_89_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_box(0);
v___x_93_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__20);
v___x_94_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v___x_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1(lean_object* v_x_98_, lean_object* v_a_99_, lean_object* v_a_100_){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_101_ = lean_unsigned_to_nat(0u);
v___x_102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0__term___u227a___00__closed__10));
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
v___x_114_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__4));
v___x_115_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__6);
v___x_116_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__7));
lean_inc(v_currMacroScope_107_);
lean_inc(v_quotContext_106_);
v___x_117_ = l_Lean_addMacroScope(v_quotContext_106_, v___x_116_, v_currMacroScope_107_);
v___x_118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21, &lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21_once, _init_lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__21);
lean_inc_n(v___x_113_, 2);
v___x_119_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_119_, 0, v___x_113_);
lean_ctor_set(v___x_119_, 1, v___x_115_);
lean_ctor_set(v___x_119_, 2, v___x_117_);
lean_ctor_set(v___x_119_, 3, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___closed__23));
v___x_121_ = l_Lean_Syntax_node2(v___x_113_, v___x_120_, v___x_109_, v___x_111_);
v___x_122_ = l_Lean_Syntax_node2(v___x_113_, v___x_114_, v___x_119_, v___x_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_100_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1___boxed(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib___private_Mathlib_Order_Zorn_0____aux__Mathlib__Order__Zorn______macroRules____private__Mathlib__Order__Zorn__0__term___u227a____1(v_x_124_, v_a_125_, v_a_126_);
lean_dec_ref(v_a_125_);
return v_res_127_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Chain(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Minimal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Zorn(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_CompleteLattice_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Minimal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Zorn(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Chain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Minimal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Zorn(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Minimal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Zorn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Zorn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Zorn(builtin);
}
#ifdef __cplusplus
}
#endif
