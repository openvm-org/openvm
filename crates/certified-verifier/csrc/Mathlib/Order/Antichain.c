// Lean compiler output
// Module: Mathlib.Order.Antichain
// Imports: public import Init public meta import Init public import Mathlib.Order.Bounds.Basic public import Mathlib.Order.Preorder.Chain
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(11, 240, 1, 62, 92, 163, 173, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Antichain"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(220, 147, 21, 54, 106, 72, 118, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(125, 157, 64, 127, 223, 252, 50, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 171, 244, 181, 254, 124, 107, 104)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__16_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__18_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "StrongLT"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(129, 66, 62, 1, 107, 224, 114, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__10_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__5));
v___x_56_ = l_String_toRawSubstring_x27(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1(lean_object* v_x_73_, lean_object* v_a_74_, lean_object* v_a_75_){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; uint8_t v___x_78_; 
v___x_76_ = lean_unsigned_to_nat(0u);
v___x_77_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10));
lean_inc(v_x_73_);
v___x_78_ = l_Lean_Syntax_isOfKind(v_x_73_, v___x_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; lean_object* v___x_80_; 
lean_dec(v_x_73_);
v___x_79_ = lean_box(1);
v___x_80_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v_a_75_);
return v___x_80_;
}
else
{
lean_object* v_quotContext_81_; lean_object* v_currMacroScope_82_; lean_object* v_ref_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; uint8_t v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v_quotContext_81_ = lean_ctor_get(v_a_74_, 1);
v_currMacroScope_82_ = lean_ctor_get(v_a_74_, 2);
v_ref_83_ = lean_ctor_get(v_a_74_, 5);
v___x_84_ = l_Lean_Syntax_getArg(v_x_73_, v___x_76_);
v___x_85_ = lean_unsigned_to_nat(2u);
v___x_86_ = l_Lean_Syntax_getArg(v_x_73_, v___x_85_);
lean_dec(v_x_73_);
v___x_87_ = 0;
v___x_88_ = l_Lean_SourceInfo_fromRef(v_ref_83_, v___x_87_);
v___x_89_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4));
v___x_90_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6, &lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__6);
v___x_91_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__7));
lean_inc(v_currMacroScope_82_);
lean_inc(v_quotContext_81_);
v___x_92_ = l_Lean_addMacroScope(v_quotContext_81_, v___x_91_, v_currMacroScope_82_);
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__11));
lean_inc_n(v___x_88_, 2);
v___x_94_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_94_, 0, v___x_88_);
lean_ctor_set(v___x_94_, 1, v___x_90_);
lean_ctor_set(v___x_94_, 2, v___x_92_);
lean_ctor_set(v___x_94_, 3, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__13));
v___x_96_ = l_Lean_Syntax_node2(v___x_88_, v___x_95_, v___x_84_, v___x_86_);
v___x_97_ = l_Lean_Syntax_node2(v___x_88_, v___x_89_, v___x_94_, v___x_96_);
v___x_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_75_);
return v___x_98_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___boxed(lean_object* v_x_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1(v_x_99_, v_a_100_, v_a_101_);
lean_dec_ref(v_a_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1(lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______macroRules____private__Mathlib__Order__Antichain__0__term___u227a____1___closed__4));
lean_inc(v_x_106_);
v___x_110_ = l_Lean_Syntax_isOfKind(v_x_106_, v___x_109_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; lean_object* v___x_112_; 
lean_dec(v_x_106_);
v___x_111_ = lean_box(0);
v___x_112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_108_);
return v___x_112_;
}
else
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_113_ = lean_unsigned_to_nat(0u);
v___x_114_ = l_Lean_Syntax_getArg(v_x_106_, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___closed__1));
lean_inc(v___x_114_);
v___x_116_ = l_Lean_Syntax_isOfKind(v___x_114_, v___x_115_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; lean_object* v___x_118_; 
lean_dec(v___x_114_);
lean_dec(v_x_106_);
v___x_117_ = lean_box(0);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v_a_108_);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_119_ = lean_unsigned_to_nat(1u);
v___x_120_ = l_Lean_Syntax_getArg(v_x_106_, v___x_119_);
lean_dec(v_x_106_);
v___x_121_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_120_);
v___x_122_ = l_Lean_Syntax_matchesNull(v___x_120_, v___x_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v___x_124_; 
lean_dec(v___x_120_);
lean_dec(v___x_114_);
v___x_123_ = lean_box(0);
v___x_124_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v_a_108_);
return v___x_124_;
}
else
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v_ref_127_; uint8_t v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_125_ = l_Lean_Syntax_getArg(v___x_120_, v___x_113_);
v___x_126_ = l_Lean_Syntax_getArg(v___x_120_, v___x_119_);
lean_dec(v___x_120_);
v_ref_127_ = l_Lean_replaceRef(v___x_114_, v_a_107_);
lean_dec(v___x_114_);
v___x_128_ = 0;
v___x_129_ = l_Lean_SourceInfo_fromRef(v_ref_127_, v___x_128_);
lean_dec(v_ref_127_);
v___x_130_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__10));
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Antichain_0__term___u227a___00__closed__13));
lean_inc(v___x_129_);
v___x_132_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_129_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = l_Lean_Syntax_node3(v___x_129_, v___x_130_, v___x_125_, v___x_132_, v___x_126_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_108_);
return v___x_134_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___private_Mathlib_Order_Antichain_0____aux__Mathlib__Order__Antichain______unexpand__StrongLT__1(v_x_135_, v_a_136_, v_a_137_);
lean_dec(v_a_136_);
return v_res_138_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Antichain(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Antichain(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Antichain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Antichain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Antichain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Antichain(builtin);
}
#ifdef __cplusplus
}
#endif
