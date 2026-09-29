// Lean compiler output
// Module: Mathlib.Order.Preorder.Chain
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Pairwise public import Mathlib.Data.Set.Notation public import Mathlib.Data.Set.Pairwise.Basic public import Mathlib.Data.SetLike.Basic public import Mathlib.Order.Directed public import Mathlib.Order.Hom.Set
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Subtype_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(11, 240, 1, 62, 92, 163, 173, 149)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(59, 206, 221, 214, 96, 24, 176, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Chain"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(138, 38, 121, 255, 131, 52, 168, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(11, 196, 18, 152, 203, 85, 50, 64)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≺_"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(179, 219, 174, 33, 177, 24, 222, 56)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≺ "};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__18_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__19_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__12_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__20_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__21_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a__ = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(100, 2, 144, 119, 127, 225, 14, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(85, 59, 115, 197, 18, 0, 59, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 117, 91, 242, 111, 51, 163, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(116, 184, 65, 206, 247, 115, 241, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(177, 16, 254, 137, 150, 147, 86, 125)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__14_value),((lean_object*)(((size_t)(1671797962) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(66, 137, 71, 42, 109, 139, 73, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__15_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(237, 164, 120, 211, 2, 197, 217, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__17_value),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(173, 85, 193, 8, 50, 4, 55, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__19_value),((lean_object*)(((size_t)(11) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(109, 154, 59, 74, 114, 225, 46, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__22_value;
static const lean_string_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsChain_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_IsChain_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instSetLike(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Flag_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Flag_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Flag_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_ofIsMaxChain(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_map___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Flag_map___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Flag_map___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Flag_map___closed__0 = (const lean_object*)&lp_mathlib_Flag_map___closed__0_value;
static const lean_ctor_object lp_mathlib_Flag_map___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Flag_map___closed__0_value),((lean_object*)&lp_mathlib_Flag_map___closed__0_value)}};
static const lean_object* lp_mathlib_Flag_map___closed__1 = (const lean_object*)&lp_mathlib_Flag_map___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Flag_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Flag_instUnique___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__5));
v___x_60_ = l_String_toRawSubstring_x27(v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1(lean_object* v_x_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_108_ = lean_unsigned_to_nat(0u);
v___x_109_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0__term___u227a___00__closed__12));
lean_inc(v_x_105_);
v___x_110_ = l_Lean_Syntax_isOfKind(v_x_105_, v___x_109_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; lean_object* v___x_112_; 
lean_dec(v_x_105_);
v___x_111_ = lean_box(1);
v___x_112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_107_);
return v___x_112_;
}
else
{
lean_object* v_quotContext_113_; lean_object* v_currMacroScope_114_; lean_object* v_ref_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_quotContext_113_ = lean_ctor_get(v_a_106_, 1);
v_currMacroScope_114_ = lean_ctor_get(v_a_106_, 2);
v_ref_115_ = lean_ctor_get(v_a_106_, 5);
v___x_116_ = l_Lean_Syntax_getArg(v_x_105_, v___x_108_);
v___x_117_ = lean_unsigned_to_nat(2u);
v___x_118_ = l_Lean_Syntax_getArg(v_x_105_, v___x_117_);
lean_dec(v_x_105_);
v___x_119_ = 0;
v___x_120_ = l_Lean_SourceInfo_fromRef(v_ref_115_, v___x_119_);
v___x_121_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__4));
v___x_122_ = lean_obj_once(&lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6, &lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__6);
v___x_123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__7));
lean_inc(v_currMacroScope_114_);
lean_inc(v_quotContext_113_);
v___x_124_ = l_Lean_addMacroScope(v_quotContext_113_, v___x_123_, v_currMacroScope_114_);
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__22));
lean_inc_n(v___x_120_, 2);
v___x_126_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_126_, 0, v___x_120_);
lean_ctor_set(v___x_126_, 1, v___x_122_);
lean_ctor_set(v___x_126_, 2, v___x_124_);
lean_ctor_set(v___x_126_, 3, v___x_125_);
v___x_127_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___closed__24));
v___x_128_ = l_Lean_Syntax_node2(v___x_120_, v___x_127_, v___x_116_, v___x_118_);
v___x_129_ = l_Lean_Syntax_node2(v___x_120_, v___x_121_, v___x_126_, v___x_128_);
v___x_130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v_a_107_);
return v___x_130_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1___boxed(lean_object* v_x_131_, lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib___private_Mathlib_Order_Preorder_Chain_0____aux__Mathlib__Order__Preorder__Chain______macroRules____private__Mathlib__Order__Preorder__Chain__0__term___u227a____1(v_x_131_, v_a_132_, v_a_133_);
lean_dec_ref(v_a_132_);
return v_res_134_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsChain_linearOrder___redArg___lam__0(lean_object* v_inst_135_, lean_object* v_x_136_, lean_object* v_y_137_){
_start:
{
lean_object* v___x_138_; uint8_t v___x_139_; 
v___x_138_ = lean_apply_2(v_inst_135_, v_x_136_, v_y_137_);
v___x_139_ = lean_unbox(v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_140_, lean_object* v_x_141_, lean_object* v_y_142_){
_start:
{
uint8_t v_res_143_; lean_object* v_r_144_; 
v_res_143_ = lp_mathlib_IsChain_linearOrder___redArg___lam__0(v_inst_140_, v_x_141_, v_y_142_);
v_r_144_ = lean_box(v_res_143_);
return v_r_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__2(lean_object* v_inst_145_, lean_object* v_a_146_, lean_object* v_b_147_){
_start:
{
lean_object* v___x_148_; uint8_t v___x_149_; 
lean_inc(v_b_147_);
lean_inc(v_a_146_);
v___x_148_ = lean_apply_2(v_inst_145_, v_a_146_, v_b_147_);
v___x_149_ = lean_unbox(v___x_148_);
if (v___x_149_ == 0)
{
lean_dec(v_b_147_);
return v_a_146_;
}
else
{
lean_dec(v_a_146_);
return v_b_147_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__1(lean_object* v_inst_150_, lean_object* v_a_151_, lean_object* v_b_152_){
_start:
{
lean_object* v___x_153_; uint8_t v___x_154_; 
lean_inc(v_b_152_);
lean_inc(v_a_151_);
v___x_153_ = lean_apply_2(v_inst_150_, v_a_151_, v_b_152_);
v___x_154_ = lean_unbox(v___x_153_);
if (v___x_154_ == 0)
{
lean_dec(v_a_151_);
return v_b_152_;
}
else
{
lean_dec(v_b_152_);
return v_a_151_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_IsChain_linearOrder___redArg___lam__3(lean_object* v___f_155_, lean_object* v_a_156_, lean_object* v_b_157_){
_start:
{
uint8_t v___x_158_; 
lean_inc(v_b_157_);
lean_inc(v_a_156_);
lean_inc_ref(v___f_155_);
v___x_158_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_155_, v_a_156_, v_b_157_);
if (v___x_158_ == 0)
{
uint8_t v___x_159_; 
v___x_159_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_155_, v_a_156_, v_b_157_);
if (v___x_159_ == 0)
{
uint8_t v___x_160_; 
v___x_160_ = 2;
return v___x_160_;
}
else
{
uint8_t v___x_161_; 
v___x_161_ = 1;
return v___x_161_;
}
}
else
{
uint8_t v___x_162_; 
lean_dec(v_b_157_);
lean_dec(v_a_156_);
lean_dec_ref(v___f_155_);
v___x_162_ = 0;
return v___x_162_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___lam__3___boxed(lean_object* v___f_163_, lean_object* v_a_164_, lean_object* v_b_165_){
_start:
{
uint8_t v_res_166_; lean_object* v_r_167_; 
v_res_166_ = lp_mathlib_IsChain_linearOrder___redArg___lam__3(v___f_163_, v_a_164_, v_b_165_);
v_r_167_ = lean_box(v_res_166_);
return v_r_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg(lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___f_170_; lean_object* v___f_171_; lean_object* v___f_172_; lean_object* v___f_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
lean_inc_ref_n(v_inst_169_, 2);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_170_, 0, v_inst_169_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(v___f_171_, 0, v_inst_169_);
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(v___f_172_, 0, v_inst_169_);
lean_inc_ref_n(v___f_170_, 3);
v___f_173_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_173_, 0, v___f_170_);
v___x_174_ = lp_mathlib_Subtype_preorder(lean_box(0), v_inst_168_, lean_box(0));
lean_inc_ref_n(v___x_174_, 2);
v___x_175_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_175_, 0, lean_box(0));
lean_closure_set(v___x_175_, 1, v___x_174_);
lean_closure_set(v___x_175_, 2, v___f_170_);
v___x_176_ = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_176_, 0, lean_box(0));
lean_closure_set(v___x_176_, 1, v___x_174_);
lean_closure_set(v___x_176_, 2, v___f_170_);
v___x_177_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_177_, 0, v___x_174_);
lean_ctor_set(v___x_177_, 1, v___f_172_);
lean_ctor_set(v___x_177_, 2, v___f_171_);
lean_ctor_set(v___x_177_, 3, v___f_173_);
lean_ctor_set(v___x_177_, 4, v___f_170_);
lean_ctor_set(v___x_177_, 5, v___x_175_);
lean_ctor_set(v___x_177_, 6, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___redArg___boxed(lean_object* v_inst_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_IsChain_linearOrder___redArg(v_inst_178_, v_inst_179_);
lean_dec_ref(v_inst_178_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder(lean_object* v_00_u03b1_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_s_184_, lean_object* v_hs_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib_IsChain_linearOrder___redArg(v_inst_182_, v_inst_183_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsChain_linearOrder___boxed(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_s_190_, lean_object* v_hs_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_IsChain_linearOrder(v_00_u03b1_187_, v_inst_188_, v_inst_189_, v_s_190_, v_hs_191_);
lean_dec_ref(v_inst_188_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instSetLike(lean_object* v_00_u03b1_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_box(0);
return v___x_195_;
}
}
static lean_object* _init_lp_mathlib_Flag_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = lean_box(0);
v___x_197_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instPartialOrder(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lean_obj_once(&lp_mathlib_Flag_instPartialOrder___closed__0, &lp_mathlib_Flag_instPartialOrder___closed__0_once, _init_lp_mathlib_Flag_instPartialOrder___closed__0);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_ofIsMaxChain(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_, lean_object* v_c_203_, lean_object* v_hc_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_box(0);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___redArg(lean_object* v_inst_206_){
_start:
{
lean_inc(v_inst_206_);
return v_inst_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___redArg___boxed(lean_object* v_inst_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Flag_instOrderTopSubtypeMem___redArg(v_inst_207_);
lean_dec(v_inst_207_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_s_212_){
_start:
{
lean_inc(v_inst_211_);
return v_inst_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderTopSubtypeMem___boxed(lean_object* v_00_u03b1_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_s_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Flag_instOrderTopSubtypeMem(v_00_u03b1_213_, v_inst_214_, v_inst_215_, v_s_216_);
lean_dec(v_inst_215_);
lean_dec_ref(v_inst_214_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___redArg(lean_object* v_inst_218_){
_start:
{
lean_inc(v_inst_218_);
return v_inst_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___redArg___boxed(lean_object* v_inst_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Flag_instOrderBotSubtypeMem___redArg(v_inst_219_);
lean_dec(v_inst_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem(lean_object* v_00_u03b1_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_s_224_){
_start:
{
lean_inc(v_inst_223_);
return v_inst_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instOrderBotSubtypeMem___boxed(lean_object* v_00_u03b1_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_s_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Flag_instOrderBotSubtypeMem(v_00_u03b1_225_, v_inst_226_, v_inst_227_, v_s_228_);
lean_dec(v_inst_227_);
lean_dec_ref(v_inst_226_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem___redArg(lean_object* v_inst_230_){
_start:
{
lean_object* v_toOrderTop_231_; lean_object* v_toOrderBot_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_239_; 
v_toOrderTop_231_ = lean_ctor_get(v_inst_230_, 0);
v_toOrderBot_232_ = lean_ctor_get(v_inst_230_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v_inst_230_);
if (v_isSharedCheck_239_ == 0)
{
v___x_234_ = v_inst_230_;
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_toOrderBot_232_);
lean_inc(v_toOrderTop_231_);
lean_dec(v_inst_230_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_239_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v___x_237_; 
if (v_isShared_235_ == 0)
{
v___x_237_ = v___x_234_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_238_; 
v_reuseFailAlloc_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_238_, 0, v_toOrderTop_231_);
lean_ctor_set(v_reuseFailAlloc_238_, 1, v_toOrderBot_232_);
v___x_237_ = v_reuseFailAlloc_238_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
return v___x_237_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem(lean_object* v_00_u03b1_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_s_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lp_mathlib_Flag_instBoundedOrderSubtypeMem___redArg(v_inst_242_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instBoundedOrderSubtypeMem___boxed(lean_object* v_00_u03b1_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_s_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Flag_instBoundedOrderSubtypeMem(v_00_u03b1_245_, v_inst_246_, v_inst_247_, v_s_248_);
lean_dec_ref(v_inst_246_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_map___lam__0(lean_object* v_s_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lean_box(0);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_map(lean_object* v_00_u03b1_255_, lean_object* v_00_u03b2_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_e_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = ((lean_object*)(lp_mathlib_Flag_map___closed__1));
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_map___boxed(lean_object* v_00_u03b1_261_, lean_object* v_00_u03b2_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_e_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Flag_map(v_00_u03b1_261_, v_00_u03b2_262_, v_inst_263_, v_inst_264_, v_e_265_);
lean_dec_ref(v_e_265_);
lean_dec_ref(v_inst_264_);
lean_dec_ref(v_inst_263_);
return v_res_266_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0(lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_a_269_, lean_object* v_b_270_){
_start:
{
lean_object* v___x_271_; uint8_t v___x_272_; 
lean_inc(v_b_270_);
lean_inc(v_a_269_);
v___x_271_ = lean_apply_2(v_inst_267_, v_a_269_, v_b_270_);
v___x_272_ = lean_unbox(v___x_271_);
if (v___x_272_ == 0)
{
lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_273_ = lean_apply_2(v_inst_268_, v_a_269_, v_b_270_);
v___x_274_ = lean_unbox(v___x_273_);
if (v___x_274_ == 0)
{
uint8_t v___x_275_; 
v___x_275_ = 2;
return v___x_275_;
}
else
{
uint8_t v___x_276_; 
v___x_276_ = 1;
return v___x_276_;
}
}
else
{
uint8_t v___x_277_; 
lean_dec(v_b_270_);
lean_dec(v_a_269_);
lean_dec_ref(v_inst_268_);
v___x_277_ = 0;
return v___x_277_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0___boxed(lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_a_280_, lean_object* v_b_281_){
_start:
{
uint8_t v_res_282_; lean_object* v_r_283_; 
v_res_282_ = lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0(v_inst_278_, v_inst_279_, v_a_280_, v_b_281_);
v_r_283_ = lean_box(v_res_282_);
return v_r_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg(lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___f_288_; lean_object* v___f_289_; lean_object* v___f_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
lean_inc_ref(v_inst_287_);
lean_inc_ref(v_inst_286_);
v___f_288_ = lean_alloc_closure((void*)(lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_288_, 0, v_inst_286_);
lean_closure_set(v___f_288_, 1, v_inst_287_);
lean_inc_ref_n(v_inst_285_, 2);
v___f_289_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__2), 3, 1);
lean_closure_set(v___f_289_, 0, v_inst_285_);
v___f_290_ = lean_alloc_closure((void*)(lp_mathlib_IsChain_linearOrder___redArg___lam__1), 3, 1);
lean_closure_set(v___f_290_, 0, v_inst_285_);
v___x_291_ = lp_mathlib_Subtype_preorder(lean_box(0), v_inst_284_, lean_box(0));
lean_inc_ref(v_inst_284_);
v___x_292_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_decidableLE___boxed), 6, 4);
lean_closure_set(v___x_292_, 0, lean_box(0));
lean_closure_set(v___x_292_, 1, v_inst_284_);
lean_closure_set(v___x_292_, 2, v_inst_285_);
lean_closure_set(v___x_292_, 3, lean_box(0));
v___x_293_ = lean_alloc_closure((void*)(l_Subtype_instDecidableEq___boxed), 5, 3);
lean_closure_set(v___x_293_, 0, lean_box(0));
lean_closure_set(v___x_293_, 1, lean_box(0));
lean_closure_set(v___x_293_, 2, v_inst_287_);
v___x_294_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_decidableLT___boxed), 6, 4);
lean_closure_set(v___x_294_, 0, lean_box(0));
lean_closure_set(v___x_294_, 1, v_inst_284_);
lean_closure_set(v___x_294_, 2, v_inst_286_);
lean_closure_set(v___x_294_, 3, lean_box(0));
v___x_295_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_295_, 0, v___x_291_);
lean_ctor_set(v___x_295_, 1, v___f_290_);
lean_ctor_set(v___x_295_, 2, v___f_289_);
lean_ctor_set(v___x_295_, 3, v___f_288_);
lean_ctor_set(v___x_295_, 4, v___x_292_);
lean_ctor_set(v___x_295_, 5, v___x_293_);
lean_ctor_set(v___x_295_, 6, v___x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq(lean_object* v_00_u03b1_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_s_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_Flag_instLinearOrderSubtypeMemOfDecidableLEOfDecidableLTOfDecidableEq___redArg(v_inst_297_, v_inst_298_, v_inst_299_, v_inst_300_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instUnique(lean_object* v_00_u03b1_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lean_box(0);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Flag_instUnique___boxed(lean_object* v_00_u03b1_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_Flag_instUnique(v_00_u03b1_306_, v_inst_307_);
lean_dec_ref(v_inst_307_);
return v_res_308_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Pairwise_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
}
#ifdef __cplusplus
}
#endif
