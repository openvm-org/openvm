// Lean compiler output
// Module: Mathlib.Tactic.RSuffices
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_rcasesPatMed;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rsuffices"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__2_value),LEAN_SCALAR_PTR_LITERAL(216, 155, 227, 138, 150, 150, 217, 29)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__9_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__17_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__25_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__27_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_rsuffices___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_rsuffices___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_rsuffices___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rsuffices;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rotateLeft"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rotate_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "obtain"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_rsuffices___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(11, 177, 143, 165, 56, 37, 104, 113)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__12(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_22_ = l_Lean_Parser_Tactic_rcasesPatMed;
v___x_23_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__11));
v___x_24_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__5));
v___x_25_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_25_, 0, v___x_24_);
lean_ctor_set(v___x_25_, 1, v___x_23_);
lean_ctor_set(v___x_25_, 2, v___x_22_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__13(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_26_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__12, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__12);
v___x_27_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__8));
v___x_28_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
lean_ctor_set(v___x_28_, 1, v___x_26_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__14(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__13, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__13);
v___x_30_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__6));
v___x_31_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__5));
v___x_32_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___x_30_);
lean_ctor_set(v___x_32_, 2, v___x_29_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__22(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_49_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__21));
v___x_50_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__14, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__14);
v___x_51_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__5));
v___x_52_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_49_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__31(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_72_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__30));
v___x_73_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__22, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__22);
v___x_74_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__5));
v___x_75_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v___x_73_);
lean_ctor_set(v___x_75_, 2, v___x_72_);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__32(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_76_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__31, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__31);
v___x_77_ = lean_unsigned_to_nat(1022u);
v___x_78_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__3));
v___x_79_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v___x_77_);
lean_ctor_set(v___x_79_, 2, v___x_76_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_rsuffices(void){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_rsuffices___closed__32, &lp_mathlib_Mathlib_Tactic_rsuffices___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_rsuffices___closed__32);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0(lean_object* v_x_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___closed__0));
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___boxed(lean_object* v_x_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0(v_x_85_);
lean_dec(v_x_85_);
return v_res_86_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19(void){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = l_Array_mkArray0(lean_box(0));
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v___x_127_; lean_object* v___y_129_; lean_object* v___y_130_; lean_object* v___y_131_; lean_object* v___y_132_; lean_object* v___y_133_; lean_object* v___y_134_; lean_object* v___y_135_; lean_object* v___y_136_; lean_object* v___y_137_; lean_object* v___y_138_; lean_object* v___y_139_; lean_object* v___y_140_; lean_object* v___y_141_; lean_object* v___y_142_; lean_object* v___y_143_; lean_object* v___x_162_; uint8_t v___x_163_; 
v___x_127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__1));
v___x_162_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_rsuffices___closed__3));
lean_inc(v_x_124_);
v___x_163_ = l_Lean_Syntax_isOfKind(v_x_124_, v___x_162_);
if (v___x_163_ == 0)
{
lean_object* v___x_164_; lean_object* v___x_165_; 
lean_dec(v_x_124_);
v___x_164_ = lean_box(1);
v___x_165_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v_a_126_);
return v___x_165_;
}
else
{
lean_object* v___y_167_; lean_object* v___y_168_; lean_object* v___y_169_; lean_object* v___y_170_; lean_object* v___y_171_; lean_object* v___y_172_; lean_object* v___y_173_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_178_; lean_object* v___y_179_; lean_object* v___y_180_; lean_object* v___y_181_; lean_object* v___y_191_; lean_object* v___y_192_; lean_object* v___y_193_; lean_object* v___y_194_; lean_object* v___y_195_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_198_; lean_object* v___y_199_; lean_object* v___y_200_; lean_object* v___y_201_; lean_object* v___y_202_; lean_object* v___y_203_; lean_object* v___y_204_; lean_object* v___y_205_; lean_object* v___x_213_; lean_object* v___y_215_; lean_object* v___y_216_; lean_object* v_bar_217_; lean_object* v___y_218_; lean_object* v___y_219_; lean_object* v___x_238_; lean_object* v___y_240_; lean_object* v___y_241_; lean_object* v_foo_242_; lean_object* v___y_243_; lean_object* v___y_244_; lean_object* v_pred_259_; lean_object* v___y_260_; lean_object* v___y_261_; lean_object* v___x_271_; uint8_t v___x_272_; 
v___x_213_ = lean_unsigned_to_nat(0u);
v___x_238_ = lean_unsigned_to_nat(1u);
v___x_271_ = l_Lean_Syntax_getArg(v_x_124_, v___x_238_);
v___x_272_ = l_Lean_Syntax_isNone(v___x_271_);
if (v___x_272_ == 0)
{
uint8_t v___x_273_; 
lean_inc(v___x_271_);
v___x_273_ = l_Lean_Syntax_matchesNull(v___x_271_, v___x_238_);
if (v___x_273_ == 0)
{
lean_object* v___x_274_; lean_object* v___x_275_; 
lean_dec(v___x_271_);
lean_dec(v_x_124_);
v___x_274_ = lean_box(1);
v___x_275_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_274_);
lean_ctor_set(v___x_275_, 1, v_a_126_);
return v___x_275_;
}
else
{
lean_object* v_pred_276_; lean_object* v___x_277_; 
v_pred_276_ = l_Lean_Syntax_getArg(v___x_271_, v___x_213_);
lean_dec(v___x_271_);
v___x_277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_277_, 0, v_pred_276_);
v_pred_259_ = v___x_277_;
v___y_260_ = v_a_125_;
v___y_261_ = v_a_126_;
goto v___jp_258_;
}
}
else
{
lean_object* v___x_278_; 
lean_dec(v___x_271_);
v___x_278_ = lean_box(0);
v_pred_259_ = v___x_278_;
v___y_260_ = v_a_125_;
v___y_261_ = v_a_126_;
goto v___jp_258_;
}
v___jp_166_:
{
lean_object* v___x_182_; lean_object* v___x_183_; 
lean_inc_ref(v___y_172_);
v___x_182_ = l_Array_append___redArg(v___y_172_, v___y_181_);
lean_dec_ref(v___y_181_);
lean_inc(v___y_170_);
lean_inc(v___y_178_);
v___x_183_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_183_, 0, v___y_178_);
lean_ctor_set(v___x_183_, 1, v___y_170_);
lean_ctor_set(v___x_183_, 2, v___x_182_);
if (lean_obj_tag(v___y_169_) == 1)
{
lean_object* v_val_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v_val_184_ = lean_ctor_get(v___y_169_, 0);
lean_inc(v_val_184_);
lean_dec_ref_known(v___y_169_, 1);
v___x_185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__4));
lean_inc_n(v___y_178_, 2);
v___x_186_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_186_, 0, v___y_178_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
lean_inc(v___y_170_);
v___x_187_ = l_Lean_Syntax_node1(v___y_178_, v___y_170_, v_val_184_);
v___x_188_ = l_Array_mkArray2___redArg(v___x_186_, v___x_187_);
v___y_129_ = v___y_167_;
v___y_130_ = v___y_168_;
v___y_131_ = v___y_170_;
v___y_132_ = v___y_171_;
v___y_133_ = v___y_172_;
v___y_134_ = v___y_173_;
v___y_135_ = v___y_174_;
v___y_136_ = v___x_183_;
v___y_137_ = v___y_175_;
v___y_138_ = v___y_176_;
v___y_139_ = v___y_177_;
v___y_140_ = v___y_178_;
v___y_141_ = v___y_179_;
v___y_142_ = v___y_180_;
v___y_143_ = v___x_188_;
goto v___jp_128_;
}
else
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0(v___y_169_);
lean_dec(v___y_169_);
v___y_129_ = v___y_167_;
v___y_130_ = v___y_168_;
v___y_131_ = v___y_170_;
v___y_132_ = v___y_171_;
v___y_133_ = v___y_172_;
v___y_134_ = v___y_173_;
v___y_135_ = v___y_174_;
v___y_136_ = v___x_183_;
v___y_137_ = v___y_175_;
v___y_138_ = v___y_176_;
v___y_139_ = v___y_177_;
v___y_140_ = v___y_178_;
v___y_141_ = v___y_179_;
v___y_142_ = v___y_180_;
v___y_143_ = v___x_189_;
goto v___jp_128_;
}
}
v___jp_190_:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
lean_inc_ref(v___y_197_);
v___x_206_ = l_Array_append___redArg(v___y_197_, v___y_205_);
lean_dec_ref(v___y_205_);
lean_inc(v___y_193_);
lean_inc(v___y_202_);
v___x_207_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_207_, 0, v___y_202_);
lean_ctor_set(v___x_207_, 1, v___y_193_);
lean_ctor_set(v___x_207_, 2, v___x_206_);
if (lean_obj_tag(v___y_196_) == 1)
{
lean_object* v_val_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v_val_208_ = lean_ctor_get(v___y_196_, 0);
lean_inc(v_val_208_);
lean_dec_ref_known(v___y_196_, 1);
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__5));
lean_inc(v___y_202_);
v___x_210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_210_, 0, v___y_202_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
v___x_211_ = l_Array_mkArray2___redArg(v___x_210_, v_val_208_);
v___y_167_ = v___y_191_;
v___y_168_ = v___y_192_;
v___y_169_ = v___y_194_;
v___y_170_ = v___y_193_;
v___y_171_ = v___y_195_;
v___y_172_ = v___y_197_;
v___y_173_ = v___y_198_;
v___y_174_ = v___y_199_;
v___y_175_ = v___y_200_;
v___y_176_ = v___y_201_;
v___y_177_ = v___x_207_;
v___y_178_ = v___y_202_;
v___y_179_ = v___y_203_;
v___y_180_ = v___y_204_;
v___y_181_ = v___x_211_;
goto v___jp_166_;
}
else
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0(v___y_196_);
lean_dec(v___y_196_);
v___y_167_ = v___y_191_;
v___y_168_ = v___y_192_;
v___y_169_ = v___y_194_;
v___y_170_ = v___y_193_;
v___y_171_ = v___y_195_;
v___y_172_ = v___y_197_;
v___y_173_ = v___y_198_;
v___y_174_ = v___y_199_;
v___y_175_ = v___y_200_;
v___y_176_ = v___y_201_;
v___y_177_ = v___x_207_;
v___y_178_ = v___y_202_;
v___y_179_ = v___y_203_;
v___y_180_ = v___y_204_;
v___y_181_ = v___x_212_;
goto v___jp_166_;
}
}
v___jp_214_:
{
lean_object* v_ref_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v_ref_220_ = lean_ctor_get(v___y_218_, 5);
v___x_221_ = 0;
v___x_222_ = l_Lean_SourceInfo_fromRef(v_ref_220_, v___x_221_);
v___x_223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__6));
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__7));
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__9));
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__10));
lean_inc_n(v___x_222_, 2);
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_222_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__12));
v___x_229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__14));
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__16));
v___x_231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__17));
v___x_232_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__18));
v___x_233_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_222_);
lean_ctor_set(v___x_233_, 1, v___x_231_);
v___x_234_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__19);
if (lean_obj_tag(v___y_216_) == 1)
{
lean_object* v_val_235_; lean_object* v___x_236_; 
v_val_235_ = lean_ctor_get(v___y_216_, 0);
lean_inc(v_val_235_);
lean_dec_ref_known(v___y_216_, 1);
v___x_236_ = l_Array_mkArray1___redArg(v_val_235_);
v___y_191_ = v___y_219_;
v___y_192_ = v___x_227_;
v___y_193_ = v___x_230_;
v___y_194_ = v_bar_217_;
v___y_195_ = v___x_223_;
v___y_196_ = v___y_215_;
v___y_197_ = v___x_234_;
v___y_198_ = v___x_228_;
v___y_199_ = v___x_225_;
v___y_200_ = v___x_224_;
v___y_201_ = v___x_232_;
v___y_202_ = v___x_222_;
v___y_203_ = v___x_233_;
v___y_204_ = v___x_229_;
v___y_205_ = v___x_236_;
goto v___jp_190_;
}
else
{
lean_object* v___x_237_; 
lean_dec(v___y_216_);
v___x_237_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___lam__0___closed__0));
v___y_191_ = v___y_219_;
v___y_192_ = v___x_227_;
v___y_193_ = v___x_230_;
v___y_194_ = v_bar_217_;
v___y_195_ = v___x_223_;
v___y_196_ = v___y_215_;
v___y_197_ = v___x_234_;
v___y_198_ = v___x_228_;
v___y_199_ = v___x_225_;
v___y_200_ = v___x_224_;
v___y_201_ = v___x_232_;
v___y_202_ = v___x_222_;
v___y_203_ = v___x_233_;
v___y_204_ = v___x_229_;
v___y_205_ = v___x_237_;
goto v___jp_190_;
}
}
v___jp_239_:
{
lean_object* v___x_245_; lean_object* v___x_246_; uint8_t v___x_247_; 
v___x_245_ = lean_unsigned_to_nat(3u);
v___x_246_ = l_Lean_Syntax_getArg(v_x_124_, v___x_245_);
lean_dec(v_x_124_);
v___x_247_ = l_Lean_Syntax_isNone(v___x_246_);
if (v___x_247_ == 0)
{
uint8_t v___x_248_; 
lean_inc(v___x_246_);
v___x_248_ = l_Lean_Syntax_matchesNull(v___x_246_, v___y_240_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; 
lean_dec(v___x_246_);
lean_dec(v_foo_242_);
lean_dec(v___y_241_);
v___x_249_ = lean_box(1);
v___x_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v___y_244_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_251_ = l_Lean_Syntax_getArg(v___x_246_, v___x_238_);
lean_dec(v___x_246_);
lean_inc(v___x_251_);
v___x_252_ = l_Lean_Syntax_matchesNull(v___x_251_, v___x_238_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; lean_object* v___x_254_; 
lean_dec(v___x_251_);
lean_dec(v_foo_242_);
lean_dec(v___y_241_);
v___x_253_ = lean_box(1);
v___x_254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v___y_244_);
return v___x_254_;
}
else
{
lean_object* v_bar_255_; lean_object* v___x_256_; 
v_bar_255_ = l_Lean_Syntax_getArg(v___x_251_, v___x_213_);
lean_dec(v___x_251_);
v___x_256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_256_, 0, v_bar_255_);
v___y_215_ = v_foo_242_;
v___y_216_ = v___y_241_;
v_bar_217_ = v___x_256_;
v___y_218_ = v___y_243_;
v___y_219_ = v___y_244_;
goto v___jp_214_;
}
}
}
else
{
lean_object* v___x_257_; 
lean_dec(v___x_246_);
v___x_257_ = lean_box(0);
v___y_215_ = v_foo_242_;
v___y_216_ = v___y_241_;
v_bar_217_ = v___x_257_;
v___y_218_ = v___y_243_;
v___y_219_ = v___y_244_;
goto v___jp_214_;
}
}
v___jp_258_:
{
lean_object* v___x_262_; lean_object* v___x_263_; uint8_t v___x_264_; 
v___x_262_ = lean_unsigned_to_nat(2u);
v___x_263_ = l_Lean_Syntax_getArg(v_x_124_, v___x_262_);
v___x_264_ = l_Lean_Syntax_isNone(v___x_263_);
if (v___x_264_ == 0)
{
uint8_t v___x_265_; 
lean_inc(v___x_263_);
v___x_265_ = l_Lean_Syntax_matchesNull(v___x_263_, v___x_262_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; lean_object* v___x_267_; 
lean_dec(v___x_263_);
lean_dec(v_pred_259_);
lean_dec(v_x_124_);
v___x_266_ = lean_box(1);
v___x_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v___y_261_);
return v___x_267_;
}
else
{
lean_object* v_foo_268_; lean_object* v___x_269_; 
v_foo_268_ = l_Lean_Syntax_getArg(v___x_263_, v___x_238_);
lean_dec(v___x_263_);
v___x_269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_269_, 0, v_foo_268_);
v___y_240_ = v___x_262_;
v___y_241_ = v_pred_259_;
v_foo_242_ = v___x_269_;
v___y_243_ = v___y_260_;
v___y_244_ = v___y_261_;
goto v___jp_239_;
}
}
else
{
lean_object* v___x_270_; 
lean_dec(v___x_263_);
v___x_270_ = lean_box(0);
v___y_240_ = v___x_262_;
v___y_241_ = v_pred_259_;
v_foo_242_ = v___x_270_;
v___y_243_ = v___y_260_;
v___y_244_ = v___y_261_;
goto v___jp_239_;
}
}
}
v___jp_128_:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
lean_inc_ref_n(v___y_133_, 2);
v___x_144_ = l_Array_append___redArg(v___y_133_, v___y_143_);
lean_dec_ref(v___y_143_);
lean_inc_n(v___y_131_, 3);
lean_inc_n(v___y_140_, 10);
v___x_145_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_145_, 0, v___y_140_);
lean_ctor_set(v___x_145_, 1, v___y_131_);
lean_ctor_set(v___x_145_, 2, v___x_144_);
lean_inc(v___y_138_);
v___x_146_ = l_Lean_Syntax_node4(v___y_140_, v___y_138_, v___y_141_, v___y_139_, v___y_136_, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__0));
v___x_148_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_148_, 0, v___y_140_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__1));
lean_inc_ref(v___y_137_);
lean_inc_ref(v___y_132_);
v___x_150_ = l_Lean_Name_mkStr4(v___y_132_, v___y_137_, v___x_127_, v___x_149_);
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__2));
v___x_152_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_152_, 0, v___y_140_);
lean_ctor_set(v___x_152_, 1, v___x_151_);
v___x_153_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_153_, 0, v___y_140_);
lean_ctor_set(v___x_153_, 1, v___y_131_);
lean_ctor_set(v___x_153_, 2, v___y_133_);
v___x_154_ = l_Lean_Syntax_node2(v___y_140_, v___x_150_, v___x_152_, v___x_153_);
v___x_155_ = l_Lean_Syntax_node3(v___y_140_, v___y_131_, v___x_146_, v___x_148_, v___x_154_);
lean_inc(v___y_142_);
v___x_156_ = l_Lean_Syntax_node1(v___y_140_, v___y_142_, v___x_155_);
lean_inc(v___y_134_);
v___x_157_ = l_Lean_Syntax_node1(v___y_140_, v___y_134_, v___x_156_);
v___x_158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___closed__3));
v___x_159_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_159_, 0, v___y_140_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
lean_inc(v___y_135_);
v___x_160_ = l_Lean_Syntax_node3(v___y_140_, v___y_135_, v___y_130_, v___x_157_, v___x_159_);
v___x_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___y_129_);
return v___x_161_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1___boxed(lean_object* v_x_279_, lean_object* v_a_280_, lean_object* v_a_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__RSuffices______macroRules__Mathlib__Tactic__rsuffices__1(v_x_279_, v_a_280_, v_a_281_);
lean_dec_ref(v_a_280_);
return v_res_282_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_rsuffices = _init_lp_mathlib_Mathlib_Tactic_rsuffices();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_rsuffices);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_RSuffices(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_RSuffices(builtin);
}
#ifdef __cplusplus
}
#endif
