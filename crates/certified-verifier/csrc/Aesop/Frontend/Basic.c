// Lean compiler output
// Module: Aesop.Frontend.Basic
// Imports: public import Init public meta import Init public import Lean.Elab.Exception
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
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_throwUnsupportedSyntax___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "bool_lit"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(126, 16, 43, 36, 193, 43, 121, 25)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(152, 61, 255, 241, 92, 248, 184, 241)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__8_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "`(bool_lit| "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(126, 16, 43, 36, 193, 43, 121, 25)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__13_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__11_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__17 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__7_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__18 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__19 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__19_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Parser_Category_Aesop_bool__lit;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "bool_litTrue"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__1_value),LEAN_SCALAR_PTR_LITERAL(116, 29, 222, 251, 164, 52, 126, 99)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__5_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litTrue = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "bool_litFalse"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_bool__lit_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 93, 156, 163, 77, 158, 127, 75)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__4_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_bool__litFalse = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Lean_Parser_Category_Aesop_bool__lit(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_box(0);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0(lean_object* v_stx_80_, lean_object* v_withRef_81_, lean_object* v___y_82_, lean_object* v_oldRef_83_){
_start:
{
lean_object* v_ref_84_; lean_object* v___x_85_; 
v_ref_84_ = l_Lean_replaceRef(v_stx_80_, v_oldRef_83_);
v___x_85_ = lean_apply_3(v_withRef_81_, lean_box(0), v_ref_84_, v___y_82_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0___boxed(lean_object* v_stx_86_, lean_object* v_withRef_87_, lean_object* v___y_88_, lean_object* v_oldRef_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0(v_stx_86_, v_withRef_87_, v___y_88_, v_oldRef_89_);
lean_dec(v_oldRef_89_);
lean_dec(v_stx_86_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit___redArg(lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_stx_94_){
_start:
{
lean_object* v___y_96_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_102_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_bool__litTrue___closed__2));
lean_inc(v_stx_94_);
v___x_103_ = l_Lean_Syntax_isOfKind(v_stx_94_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_bool__litFalse___closed__1));
lean_inc(v_stx_94_);
v___x_105_ = l_Lean_Syntax_isOfKind(v_stx_94_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; 
v___x_106_ = l_Lean_Elab_throwUnsupportedSyntax___redArg(v_inst_93_);
v___y_96_ = v___x_106_;
goto v___jp_95_;
}
else
{
lean_object* v_toApplicative_107_; lean_object* v_toPure_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec_ref(v_inst_93_);
v_toApplicative_107_ = lean_ctor_get(v_inst_91_, 0);
v_toPure_108_ = lean_ctor_get(v_toApplicative_107_, 1);
v___x_109_ = lean_box(v___x_103_);
lean_inc(v_toPure_108_);
v___x_110_ = lean_apply_2(v_toPure_108_, lean_box(0), v___x_109_);
v___y_96_ = v___x_110_;
goto v___jp_95_;
}
}
else
{
lean_object* v_toApplicative_111_; lean_object* v_toPure_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
lean_dec_ref(v_inst_93_);
v_toApplicative_111_ = lean_ctor_get(v_inst_91_, 0);
v_toPure_112_ = lean_ctor_get(v_toApplicative_111_, 1);
v___x_113_ = lean_box(v___x_103_);
lean_inc(v_toPure_112_);
v___x_114_ = lean_apply_2(v_toPure_112_, lean_box(0), v___x_113_);
v___y_96_ = v___x_114_;
goto v___jp_95_;
}
v___jp_95_:
{
lean_object* v_toBind_97_; lean_object* v_getRef_98_; lean_object* v_withRef_99_; lean_object* v___f_100_; lean_object* v___x_101_; 
v_toBind_97_ = lean_ctor_get(v_inst_91_, 1);
lean_inc(v_toBind_97_);
lean_dec_ref(v_inst_91_);
v_getRef_98_ = lean_ctor_get(v_inst_92_, 0);
lean_inc(v_getRef_98_);
v_withRef_99_ = lean_ctor_get(v_inst_92_, 1);
lean_inc(v_withRef_99_);
lean_dec_ref(v_inst_92_);
v___f_100_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_elabBoolLit___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_100_, 0, v_stx_94_);
lean_closure_set(v___f_100_, 1, v_withRef_99_);
lean_closure_set(v___f_100_, 2, v___y_96_);
v___x_101_ = lean_apply_4(v_toBind_97_, lean_box(0), lean_box(0), v_getRef_98_, v___f_100_);
return v___x_101_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabBoolLit(lean_object* v_m_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_stx_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_aesop_Aesop_Frontend_elabBoolLit___redArg(v_inst_116_, v_inst_117_, v_inst_118_, v_stx_119_);
return v___x_120_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Exception(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Lean_Parser_Category_Aesop_bool__lit = _init_lp_aesop_Lean_Parser_Category_Aesop_bool__lit();
lean_mark_persistent(lp_aesop_Lean_Parser_Category_Aesop_bool__lit);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Exception(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
