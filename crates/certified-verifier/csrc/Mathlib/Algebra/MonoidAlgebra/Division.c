// Lean compiler output
// Module: Mathlib.Algebra.MonoidAlgebra.Division
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.End public import Mathlib.Algebra.MonoidAlgebra.Defs
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 212, 98, 212, 98, 99, 115, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "MonoidAlgebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(27, 119, 153, 233, 46, 30, 196, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Division"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(29, 83, 163, 131, 71, 251, 127, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(40, 89, 148, 7, 243, 18, 144, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidAlgebra"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(64, 174, 126, 188, 209, 63, 239, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term_/ᵒᶠ_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(16, 234, 150, 190, 28, 220, 64, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " /ᵒᶠ "};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__17_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__19_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__20_value),((lean_object*)(((size_t)(71) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__18_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__22_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__23_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0__ = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "divOf"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(74, 192, 73, 156, 206, 191, 125, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 199, 239, 38, 63, 229, 227, 206)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(142, 135, 209, 50, 158, 27, 153, 23)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term_%ᵒᶠ_"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 204, 18, 238, 145, 92, 119, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 5, .m_data = " %ᵒᶠ "};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__16_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0__ = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "modOf"};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 162, 193, 131, 32, 22, 71, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 199, 239, 38, 63, 229, 227, 206)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(222, 156, 223, 209, 53, 246, 141, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__modOf__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__modOf__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__5));
v___x_64_ = l_String_toRawSubstring_x27(v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; uint8_t v___x_84_; 
v___x_82_ = lean_unsigned_to_nat(0u);
v___x_83_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14));
lean_inc(v_x_79_);
v___x_84_ = l_Lean_Syntax_isOfKind(v_x_79_, v___x_83_);
if (v___x_84_ == 0)
{
lean_object* v___x_85_; lean_object* v___x_86_; 
lean_dec(v_x_79_);
v___x_85_ = lean_box(1);
v___x_86_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v_a_81_);
return v___x_86_;
}
else
{
lean_object* v_quotContext_87_; lean_object* v_currMacroScope_88_; lean_object* v_ref_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; uint8_t v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v_quotContext_87_ = lean_ctor_get(v_a_80_, 1);
v_currMacroScope_88_ = lean_ctor_get(v_a_80_, 2);
v_ref_89_ = lean_ctor_get(v_a_80_, 5);
v___x_90_ = l_Lean_Syntax_getArg(v_x_79_, v___x_82_);
v___x_91_ = lean_unsigned_to_nat(2u);
v___x_92_ = l_Lean_Syntax_getArg(v_x_79_, v___x_91_);
lean_dec(v_x_79_);
v___x_93_ = 0;
v___x_94_ = l_Lean_SourceInfo_fromRef(v_ref_89_, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4));
v___x_96_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6, &lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__6);
v___x_97_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__7));
lean_inc(v_currMacroScope_88_);
lean_inc(v_quotContext_87_);
v___x_98_ = l_Lean_addMacroScope(v_quotContext_87_, v___x_97_, v_currMacroScope_88_);
v___x_99_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__10));
lean_inc_n(v___x_94_, 2);
v___x_100_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_100_, 0, v___x_94_);
lean_ctor_set(v___x_100_, 1, v___x_96_);
lean_ctor_set(v___x_100_, 2, v___x_98_);
lean_ctor_set(v___x_100_, 3, v___x_99_);
v___x_101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__12));
v___x_102_ = l_Lean_Syntax_node2(v___x_94_, v___x_101_, v___x_90_, v___x_92_);
v___x_103_ = l_Lean_Syntax_node2(v___x_94_, v___x_95_, v___x_100_, v___x_102_);
v___x_104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v_a_81_);
return v___x_104_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___boxed(lean_object* v_x_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1(v_x_105_, v_a_106_, v_a_107_);
lean_dec_ref(v_a_106_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1(lean_object* v_x_112_, lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_115_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4));
lean_inc(v_x_112_);
v___x_116_ = l_Lean_Syntax_isOfKind(v_x_112_, v___x_115_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; lean_object* v___x_118_; 
lean_dec(v_x_112_);
v___x_117_ = lean_box(0);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v_a_114_);
return v___x_118_;
}
else
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_119_ = lean_unsigned_to_nat(0u);
v___x_120_ = l_Lean_Syntax_getArg(v_x_112_, v___x_119_);
v___x_121_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__1));
lean_inc(v___x_120_);
v___x_122_ = l_Lean_Syntax_isOfKind(v___x_120_, v___x_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v___x_124_; 
lean_dec(v___x_120_);
lean_dec(v_x_112_);
v___x_123_ = lean_box(0);
v___x_124_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v_a_114_);
return v___x_124_;
}
else
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v___x_125_ = lean_unsigned_to_nat(1u);
v___x_126_ = l_Lean_Syntax_getArg(v_x_112_, v___x_125_);
lean_dec(v_x_112_);
v___x_127_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_126_);
v___x_128_ = l_Lean_Syntax_matchesNull(v___x_126_, v___x_127_);
if (v___x_128_ == 0)
{
lean_object* v___x_129_; lean_object* v___x_130_; 
lean_dec(v___x_126_);
lean_dec(v___x_120_);
v___x_129_ = lean_box(0);
v___x_130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v_a_114_);
return v___x_130_;
}
else
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v_ref_133_; uint8_t v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_131_ = l_Lean_Syntax_getArg(v___x_126_, v___x_119_);
v___x_132_ = l_Lean_Syntax_getArg(v___x_126_, v___x_125_);
lean_dec(v___x_126_);
v_ref_133_ = l_Lean_replaceRef(v___x_120_, v_a_113_);
lean_dec(v___x_120_);
v___x_134_ = 0;
v___x_135_ = l_Lean_SourceInfo_fromRef(v_ref_133_, v___x_134_);
lean_dec(v_ref_133_);
v___x_136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__14));
v___x_137_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x2f_u1d52_u1da0___00__closed__17));
lean_inc(v___x_135_);
v___x_138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_135_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = l_Lean_Syntax_node3(v___x_135_, v___x_136_, v___x_131_, v___x_138_, v___x_132_);
v___x_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_a_114_);
return v___x_140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___boxed(lean_object* v_x_141_, lean_object* v_a_142_, lean_object* v_a_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1(v_x_141_, v_a_142_, v_a_143_);
lean_dec(v_a_142_);
return v_res_144_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__0));
v___x_163_ = l_String_toRawSubstring_x27(v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1(lean_object* v_x_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_178_ = lean_unsigned_to_nat(0u);
v___x_179_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1));
lean_inc(v_x_175_);
v___x_180_ = l_Lean_Syntax_isOfKind(v_x_175_, v___x_179_);
if (v___x_180_ == 0)
{
lean_object* v___x_181_; lean_object* v___x_182_; 
lean_dec(v_x_175_);
v___x_181_ = lean_box(1);
v___x_182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v_a_177_);
return v___x_182_;
}
else
{
lean_object* v_quotContext_183_; lean_object* v_currMacroScope_184_; lean_object* v_ref_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; uint8_t v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v_quotContext_183_ = lean_ctor_get(v_a_176_, 1);
v_currMacroScope_184_ = lean_ctor_get(v_a_176_, 2);
v_ref_185_ = lean_ctor_get(v_a_176_, 5);
v___x_186_ = l_Lean_Syntax_getArg(v_x_175_, v___x_178_);
v___x_187_ = lean_unsigned_to_nat(2u);
v___x_188_ = l_Lean_Syntax_getArg(v_x_175_, v___x_187_);
lean_dec(v_x_175_);
v___x_189_ = 0;
v___x_190_ = l_Lean_SourceInfo_fromRef(v_ref_185_, v___x_189_);
v___x_191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4));
v___x_192_ = lean_obj_once(&lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1, &lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__1);
v___x_193_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__2));
lean_inc(v_currMacroScope_184_);
lean_inc(v_quotContext_183_);
v___x_194_ = l_Lean_addMacroScope(v_quotContext_183_, v___x_193_, v_currMacroScope_184_);
v___x_195_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___closed__5));
lean_inc_n(v___x_190_, 2);
v___x_196_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_196_, 0, v___x_190_);
lean_ctor_set(v___x_196_, 1, v___x_192_);
lean_ctor_set(v___x_196_, 2, v___x_194_);
lean_ctor_set(v___x_196_, 3, v___x_195_);
v___x_197_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__12));
v___x_198_ = l_Lean_Syntax_node2(v___x_190_, v___x_197_, v___x_186_, v___x_188_);
v___x_199_ = l_Lean_Syntax_node2(v___x_190_, v___x_191_, v___x_196_, v___x_198_);
v___x_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v_a_177_);
return v___x_200_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1___boxed(lean_object* v_x_201_, lean_object* v_a_202_, lean_object* v_a_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x25_u1d52_u1da0____1(v_x_201_, v_a_202_, v_a_203_);
lean_dec_ref(v_a_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__modOf__1(lean_object* v_x_205_, lean_object* v_a_206_, lean_object* v_a_207_){
_start:
{
lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_208_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______macroRules____private__Mathlib__Algebra__MonoidAlgebra__Division__0__AddMonoidAlgebra__term___x2f_u1d52_u1da0____1___closed__4));
lean_inc(v_x_205_);
v___x_209_ = l_Lean_Syntax_isOfKind(v_x_205_, v___x_208_);
if (v___x_209_ == 0)
{
lean_object* v___x_210_; lean_object* v___x_211_; 
lean_dec(v_x_205_);
v___x_210_ = lean_box(0);
v___x_211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v_a_207_);
return v___x_211_;
}
else
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; 
v___x_212_ = lean_unsigned_to_nat(0u);
v___x_213_ = l_Lean_Syntax_getArg(v_x_205_, v___x_212_);
v___x_214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__divOf__1___closed__1));
lean_inc(v___x_213_);
v___x_215_ = l_Lean_Syntax_isOfKind(v___x_213_, v___x_214_);
if (v___x_215_ == 0)
{
lean_object* v___x_216_; lean_object* v___x_217_; 
lean_dec(v___x_213_);
lean_dec(v_x_205_);
v___x_216_ = lean_box(0);
v___x_217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_216_);
lean_ctor_set(v___x_217_, 1, v_a_207_);
return v___x_217_;
}
else
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_218_ = lean_unsigned_to_nat(1u);
v___x_219_ = l_Lean_Syntax_getArg(v_x_205_, v___x_218_);
lean_dec(v_x_205_);
v___x_220_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_219_);
v___x_221_ = l_Lean_Syntax_matchesNull(v___x_219_, v___x_220_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
lean_dec(v___x_219_);
lean_dec(v___x_213_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_a_207_);
return v___x_223_;
}
else
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v_ref_226_; uint8_t v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_224_ = l_Lean_Syntax_getArg(v___x_219_, v___x_212_);
v___x_225_ = l_Lean_Syntax_getArg(v___x_219_, v___x_218_);
lean_dec(v___x_219_);
v_ref_226_ = l_Lean_replaceRef(v___x_213_, v_a_206_);
lean_dec(v___x_213_);
v___x_227_ = 0;
v___x_228_ = l_Lean_SourceInfo_fromRef(v_ref_226_, v___x_227_);
lean_dec(v_ref_226_);
v___x_229_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__1));
v___x_230_ = ((lean_object*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra_term___x25_u1d52_u1da0___00__closed__2));
lean_inc(v___x_228_);
v___x_231_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_228_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
v___x_232_ = l_Lean_Syntax_node3(v___x_228_, v___x_229_, v___x_224_, v___x_231_, v___x_225_);
v___x_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v_a_207_);
return v___x_233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__modOf__1___boxed(lean_object* v_x_234_, lean_object* v_a_235_, lean_object* v_a_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Division_0__AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Division______unexpand__AddMonoidAlgebra__modOf__1(v_x_234_, v_a_235_, v_a_236_);
lean_dec(v_a_235_);
return v_res_237_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Division(builtin);
}
#ifdef __cplusplus
}
#endif
