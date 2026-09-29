// Lean compiler output
// Module: Mathlib.Algebra.MonoidAlgebra.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Action.TransferInstance public import Mathlib.Algebra.Module.Defs public import Mathlib.Data.Finsupp.Basic public import Mathlib.Data.Finsupp.SMulWithZero
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
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delabApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Finsupp_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidAlgebra"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term__[_]"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 199, 239, 38, 63, 229, 227, 206)}};
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(171, 38, 238, 152, 160, 20, 188, 88)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__3 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "noWs"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__5 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__5_value),LEAN_SCALAR_PTR_LITERAL(92, 29, 204, 148, 167, 109, 242, 21)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__6 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__6_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__7 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__7_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__9 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__9_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__7_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__9_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__10 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__10_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__11 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__12 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__13 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__10_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__13_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__14 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__14_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__16 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__14_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__16_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__17 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__17_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__17_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__18 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d = (const lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__18_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__0_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__1 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__2 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__2_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__3 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4_value;
static lean_once_cell_t lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 199, 239, 38, 63, 229, 227, 206)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__7 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__7_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__8 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__9 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__7_value),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__9_value)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__10 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__10_value;
static const lean_string_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__11 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__11_value;
static const lean_ctor_object lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__12 = (const lean_object*)&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_unexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_delabOfCoeff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_delabOfCoeff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_delabOfCoeff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_delabOfCoeff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "MonoidAlgebra"};
static const lean_object* lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(158, 117, 200, 219, 179, 188, 111, 163)}};
static const lean_ctor_object lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1_value_aux_0),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(6, 111, 194, 228, 94, 82, 240, 4)}};
static const lean_object* lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__17_value)}};
static const lean_object* lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_MonoidAlgebra_term_____x5b___x5d = (const lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__2_value;
static lean_once_cell_t lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0;
static const lean_ctor_object lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(158, 117, 200, 219, 179, 188, 111, 163)}};
static const lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1 = (const lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__2 = (const lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__2_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1_value)}};
static const lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__3 = (const lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__4 = (const lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__4_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__2_value),((lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__4_value)}};
static const lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__5 = (const lean_object*)&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_unexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MonoidAlgebra_coeffEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___closed__0 = (const lean_object*)&lp_mathlib_MonoidAlgebra_coeffEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_MonoidAlgebra_coeffEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___closed__1 = (const lean_object*)&lp_mathlib_MonoidAlgebra_coeffEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_MonoidAlgebra_coeffEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MonoidAlgebra_coeffEquiv___closed__0_value),((lean_object*)&lp_mathlib_MonoidAlgebra_coeffEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___closed__2 = (const lean_object*)&lp_mathlib_MonoidAlgebra_coeffEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_coeffEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_coeffEquiv___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidAlgebra_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__0));
v___x_54_ = l_String_toRawSubstring_x27(v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1(lean_object* v_x_71_, lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_74_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2));
lean_inc(v_x_71_);
v___x_75_ = l_Lean_Syntax_isOfKind(v_x_71_, v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; lean_object* v___x_77_; 
lean_dec(v_x_71_);
v___x_76_ = lean_box(1);
v___x_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v_a_73_);
return v___x_77_;
}
else
{
lean_object* v_quotContext_78_; lean_object* v_currMacroScope_79_; lean_object* v_ref_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_quotContext_78_ = lean_ctor_get(v_a_72_, 1);
v_currMacroScope_79_ = lean_ctor_get(v_a_72_, 2);
v_ref_80_ = lean_ctor_get(v_a_72_, 5);
v___x_81_ = lean_unsigned_to_nat(0u);
v___x_82_ = l_Lean_Syntax_getArg(v_x_71_, v___x_81_);
v___x_83_ = lean_unsigned_to_nat(2u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_71_, v___x_83_);
lean_dec(v_x_71_);
v___x_85_ = 0;
v___x_86_ = l_Lean_SourceInfo_fromRef(v_ref_80_, v___x_85_);
v___x_87_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4));
v___x_88_ = lean_obj_once(&lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5, &lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5_once, _init_lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__5);
v___x_89_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__6));
lean_inc(v_currMacroScope_79_);
lean_inc(v_quotContext_78_);
v___x_90_ = l_Lean_addMacroScope(v_quotContext_78_, v___x_89_, v_currMacroScope_79_);
v___x_91_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__10));
lean_inc_n(v___x_86_, 2);
v___x_92_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_92_, 0, v___x_86_);
lean_ctor_set(v___x_92_, 1, v___x_88_);
lean_ctor_set(v___x_92_, 2, v___x_90_);
lean_ctor_set(v___x_92_, 3, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__12));
v___x_94_ = l_Lean_Syntax_node2(v___x_86_, v___x_93_, v___x_82_, v___x_84_);
v___x_95_ = l_Lean_Syntax_node2(v___x_86_, v___x_87_, v___x_92_, v___x_94_);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_73_);
return v___x_96_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___boxed(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1(v_x_97_, v_a_98_, v_a_99_);
lean_dec_ref(v_a_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_unexpander(lean_object* v_x_101_, lean_object* v_a_102_, lean_object* v_a_103_){
_start:
{
lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4));
lean_inc(v_x_101_);
v___x_105_ = l_Lean_Syntax_isOfKind(v_x_101_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v_x_101_);
v___x_106_ = lean_box(0);
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_103_);
return v___x_107_;
}
else
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_108_ = lean_unsigned_to_nat(1u);
v___x_109_ = l_Lean_Syntax_getArg(v_x_101_, v___x_108_);
lean_dec(v_x_101_);
v___x_110_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_109_);
v___x_111_ = l_Lean_Syntax_matchesNull(v___x_109_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v___x_109_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_103_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = l_Lean_Syntax_getArg(v___x_109_, v___x_114_);
v___x_116_ = l_Lean_Syntax_getArg(v___x_109_, v___x_108_);
lean_dec(v___x_109_);
v___x_117_ = 0;
v___x_118_ = l_Lean_SourceInfo_fromRef(v_a_102_, v___x_117_);
v___x_119_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__2));
v___x_120_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8));
lean_inc_n(v___x_118_, 2);
v___x_121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_118_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15));
v___x_123_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_118_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
v___x_124_ = l_Lean_Syntax_node4(v___x_118_, v___x_119_, v___x_115_, v___x_121_, v___x_116_, v___x_123_);
v___x_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_103_);
return v___x_125_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_unexpander___boxed(lean_object* v_x_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_AddMonoidAlgebra_unexpander(v_x_126_, v_a_127_, v_a_128_);
lean_dec(v_a_127_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_delabOfCoeff(lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = l_Lean_PrettyPrinter_Delaborator_delabApp(v_a_130_, v_a_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_delabOfCoeff___boxed(lean_object* v_a_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_AddMonoidAlgebra_delabOfCoeff(v_a_138_, v_a_139_, v_a_140_, v_a_141_, v_a_142_, v_a_143_);
lean_dec(v_a_143_);
lean_dec_ref(v_a_142_);
lean_dec(v_a_141_);
lean_dec_ref(v_a_140_);
lean_dec(v_a_139_);
lean_dec_ref(v_a_138_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_delabOfCoeff(lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = l_Lean_PrettyPrinter_Delaborator_delabApp(v_a_146_, v_a_147_, v_a_148_, v_a_149_, v_a_150_, v_a_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_delabOfCoeff___boxed(lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_MonoidAlgebra_delabOfCoeff(v_a_154_, v_a_155_, v_a_156_, v_a_157_, v_a_158_, v_a_159_);
lean_dec(v_a_159_);
lean_dec_ref(v_a_158_);
lean_dec(v_a_157_);
lean_dec_ref(v_a_156_);
lean_dec(v_a_155_);
lean_dec_ref(v_a_154_);
return v_res_161_;
}
}
static lean_object* _init_lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = ((lean_object*)(lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__0));
v___x_173_ = l_String_toRawSubstring_x27(v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1(lean_object* v_x_187_, lean_object* v_a_188_, lean_object* v_a_189_){
_start:
{
lean_object* v___x_190_; uint8_t v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1));
lean_inc(v_x_187_);
v___x_191_ = l_Lean_Syntax_isOfKind(v_x_187_, v___x_190_);
if (v___x_191_ == 0)
{
lean_object* v___x_192_; lean_object* v___x_193_; 
lean_dec(v_x_187_);
v___x_192_ = lean_box(1);
v___x_193_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_192_);
lean_ctor_set(v___x_193_, 1, v_a_189_);
return v___x_193_;
}
else
{
lean_object* v_quotContext_194_; lean_object* v_currMacroScope_195_; lean_object* v_ref_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; uint8_t v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v_quotContext_194_ = lean_ctor_get(v_a_188_, 1);
v_currMacroScope_195_ = lean_ctor_get(v_a_188_, 2);
v_ref_196_ = lean_ctor_get(v_a_188_, 5);
v___x_197_ = lean_unsigned_to_nat(0u);
v___x_198_ = l_Lean_Syntax_getArg(v_x_187_, v___x_197_);
v___x_199_ = lean_unsigned_to_nat(2u);
v___x_200_ = l_Lean_Syntax_getArg(v_x_187_, v___x_199_);
lean_dec(v_x_187_);
v___x_201_ = 0;
v___x_202_ = l_Lean_SourceInfo_fromRef(v_ref_196_, v___x_201_);
v___x_203_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4));
v___x_204_ = lean_obj_once(&lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0, &lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0_once, _init_lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__0);
v___x_205_ = ((lean_object*)(lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__1));
lean_inc(v_currMacroScope_195_);
lean_inc(v_quotContext_194_);
v___x_206_ = l_Lean_addMacroScope(v_quotContext_194_, v___x_205_, v_currMacroScope_195_);
v___x_207_ = ((lean_object*)(lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___closed__5));
lean_inc_n(v___x_202_, 2);
v___x_208_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_208_, 0, v___x_202_);
lean_ctor_set(v___x_208_, 1, v___x_204_);
lean_ctor_set(v___x_208_, 2, v___x_206_);
lean_ctor_set(v___x_208_, 3, v___x_207_);
v___x_209_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__12));
v___x_210_ = l_Lean_Syntax_node2(v___x_202_, v___x_209_, v___x_198_, v___x_200_);
v___x_211_ = l_Lean_Syntax_node2(v___x_202_, v___x_203_, v___x_208_, v___x_210_);
v___x_212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v_a_189_);
return v___x_212_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1___boxed(lean_object* v_x_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_MonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__MonoidAlgebra__term_____x5b___x5d__1(v_x_213_, v_a_214_, v_a_215_);
lean_dec_ref(v_a_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_unexpander(lean_object* v_x_217_, lean_object* v_a_218_, lean_object* v_a_219_){
_start:
{
lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_220_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra___aux__Mathlib__Algebra__MonoidAlgebra__Defs______macroRules__AddMonoidAlgebra__term_____x5b___x5d__1___closed__4));
lean_inc(v_x_217_);
v___x_221_ = l_Lean_Syntax_isOfKind(v_x_217_, v___x_220_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
lean_dec(v_x_217_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_a_219_);
return v___x_223_;
}
else
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; uint8_t v___x_227_; 
v___x_224_ = lean_unsigned_to_nat(1u);
v___x_225_ = l_Lean_Syntax_getArg(v_x_217_, v___x_224_);
lean_dec(v_x_217_);
v___x_226_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_225_);
v___x_227_ = l_Lean_Syntax_matchesNull(v___x_225_, v___x_226_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; lean_object* v___x_229_; 
lean_dec(v___x_225_);
v___x_228_ = lean_box(0);
v___x_229_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v_a_219_);
return v___x_229_;
}
else
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; uint8_t v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_230_ = lean_unsigned_to_nat(0u);
v___x_231_ = l_Lean_Syntax_getArg(v___x_225_, v___x_230_);
v___x_232_ = l_Lean_Syntax_getArg(v___x_225_, v___x_224_);
lean_dec(v___x_225_);
v___x_233_ = 0;
v___x_234_ = l_Lean_SourceInfo_fromRef(v_a_218_, v___x_233_);
v___x_235_ = ((lean_object*)(lp_mathlib_MonoidAlgebra_term_____x5b___x5d___closed__1));
v___x_236_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__8));
lean_inc_n(v___x_234_, 2);
v___x_237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_234_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = ((lean_object*)(lp_mathlib_AddMonoidAlgebra_term_____x5b___x5d___closed__15));
v___x_239_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_239_, 0, v___x_234_);
lean_ctor_set(v___x_239_, 1, v___x_238_);
v___x_240_ = l_Lean_Syntax_node4(v___x_234_, v___x_235_, v___x_231_, v___x_237_, v___x_232_, v___x_239_);
v___x_241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_a_219_);
return v___x_241_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_unexpander___boxed(lean_object* v_x_242_, lean_object* v_a_243_, lean_object* v_a_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_MonoidAlgebra_unexpander(v_x_242_, v_a_243_, v_a_244_);
lean_dec(v_a_243_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0(lean_object* v_self_246_){
_start:
{
lean_inc_ref(v_self_246_);
return v_self_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0___boxed(lean_object* v_self_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_MonoidAlgebra_coeffEquiv___lam__0(v_self_247_);
lean_dec_ref(v_self_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1(lean_object* v_coeff_249_){
_start:
{
lean_inc_ref(v_coeff_249_);
return v_coeff_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1___boxed(lean_object* v_coeff_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_MonoidAlgebra_coeffEquiv___lam__1(v_coeff_250_);
lean_dec_ref(v_coeff_250_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv(lean_object* v_R_257_, lean_object* v_M_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = ((lean_object*)(lp_mathlib_MonoidAlgebra_coeffEquiv___closed__2));
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_coeffEquiv___boxed(lean_object* v_R_261_, lean_object* v_M_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_MonoidAlgebra_coeffEquiv(v_R_261_, v_M_262_, v_inst_263_);
lean_dec_ref(v_inst_263_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_coeffEquiv(lean_object* v_R_265_, lean_object* v_M_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = ((lean_object*)(lp_mathlib_MonoidAlgebra_coeffEquiv___closed__2));
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_coeffEquiv___boxed(lean_object* v_R_269_, lean_object* v_M_270_, lean_object* v_inst_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_AddMonoidAlgebra_coeffEquiv(v_R_269_, v_M_270_, v_inst_271_);
lean_dec_ref(v_inst_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0(lean_object* v_toZero_273_, lean_object* v_x_274_){
_start:
{
lean_inc(v_toZero_273_);
return v_toZero_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed(lean_object* v_toZero_275_, lean_object* v_x_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0(v_toZero_275_, v_x_276_);
lean_dec(v_x_276_);
lean_dec(v_toZero_275_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited___redArg(lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v_toFun_281_; lean_object* v___x_282_; lean_object* v_toZero_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_293_; 
v___x_279_ = lp_mathlib_MonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_278_);
v___x_280_ = lp_mathlib_Equiv_symm___redArg(v___x_279_);
v_toFun_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_toFun_281_);
lean_dec_ref(v___x_280_);
v___x_282_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_278_);
v_toZero_283_ = lean_ctor_get(v___x_282_, 1);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_293_ == 0)
{
lean_object* v_unused_294_; 
v_unused_294_ = lean_ctor_get(v___x_282_, 0);
lean_dec(v_unused_294_);
v___x_285_ = v___x_282_;
v_isShared_286_ = v_isSharedCheck_293_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_toZero_283_);
lean_dec(v___x_282_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_293_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___f_287_; lean_object* v___x_288_; lean_object* v___x_290_; 
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_287_, 0, v_toZero_283_);
v___x_288_ = lean_box(0);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 1, v___f_287_);
lean_ctor_set(v___x_285_, 0, v___x_288_);
v___x_290_ = v___x_285_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___x_288_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v___f_287_);
v___x_290_ = v_reuseFailAlloc_292_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_291_; 
v___x_291_ = lean_apply_1(v_toFun_281_, v___x_290_);
return v___x_291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instInhabited(lean_object* v_R_295_, lean_object* v_M_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_mathlib_MonoidAlgebra_instInhabited___redArg(v_inst_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instInhabited___redArg(lean_object* v_inst_299_){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v_toFun_302_; lean_object* v___x_303_; lean_object* v_toZero_304_; lean_object* v___x_306_; uint8_t v_isShared_307_; uint8_t v_isSharedCheck_314_; 
v___x_300_ = lp_mathlib_AddMonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_299_);
v___x_301_ = lp_mathlib_Equiv_symm___redArg(v___x_300_);
v_toFun_302_ = lean_ctor_get(v___x_301_, 0);
lean_inc(v_toFun_302_);
lean_dec_ref(v___x_301_);
v___x_303_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_299_);
v_toZero_304_ = lean_ctor_get(v___x_303_, 1);
v_isSharedCheck_314_ = !lean_is_exclusive(v___x_303_);
if (v_isSharedCheck_314_ == 0)
{
lean_object* v_unused_315_; 
v_unused_315_ = lean_ctor_get(v___x_303_, 0);
lean_dec(v_unused_315_);
v___x_306_ = v___x_303_;
v_isShared_307_ = v_isSharedCheck_314_;
goto v_resetjp_305_;
}
else
{
lean_inc(v_toZero_304_);
lean_dec(v___x_303_);
v___x_306_ = lean_box(0);
v_isShared_307_ = v_isSharedCheck_314_;
goto v_resetjp_305_;
}
v_resetjp_305_:
{
lean_object* v___f_308_; lean_object* v___x_309_; lean_object* v___x_311_; 
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_308_, 0, v_toZero_304_);
v___x_309_ = lean_box(0);
if (v_isShared_307_ == 0)
{
lean_ctor_set(v___x_306_, 1, v___f_308_);
lean_ctor_set(v___x_306_, 0, v___x_309_);
v___x_311_ = v___x_306_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_309_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v___f_308_);
v___x_311_ = v_reuseFailAlloc_313_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
lean_object* v___x_312_; 
v___x_312_ = lean_apply_1(v_toFun_302_, v___x_311_);
return v___x_312_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instInhabited(lean_object* v_R_316_, lean_object* v_M_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_mathlib_AddMonoidAlgebra_instInhabited___redArg(v_inst_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instUnique___redArg(lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; lean_object* v_toZero_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v_toFun_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_335_; 
lean_inc_ref(v_inst_320_);
v___x_321_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_320_);
v_toZero_322_ = lean_ctor_get(v___x_321_, 1);
lean_inc(v_toZero_322_);
lean_dec_ref(v___x_321_);
v___x_323_ = lp_mathlib_MonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_320_);
lean_dec_ref(v_inst_320_);
v___x_324_ = lp_mathlib_Equiv_symm___redArg(v___x_323_);
v_toFun_325_ = lean_ctor_get(v___x_324_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_335_ == 0)
{
lean_object* v_unused_336_; 
v_unused_336_ = lean_ctor_get(v___x_324_, 1);
lean_dec(v_unused_336_);
v___x_327_ = v___x_324_;
v_isShared_328_ = v_isSharedCheck_335_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_toFun_325_);
lean_dec(v___x_324_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_335_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
lean_object* v___f_329_; lean_object* v___x_330_; lean_object* v___x_332_; 
v___f_329_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_329_, 0, v_toZero_322_);
v___x_330_ = lean_box(0);
if (v_isShared_328_ == 0)
{
lean_ctor_set(v___x_327_, 1, v___f_329_);
lean_ctor_set(v___x_327_, 0, v___x_330_);
v___x_332_ = v___x_327_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_330_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v___f_329_);
v___x_332_ = v_reuseFailAlloc_334_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
lean_object* v___x_333_; 
v___x_333_ = lean_apply_1(v_toFun_325_, v___x_332_);
return v___x_333_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instUnique(lean_object* v_R_337_, lean_object* v_M_338_, lean_object* v_inst_339_, lean_object* v_inst_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_MonoidAlgebra_instUnique___redArg(v_inst_339_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instUnique___redArg(lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; lean_object* v_toZero_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v_toFun_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_357_; 
lean_inc_ref(v_inst_342_);
v___x_343_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_342_);
v_toZero_344_ = lean_ctor_get(v___x_343_, 1);
lean_inc(v_toZero_344_);
lean_dec_ref(v___x_343_);
v___x_345_ = lp_mathlib_AddMonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_342_);
lean_dec_ref(v_inst_342_);
v___x_346_ = lp_mathlib_Equiv_symm___redArg(v___x_345_);
v_toFun_347_ = lean_ctor_get(v___x_346_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_346_);
if (v_isSharedCheck_357_ == 0)
{
lean_object* v_unused_358_; 
v_unused_358_ = lean_ctor_get(v___x_346_, 1);
lean_dec(v_unused_358_);
v___x_349_ = v___x_346_;
v_isShared_350_ = v_isSharedCheck_357_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_toFun_347_);
lean_dec(v___x_346_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_357_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___f_351_; lean_object* v___x_352_; lean_object* v___x_354_; 
v___f_351_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_351_, 0, v_toZero_344_);
v___x_352_ = lean_box(0);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v___f_351_);
lean_ctor_set(v___x_349_, 0, v___x_352_);
v___x_354_ = v___x_349_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v___x_352_);
lean_ctor_set(v_reuseFailAlloc_356_, 1, v___f_351_);
v___x_354_ = v_reuseFailAlloc_356_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
lean_object* v___x_355_; 
v___x_355_ = lean_apply_1(v_toFun_347_, v___x_354_);
return v___x_355_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instUnique(lean_object* v_R_359_, lean_object* v_M_360_, lean_object* v_inst_361_, lean_object* v_inst_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = lp_mathlib_AddMonoidAlgebra_instUnique___redArg(v_inst_361_);
return v___x_363_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0(lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_a_366_, lean_object* v_b_367_){
_start:
{
uint8_t v___x_368_; 
v___x_368_ = lp_mathlib_Finsupp_instDecidableEq___redArg(v_inst_364_, v_inst_365_, v_a_366_, v_b_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0___boxed(lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_a_371_, lean_object* v_b_372_){
_start:
{
uint8_t v_res_373_; lean_object* v_r_374_; 
v_res_373_ = lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0(v_inst_369_, v_inst_370_, v_a_371_, v_b_372_);
v_r_374_ = lean_box(v_res_373_);
return v_r_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__1(lean_object* v___x_375_, lean_object* v___y_376_){
_start:
{
lean_object* v_toFun_377_; lean_object* v___x_378_; 
v_toFun_377_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_toFun_377_);
lean_dec_ref(v___x_375_);
v___x_378_ = lean_apply_1(v_toFun_377_, v___y_376_);
return v___x_378_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq___redArg(lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_a_382_, lean_object* v_b_383_){
_start:
{
lean_object* v___f_384_; lean_object* v___x_385_; lean_object* v___f_386_; uint8_t v___x_387_; 
v___f_384_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_384_, 0, v_inst_381_);
lean_closure_set(v___f_384_, 1, v_inst_380_);
v___x_385_ = lp_mathlib_MonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_379_);
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__1), 2, 1);
lean_closure_set(v___f_386_, 0, v___x_385_);
v___x_387_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_386_, v___f_384_, v_a_382_, v_b_383_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___boxed(lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_a_391_, lean_object* v_b_392_){
_start:
{
uint8_t v_res_393_; lean_object* v_r_394_; 
v_res_393_ = lp_mathlib_MonoidAlgebra_instDecidableEq___redArg(v_inst_388_, v_inst_389_, v_inst_390_, v_a_391_, v_b_392_);
lean_dec_ref(v_inst_388_);
v_r_394_ = lean_box(v_res_393_);
return v_r_394_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidAlgebra_instDecidableEq(lean_object* v_R_395_, lean_object* v_M_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_a_400_, lean_object* v_b_401_){
_start:
{
uint8_t v___x_402_; 
v___x_402_ = lp_mathlib_MonoidAlgebra_instDecidableEq___redArg(v_inst_397_, v_inst_398_, v_inst_399_, v_a_400_, v_b_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidAlgebra_instDecidableEq___boxed(lean_object* v_R_403_, lean_object* v_M_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_a_408_, lean_object* v_b_409_){
_start:
{
uint8_t v_res_410_; lean_object* v_r_411_; 
v_res_410_ = lp_mathlib_MonoidAlgebra_instDecidableEq(v_R_403_, v_M_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_a_408_, v_b_409_);
lean_dec_ref(v_inst_405_);
v_r_411_ = lean_box(v_res_410_);
return v_r_411_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_a_415_, lean_object* v_b_416_){
_start:
{
lean_object* v___f_417_; lean_object* v___x_418_; lean_object* v___f_419_; uint8_t v___x_420_; 
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_417_, 0, v_inst_414_);
lean_closure_set(v___f_417_, 1, v_inst_413_);
v___x_418_ = lp_mathlib_AddMonoidAlgebra_coeffEquiv(lean_box(0), lean_box(0), v_inst_412_);
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_MonoidAlgebra_instDecidableEq___redArg___lam__1), 2, 1);
lean_closure_set(v___f_419_, 0, v___x_418_);
v___x_420_ = lp_mathlib_Function_Injective_decidableEq___redArg(v___f_419_, v___f_417_, v_a_415_, v_b_416_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg___boxed(lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_a_424_, lean_object* v_b_425_){
_start:
{
uint8_t v_res_426_; lean_object* v_r_427_; 
v_res_426_ = lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(v_inst_421_, v_inst_422_, v_inst_423_, v_a_424_, v_b_425_);
lean_dec_ref(v_inst_421_);
v_r_427_ = lean_box(v_res_426_);
return v_r_427_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidAlgebra_instDecidableEq(lean_object* v_R_428_, lean_object* v_M_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_a_433_, lean_object* v_b_434_){
_start:
{
uint8_t v___x_435_; 
v___x_435_ = lp_mathlib_AddMonoidAlgebra_instDecidableEq___redArg(v_inst_430_, v_inst_431_, v_inst_432_, v_a_433_, v_b_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidAlgebra_instDecidableEq___boxed(lean_object* v_R_436_, lean_object* v_M_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_a_441_, lean_object* v_b_442_){
_start:
{
uint8_t v_res_443_; lean_object* v_r_444_; 
v_res_443_ = lp_mathlib_AddMonoidAlgebra_instDecidableEq(v_R_436_, v_M_437_, v_inst_438_, v_inst_439_, v_inst_440_, v_a_441_, v_b_442_);
lean_dec_ref(v_inst_438_);
v_r_444_ = lean_box(v_res_443_);
return v_r_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg___lam__0(lean_object* v_inst_445_, lean_object* v_m_446_, lean_object* v_a_447_){
_start:
{
lean_object* v_val_448_; lean_object* v___x_449_; 
v_val_448_ = lean_ctor_get(v_m_446_, 0);
lean_inc(v_val_448_);
lean_dec_ref(v_m_446_);
v___x_449_ = lean_apply_2(v_inst_445_, v_val_448_, v_a_447_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg(lean_object* v_inst_450_){
_start:
{
lean_object* v___f_451_; 
v___f_451_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg___lam__0), 3, 1);
lean_closure_set(v___f_451_, 0, v_inst_450_);
return v___f_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits(lean_object* v_M_452_, lean_object* v_00_u03b1_453_, lean_object* v_inst_454_, lean_object* v_inst_455_){
_start:
{
lean_object* v___f_456_; 
v___f_456_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___redArg___lam__0), 3, 1);
lean_closure_set(v___f_456_, 0, v_inst_455_);
return v___f_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits___boxed(lean_object* v_M_457_, lean_object* v_00_u03b1_458_, lean_object* v_inst_459_, lean_object* v_inst_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib___private_Mathlib_Algebra_MonoidAlgebra_Defs_0__MonoidAlgebra_instSMulUnits(v_M_457_, v_00_u03b1_458_, v_inst_459_, v_inst_460_);
lean_dec_ref(v_inst_459_);
return v_res_461_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_TransferInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_MonoidAlgebra_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
