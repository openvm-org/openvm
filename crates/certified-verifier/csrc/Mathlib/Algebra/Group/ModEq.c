// Lean compiler output
// Module: Mathlib.Algebra.Group.ModEq
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs import Mathlib.Algebra.Group.Torsion import Mathlib.Tactic.TermCongr import Mathlib.Tactic.Use
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__0 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__0_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 14, .m_data = "term_≡_[PMOD_]"};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__1 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(28, 66, 9, 17, 212, 36, 3, 48)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__3 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≡ "};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__5 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__5_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__6 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__6_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__7 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__8 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__9 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__9_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__6_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__9_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__10 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__10_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " [PMOD "};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__11 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__12 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__10_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__12_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__13 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__13_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__9_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__14 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__14_value;
static const lean_string_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__15 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__15_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__15_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__16 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__4_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__14_value),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__16_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__17 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__17_value;
static const lean_ctor_object lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__17_value)}};
static const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__18 = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d = (const lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__18_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__0 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__0_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__1 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__2 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__2_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__3 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ModEq"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5_value;
static lean_once_cell_t lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(168, 164, 143, 99, 107, 224, 167, 149)}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__7 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__7_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 118, 107, 168, 157, 101, 14, 14)}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__9 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__10 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__10_value;
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__11 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__11_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__12 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__0 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__0_value;
static const lean_ctor_object lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__1 = (const lean_object*)&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__5));
v___x_57_ = l_String_toRawSubstring_x27(v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1(lean_object* v_x_72_, lean_object* v_a_73_, lean_object* v_a_74_){
_start:
{
lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2));
lean_inc(v_x_72_);
v___x_76_ = l_Lean_Syntax_isOfKind(v_x_72_, v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; lean_object* v___x_78_; 
lean_dec(v_x_72_);
v___x_77_ = lean_box(1);
v___x_78_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_74_);
return v___x_78_;
}
else
{
lean_object* v_quotContext_79_; lean_object* v_currMacroScope_80_; lean_object* v_ref_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v_quotContext_79_ = lean_ctor_get(v_a_73_, 1);
v_currMacroScope_80_ = lean_ctor_get(v_a_73_, 2);
v_ref_81_ = lean_ctor_get(v_a_73_, 5);
v___x_82_ = lean_unsigned_to_nat(0u);
v___x_83_ = l_Lean_Syntax_getArg(v_x_72_, v___x_82_);
v___x_84_ = lean_unsigned_to_nat(2u);
v___x_85_ = l_Lean_Syntax_getArg(v_x_72_, v___x_84_);
v___x_86_ = lean_unsigned_to_nat(4u);
v___x_87_ = l_Lean_Syntax_getArg(v_x_72_, v___x_86_);
lean_dec(v_x_72_);
v___x_88_ = 0;
v___x_89_ = l_Lean_SourceInfo_fromRef(v_ref_81_, v___x_88_);
v___x_90_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4));
v___x_91_ = lean_obj_once(&lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6, &lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6_once, _init_lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__6);
v___x_92_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__7));
lean_inc(v_currMacroScope_80_);
lean_inc(v_quotContext_79_);
v___x_93_ = l_Lean_addMacroScope(v_quotContext_79_, v___x_92_, v_currMacroScope_80_);
v___x_94_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__10));
lean_inc_n(v___x_89_, 2);
v___x_95_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_95_, 0, v___x_89_);
lean_ctor_set(v___x_95_, 1, v___x_91_);
lean_ctor_set(v___x_95_, 2, v___x_93_);
lean_ctor_set(v___x_95_, 3, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__12));
v___x_97_ = l_Lean_Syntax_node3(v___x_89_, v___x_96_, v___x_87_, v___x_83_, v___x_85_);
v___x_98_ = l_Lean_Syntax_node2(v___x_89_, v___x_90_, v___x_95_, v___x_97_);
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_74_);
return v___x_99_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___boxed(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1(v_x_100_, v_a_101_, v_a_102_);
lean_dec_ref(v_a_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1(lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______macroRules__AddCommGroup__term___u2261___x5bPMOD___x5d__1___closed__4));
lean_inc(v_x_107_);
v___x_111_ = l_Lean_Syntax_isOfKind(v_x_107_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v_x_107_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_109_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_114_ = lean_unsigned_to_nat(0u);
v___x_115_ = l_Lean_Syntax_getArg(v_x_107_, v___x_114_);
v___x_116_ = ((lean_object*)(lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___closed__1));
lean_inc(v___x_115_);
v___x_117_ = l_Lean_Syntax_isOfKind(v___x_115_, v___x_116_);
if (v___x_117_ == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v___x_115_);
lean_dec(v_x_107_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_109_);
return v___x_119_;
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_120_ = lean_unsigned_to_nat(1u);
v___x_121_ = l_Lean_Syntax_getArg(v_x_107_, v___x_120_);
lean_dec(v_x_107_);
v___x_122_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_121_);
v___x_123_ = l_Lean_Syntax_matchesNull(v___x_121_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_dec(v___x_121_);
lean_dec(v___x_115_);
v___x_124_ = lean_box(0);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v_a_109_);
return v___x_125_;
}
else
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v_ref_130_; uint8_t v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_126_ = l_Lean_Syntax_getArg(v___x_121_, v___x_114_);
v___x_127_ = l_Lean_Syntax_getArg(v___x_121_, v___x_120_);
v___x_128_ = lean_unsigned_to_nat(2u);
v___x_129_ = l_Lean_Syntax_getArg(v___x_121_, v___x_128_);
lean_dec(v___x_121_);
v_ref_130_ = l_Lean_replaceRef(v___x_115_, v_a_108_);
lean_dec(v___x_115_);
v___x_131_ = 0;
v___x_132_ = l_Lean_SourceInfo_fromRef(v_ref_130_, v___x_131_);
lean_dec(v_ref_130_);
v___x_133_ = ((lean_object*)(lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__2));
v___x_134_ = ((lean_object*)(lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__5));
lean_inc_n(v___x_132_, 3);
v___x_135_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_132_);
lean_ctor_set(v___x_135_, 1, v___x_134_);
v___x_136_ = ((lean_object*)(lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__11));
v___x_137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_132_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = ((lean_object*)(lp_mathlib_AddCommGroup_term___u2261___x5bPMOD___x5d___closed__15));
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_132_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = l_Lean_Syntax_node6(v___x_132_, v___x_133_, v___x_127_, v___x_135_, v___x_129_, v___x_137_, v___x_126_, v___x_139_);
v___x_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
lean_ctor_set(v___x_141_, 1, v_a_109_);
return v___x_141_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1___boxed(lean_object* v_x_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_AddCommGroup___aux__Mathlib__Algebra__Group__ModEq______unexpand__AddCommGroup__ModEq__1(v_x_142_, v_a_143_, v_a_144_);
lean_dec(v_a_143_);
return v_res_145_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_ModEq(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_ModEq(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_ModEq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_ModEq(builtin);
}
#ifdef __cplusplus
}
#endif
