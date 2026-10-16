// Lean compiler output
// Module: Mathlib.Data.Int.ModEq
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.ModEq import Mathlib.Data.Int.Cast.Lemmas
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
lean_object* lean_int_emod(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 14, .m_data = "term_≡_[ZMOD_]"};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__0 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 117, 211, 115, 37, 198, 98, 111)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1_value;
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__2 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value;
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≡ "};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__4 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__4_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__5 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__5_value;
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__6 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__7 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__8 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__5_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__9 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__9_value;
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " [ZMOD "};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__10 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__10_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__10_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__11 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__9_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__12 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__12_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__13 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__13_value;
static const lean_string_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__14 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__14_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__14_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__15 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__15_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__3_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__13_value),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__15_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__16 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_term___u2261___x5bZMOD___x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__16_value)}};
static const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d___closed__17 = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2261___x5bZMOD___x5d = (const lean_object*)&lp_mathlib_term___u2261___x5bZMOD___x5d___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Int.ModEq"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ModEq"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(166, 3, 119, 197, 26, 253, 47, 159)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidableModEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidableModEq___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__5));
v___x_55_ = l_String_toRawSubstring_x27(v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1(lean_object* v_x_70_, lean_object* v_a_71_, lean_object* v_a_72_){
_start:
{
lean_object* v___x_73_; uint8_t v___x_74_; 
v___x_73_ = ((lean_object*)(lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1));
lean_inc(v_x_70_);
v___x_74_ = l_Lean_Syntax_isOfKind(v_x_70_, v___x_73_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; lean_object* v___x_76_; 
lean_dec(v_x_70_);
v___x_75_ = lean_box(1);
v___x_76_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v_a_72_);
return v___x_76_;
}
else
{
lean_object* v_quotContext_77_; lean_object* v_currMacroScope_78_; lean_object* v_ref_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v_quotContext_77_ = lean_ctor_get(v_a_71_, 1);
v_currMacroScope_78_ = lean_ctor_get(v_a_71_, 2);
v_ref_79_ = lean_ctor_get(v_a_71_, 5);
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = l_Lean_Syntax_getArg(v_x_70_, v___x_80_);
v___x_82_ = lean_unsigned_to_nat(2u);
v___x_83_ = l_Lean_Syntax_getArg(v_x_70_, v___x_82_);
v___x_84_ = lean_unsigned_to_nat(4u);
v___x_85_ = l_Lean_Syntax_getArg(v_x_70_, v___x_84_);
lean_dec(v_x_70_);
v___x_86_ = 0;
v___x_87_ = l_Lean_SourceInfo_fromRef(v_ref_79_, v___x_86_);
v___x_88_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4));
v___x_89_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6, &lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__6);
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__9));
lean_inc(v_currMacroScope_78_);
lean_inc(v_quotContext_77_);
v___x_91_ = l_Lean_addMacroScope(v_quotContext_77_, v___x_90_, v_currMacroScope_78_);
v___x_92_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__11));
lean_inc_n(v___x_87_, 2);
v___x_93_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_93_, 0, v___x_87_);
lean_ctor_set(v___x_93_, 1, v___x_89_);
lean_ctor_set(v___x_93_, 2, v___x_91_);
lean_ctor_set(v___x_93_, 3, v___x_92_);
v___x_94_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__13));
v___x_95_ = l_Lean_Syntax_node3(v___x_87_, v___x_94_, v___x_85_, v___x_81_, v___x_83_);
v___x_96_ = l_Lean_Syntax_node2(v___x_87_, v___x_88_, v___x_93_, v___x_95_);
v___x_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v_a_72_);
return v___x_97_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___boxed(lean_object* v_x_98_, lean_object* v_a_99_, lean_object* v_a_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1(v_x_98_, v_a_99_, v_a_100_);
lean_dec_ref(v_a_99_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1(lean_object* v_x_105_, lean_object* v_a_106_, lean_object* v_a_107_){
_start:
{
lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_108_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______macroRules__term___u2261___x5bZMOD___x5d__1___closed__4));
lean_inc(v_x_105_);
v___x_109_ = l_Lean_Syntax_isOfKind(v_x_105_, v___x_108_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; lean_object* v___x_111_; 
lean_dec(v_x_105_);
v___x_110_ = lean_box(0);
v___x_111_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v_a_107_);
return v___x_111_;
}
else
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = l_Lean_Syntax_getArg(v_x_105_, v___x_112_);
v___x_114_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___closed__1));
lean_inc(v___x_113_);
v___x_115_ = l_Lean_Syntax_isOfKind(v___x_113_, v___x_114_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_117_; 
lean_dec(v___x_113_);
lean_dec(v_x_105_);
v___x_116_ = lean_box(0);
v___x_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_107_);
return v___x_117_;
}
else
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_118_ = lean_unsigned_to_nat(1u);
v___x_119_ = l_Lean_Syntax_getArg(v_x_105_, v___x_118_);
lean_dec(v_x_105_);
v___x_120_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_119_);
v___x_121_ = l_Lean_Syntax_matchesNull(v___x_119_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec(v___x_119_);
lean_dec(v___x_113_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_107_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v_ref_128_; uint8_t v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_124_ = l_Lean_Syntax_getArg(v___x_119_, v___x_112_);
v___x_125_ = l_Lean_Syntax_getArg(v___x_119_, v___x_118_);
v___x_126_ = lean_unsigned_to_nat(2u);
v___x_127_ = l_Lean_Syntax_getArg(v___x_119_, v___x_126_);
lean_dec(v___x_119_);
v_ref_128_ = l_Lean_replaceRef(v___x_113_, v_a_106_);
lean_dec(v___x_113_);
v___x_129_ = 0;
v___x_130_ = l_Lean_SourceInfo_fromRef(v_ref_128_, v___x_129_);
lean_dec(v_ref_128_);
v___x_131_ = ((lean_object*)(lp_mathlib_term___u2261___x5bZMOD___x5d___closed__1));
v___x_132_ = ((lean_object*)(lp_mathlib_term___u2261___x5bZMOD___x5d___closed__4));
lean_inc_n(v___x_130_, 3);
v___x_133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_130_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
v___x_134_ = ((lean_object*)(lp_mathlib_term___u2261___x5bZMOD___x5d___closed__10));
v___x_135_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_130_);
lean_ctor_set(v___x_135_, 1, v___x_134_);
v___x_136_ = ((lean_object*)(lp_mathlib_term___u2261___x5bZMOD___x5d___closed__14));
v___x_137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_130_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = l_Lean_Syntax_node6(v___x_130_, v___x_131_, v___x_125_, v___x_133_, v___x_127_, v___x_135_, v___x_124_, v___x_137_);
v___x_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v_a_107_);
return v___x_139_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1___boxed(lean_object* v_x_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib___aux__Mathlib__Data__Int__ModEq______unexpand__Int__ModEq__1(v_x_140_, v_a_141_, v_a_142_);
lean_dec(v_a_141_);
return v_res_143_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidableModEq(lean_object* v_n_144_, lean_object* v_a_145_, lean_object* v_b_146_){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; uint8_t v___x_149_; 
v___x_147_ = lean_int_emod(v_a_145_, v_n_144_);
v___x_148_ = lean_int_emod(v_b_146_, v_n_144_);
v___x_149_ = lean_int_dec_eq(v___x_147_, v___x_148_);
lean_dec(v___x_148_);
lean_dec(v___x_147_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidableModEq___boxed(lean_object* v_n_150_, lean_object* v_a_151_, lean_object* v_b_152_){
_start:
{
uint8_t v_res_153_; lean_object* v_r_154_; 
v_res_153_ = lp_mathlib_Int_instDecidableModEq(v_n_150_, v_a_151_, v_b_152_);
lean_dec(v_b_152_);
lean_dec(v_a_151_);
lean_dec(v_n_150_);
v_r_154_ = lean_box(v_res_153_);
return v_r_154_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_ModEq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_ModEq(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_ModEq(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_ModEq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_ModEq(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_ModEq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_ModEq(builtin);
}
#ifdef __cplusplus
}
#endif
