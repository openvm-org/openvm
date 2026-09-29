// Lean compiler output
// Module: Mathlib.Order.RelIso.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Embedding.Basic public import Mathlib.Order.RelClasses
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
lean_object* l_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_emptySum(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumEmpty(lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ofUnique___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_refl___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_term___u2192r___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_→r_"};
static const lean_object* lp_mathlib_term___u2192r___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192r___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(202, 19, 85, 120, 101, 228, 31, 249)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192r___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192r___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192r___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192r___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →r "};
static const lean_object* lp_mathlib_term___u2192r___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192r___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192r___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192r___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192r___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192r___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192r___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192r___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192r___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192r__ = (const lean_object*)&lp_mathlib_term___u2192r___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RelHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(57, 112, 46, 5, 50, 75, 84, 50)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_RelHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelHom_id___closed__0 = (const lean_object*)&lp_mathlib_RelHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelHom_swapEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelHom_swap, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelHom_swapEquiv___closed__0 = (const lean_object*)&lp_mathlib_RelHom_swapEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_RelHom_swapEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelHom_swapEquiv___closed__0_value),((lean_object*)&lp_mathlib_RelHom_swapEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_RelHom_swapEquiv___closed__1 = (const lean_object*)&lp_mathlib_RelHom_swapEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swapEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u21aar___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_↪r_"};
static const lean_object* lp_mathlib_term___u21aar___00__closed__0 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u21aar___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21aar___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 57, 63, 179, 181, 39, 5, 248)}};
static const lean_object* lp_mathlib_term___u21aar___00__closed__1 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u21aar___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ↪r "};
static const lean_object* lp_mathlib_term___u21aar___00__closed__2 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u21aar___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u21aar___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u21aar___00__closed__3 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u21aar___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21aar___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192r___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u21aar___00__closed__4 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u21aar___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u21aar___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21aar___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u21aar___00__closed__5 = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u21aar__ = (const lean_object*)&lp_mathlib_term___u21aar___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "RelEmbedding"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(228, 1, 194, 179, 83, 180, 169, 133)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelEmbedding__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelEmbedding__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_instCoeRelHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelEmbedding_toRelHom___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelEmbedding_instCoeRelHom___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_instCoeRelHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instCoeRelHom(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_refl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_refl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelEmbedding_refl___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_refl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_refl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_swapEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelEmbedding_swap___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelEmbedding_swapEquiv___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_swapEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_RelEmbedding_swapEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelEmbedding_swapEquiv___closed__0_value),((lean_object*)&lp_mathlib_RelEmbedding_swapEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_RelEmbedding_swapEquiv___closed__1 = (const lean_object*)&lp_mathlib_RelEmbedding_swapEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swapEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subtype_relEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subtype_relEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subtype_relEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_relEmbedding(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_mkRelHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_mkRelHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_ofIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelEmbedding_ofIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_ofIsEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInl___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_sumLiftRelInl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelEmbedding_sumLiftRelInl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelEmbedding_sumLiftRelInl___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_sumLiftRelInl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInr___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_RelEmbedding_sumLiftRelInr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelEmbedding_sumLiftRelInr___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelEmbedding_sumLiftRelInr___closed__0 = (const lean_object*)&lp_mathlib_RelEmbedding_sumLiftRelInr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexInl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexInr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243r___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≃r_"};
static const lean_object* lp_mathlib_term___u2243r___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243r___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243r___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 128, 166, 76, 143, 235, 217, 17)}};
static const lean_object* lp_mathlib_term___u2243r___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243r___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≃r "};
static const lean_object* lp_mathlib_term___u2243r___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243r___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243r___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243r___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243r___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192r___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243r___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192r___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243r___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243r___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243r___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243r___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243r___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243r__ = (const lean_object*)&lp_mathlib_term___u2243r___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RelIso"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 94, 135, 138, 246, 71, 186, 130)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelIso__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelIso__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toRelEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toRelEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelIso_instCoeOutRelEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelIso_toRelEmbedding, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelIso_instCoeOutRelEmbedding___closed__0 = (const lean_object*)&lp_mathlib_RelIso_instCoeOutRelEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instCoeOutRelEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_refl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instInhabited(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_cast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_cast___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelIso_swapEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelIso_swap, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelIso_swapEquiv___closed__0 = (const lean_object*)&lp_mathlib_RelIso_swapEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_RelIso_swapEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelIso_swapEquiv___closed__0_value),((lean_object*)&lp_mathlib_RelIso_swapEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_RelIso_swapEquiv___closed__1 = (const lean_object*)&lp_mathlib_RelIso_swapEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swapEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_compl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_compl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelIso_complEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelIso_complEquiv___closed__0 = (const lean_object*)&lp_mathlib_RelIso_complEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_RelIso_complEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelIso_compl, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_RelIso_complEquiv___closed__1 = (const lean_object*)&lp_mathlib_RelIso_complEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_RelIso_complEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelIso_complEquiv___closed__1_value),((lean_object*)&lp_mathlib_RelIso_complEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_RelIso_complEquiv___closed__2 = (const lean_object*)&lp_mathlib_RelIso_complEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_complEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_prodLexCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_prodLexCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_sumLexEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_sumLexEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_RelIso_emptySumLex___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RelIso_emptySumLex___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_RelIso_emptySumLex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfIrrefl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfIrrefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfRefl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfRefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__5));
v___x_36_ = l_String_toRawSubstring_x27(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib_term___u2192r___00__closed__1));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(1);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_quotContext_60_; lean_object* v_currMacroScope_61_; lean_object* v_ref_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_quotContext_60_ = lean_ctor_get(v_a_54_, 1);
v_currMacroScope_61_ = lean_ctor_get(v_a_54_, 2);
v_ref_62_ = lean_ctor_get(v_a_54_, 5);
v___x_63_ = lean_unsigned_to_nat(0u);
v___x_64_ = l_Lean_Syntax_getArg(v_x_53_, v___x_63_);
v___x_65_ = lean_unsigned_to_nat(2u);
v___x_66_ = l_Lean_Syntax_getArg(v_x_53_, v___x_65_);
lean_dec(v_x_53_);
v___x_67_ = 0;
v___x_68_ = l_Lean_SourceInfo_fromRef(v_ref_62_, v___x_67_);
v___x_69_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
v___x_70_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6, &lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__6);
v___x_71_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__7));
lean_inc(v_currMacroScope_61_);
lean_inc(v_quotContext_60_);
v___x_72_ = l_Lean_addMacroScope(v_quotContext_60_, v___x_71_, v_currMacroScope_61_);
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__11));
lean_inc_n(v___x_68_, 2);
v___x_74_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_74_, 0, v___x_68_);
lean_ctor_set(v___x_74_, 1, v___x_70_);
lean_ctor_set(v___x_74_, 2, v___x_72_);
lean_ctor_set(v___x_74_, 3, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13));
v___x_76_ = l_Lean_Syntax_node2(v___x_68_, v___x_75_, v___x_64_, v___x_66_);
v___x_77_ = l_Lean_Syntax_node2(v___x_68_, v___x_69_, v___x_74_, v___x_76_);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_55_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___boxed(lean_object* v_x_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1(v_x_79_, v_a_80_, v_a_81_);
lean_dec_ref(v_a_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
lean_inc(v_x_86_);
v___x_90_ = l_Lean_Syntax_isOfKind(v_x_86_, v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v_x_86_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v_a_88_);
return v___x_92_;
}
else
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_86_, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1));
lean_inc(v___x_94_);
v___x_96_ = l_Lean_Syntax_isOfKind(v___x_94_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v___x_94_);
lean_dec(v_x_86_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_88_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_99_ = lean_unsigned_to_nat(1u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_86_, v___x_99_);
lean_dec(v_x_86_);
v___x_101_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_100_);
v___x_102_ = l_Lean_Syntax_matchesNull(v___x_100_, v___x_101_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v___x_100_);
lean_dec(v___x_94_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v_a_88_);
return v___x_104_;
}
else
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v_ref_107_; uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_105_ = l_Lean_Syntax_getArg(v___x_100_, v___x_93_);
v___x_106_ = l_Lean_Syntax_getArg(v___x_100_, v___x_99_);
lean_dec(v___x_100_);
v_ref_107_ = l_Lean_replaceRef(v___x_94_, v_a_87_);
lean_dec(v___x_94_);
v___x_108_ = 0;
v___x_109_ = l_Lean_SourceInfo_fromRef(v_ref_107_, v___x_108_);
lean_dec(v_ref_107_);
v___x_110_ = ((lean_object*)(lp_mathlib_term___u2192r___00__closed__1));
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192r___00__closed__4));
lean_inc(v___x_109_);
v___x_112_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_109_);
lean_ctor_set(v___x_112_, 1, v___x_111_);
v___x_113_ = l_Lean_Syntax_node3(v___x_109_, v___x_110_, v___x_105_, v___x_112_, v___x_106_);
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_a_88_);
return v___x_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___boxed(lean_object* v_x_115_, lean_object* v_a_116_, lean_object* v_a_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1(v_x_115_, v_a_116_, v_a_117_);
lean_dec(v_a_116_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id___lam__0(lean_object* v_x_119_){
_start:
{
lean_inc(v_x_119_);
return v_x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id___lam__0___boxed(lean_object* v_x_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_RelHom_id___lam__0(v_x_120_);
lean_dec(v_x_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_id(lean_object* v_00_u03b1_123_, lean_object* v_r_124_){
_start:
{
lean_object* v___f_125_; 
v___f_125_ = ((lean_object*)(lp_mathlib_RelHom_id___closed__0));
return v___f_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp___redArg___lam__0(lean_object* v_f_126_, lean_object* v_g_127_, lean_object* v_x_128_){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lean_apply_1(v_f_126_, v_x_128_);
v___x_130_ = lean_apply_1(v_g_127_, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp___redArg(lean_object* v_g_131_, lean_object* v_f_132_){
_start:
{
lean_object* v___f_133_; 
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_133_, 0, v_f_132_);
lean_closure_set(v___f_133_, 1, v_g_131_);
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_comp(lean_object* v_00_u03b1_134_, lean_object* v_00_u03b2_135_, lean_object* v_00_u03b3_136_, lean_object* v_r_137_, lean_object* v_s_138_, lean_object* v_t_139_, lean_object* v_g_140_, lean_object* v_f_141_){
_start:
{
lean_object* v___f_142_; 
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_142_, 0, v_f_141_);
lean_closure_set(v___f_142_, 1, v_g_140_);
return v___f_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap___redArg___lam__0(lean_object* v_f_143_, lean_object* v___y_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_f_143_, v___y_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap___redArg(lean_object* v_f_146_){
_start:
{
lean_object* v___f_147_; 
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_swap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_147_, 0, v_f_146_);
return v___f_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swap(lean_object* v_00_u03b1_148_, lean_object* v_00_u03b2_149_, lean_object* v_r_150_, lean_object* v_s_151_, lean_object* v_f_152_){
_start:
{
lean_object* v___f_153_; 
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_swap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_153_, 0, v_f_152_);
return v___f_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_swapEquiv(lean_object* v_00_u03b1_157_, lean_object* v_00_u03b2_158_, lean_object* v_r_159_, lean_object* v_s_160_){
_start:
{
lean_object* v___x_161_; 
v___x_161_ = ((lean_object*)(lp_mathlib_RelHom_swapEquiv___closed__1));
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___redArg(lean_object* v_f_162_){
_start:
{
lean_inc(v_f_162_);
return v_f_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___redArg___boxed(lean_object* v_f_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_RelHom_preimage___redArg(v_f_163_);
lean_dec(v_f_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage(lean_object* v_00_u03b1_165_, lean_object* v_00_u03b2_166_, lean_object* v_f_167_, lean_object* v_s_168_){
_start:
{
lean_inc(v_f_167_);
return v_f_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_preimage___boxed(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_f_171_, lean_object* v_s_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_RelHom_preimage(v_00_u03b1_169_, v_00_u03b2_170_, v_f_171_, v_s_172_);
lean_dec(v_f_171_);
return v_res_173_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__0));
v___x_191_ = l_String_toRawSubstring_x27(v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1(lean_object* v_x_205_, lean_object* v_a_206_, lean_object* v_a_207_){
_start:
{
lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_208_ = ((lean_object*)(lp_mathlib_term___u21aar___00__closed__1));
lean_inc(v_x_205_);
v___x_209_ = l_Lean_Syntax_isOfKind(v_x_205_, v___x_208_);
if (v___x_209_ == 0)
{
lean_object* v___x_210_; lean_object* v___x_211_; 
lean_dec(v_x_205_);
v___x_210_ = lean_box(1);
v___x_211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v_a_207_);
return v___x_211_;
}
else
{
lean_object* v_quotContext_212_; lean_object* v_currMacroScope_213_; lean_object* v_ref_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; uint8_t v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_quotContext_212_ = lean_ctor_get(v_a_206_, 1);
v_currMacroScope_213_ = lean_ctor_get(v_a_206_, 2);
v_ref_214_ = lean_ctor_get(v_a_206_, 5);
v___x_215_ = lean_unsigned_to_nat(0u);
v___x_216_ = l_Lean_Syntax_getArg(v_x_205_, v___x_215_);
v___x_217_ = lean_unsigned_to_nat(2u);
v___x_218_ = l_Lean_Syntax_getArg(v_x_205_, v___x_217_);
lean_dec(v_x_205_);
v___x_219_ = 0;
v___x_220_ = l_Lean_SourceInfo_fromRef(v_ref_214_, v___x_219_);
v___x_221_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
v___x_222_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1, &lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__1);
v___x_223_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__2));
lean_inc(v_currMacroScope_213_);
lean_inc(v_quotContext_212_);
v___x_224_ = l_Lean_addMacroScope(v_quotContext_212_, v___x_223_, v_currMacroScope_213_);
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___closed__6));
lean_inc_n(v___x_220_, 2);
v___x_226_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_226_, 0, v___x_220_);
lean_ctor_set(v___x_226_, 1, v___x_222_);
lean_ctor_set(v___x_226_, 2, v___x_224_);
lean_ctor_set(v___x_226_, 3, v___x_225_);
v___x_227_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13));
v___x_228_ = l_Lean_Syntax_node2(v___x_220_, v___x_227_, v___x_216_, v___x_218_);
v___x_229_ = l_Lean_Syntax_node2(v___x_220_, v___x_221_, v___x_226_, v___x_228_);
v___x_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_a_207_);
return v___x_230_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1___boxed(lean_object* v_x_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u21aar____1(v_x_231_, v_a_232_, v_a_233_);
lean_dec_ref(v_a_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelEmbedding__1(lean_object* v_x_235_, lean_object* v_a_236_, lean_object* v_a_237_){
_start:
{
lean_object* v___x_238_; uint8_t v___x_239_; 
v___x_238_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
lean_inc(v_x_235_);
v___x_239_ = l_Lean_Syntax_isOfKind(v_x_235_, v___x_238_);
if (v___x_239_ == 0)
{
lean_object* v___x_240_; lean_object* v___x_241_; 
lean_dec(v_x_235_);
v___x_240_ = lean_box(0);
v___x_241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v_a_237_);
return v___x_241_;
}
else
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_242_ = lean_unsigned_to_nat(0u);
v___x_243_ = l_Lean_Syntax_getArg(v_x_235_, v___x_242_);
v___x_244_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1));
lean_inc(v___x_243_);
v___x_245_ = l_Lean_Syntax_isOfKind(v___x_243_, v___x_244_);
if (v___x_245_ == 0)
{
lean_object* v___x_246_; lean_object* v___x_247_; 
lean_dec(v___x_243_);
lean_dec(v_x_235_);
v___x_246_ = lean_box(0);
v___x_247_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set(v___x_247_, 1, v_a_237_);
return v___x_247_;
}
else
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; uint8_t v___x_251_; 
v___x_248_ = lean_unsigned_to_nat(1u);
v___x_249_ = l_Lean_Syntax_getArg(v_x_235_, v___x_248_);
lean_dec(v_x_235_);
v___x_250_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_249_);
v___x_251_ = l_Lean_Syntax_matchesNull(v___x_249_, v___x_250_);
if (v___x_251_ == 0)
{
lean_object* v___x_252_; lean_object* v___x_253_; 
lean_dec(v___x_249_);
lean_dec(v___x_243_);
v___x_252_ = lean_box(0);
v___x_253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set(v___x_253_, 1, v_a_237_);
return v___x_253_;
}
else
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v_ref_256_; uint8_t v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_254_ = l_Lean_Syntax_getArg(v___x_249_, v___x_242_);
v___x_255_ = l_Lean_Syntax_getArg(v___x_249_, v___x_248_);
lean_dec(v___x_249_);
v_ref_256_ = l_Lean_replaceRef(v___x_243_, v_a_236_);
lean_dec(v___x_243_);
v___x_257_ = 0;
v___x_258_ = l_Lean_SourceInfo_fromRef(v_ref_256_, v___x_257_);
lean_dec(v_ref_256_);
v___x_259_ = ((lean_object*)(lp_mathlib_term___u21aar___00__closed__1));
v___x_260_ = ((lean_object*)(lp_mathlib_term___u21aar___00__closed__2));
lean_inc(v___x_258_);
v___x_261_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_258_);
lean_ctor_set(v___x_261_, 1, v___x_260_);
v___x_262_ = l_Lean_Syntax_node3(v___x_258_, v___x_259_, v___x_254_, v___x_261_, v___x_255_);
v___x_263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_237_);
return v___x_263_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelEmbedding__1___boxed(lean_object* v_x_264_, lean_object* v_a_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelEmbedding__1(v_x_264_, v_a_265_, v_a_266_);
lean_dec(v_a_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___redArg(lean_object* v_f_268_){
_start:
{
lean_inc(v_f_268_);
return v_f_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___redArg___boxed(lean_object* v_f_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_RelEmbedding_toRelHom___redArg(v_f_269_);
lean_dec(v_f_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom(lean_object* v_00_u03b1_271_, lean_object* v_00_u03b2_272_, lean_object* v_r_273_, lean_object* v_s_274_, lean_object* v_f_275_){
_start:
{
lean_inc(v_f_275_);
return v_f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toRelHom___boxed(lean_object* v_00_u03b1_276_, lean_object* v_00_u03b2_277_, lean_object* v_r_278_, lean_object* v_s_279_, lean_object* v_f_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_RelEmbedding_toRelHom(v_00_u03b1_276_, v_00_u03b2_277_, v_r_278_, v_s_279_, v_f_280_);
lean_dec(v_f_280_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instCoeRelHom(lean_object* v_00_u03b1_283_, lean_object* v_00_u03b2_284_, lean_object* v_r_285_, lean_object* v_s_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = ((lean_object*)(lp_mathlib_RelEmbedding_instCoeRelHom___closed__0));
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_refl(lean_object* v_00_u03b1_289_, lean_object* v_r_290_){
_start:
{
lean_object* v___f_291_; 
v___f_291_ = ((lean_object*)(lp_mathlib_RelEmbedding_refl___closed__0));
return v___f_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_trans___redArg(lean_object* v_f_292_, lean_object* v_g_293_){
_start:
{
lean_object* v___f_294_; 
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_294_, 0, v_f_292_);
lean_closure_set(v___f_294_, 1, v_g_293_);
return v___f_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_trans(lean_object* v_00_u03b1_295_, lean_object* v_00_u03b2_296_, lean_object* v_00_u03b3_297_, lean_object* v_r_298_, lean_object* v_s_299_, lean_object* v_t_300_, lean_object* v_f_301_, lean_object* v_g_302_){
_start:
{
lean_object* v___f_303_; 
v___f_303_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_303_, 0, v_f_301_);
lean_closure_set(v___f_303_, 1, v_g_302_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_instInhabited(lean_object* v_00_u03b1_304_, lean_object* v_r_305_){
_start:
{
lean_object* v___f_306_; 
v___f_306_ = ((lean_object*)(lp_mathlib_RelEmbedding_refl___closed__0));
return v___f_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___redArg(lean_object* v_f_307_){
_start:
{
lean_inc(v_f_307_);
return v_f_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___redArg___boxed(lean_object* v_f_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_RelEmbedding_swap___redArg(v_f_308_);
lean_dec(v_f_308_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap(lean_object* v_00_u03b1_310_, lean_object* v_00_u03b2_311_, lean_object* v_r_312_, lean_object* v_s_313_, lean_object* v_f_314_){
_start:
{
lean_inc(v_f_314_);
return v_f_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swap___boxed(lean_object* v_00_u03b1_315_, lean_object* v_00_u03b2_316_, lean_object* v_r_317_, lean_object* v_s_318_, lean_object* v_f_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_RelEmbedding_swap(v_00_u03b1_315_, v_00_u03b2_316_, v_r_317_, v_s_318_, v_f_319_);
lean_dec(v_f_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_swapEquiv(lean_object* v_00_u03b1_324_, lean_object* v_00_u03b2_325_, lean_object* v_r_326_, lean_object* v_s_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = ((lean_object*)(lp_mathlib_RelEmbedding_swapEquiv___closed__1));
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___redArg(lean_object* v_f_329_){
_start:
{
lean_inc(v_f_329_);
return v_f_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___redArg___boxed(lean_object* v_f_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_RelEmbedding_preimage___redArg(v_f_330_);
lean_dec(v_f_330_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage(lean_object* v_00_u03b1_332_, lean_object* v_00_u03b2_333_, lean_object* v_f_334_, lean_object* v_s_335_){
_start:
{
lean_inc(v_f_334_);
return v_f_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_preimage___boxed(lean_object* v_00_u03b1_336_, lean_object* v_00_u03b2_337_, lean_object* v_f_338_, lean_object* v_s_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_RelEmbedding_preimage(v_00_u03b1_336_, v_00_u03b2_337_, v_f_338_, v_s_339_);
lean_dec(v_f_338_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_relEmbedding(lean_object* v_X_342_, lean_object* v_r_343_, lean_object* v_p_344_){
_start:
{
lean_object* v___f_345_; 
v___f_345_ = ((lean_object*)(lp_mathlib_Subtype_relEmbedding___closed__0));
return v___f_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_mkRelHom___redArg(lean_object* v_x_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lean_alloc_closure((void*)(l_Quotient_mk___boxed), 3, 2);
lean_closure_set(v___x_347_, 0, lean_box(0));
lean_closure_set(v___x_347_, 1, v_x_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_mkRelHom(lean_object* v_00_u03b1_348_, lean_object* v_x_349_, lean_object* v_r_350_, lean_object* v_H_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lean_alloc_closure((void*)(l_Quotient_mk___boxed), 3, 2);
lean_closure_set(v___x_352_, 0, lean_box(0));
lean_closure_set(v___x_352_, 1, v_x_349_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___redArg(lean_object* v_f_353_){
_start:
{
lean_inc(v_f_353_);
return v_f_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___redArg___boxed(lean_object* v_f_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib_RelEmbedding_ofMapRelIff___redArg(v_f_354_);
lean_dec(v_f_354_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff(lean_object* v_00_u03b1_356_, lean_object* v_00_u03b2_357_, lean_object* v_r_358_, lean_object* v_s_359_, lean_object* v_f_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_hf_363_){
_start:
{
lean_inc(v_f_360_);
return v_f_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMapRelIff___boxed(lean_object* v_00_u03b1_364_, lean_object* v_00_u03b2_365_, lean_object* v_r_366_, lean_object* v_s_367_, lean_object* v_f_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_hf_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_mathlib_RelEmbedding_ofMapRelIff(v_00_u03b1_364_, v_00_u03b2_365_, v_r_366_, v_s_367_, v_f_368_, v_inst_369_, v_inst_370_, v_hf_371_);
lean_dec(v_f_368_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___redArg(lean_object* v_f_373_){
_start:
{
lean_inc(v_f_373_);
return v_f_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___redArg___boxed(lean_object* v_f_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_RelEmbedding_ofMonotone___redArg(v_f_374_);
lean_dec(v_f_374_);
return v_res_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone(lean_object* v_00_u03b1_376_, lean_object* v_00_u03b2_377_, lean_object* v_r_378_, lean_object* v_s_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_f_382_, lean_object* v_H_383_){
_start:
{
lean_inc(v_f_382_);
return v_f_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofMonotone___boxed(lean_object* v_00_u03b1_384_, lean_object* v_00_u03b2_385_, lean_object* v_r_386_, lean_object* v_s_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_f_390_, lean_object* v_H_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_mathlib_RelEmbedding_ofMonotone(v_00_u03b1_384_, v_00_u03b2_385_, v_r_386_, v_s_387_, v_inst_388_, v_inst_389_, v_f_390_, v_H_391_);
lean_dec(v_f_390_);
return v_res_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofIsEmpty(lean_object* v_00_u03b1_394_, lean_object* v_00_u03b2_395_, lean_object* v_r_396_, lean_object* v_s_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___f_399_; 
v___f_399_ = ((lean_object*)(lp_mathlib_RelEmbedding_ofIsEmpty___closed__0));
return v___f_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInl___lam__0(lean_object* v_val_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_401_, 0, v_val_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInl(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_r_405_, lean_object* v_s_406_){
_start:
{
lean_object* v___f_407_; 
v___f_407_ = ((lean_object*)(lp_mathlib_RelEmbedding_sumLiftRelInl___closed__0));
return v___f_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInr___lam__0(lean_object* v_val_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_409_, 0, v_val_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelInr(lean_object* v_00_u03b1_411_, lean_object* v_00_u03b2_412_, lean_object* v_r_413_, lean_object* v_s_414_){
_start:
{
lean_object* v___f_415_; 
v___f_415_ = ((lean_object*)(lp_mathlib_RelEmbedding_sumLiftRelInr___closed__0));
return v___f_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap___redArg___lam__1(lean_object* v_g_416_, lean_object* v___y_417_){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = lean_apply_1(v_g_416_, v___y_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap___redArg(lean_object* v_f_419_, lean_object* v_g_420_){
_start:
{
lean_object* v___f_421_; lean_object* v___f_422_; lean_object* v___x_423_; 
v___f_421_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_swap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_421_, 0, v_f_419_);
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_sumLiftRelMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_422_, 0, v_g_420_);
v___x_423_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_423_, 0, lean_box(0));
lean_closure_set(v___x_423_, 1, lean_box(0));
lean_closure_set(v___x_423_, 2, lean_box(0));
lean_closure_set(v___x_423_, 3, lean_box(0));
lean_closure_set(v___x_423_, 4, v___f_421_);
lean_closure_set(v___x_423_, 5, v___f_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLiftRelMap(lean_object* v_00_u03b1_424_, lean_object* v_00_u03b2_425_, lean_object* v_00_u03b3_426_, lean_object* v_00_u03b4_427_, lean_object* v_r_428_, lean_object* v_s_429_, lean_object* v_t_430_, lean_object* v_u_431_, lean_object* v_f_432_, lean_object* v_g_433_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_RelEmbedding_sumLiftRelMap___redArg(v_f_432_, v_g_433_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexInl(lean_object* v_00_u03b1_435_, lean_object* v_00_u03b2_436_, lean_object* v_r_437_, lean_object* v_s_438_){
_start:
{
lean_object* v___f_439_; 
v___f_439_ = ((lean_object*)(lp_mathlib_RelEmbedding_sumLiftRelInl___closed__0));
return v___f_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexInr(lean_object* v_00_u03b1_440_, lean_object* v_00_u03b2_441_, lean_object* v_r_442_, lean_object* v_s_443_){
_start:
{
lean_object* v___f_444_; 
v___f_444_ = ((lean_object*)(lp_mathlib_RelEmbedding_sumLiftRelInr___closed__0));
return v___f_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexMap___redArg(lean_object* v_f_445_, lean_object* v_g_446_){
_start:
{
lean_object* v___f_447_; lean_object* v___f_448_; lean_object* v___x_449_; 
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_swap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_447_, 0, v_f_445_);
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_sumLiftRelMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_448_, 0, v_g_446_);
v___x_449_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_449_, 0, lean_box(0));
lean_closure_set(v___x_449_, 1, lean_box(0));
lean_closure_set(v___x_449_, 2, lean_box(0));
lean_closure_set(v___x_449_, 3, lean_box(0));
lean_closure_set(v___x_449_, 4, v___f_447_);
lean_closure_set(v___x_449_, 5, v___f_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_sumLexMap(lean_object* v_00_u03b1_450_, lean_object* v_00_u03b2_451_, lean_object* v_00_u03b3_452_, lean_object* v_00_u03b4_453_, lean_object* v_r_454_, lean_object* v_s_455_, lean_object* v_t_456_, lean_object* v_u_457_, lean_object* v_f_458_, lean_object* v_g_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_RelEmbedding_sumLexMap___redArg(v_f_458_, v_g_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft___redArg___lam__0(lean_object* v_a_461_, lean_object* v_snd_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_463_, 0, v_a_461_);
lean_ctor_set(v___x_463_, 1, v_snd_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft___redArg(lean_object* v_a_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_prodLexMkLeft___redArg___lam__0), 2, 1);
lean_closure_set(v___f_465_, 0, v_a_464_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkLeft(lean_object* v_00_u03b1_466_, lean_object* v_00_u03b2_467_, lean_object* v_r_468_, lean_object* v_s_469_, lean_object* v_a_470_, lean_object* v_h_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_prodLexMkLeft___redArg___lam__0), 2, 1);
lean_closure_set(v___f_472_, 0, v_a_470_);
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight___redArg___lam__0(lean_object* v_b_473_, lean_object* v_a_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_475_, 0, v_a_474_);
lean_ctor_set(v___x_475_, 1, v_b_473_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight___redArg(lean_object* v_b_476_){
_start:
{
lean_object* v___f_477_; 
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_prodLexMkRight___redArg___lam__0), 2, 1);
lean_closure_set(v___f_477_, 0, v_b_476_);
return v___f_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMkRight(lean_object* v_00_u03b1_478_, lean_object* v_00_u03b2_479_, lean_object* v_s_480_, lean_object* v_r_481_, lean_object* v_b_482_, lean_object* v_h_483_){
_start:
{
lean_object* v___f_484_; 
v___f_484_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_prodLexMkRight___redArg___lam__0), 2, 1);
lean_closure_set(v___f_484_, 0, v_b_482_);
return v___f_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMap___redArg(lean_object* v_f_485_, lean_object* v_g_486_){
_start:
{
lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___x_489_; 
v___f_487_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_swap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_487_, 0, v_f_485_);
v___f_488_ = lean_alloc_closure((void*)(lp_mathlib_RelEmbedding_sumLiftRelMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_488_, 0, v_g_486_);
v___x_489_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_489_, 0, lean_box(0));
lean_closure_set(v___x_489_, 1, lean_box(0));
lean_closure_set(v___x_489_, 2, lean_box(0));
lean_closure_set(v___x_489_, 3, lean_box(0));
lean_closure_set(v___x_489_, 4, v___f_487_);
lean_closure_set(v___x_489_, 5, v___f_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_prodLexMap(lean_object* v_00_u03b1_490_, lean_object* v_00_u03b2_491_, lean_object* v_00_u03b3_492_, lean_object* v_00_u03b4_493_, lean_object* v_r_494_, lean_object* v_s_495_, lean_object* v_t_496_, lean_object* v_u_497_, lean_object* v_f_498_, lean_object* v_g_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_mathlib_RelEmbedding_prodLexMap___redArg(v_f_498_, v_g_499_);
return v___x_500_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1(void){
_start:
{
lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_517_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__0));
v___x_518_ = l_String_toRawSubstring_x27(v___x_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1(lean_object* v_x_532_, lean_object* v_a_533_, lean_object* v_a_534_){
_start:
{
lean_object* v___x_535_; uint8_t v___x_536_; 
v___x_535_ = ((lean_object*)(lp_mathlib_term___u2243r___00__closed__1));
lean_inc(v_x_532_);
v___x_536_ = l_Lean_Syntax_isOfKind(v_x_532_, v___x_535_);
if (v___x_536_ == 0)
{
lean_object* v___x_537_; lean_object* v___x_538_; 
lean_dec(v_x_532_);
v___x_537_ = lean_box(1);
v___x_538_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_538_, 0, v___x_537_);
lean_ctor_set(v___x_538_, 1, v_a_534_);
return v___x_538_;
}
else
{
lean_object* v_quotContext_539_; lean_object* v_currMacroScope_540_; lean_object* v_ref_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; uint8_t v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v_quotContext_539_ = lean_ctor_get(v_a_533_, 1);
v_currMacroScope_540_ = lean_ctor_get(v_a_533_, 2);
v_ref_541_ = lean_ctor_get(v_a_533_, 5);
v___x_542_ = lean_unsigned_to_nat(0u);
v___x_543_ = l_Lean_Syntax_getArg(v_x_532_, v___x_542_);
v___x_544_ = lean_unsigned_to_nat(2u);
v___x_545_ = l_Lean_Syntax_getArg(v_x_532_, v___x_544_);
lean_dec(v_x_532_);
v___x_546_ = 0;
v___x_547_ = l_Lean_SourceInfo_fromRef(v_ref_541_, v___x_546_);
v___x_548_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
v___x_549_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1, &lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__1);
v___x_550_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__2));
lean_inc(v_currMacroScope_540_);
lean_inc(v_quotContext_539_);
v___x_551_ = l_Lean_addMacroScope(v_quotContext_539_, v___x_550_, v_currMacroScope_540_);
v___x_552_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___closed__6));
lean_inc_n(v___x_547_, 2);
v___x_553_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_553_, 0, v___x_547_);
lean_ctor_set(v___x_553_, 1, v___x_549_);
lean_ctor_set(v___x_553_, 2, v___x_551_);
lean_ctor_set(v___x_553_, 3, v___x_552_);
v___x_554_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__13));
v___x_555_ = l_Lean_Syntax_node2(v___x_547_, v___x_554_, v___x_543_, v___x_545_);
v___x_556_ = l_Lean_Syntax_node2(v___x_547_, v___x_548_, v___x_553_, v___x_555_);
v___x_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_557_, 0, v___x_556_);
lean_ctor_set(v___x_557_, 1, v_a_534_);
return v___x_557_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1___boxed(lean_object* v_x_558_, lean_object* v_a_559_, lean_object* v_a_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2243r____1(v_x_558_, v_a_559_, v_a_560_);
lean_dec_ref(v_a_559_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelIso__1(lean_object* v_x_562_, lean_object* v_a_563_, lean_object* v_a_564_){
_start:
{
lean_object* v___x_565_; uint8_t v___x_566_; 
v___x_565_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______macroRules__term___u2192r____1___closed__4));
lean_inc(v_x_562_);
v___x_566_ = l_Lean_Syntax_isOfKind(v_x_562_, v___x_565_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; lean_object* v___x_568_; 
lean_dec(v_x_562_);
v___x_567_ = lean_box(0);
v___x_568_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_568_, 0, v___x_567_);
lean_ctor_set(v___x_568_, 1, v_a_564_);
return v___x_568_;
}
else
{
lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; uint8_t v___x_572_; 
v___x_569_ = lean_unsigned_to_nat(0u);
v___x_570_ = l_Lean_Syntax_getArg(v_x_562_, v___x_569_);
v___x_571_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelHom__1___closed__1));
lean_inc(v___x_570_);
v___x_572_ = l_Lean_Syntax_isOfKind(v___x_570_, v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; lean_object* v___x_574_; 
lean_dec(v___x_570_);
lean_dec(v_x_562_);
v___x_573_ = lean_box(0);
v___x_574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_574_, 0, v___x_573_);
lean_ctor_set(v___x_574_, 1, v_a_564_);
return v___x_574_;
}
else
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; uint8_t v___x_578_; 
v___x_575_ = lean_unsigned_to_nat(1u);
v___x_576_ = l_Lean_Syntax_getArg(v_x_562_, v___x_575_);
lean_dec(v_x_562_);
v___x_577_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_576_);
v___x_578_ = l_Lean_Syntax_matchesNull(v___x_576_, v___x_577_);
if (v___x_578_ == 0)
{
lean_object* v___x_579_; lean_object* v___x_580_; 
lean_dec(v___x_576_);
lean_dec(v___x_570_);
v___x_579_ = lean_box(0);
v___x_580_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_580_, 0, v___x_579_);
lean_ctor_set(v___x_580_, 1, v_a_564_);
return v___x_580_;
}
else
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v_ref_583_; uint8_t v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_581_ = l_Lean_Syntax_getArg(v___x_576_, v___x_569_);
v___x_582_ = l_Lean_Syntax_getArg(v___x_576_, v___x_575_);
lean_dec(v___x_576_);
v_ref_583_ = l_Lean_replaceRef(v___x_570_, v_a_563_);
lean_dec(v___x_570_);
v___x_584_ = 0;
v___x_585_ = l_Lean_SourceInfo_fromRef(v_ref_583_, v___x_584_);
lean_dec(v_ref_583_);
v___x_586_ = ((lean_object*)(lp_mathlib_term___u2243r___00__closed__1));
v___x_587_ = ((lean_object*)(lp_mathlib_term___u2243r___00__closed__2));
lean_inc(v___x_585_);
v___x_588_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_585_);
lean_ctor_set(v___x_588_, 1, v___x_587_);
v___x_589_ = l_Lean_Syntax_node3(v___x_585_, v___x_586_, v___x_581_, v___x_588_, v___x_582_);
v___x_590_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_590_, 0, v___x_589_);
lean_ctor_set(v___x_590_, 1, v_a_564_);
return v___x_590_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelIso__1___boxed(lean_object* v_x_591_, lean_object* v_a_592_, lean_object* v_a_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib___aux__Mathlib__Order__RelIso__Basic______unexpand__RelIso__1(v_x_591_, v_a_592_, v_a_593_);
lean_dec(v_a_592_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toRelEmbedding___redArg(lean_object* v_f_595_){
_start:
{
lean_object* v___f_596_; 
v___f_596_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_596_, 0, v_f_595_);
return v___f_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toRelEmbedding(lean_object* v_00_u03b1_597_, lean_object* v_00_u03b2_598_, lean_object* v_r_599_, lean_object* v_s_600_, lean_object* v_f_601_){
_start:
{
lean_object* v___f_602_; 
v___f_602_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_602_, 0, v_f_601_);
return v___f_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instCoeOutRelEmbedding(lean_object* v_00_u03b1_604_, lean_object* v_00_u03b2_605_, lean_object* v_r_606_, lean_object* v_s_607_){
_start:
{
lean_object* v___x_608_; 
v___x_608_ = ((lean_object*)(lp_mathlib_RelIso_instCoeOutRelEmbedding___closed__0));
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_symm___redArg(lean_object* v_f_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lp_mathlib_Equiv_symm___redArg(v_f_609_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_symm(lean_object* v_00_u03b1_611_, lean_object* v_00_u03b2_612_, lean_object* v_r_613_, lean_object* v_s_614_, lean_object* v_f_615_){
_start:
{
lean_object* v___x_616_; 
v___x_616_ = lp_mathlib_Equiv_symm___redArg(v_f_615_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_apply___redArg(lean_object* v_h_617_, lean_object* v_a_618_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_h_617_, v_a_618_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_apply(lean_object* v_00_u03b1_620_, lean_object* v_00_u03b2_621_, lean_object* v_r_622_, lean_object* v_s_623_, lean_object* v_h_624_, lean_object* v_a_625_){
_start:
{
lean_object* v___x_626_; 
v___x_626_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_h_624_, v_a_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_symm__apply___redArg(lean_object* v_h_627_, lean_object* v_a_628_){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_629_ = lp_mathlib_Equiv_symm___redArg(v_h_627_);
v___x_630_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_629_, v_a_628_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_Simps_symm__apply(lean_object* v_00_u03b1_631_, lean_object* v_00_u03b2_632_, lean_object* v_r_633_, lean_object* v_s_634_, lean_object* v_h_635_, lean_object* v_a_636_){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = lp_mathlib_RelIso_Simps_symm__apply___redArg(v_h_635_, v_a_636_);
return v___x_637_;
}
}
static lean_object* _init_lp_mathlib_RelIso_refl___closed__0(void){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_refl(lean_object* v_00_u03b1_639_, lean_object* v_r_640_){
_start:
{
lean_object* v___x_641_; 
v___x_641_ = lean_obj_once(&lp_mathlib_RelIso_refl___closed__0, &lp_mathlib_RelIso_refl___closed__0_once, _init_lp_mathlib_RelIso_refl___closed__0);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_trans___redArg(lean_object* v_f_u2081_642_, lean_object* v_f_u2082_643_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lp_mathlib_Equiv_trans___redArg(v_f_u2081_642_, v_f_u2082_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_trans(lean_object* v_00_u03b1_645_, lean_object* v_00_u03b2_646_, lean_object* v_00_u03b3_647_, lean_object* v_r_648_, lean_object* v_s_649_, lean_object* v_t_650_, lean_object* v_f_u2081_651_, lean_object* v_f_u2082_652_){
_start:
{
lean_object* v___x_653_; 
v___x_653_ = lp_mathlib_Equiv_trans___redArg(v_f_u2081_651_, v_f_u2082_652_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_instInhabited(lean_object* v_00_u03b1_654_, lean_object* v_r_655_){
_start:
{
lean_object* v___x_656_; 
v___x_656_ = lean_obj_once(&lp_mathlib_RelIso_refl___closed__0, &lp_mathlib_RelIso_refl___closed__0_once, _init_lp_mathlib_RelIso_refl___closed__0);
return v___x_656_;
}
}
static lean_object* _init_lp_mathlib_RelIso_cast___closed__0(void){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_cast(lean_object* v_00_u03b1_658_, lean_object* v_00_u03b2_659_, lean_object* v_r_660_, lean_object* v_s_661_, lean_object* v_h_u2081_662_, lean_object* v_h_u2082_663_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lean_obj_once(&lp_mathlib_RelIso_cast___closed__0, &lp_mathlib_RelIso_cast___closed__0_once, _init_lp_mathlib_RelIso_cast___closed__0);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__0(lean_object* v_f_665_, lean_object* v___y_666_){
_start:
{
lean_object* v___x_667_; 
v___x_667_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_665_, v___y_666_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__1(lean_object* v_f_668_, lean_object* v___y_669_){
_start:
{
lean_object* v___x_670_; lean_object* v_toFun_671_; lean_object* v___x_672_; 
v___x_670_ = lp_mathlib_Equiv_symm___redArg(v_f_668_);
v_toFun_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_toFun_671_);
lean_dec_ref(v___x_670_);
v___x_672_ = lean_apply_1(v_toFun_671_, v___y_669_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(lean_object* v_f_673_){
_start:
{
lean_object* v___f_674_; lean_object* v___f_675_; lean_object* v___x_676_; 
lean_inc_ref(v_f_673_);
v___f_674_ = lean_alloc_closure((void*)(lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_674_, 0, v_f_673_);
v___f_675_ = lean_alloc_closure((void*)(lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg___lam__1), 2, 1);
lean_closure_set(v___f_675_, 0, v_f_673_);
v___x_676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_676_, 0, v___f_674_);
lean_ctor_set(v___x_676_, 1, v___f_675_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0(lean_object* v_00_u03b1_677_, lean_object* v_00_u03b2_678_, lean_object* v_f_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(v_f_679_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swap___redArg(lean_object* v_f_681_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(v_f_681_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swap(lean_object* v_00_u03b1_683_, lean_object* v_00_u03b2_684_, lean_object* v_r_685_, lean_object* v_s_686_, lean_object* v_f_687_){
_start:
{
lean_object* v___x_688_; 
v___x_688_ = lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(v_f_687_);
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_swapEquiv(lean_object* v_00_u03b1_692_, lean_object* v_00_u03b2_693_, lean_object* v_r_694_, lean_object* v_s_695_){
_start:
{
lean_object* v___x_696_; 
v___x_696_ = ((lean_object*)(lp_mathlib_RelIso_swapEquiv___closed__1));
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_compl___redArg(lean_object* v_f_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(v_f_697_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_compl(lean_object* v_00_u03b1_699_, lean_object* v_00_u03b2_700_, lean_object* v_r_701_, lean_object* v_s_702_, lean_object* v_f_703_){
_start:
{
lean_object* v___x_704_; 
v___x_704_ = lp_mathlib_EquivLike_toEquiv___at___00RelIso_swap_spec__0___redArg(v_f_703_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_complEquiv(lean_object* v_00_u03b1_710_, lean_object* v_00_u03b2_711_, lean_object* v_r_712_, lean_object* v_s_713_){
_start:
{
lean_object* v___x_714_; 
v___x_714_ = ((lean_object*)(lp_mathlib_RelIso_complEquiv___closed__2));
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy___redArg(lean_object* v_f_715_, lean_object* v_g_716_){
_start:
{
lean_object* v___x_717_; 
v___x_717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_717_, 0, v_f_715_);
lean_ctor_set(v___x_717_, 1, v_g_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy(lean_object* v_00_u03b1_718_, lean_object* v_00_u03b2_719_, lean_object* v_r_720_, lean_object* v_s_721_, lean_object* v_e_722_, lean_object* v_f_723_, lean_object* v_g_724_, lean_object* v_hf_725_, lean_object* v_hg_726_){
_start:
{
lean_object* v___x_727_; 
v___x_727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_727_, 0, v_f_723_);
lean_ctor_set(v___x_727_, 1, v_g_724_);
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_copy___boxed(lean_object* v_00_u03b1_728_, lean_object* v_00_u03b2_729_, lean_object* v_r_730_, lean_object* v_s_731_, lean_object* v_e_732_, lean_object* v_f_733_, lean_object* v_g_734_, lean_object* v_hf_735_, lean_object* v_hg_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_mathlib_RelIso_copy(v_00_u03b1_728_, v_00_u03b2_729_, v_r_730_, v_s_731_, v_e_732_, v_f_733_, v_g_734_, v_hf_735_, v_hg_736_);
lean_dec_ref(v_e_732_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___redArg(lean_object* v_f_738_){
_start:
{
lean_inc_ref(v_f_738_);
return v_f_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___redArg___boxed(lean_object* v_f_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_mathlib_RelIso_preimage___redArg(v_f_739_);
lean_dec_ref(v_f_739_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage(lean_object* v_00_u03b1_741_, lean_object* v_00_u03b2_742_, lean_object* v_f_743_, lean_object* v_s_744_){
_start:
{
lean_inc_ref(v_f_743_);
return v_f_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_preimage___boxed(lean_object* v_00_u03b1_745_, lean_object* v_00_u03b2_746_, lean_object* v_f_747_, lean_object* v_s_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_mathlib_RelIso_preimage(v_00_u03b1_745_, v_00_u03b2_746_, v_f_747_, v_s_748_);
lean_dec_ref(v_f_747_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg___lam__0(lean_object* v_e_u2082_750_, lean_object* v_e_u2081_751_, lean_object* v_f_u2081_752_, lean_object* v___y_753_){
_start:
{
lean_object* v___f_754_; lean_object* v___x_755_; lean_object* v___f_756_; lean_object* v___f_757_; lean_object* v___x_758_; 
v___f_754_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_754_, 0, v_e_u2082_750_);
v___x_755_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_751_);
v___f_756_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_756_, 0, v___x_755_);
v___f_757_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_757_, 0, v___f_756_);
lean_closure_set(v___f_757_, 1, v_f_u2081_752_);
v___x_758_ = lp_mathlib_RelHom_comp___redArg___lam__0(v___f_757_, v___f_754_, v___y_753_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg___lam__1(lean_object* v_e_u2082_759_, lean_object* v_e_u2081_760_, lean_object* v_f_u2082_761_, lean_object* v___y_762_){
_start:
{
lean_object* v___x_763_; lean_object* v___f_764_; lean_object* v___f_765_; lean_object* v___f_766_; lean_object* v___x_767_; 
v___x_763_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_759_);
v___f_764_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_764_, 0, v___x_763_);
v___f_765_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_765_, 0, v_e_u2081_760_);
v___f_766_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_766_, 0, v___f_765_);
lean_closure_set(v___f_766_, 1, v_f_u2082_761_);
v___x_767_ = lp_mathlib_RelHom_comp___redArg___lam__0(v___f_766_, v___f_764_, v___y_762_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr___redArg(lean_object* v_e_u2081_768_, lean_object* v_e_u2082_769_){
_start:
{
lean_object* v___f_770_; lean_object* v___f_771_; lean_object* v___x_772_; 
lean_inc_ref(v_e_u2081_768_);
lean_inc_ref(v_e_u2082_769_);
v___f_770_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relHomCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_770_, 0, v_e_u2082_769_);
lean_closure_set(v___f_770_, 1, v_e_u2081_768_);
v___f_771_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relHomCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_771_, 0, v_e_u2082_769_);
lean_closure_set(v___f_771_, 1, v_e_u2081_768_);
v___x_772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_772_, 0, v___f_770_);
lean_ctor_set(v___x_772_, 1, v___f_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relHomCongr(lean_object* v_00_u03b1_u2081_773_, lean_object* v_00_u03b2_u2081_774_, lean_object* v_00_u03b1_u2082_775_, lean_object* v_00_u03b2_u2082_776_, lean_object* v_r_u2081_777_, lean_object* v_s_u2081_778_, lean_object* v_r_u2082_779_, lean_object* v_s_u2082_780_, lean_object* v_e_u2081_781_, lean_object* v_e_u2082_782_){
_start:
{
lean_object* v___x_783_; 
v___x_783_ = lp_mathlib_RelIso_relHomCongr___redArg(v_e_u2081_781_, v_e_u2082_782_);
return v___x_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__0(lean_object* v_e_u2081_784_, lean_object* v_e_u2082_785_, lean_object* v_f_u2081_786_, lean_object* v___y_787_){
_start:
{
lean_object* v___x_788_; lean_object* v___f_789_; lean_object* v___f_790_; lean_object* v___f_791_; lean_object* v___x_792_; 
v___x_788_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_784_);
v___f_789_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_789_, 0, v___x_788_);
v___f_790_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_790_, 0, v___f_789_);
lean_closure_set(v___f_790_, 1, v_f_u2081_786_);
v___f_791_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_791_, 0, v_e_u2082_785_);
v___x_792_ = lp_mathlib_Function_Embedding_trans___redArg___lam__0(v___f_790_, v___f_791_, v___y_787_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__1(lean_object* v_e_u2081_793_, lean_object* v_e_u2082_794_, lean_object* v_f_u2082_795_, lean_object* v___y_796_){
_start:
{
lean_object* v___f_797_; lean_object* v___f_798_; lean_object* v___x_799_; lean_object* v___f_800_; lean_object* v___x_801_; 
v___f_797_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_797_, 0, v_e_u2081_793_);
v___f_798_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_798_, 0, v___f_797_);
lean_closure_set(v___f_798_, 1, v_f_u2082_795_);
v___x_799_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_794_);
v___f_800_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_800_, 0, v___x_799_);
v___x_801_ = lp_mathlib_Function_Embedding_trans___redArg___lam__0(v___f_798_, v___f_800_, v___y_796_);
return v___x_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg(lean_object* v_e_u2081_802_, lean_object* v_e_u2082_803_){
_start:
{
lean_object* v___f_804_; lean_object* v___f_805_; lean_object* v___x_806_; 
lean_inc_ref(v_e_u2082_803_);
lean_inc_ref(v_e_u2081_802_);
v___f_804_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_804_, 0, v_e_u2081_802_);
lean_closure_set(v___f_804_, 1, v_e_u2082_803_);
v___f_805_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relEmbeddingCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_805_, 0, v_e_u2081_802_);
lean_closure_set(v___f_805_, 1, v_e_u2082_803_);
v___x_806_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_806_, 0, v___f_804_);
lean_ctor_set(v___x_806_, 1, v___f_805_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relEmbeddingCongr(lean_object* v_00_u03b1_u2081_807_, lean_object* v_00_u03b2_u2081_808_, lean_object* v_00_u03b1_u2082_809_, lean_object* v_00_u03b2_u2082_810_, lean_object* v_r_u2081_811_, lean_object* v_s_u2081_812_, lean_object* v_r_u2082_813_, lean_object* v_s_u2082_814_, lean_object* v_e_u2081_815_, lean_object* v_e_u2082_816_){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = lp_mathlib_RelIso_relEmbeddingCongr___redArg(v_e_u2081_815_, v_e_u2082_816_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg___lam__0(lean_object* v_e_u2081_818_, lean_object* v_e_u2082_819_, lean_object* v_f_u2081_820_){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_821_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_818_);
v___x_822_ = lp_mathlib_Equiv_trans___redArg(v___x_821_, v_f_u2081_820_);
v___x_823_ = lp_mathlib_Equiv_trans___redArg(v___x_822_, v_e_u2082_819_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg___lam__1(lean_object* v_e_u2081_824_, lean_object* v_e_u2082_825_, lean_object* v_f_u2082_826_){
_start:
{
lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_827_ = lp_mathlib_Equiv_trans___redArg(v_e_u2081_824_, v_f_u2082_826_);
v___x_828_ = lp_mathlib_Equiv_symm___redArg(v_e_u2082_825_);
v___x_829_ = lp_mathlib_Equiv_trans___redArg(v___x_827_, v___x_828_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr___redArg(lean_object* v_e_u2081_830_, lean_object* v_e_u2082_831_){
_start:
{
lean_object* v___f_832_; lean_object* v___f_833_; lean_object* v___x_834_; 
lean_inc_ref(v_e_u2082_831_);
lean_inc_ref(v_e_u2081_830_);
v___f_832_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relIsoCongr___redArg___lam__0), 3, 2);
lean_closure_set(v___f_832_, 0, v_e_u2081_830_);
lean_closure_set(v___f_832_, 1, v_e_u2082_831_);
v___f_833_ = lean_alloc_closure((void*)(lp_mathlib_RelIso_relIsoCongr___redArg___lam__1), 3, 2);
lean_closure_set(v___f_833_, 0, v_e_u2081_830_);
lean_closure_set(v___f_833_, 1, v_e_u2082_831_);
v___x_834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_834_, 0, v___f_832_);
lean_ctor_set(v___x_834_, 1, v___f_833_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoCongr(lean_object* v_00_u03b1_u2081_835_, lean_object* v_00_u03b2_u2081_836_, lean_object* v_00_u03b1_u2082_837_, lean_object* v_00_u03b2_u2082_838_, lean_object* v_r_u2081_839_, lean_object* v_s_u2081_840_, lean_object* v_r_u2082_841_, lean_object* v_s_u2082_842_, lean_object* v_e_u2081_843_, lean_object* v_e_u2082_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_mathlib_RelIso_relIsoCongr___redArg(v_e_u2081_843_, v_e_u2082_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexCongr___redArg(lean_object* v_e_u2081_846_, lean_object* v_e_u2082_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = lp_mathlib_Equiv_sumCongr___redArg(v_e_u2081_846_, v_e_u2082_847_);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexCongr(lean_object* v_00_u03b1_u2081_849_, lean_object* v_00_u03b1_u2082_850_, lean_object* v_00_u03b2_u2081_851_, lean_object* v_00_u03b2_u2082_852_, lean_object* v_r_u2081_853_, lean_object* v_r_u2082_854_, lean_object* v_s_u2081_855_, lean_object* v_s_u2082_856_, lean_object* v_e_u2081_857_, lean_object* v_e_u2082_858_){
_start:
{
lean_object* v___x_859_; 
v___x_859_ = lp_mathlib_Equiv_sumCongr___redArg(v_e_u2081_857_, v_e_u2082_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_prodLexCongr___redArg(lean_object* v_e_u2081_860_, lean_object* v_e_u2082_861_){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lp_mathlib_Equiv_prodCongr___redArg(v_e_u2081_860_, v_e_u2082_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_prodLexCongr(lean_object* v_00_u03b1_u2081_863_, lean_object* v_00_u03b1_u2082_864_, lean_object* v_00_u03b2_u2081_865_, lean_object* v_00_u03b2_u2082_866_, lean_object* v_r_u2081_867_, lean_object* v_r_u2082_868_, lean_object* v_s_u2081_869_, lean_object* v_s_u2082_870_, lean_object* v_e_u2081_871_, lean_object* v_e_u2082_872_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_mathlib_Equiv_prodCongr___redArg(v_e_u2081_871_, v_e_u2082_872_);
return v___x_873_;
}
}
static lean_object* _init_lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0(void){
_start:
{
lean_object* v___x_874_; 
v___x_874_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_relIsoOfIsEmpty(lean_object* v_00_u03b1_875_, lean_object* v_00_u03b2_876_, lean_object* v_r_877_, lean_object* v_s_878_, lean_object* v_inst_879_, lean_object* v_inst_880_){
_start:
{
lean_object* v___x_881_; 
v___x_881_ = lean_obj_once(&lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0, &lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0_once, _init_lp_mathlib_RelIso_relIsoOfIsEmpty___closed__0);
return v___x_881_;
}
}
static lean_object* _init_lp_mathlib_RelIso_sumLexEmpty___closed__0(void){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_mathlib_Equiv_sumEmpty(lean_box(0), lean_box(0), lean_box(0));
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_sumLexEmpty(lean_object* v_00_u03b1_883_, lean_object* v_00_u03b2_884_, lean_object* v_r_885_, lean_object* v_s_886_, lean_object* v_inst_887_){
_start:
{
lean_object* v___x_888_; 
v___x_888_ = lean_obj_once(&lp_mathlib_RelIso_sumLexEmpty___closed__0, &lp_mathlib_RelIso_sumLexEmpty___closed__0_once, _init_lp_mathlib_RelIso_sumLexEmpty___closed__0);
return v___x_888_;
}
}
static lean_object* _init_lp_mathlib_RelIso_emptySumLex___closed__0(void){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = lp_mathlib_Equiv_emptySum(lean_box(0), lean_box(0), lean_box(0));
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_emptySumLex(lean_object* v_00_u03b1_890_, lean_object* v_00_u03b2_891_, lean_object* v_r_892_, lean_object* v_s_893_, lean_object* v_inst_894_){
_start:
{
lean_object* v___x_895_; 
v___x_895_ = lean_obj_once(&lp_mathlib_RelIso_emptySumLex___closed__0, &lp_mathlib_RelIso_emptySumLex___closed__0_once, _init_lp_mathlib_RelIso_emptySumLex___closed__0);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfIrrefl___redArg(lean_object* v_inst_896_, lean_object* v_inst_897_){
_start:
{
lean_object* v___x_898_; 
v___x_898_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_896_, v_inst_897_);
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfIrrefl(lean_object* v_00_u03b1_899_, lean_object* v_00_u03b2_900_, lean_object* v_r_901_, lean_object* v_s_902_, lean_object* v_inst_903_, lean_object* v_inst_904_, lean_object* v_inst_905_, lean_object* v_inst_906_){
_start:
{
lean_object* v___x_907_; 
v___x_907_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_905_, v_inst_906_);
return v___x_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfRefl___redArg(lean_object* v_inst_908_, lean_object* v_inst_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_908_, v_inst_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofUniqueOfRefl(lean_object* v_00_u03b1_911_, lean_object* v_00_u03b2_912_, lean_object* v_r_913_, lean_object* v_s_914_, lean_object* v_inst_915_, lean_object* v_inst_916_, lean_object* v_inst_917_, lean_object* v_inst_918_){
_start:
{
lean_object* v___x_919_; 
v___x_919_ = lp_mathlib_Equiv_ofUnique___redArg(v_inst_917_, v_inst_918_);
return v___x_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___redArg(lean_object* v_f_920_){
_start:
{
lean_inc(v_f_920_);
return v_f_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___redArg___boxed(lean_object* v_f_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_mathlib_RelHom_toMap___redArg(v_f_921_);
lean_dec(v_f_921_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap(lean_object* v_00_u03b1_923_, lean_object* v_00_u03b2_924_, lean_object* v_r_925_, lean_object* v_f_926_){
_start:
{
lean_inc(v_f_926_);
return v_f_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toMap___boxed(lean_object* v_00_u03b1_927_, lean_object* v_00_u03b2_928_, lean_object* v_r_929_, lean_object* v_f_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_mathlib_RelHom_toMap(v_00_u03b1_927_, v_00_u03b2_928_, v_r_929_, v_f_930_);
lean_dec(v_f_930_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___redArg(lean_object* v_f_932_){
_start:
{
lean_inc(v_f_932_);
return v_f_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___redArg___boxed(lean_object* v_f_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_mathlib_RelEmbedding_toMap___redArg(v_f_933_);
lean_dec(v_f_933_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap(lean_object* v_00_u03b1_935_, lean_object* v_00_u03b2_936_, lean_object* v_r_937_, lean_object* v_f_938_){
_start:
{
lean_inc(v_f_938_);
return v_f_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_toMap___boxed(lean_object* v_00_u03b1_939_, lean_object* v_00_u03b2_940_, lean_object* v_r_941_, lean_object* v_f_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_mathlib_RelEmbedding_toMap(v_00_u03b1_939_, v_00_u03b2_940_, v_r_941_, v_f_942_);
lean_dec(v_f_942_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___redArg(lean_object* v_f_944_){
_start:
{
lean_inc_ref(v_f_944_);
return v_f_944_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___redArg___boxed(lean_object* v_f_945_){
_start:
{
lean_object* v_res_946_; 
v_res_946_ = lp_mathlib_RelIso_toMap___redArg(v_f_945_);
lean_dec_ref(v_f_945_);
return v_res_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap(lean_object* v_00_u03b1_947_, lean_object* v_00_u03b2_948_, lean_object* v_r_949_, lean_object* v_f_950_){
_start:
{
lean_inc_ref(v_f_950_);
return v_f_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_toMap___boxed(lean_object* v_00_u03b1_951_, lean_object* v_00_u03b2_952_, lean_object* v_r_953_, lean_object* v_f_954_){
_start:
{
lean_object* v_res_955_; 
v_res_955_ = lp_mathlib_RelIso_toMap(v_00_u03b1_951_, v_00_u03b2_952_, v_r_953_, v_f_954_);
lean_dec_ref(v_f_954_);
return v_res_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___redArg(lean_object* v_f_956_){
_start:
{
lean_inc(v_f_956_);
return v_f_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___redArg___boxed(lean_object* v_f_957_){
_start:
{
lean_object* v_res_958_; 
v_res_958_ = lp_mathlib_RelHom_ofOnFun___redArg(v_f_957_);
lean_dec(v_f_957_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun(lean_object* v_00_u03b1_959_, lean_object* v_00_u03b2_960_, lean_object* v_r_961_, lean_object* v_f_962_){
_start:
{
lean_inc(v_f_962_);
return v_f_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_ofOnFun___boxed(lean_object* v_00_u03b1_963_, lean_object* v_00_u03b2_964_, lean_object* v_r_965_, lean_object* v_f_966_){
_start:
{
lean_object* v_res_967_; 
v_res_967_ = lp_mathlib_RelHom_ofOnFun(v_00_u03b1_963_, v_00_u03b2_964_, v_r_965_, v_f_966_);
lean_dec(v_f_966_);
return v_res_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___redArg(lean_object* v_f_968_){
_start:
{
lean_inc(v_f_968_);
return v_f_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___redArg___boxed(lean_object* v_f_969_){
_start:
{
lean_object* v_res_970_; 
v_res_970_ = lp_mathlib_RelEmbedding_ofOnFun___redArg(v_f_969_);
lean_dec(v_f_969_);
return v_res_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun(lean_object* v_00_u03b1_971_, lean_object* v_00_u03b2_972_, lean_object* v_r_973_, lean_object* v_f_974_){
_start:
{
lean_inc(v_f_974_);
return v_f_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_ofOnFun___boxed(lean_object* v_00_u03b1_975_, lean_object* v_00_u03b2_976_, lean_object* v_r_977_, lean_object* v_f_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_RelEmbedding_ofOnFun(v_00_u03b1_975_, v_00_u03b2_976_, v_r_977_, v_f_978_);
lean_dec(v_f_978_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___redArg(lean_object* v_f_980_){
_start:
{
lean_inc_ref(v_f_980_);
return v_f_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___redArg___boxed(lean_object* v_f_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_mathlib_RelIso_ofOnFun___redArg(v_f_981_);
lean_dec_ref(v_f_981_);
return v_res_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun(lean_object* v_00_u03b1_983_, lean_object* v_00_u03b2_984_, lean_object* v_r_985_, lean_object* v_f_986_){
_start:
{
lean_inc_ref(v_f_986_);
return v_f_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelIso_ofOnFun___boxed(lean_object* v_00_u03b1_987_, lean_object* v_00_u03b2_988_, lean_object* v_r_989_, lean_object* v_f_990_){
_start:
{
lean_object* v_res_991_; 
v_res_991_ = lp_mathlib_RelIso_ofOnFun(v_00_u03b1_987_, v_00_u03b2_988_, v_r_989_, v_f_990_);
lean_dec_ref(v_f_990_);
return v_res_991_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
