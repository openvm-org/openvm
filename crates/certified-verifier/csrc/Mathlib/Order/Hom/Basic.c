// Lean compiler output
// Module: Mathlib.Order.Hom.Basic
// Imports: public import Init public meta import Init public import Mathlib.Order.Disjoint public import Mathlib.Order.RelIso.Basic public import Mathlib.Tactic.Monotonicity.Attr public import Mathlib.Tactic.PPWithUniv
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
lean_object* lp_mathlib_RelIso_relEmbeddingCongr___redArg(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_refl___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_piUnique___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* lp_mathlib_RelIso_relIsoCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodAssoc(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_impEmbedding___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_term___u2192o___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_→o_"};
static const lean_object* lp_mathlib_term___u2192o___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192o___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(18, 34, 173, 10, 172, 12, 235, 162)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2192o___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2192o___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192o___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2192o___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " →o "};
static const lean_object* lp_mathlib_term___u2192o___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2192o___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2192o___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192o___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__7_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2192o___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2192o___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2192o___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2192o___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2192o___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2192o__ = (const lean_object*)&lp_mathlib_term___u2192o___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderHom"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(8, 253, 133, 249, 118, 233, 61, 235)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u21aao___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_↪o_"};
static const lean_object* lp_mathlib_term___u21aao___00__closed__0 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u21aao___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21aao___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(148, 164, 227, 86, 250, 169, 87, 8)}};
static const lean_object* lp_mathlib_term___u21aao___00__closed__1 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u21aao___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ↪o "};
static const lean_object* lp_mathlib_term___u21aao___00__closed__2 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u21aao___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u21aao___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u21aao___00__closed__3 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u21aao___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u21aao___00__closed__4 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u21aao___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21aao___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21aao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u21aao___00__closed__5 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u21aao___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u21aao___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21aao___00__closed__5_value)}};
static const lean_object* lp_mathlib_term___u21aao___00__closed__6 = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u21aao__ = (const lean_object*)&lp_mathlib_term___u21aao___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "OrderEmbedding"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 170, 146, 165, 33, 84, 52, 200)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderEmbedding__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderEmbedding__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243o___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≃o_"};
static const lean_object* lp_mathlib_term___u2243o___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243o___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243o___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 81, 64, 32, 157, 16, 195, 78)}};
static const lean_object* lp_mathlib_term___u2243o___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243o___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≃o "};
static const lean_object* lp_mathlib_term___u2243o___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243o___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243o___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243o___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243o___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2192o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243o___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21aao___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243o___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243o___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243o___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243o___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243o___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243o__ = (const lean_object*)&lp_mathlib_term___u2243o___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderIso"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 144, 236, 224, 220, 223, 87, 89)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderIso__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderIso__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIsoClass_toOrderIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIsoClass_toOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderIsoOfOrderIsoClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderIsoOfOrderIsoClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_instCoeTCOrderHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_instCoeTCOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderHom_instDecidableEqOfForall___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instDecidableEqOfForall___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderHom_instDecidableEqOfForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instDecidableEqOfForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_OrderHom_id___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_equivRelHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_equivRelHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_equivRelHom___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_equivRelHom___closed__0_value;
static const lean_ctor_object lp_mathlib_OrderHom_equivRelHom___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderHom_equivRelHom___closed__0_value),((lean_object*)&lp_mathlib_OrderHom_equivRelHom___closed__0_value)}};
static const lean_object* lp_mathlib_OrderHom_equivRelHom___closed__1 = (const lean_object*)&lp_mathlib_OrderHom_equivRelHom___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_OrderHom_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OrderHom_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_curry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_curry___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_curry___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_curry___closed__0_value;
static const lean_closure_object lp_mathlib_OrderHom_curry___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_curry___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_curry___closed__1 = (const lean_object*)&lp_mathlib_OrderHom_curry___closed__1_value;
static const lean_ctor_object lp_mathlib_OrderHom_curry___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderHom_curry___closed__0_value),((lean_object*)&lp_mathlib_OrderHom_curry___closed__1_value)}};
static const lean_object* lp_mathlib_OrderHom_curry___closed__2 = (const lean_object*)&lp_mathlib_OrderHom_curry___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_comp_u2098___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_comp_u2098___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_comp_u2098___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_comp_u2098___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_const___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_const___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_const___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_const___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_prod_u2098___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_prod_u2098___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_prod_u2098___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_prod_u2098___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_diag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_prod___redArg___lam__0, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_OrderHom_id___closed__0_value),((lean_object*)&lp_mathlib_OrderHom_id___closed__0_value)} };
static const lean_object* lp_mathlib_OrderHom_diag___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_diag___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_diag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_diag___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_fst___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_snd___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_prodIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_prodIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_prodIso___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_prodIso___closed__0_value;
static const lean_ctor_object lp_mathlib_OrderHom_prodIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderHom_prodIso___closed__0_value),((lean_object*)&lp_mathlib_OrderHom_prod_u2098___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_OrderHom_prodIso___closed__1 = (const lean_object*)&lp_mathlib_OrderHom_prodIso___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_coeFnHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_coeFnHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_piIso___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_piIso___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_piIso___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_piIso___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_Subtype_val___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_Subtype_val___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_Subtype_val___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_Subtype_val___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subtype_orderEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subtype_impEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subtype_orderEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subtype_orderEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_unique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_unique___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderHom_dual___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderHom_dual___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderHom_dual___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_dual___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderHom_dual___closed__0 = (const lean_object*)&lp_mathlib_OrderHom_dual___closed__0_value;
static const lean_ctor_object lp_mathlib_OrderHom_dual___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderHom_dual___closed__0_value),((lean_object*)&lp_mathlib_OrderHom_dual___closed__0_value)}};
static const lean_object* lp_mathlib_OrderHom_dual___closed__1 = (const lean_object*)&lp_mathlib_OrderHom_dual___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderHom_dualIso___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderHom_dualIso___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderEmbedding_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_refl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderEmbedding_id___closed__0 = (const lean_object*)&lp_mathlib_OrderEmbedding_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderEmbedding_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderEmbedding_subtype___closed__0 = (const lean_object*)&lp_mathlib_OrderEmbedding_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_OrderEmbedding_ofIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_OrderEmbedding_ofIsEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toOrderEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toOrderEmbedding(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_refl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderIso_conj___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_toEmbedding___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderIso_conj___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderIso_conj___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_OrderIso_conj___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderIso_conj___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderIso_conj___redArg___closed__1 = (const lean_object*)&lp_mathlib_OrderIso_conj___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_OrderIso_conj___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderIso_conj___redArg___closed__0_value),((lean_object*)&lp_mathlib_OrderIso_conj___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_OrderIso_conj___redArg___closed__2 = (const lean_object*)&lp_mathlib_OrderIso_conj___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderEmbeddingCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderEmbeddingCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderIsoCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderIsoCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_prodComm___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_prodComm___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_prodComm(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_prodAssoc___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_prodAssoc___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_prodAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualDual(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderIso_ofIsEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderIso_ofIsEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofIsEmpty___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_ULift_orderIso___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ULift_orderIso___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ULift_orderIso(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_orderIso___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__5));
v___x_37_ = l_String_toRawSubstring_x27(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_term___u2192o___00__closed__1));
lean_inc(v_x_54_);
v___x_58_ = l_Lean_Syntax_isOfKind(v_x_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_x_54_);
v___x_59_ = lean_box(1);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
return v___x_60_;
}
else
{
lean_object* v_quotContext_61_; lean_object* v_currMacroScope_62_; lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_quotContext_61_ = lean_ctor_get(v_a_55_, 1);
v_currMacroScope_62_ = lean_ctor_get(v_a_55_, 2);
v_ref_63_ = lean_ctor_get(v_a_55_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_54_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_54_, v___x_66_);
lean_dec(v_x_54_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6, &lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__11));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
lean_inc(v_x_87_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_x_87_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_93_; 
lean_dec(v_x_87_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_87_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1));
lean_inc(v___x_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_95_);
lean_dec(v_x_87_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_89_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_87_, v___x_100_);
lean_dec(v_x_87_);
v___x_102_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v___x_95_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_89_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_106_ = l_Lean_Syntax_getArg(v___x_101_, v___x_94_);
v___x_107_ = l_Lean_Syntax_getArg(v___x_101_, v___x_100_);
lean_dec(v___x_101_);
v_ref_108_ = l_Lean_replaceRef(v___x_95_, v_a_88_);
lean_dec(v___x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
lean_dec(v_ref_108_);
v___x_111_ = ((lean_object*)(lp_mathlib_term___u2192o___00__closed__1));
v___x_112_ = ((lean_object*)(lp_mathlib_term___u2192o___00__closed__4));
lean_inc(v___x_110_);
v___x_113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = l_Lean_Syntax_node3(v___x_110_, v___x_111_, v___x_106_, v___x_113_, v___x_107_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_89_);
return v___x_115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_139_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__0));
v___x_140_ = l_String_toRawSubstring_x27(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1(lean_object* v_x_149_, lean_object* v_a_150_, lean_object* v_a_151_){
_start:
{
lean_object* v___x_152_; uint8_t v___x_153_; 
v___x_152_ = ((lean_object*)(lp_mathlib_term___u21aao___00__closed__1));
lean_inc(v_x_149_);
v___x_153_ = l_Lean_Syntax_isOfKind(v_x_149_, v___x_152_);
if (v___x_153_ == 0)
{
lean_object* v___x_154_; lean_object* v___x_155_; 
lean_dec(v_x_149_);
v___x_154_ = lean_box(1);
v___x_155_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
lean_ctor_set(v___x_155_, 1, v_a_151_);
return v___x_155_;
}
else
{
lean_object* v_quotContext_156_; lean_object* v_currMacroScope_157_; lean_object* v_ref_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; uint8_t v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v_quotContext_156_ = lean_ctor_get(v_a_150_, 1);
v_currMacroScope_157_ = lean_ctor_get(v_a_150_, 2);
v_ref_158_ = lean_ctor_get(v_a_150_, 5);
v___x_159_ = lean_unsigned_to_nat(0u);
v___x_160_ = l_Lean_Syntax_getArg(v_x_149_, v___x_159_);
v___x_161_ = lean_unsigned_to_nat(2u);
v___x_162_ = l_Lean_Syntax_getArg(v_x_149_, v___x_161_);
lean_dec(v_x_149_);
v___x_163_ = 0;
v___x_164_ = l_Lean_SourceInfo_fromRef(v_ref_158_, v___x_163_);
v___x_165_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
v___x_166_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__1);
v___x_167_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__2));
lean_inc(v_currMacroScope_157_);
lean_inc(v_quotContext_156_);
v___x_168_ = l_Lean_addMacroScope(v_quotContext_156_, v___x_167_, v_currMacroScope_157_);
v___x_169_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___closed__4));
lean_inc_n(v___x_164_, 2);
v___x_170_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_170_, 0, v___x_164_);
lean_ctor_set(v___x_170_, 1, v___x_166_);
lean_ctor_set(v___x_170_, 2, v___x_168_);
lean_ctor_set(v___x_170_, 3, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13));
v___x_172_ = l_Lean_Syntax_node2(v___x_164_, v___x_171_, v___x_160_, v___x_162_);
v___x_173_ = l_Lean_Syntax_node2(v___x_164_, v___x_165_, v___x_170_, v___x_172_);
v___x_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_151_);
return v___x_174_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1___boxed(lean_object* v_x_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u21aao____1(v_x_175_, v_a_176_, v_a_177_);
lean_dec_ref(v_a_176_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderEmbedding__1(lean_object* v_x_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v___x_182_; uint8_t v___x_183_; 
v___x_182_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
lean_inc(v_x_179_);
v___x_183_ = l_Lean_Syntax_isOfKind(v_x_179_, v___x_182_);
if (v___x_183_ == 0)
{
lean_object* v___x_184_; lean_object* v___x_185_; 
lean_dec(v_x_179_);
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
lean_ctor_set(v___x_185_, 1, v_a_181_);
return v___x_185_;
}
else
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; uint8_t v___x_189_; 
v___x_186_ = lean_unsigned_to_nat(0u);
v___x_187_ = l_Lean_Syntax_getArg(v_x_179_, v___x_186_);
v___x_188_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1));
lean_inc(v___x_187_);
v___x_189_ = l_Lean_Syntax_isOfKind(v___x_187_, v___x_188_);
if (v___x_189_ == 0)
{
lean_object* v___x_190_; lean_object* v___x_191_; 
lean_dec(v___x_187_);
lean_dec(v_x_179_);
v___x_190_ = lean_box(0);
v___x_191_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v_a_181_);
return v___x_191_;
}
else
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_192_ = lean_unsigned_to_nat(1u);
v___x_193_ = l_Lean_Syntax_getArg(v_x_179_, v___x_192_);
lean_dec(v_x_179_);
v___x_194_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_193_);
v___x_195_ = l_Lean_Syntax_matchesNull(v___x_193_, v___x_194_);
if (v___x_195_ == 0)
{
lean_object* v___x_196_; lean_object* v___x_197_; 
lean_dec(v___x_193_);
lean_dec(v___x_187_);
v___x_196_ = lean_box(0);
v___x_197_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v_a_181_);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_ref_200_; uint8_t v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_198_ = l_Lean_Syntax_getArg(v___x_193_, v___x_186_);
v___x_199_ = l_Lean_Syntax_getArg(v___x_193_, v___x_192_);
lean_dec(v___x_193_);
v_ref_200_ = l_Lean_replaceRef(v___x_187_, v_a_180_);
lean_dec(v___x_187_);
v___x_201_ = 0;
v___x_202_ = l_Lean_SourceInfo_fromRef(v_ref_200_, v___x_201_);
lean_dec(v_ref_200_);
v___x_203_ = ((lean_object*)(lp_mathlib_term___u21aao___00__closed__1));
v___x_204_ = ((lean_object*)(lp_mathlib_term___u21aao___00__closed__2));
lean_inc(v___x_202_);
v___x_205_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_202_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = l_Lean_Syntax_node3(v___x_202_, v___x_203_, v___x_198_, v___x_205_, v___x_199_);
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v_a_181_);
return v___x_207_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderEmbedding__1___boxed(lean_object* v_x_208_, lean_object* v_a_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderEmbedding__1(v_x_208_, v_a_209_, v_a_210_);
lean_dec(v_a_209_);
return v_res_211_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1(void){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__0));
v___x_229_ = l_String_toRawSubstring_x27(v___x_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1(lean_object* v_x_238_, lean_object* v_a_239_, lean_object* v_a_240_){
_start:
{
lean_object* v___x_241_; uint8_t v___x_242_; 
v___x_241_ = ((lean_object*)(lp_mathlib_term___u2243o___00__closed__1));
lean_inc(v_x_238_);
v___x_242_ = l_Lean_Syntax_isOfKind(v_x_238_, v___x_241_);
if (v___x_242_ == 0)
{
lean_object* v___x_243_; lean_object* v___x_244_; 
lean_dec(v_x_238_);
v___x_243_ = lean_box(1);
v___x_244_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v_a_240_);
return v___x_244_;
}
else
{
lean_object* v_quotContext_245_; lean_object* v_currMacroScope_246_; lean_object* v_ref_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; uint8_t v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v_quotContext_245_ = lean_ctor_get(v_a_239_, 1);
v_currMacroScope_246_ = lean_ctor_get(v_a_239_, 2);
v_ref_247_ = lean_ctor_get(v_a_239_, 5);
v___x_248_ = lean_unsigned_to_nat(0u);
v___x_249_ = l_Lean_Syntax_getArg(v_x_238_, v___x_248_);
v___x_250_ = lean_unsigned_to_nat(2u);
v___x_251_ = l_Lean_Syntax_getArg(v_x_238_, v___x_250_);
lean_dec(v_x_238_);
v___x_252_ = 0;
v___x_253_ = l_Lean_SourceInfo_fromRef(v_ref_247_, v___x_252_);
v___x_254_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
v___x_255_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__1);
v___x_256_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__2));
lean_inc(v_currMacroScope_246_);
lean_inc(v_quotContext_245_);
v___x_257_ = l_Lean_addMacroScope(v_quotContext_245_, v___x_256_, v_currMacroScope_246_);
v___x_258_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___closed__4));
lean_inc_n(v___x_253_, 2);
v___x_259_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_259_, 0, v___x_253_);
lean_ctor_set(v___x_259_, 1, v___x_255_);
lean_ctor_set(v___x_259_, 2, v___x_257_);
lean_ctor_set(v___x_259_, 3, v___x_258_);
v___x_260_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__13));
v___x_261_ = l_Lean_Syntax_node2(v___x_253_, v___x_260_, v___x_249_, v___x_251_);
v___x_262_ = l_Lean_Syntax_node2(v___x_253_, v___x_254_, v___x_259_, v___x_261_);
v___x_263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_240_);
return v___x_263_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1___boxed(lean_object* v_x_264_, lean_object* v_a_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2243o____1(v_x_264_, v_a_265_, v_a_266_);
lean_dec_ref(v_a_265_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderIso__1(lean_object* v_x_268_, lean_object* v_a_269_, lean_object* v_a_270_){
_start:
{
lean_object* v___x_271_; uint8_t v___x_272_; 
v___x_271_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______macroRules__term___u2192o____1___closed__4));
lean_inc(v_x_268_);
v___x_272_ = l_Lean_Syntax_isOfKind(v_x_268_, v___x_271_);
if (v___x_272_ == 0)
{
lean_object* v___x_273_; lean_object* v___x_274_; 
lean_dec(v_x_268_);
v___x_273_ = lean_box(0);
v___x_274_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_274_, 0, v___x_273_);
lean_ctor_set(v___x_274_, 1, v_a_270_);
return v___x_274_;
}
else
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; uint8_t v___x_278_; 
v___x_275_ = lean_unsigned_to_nat(0u);
v___x_276_ = l_Lean_Syntax_getArg(v_x_268_, v___x_275_);
v___x_277_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderHom__1___closed__1));
lean_inc(v___x_276_);
v___x_278_ = l_Lean_Syntax_isOfKind(v___x_276_, v___x_277_);
if (v___x_278_ == 0)
{
lean_object* v___x_279_; lean_object* v___x_280_; 
lean_dec(v___x_276_);
lean_dec(v_x_268_);
v___x_279_ = lean_box(0);
v___x_280_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set(v___x_280_, 1, v_a_270_);
return v___x_280_;
}
else
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; uint8_t v___x_284_; 
v___x_281_ = lean_unsigned_to_nat(1u);
v___x_282_ = l_Lean_Syntax_getArg(v_x_268_, v___x_281_);
lean_dec(v_x_268_);
v___x_283_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_282_);
v___x_284_ = l_Lean_Syntax_matchesNull(v___x_282_, v___x_283_);
if (v___x_284_ == 0)
{
lean_object* v___x_285_; lean_object* v___x_286_; 
lean_dec(v___x_282_);
lean_dec(v___x_276_);
v___x_285_ = lean_box(0);
v___x_286_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
lean_ctor_set(v___x_286_, 1, v_a_270_);
return v___x_286_;
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v_ref_289_; uint8_t v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_287_ = l_Lean_Syntax_getArg(v___x_282_, v___x_275_);
v___x_288_ = l_Lean_Syntax_getArg(v___x_282_, v___x_281_);
lean_dec(v___x_282_);
v_ref_289_ = l_Lean_replaceRef(v___x_276_, v_a_269_);
lean_dec(v___x_276_);
v___x_290_ = 0;
v___x_291_ = l_Lean_SourceInfo_fromRef(v_ref_289_, v___x_290_);
lean_dec(v_ref_289_);
v___x_292_ = ((lean_object*)(lp_mathlib_term___u2243o___00__closed__1));
v___x_293_ = ((lean_object*)(lp_mathlib_term___u2243o___00__closed__2));
lean_inc(v___x_291_);
v___x_294_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_291_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = l_Lean_Syntax_node3(v___x_291_, v___x_292_, v___x_287_, v___x_294_, v___x_288_);
v___x_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
lean_ctor_set(v___x_296_, 1, v_a_270_);
return v___x_296_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderIso__1___boxed(lean_object* v_x_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib___aux__Mathlib__Order__Hom__Basic______unexpand__OrderIso__1(v_x_297_, v_a_298_, v_a_299_);
lean_dec(v_a_298_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofClass___redArg(lean_object* v_inst_301_, lean_object* v_f_302_){
_start:
{
lean_object* v___x_303_; 
v___x_303_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_301_, v_f_302_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofClass(lean_object* v_F_304_, lean_object* v_00_u03b1_305_, lean_object* v_00_u03b2_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_f_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_309_, v_f_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIsoClass_toOrderIso___redArg(lean_object* v_inst_313_, lean_object* v_f_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_313_, v_f_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIsoClass_toOrderIso(lean_object* v_F_316_, lean_object* v_00_u03b1_317_, lean_object* v_00_u03b2_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_f_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_321_, v_f_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderIsoOfOrderIsoClass___redArg(lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_ofClass), 8, 7);
lean_closure_set(v___x_328_, 0, lean_box(0));
lean_closure_set(v___x_328_, 1, lean_box(0));
lean_closure_set(v___x_328_, 2, lean_box(0));
lean_closure_set(v___x_328_, 3, v_inst_325_);
lean_closure_set(v___x_328_, 4, v_inst_326_);
lean_closure_set(v___x_328_, 5, v_inst_327_);
lean_closure_set(v___x_328_, 6, lean_box(0));
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCOrderIsoOfOrderIsoClass(lean_object* v_F_329_, lean_object* v_00_u03b1_330_, lean_object* v_00_u03b2_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_ofClass), 8, 7);
lean_closure_set(v___x_336_, 0, lean_box(0));
lean_closure_set(v___x_336_, 1, lean_box(0));
lean_closure_set(v___x_336_, 2, lean_box(0));
lean_closure_set(v___x_336_, 3, v_inst_332_);
lean_closure_set(v___x_336_, 4, v_inst_333_);
lean_closure_set(v___x_336_, 5, v_inst_334_);
lean_closure_set(v___x_336_, 6, lean_box(0));
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass___redArg(lean_object* v_inst_337_, lean_object* v_f_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lean_apply_1(v_inst_337_, v_f_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass(lean_object* v_F_340_, lean_object* v_00_u03b1_341_, lean_object* v_00_u03b2_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_f_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lean_apply_1(v_inst_345_, v_f_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_ofClass___boxed(lean_object* v_F_349_, lean_object* v_00_u03b1_350_, lean_object* v_00_u03b2_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_f_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_OrderHom_ofClass(v_F_349_, v_00_u03b1_350_, v_00_u03b2_351_, v_inst_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_f_356_);
lean_dec_ref(v_inst_353_);
lean_dec_ref(v_inst_352_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_instCoeTCOrderHom___redArg(lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_ofClass___boxed), 8, 7);
lean_closure_set(v___x_361_, 0, lean_box(0));
lean_closure_set(v___x_361_, 1, lean_box(0));
lean_closure_set(v___x_361_, 2, lean_box(0));
lean_closure_set(v___x_361_, 3, v_inst_358_);
lean_closure_set(v___x_361_, 4, v_inst_359_);
lean_closure_set(v___x_361_, 5, v_inst_360_);
lean_closure_set(v___x_361_, 6, lean_box(0));
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_instCoeTCOrderHom(lean_object* v_F_362_, lean_object* v_00_u03b1_363_, lean_object* v_00_u03b2_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_ofClass___boxed), 8, 7);
lean_closure_set(v___x_369_, 0, lean_box(0));
lean_closure_set(v___x_369_, 1, lean_box(0));
lean_closure_set(v___x_369_, 2, lean_box(0));
lean_closure_set(v___x_369_, 3, v_inst_365_);
lean_closure_set(v___x_369_, 4, v_inst_366_);
lean_closure_set(v___x_369_, 5, v_inst_367_);
lean_closure_set(v___x_369_, 6, lean_box(0));
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom___redArg(lean_object* v_inst_370_, lean_object* v_f_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lean_apply_1(v_inst_370_, v_f_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom(lean_object* v_F_373_, lean_object* v_00_u03b1_374_, lean_object* v_00_u03b2_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_f_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lean_apply_1(v_inst_378_, v_f_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toOrderHom___boxed(lean_object* v_F_382_, lean_object* v_00_u03b1_383_, lean_object* v_00_u03b2_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_f_389_){
_start:
{
lean_object* v_res_390_; 
v_res_390_ = lp_mathlib_OrderHomClass_toOrderHom(v_F_382_, v_00_u03b1_383_, v_00_u03b2_384_, v_inst_385_, v_inst_386_, v_inst_387_, v_inst_388_, v_f_389_);
lean_dec_ref(v_inst_386_);
lean_dec_ref(v_inst_385_);
return v_res_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe___redArg(lean_object* v_f_391_, lean_object* v_a_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_apply_1(v_f_391_, v_a_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe(lean_object* v_00_u03b1_394_, lean_object* v_00_u03b2_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_f_398_, lean_object* v_a_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_apply_1(v_f_398_, v_a_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Simps_coe___boxed(lean_object* v_00_u03b1_401_, lean_object* v_00_u03b2_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_f_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_OrderHom_Simps_coe(v_00_u03b1_401_, v_00_u03b2_402_, v_inst_403_, v_inst_404_, v_f_405_, v_a_406_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___redArg(lean_object* v_f_x27_408_){
_start:
{
lean_inc(v_f_x27_408_);
return v_f_x27_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___redArg___boxed(lean_object* v_f_x27_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_OrderHom_copy___redArg(v_f_x27_409_);
lean_dec(v_f_x27_409_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy(lean_object* v_00_u03b1_411_, lean_object* v_00_u03b2_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_f_415_, lean_object* v_f_x27_416_, lean_object* v_h_417_){
_start:
{
lean_inc(v_f_x27_416_);
return v_f_x27_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_copy___boxed(lean_object* v_00_u03b1_418_, lean_object* v_00_u03b2_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_f_422_, lean_object* v_f_x27_423_, lean_object* v_h_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_OrderHom_copy(v_00_u03b1_418_, v_00_u03b2_419_, v_inst_420_, v_inst_421_, v_f_422_, v_f_x27_423_, v_h_424_);
lean_dec(v_f_x27_423_);
lean_dec(v_f_422_);
lean_dec_ref(v_inst_421_);
lean_dec_ref(v_inst_420_);
return v_res_425_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderHom_instDecidableEqOfForall___redArg(lean_object* v_inst_426_, lean_object* v_a_427_, lean_object* v_b_428_){
_start:
{
lean_object* v___x_429_; uint8_t v___x_430_; 
v___x_429_ = lean_apply_2(v_inst_426_, v_a_427_, v_b_428_);
v___x_430_ = lean_unbox(v___x_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instDecidableEqOfForall___redArg___boxed(lean_object* v_inst_431_, lean_object* v_a_432_, lean_object* v_b_433_){
_start:
{
uint8_t v_res_434_; lean_object* v_r_435_; 
v_res_434_ = lp_mathlib_OrderHom_instDecidableEqOfForall___redArg(v_inst_431_, v_a_432_, v_b_433_);
v_r_435_ = lean_box(v_res_434_);
return v_r_435_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderHom_instDecidableEqOfForall(lean_object* v_00_u03b1_436_, lean_object* v_00_u03b2_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_a_441_, lean_object* v_b_442_){
_start:
{
lean_object* v___x_443_; uint8_t v___x_444_; 
v___x_443_ = lean_apply_2(v_inst_440_, v_a_441_, v_b_442_);
v___x_444_ = lean_unbox(v___x_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instDecidableEqOfForall___boxed(lean_object* v_00_u03b1_445_, lean_object* v_00_u03b2_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_a_450_, lean_object* v_b_451_){
_start:
{
uint8_t v_res_452_; lean_object* v_r_453_; 
v_res_452_ = lp_mathlib_OrderHom_instDecidableEqOfForall(v_00_u03b1_445_, v_00_u03b2_446_, v_inst_447_, v_inst_448_, v_inst_449_, v_a_450_, v_b_451_);
lean_dec_ref(v_inst_448_);
lean_dec_ref(v_inst_447_);
v_r_453_ = lean_box(v_res_452_);
return v_r_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_id(lean_object* v_00_u03b1_455_, lean_object* v_inst_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = ((lean_object*)(lp_mathlib_OrderHom_id___closed__0));
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_id___boxed(lean_object* v_00_u03b1_458_, lean_object* v_inst_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_OrderHom_id(v_00_u03b1_458_, v_inst_459_);
lean_dec_ref(v_inst_459_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInhabited(lean_object* v_00_u03b1_461_, lean_object* v_inst_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = ((lean_object*)(lp_mathlib_OrderHom_id___closed__0));
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instInhabited___boxed(lean_object* v_00_u03b1_464_, lean_object* v_inst_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib_OrderHom_instInhabited(v_00_u03b1_464_, v_inst_465_);
lean_dec_ref(v_inst_465_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom___lam__0(lean_object* v_f_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lean_apply_1(v_f_467_, v___y_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom(lean_object* v_00_u03b1_473_, lean_object* v_00_u03b2_474_, lean_object* v_inst_475_, lean_object* v_inst_476_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = ((lean_object*)(lp_mathlib_OrderHom_equivRelHom___closed__1));
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_equivRelHom___boxed(lean_object* v_00_u03b1_478_, lean_object* v_00_u03b2_479_, lean_object* v_inst_480_, lean_object* v_inst_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_OrderHom_equivRelHom(v_00_u03b1_478_, v_00_u03b2_479_, v_inst_480_, v_inst_481_);
lean_dec_ref(v_inst_481_);
lean_dec_ref(v_inst_480_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPreorder(lean_object* v_00_u03b1_486_, lean_object* v_00_u03b2_487_, lean_object* v_inst_488_, lean_object* v_inst_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = ((lean_object*)(lp_mathlib_OrderHom_instPreorder___closed__0));
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPreorder___boxed(lean_object* v_00_u03b1_491_, lean_object* v_00_u03b2_492_, lean_object* v_inst_493_, lean_object* v_inst_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_mathlib_OrderHom_instPreorder(v_00_u03b1_491_, v_00_u03b2_492_, v_inst_493_, v_inst_494_);
lean_dec_ref(v_inst_494_);
lean_dec_ref(v_inst_493_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPartialOrder(lean_object* v_00_u03b1_496_, lean_object* v_inst_497_, lean_object* v_00_u03b2_498_, lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = ((lean_object*)(lp_mathlib_OrderHom_instPreorder___closed__0));
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_instPartialOrder___boxed(lean_object* v_00_u03b1_501_, lean_object* v_inst_502_, lean_object* v_00_u03b2_503_, lean_object* v_inst_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_OrderHom_instPartialOrder(v_00_u03b1_501_, v_inst_502_, v_00_u03b2_503_, v_inst_504_);
lean_dec_ref(v_inst_504_);
lean_dec_ref(v_inst_502_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___lam__0(lean_object* v_f_506_, lean_object* v___y_507_, lean_object* v___y_508_){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_509_, 0, v___y_507_);
lean_ctor_set(v___x_509_, 1, v___y_508_);
v___x_510_ = lean_apply_1(v_f_506_, v___x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___lam__1(lean_object* v_f_511_, lean_object* v___y_512_){
_start:
{
lean_object* v_fst_513_; lean_object* v_snd_514_; lean_object* v___x_515_; 
v_fst_513_ = lean_ctor_get(v___y_512_, 0);
lean_inc(v_fst_513_);
v_snd_514_ = lean_ctor_get(v___y_512_, 1);
lean_inc(v_snd_514_);
lean_dec_ref(v___y_512_);
v___x_515_ = lean_apply_2(v_f_511_, v_fst_513_, v_snd_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry(lean_object* v_00_u03b1_521_, lean_object* v_00_u03b2_522_, lean_object* v_00_u03b3_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_inst_526_){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = ((lean_object*)(lp_mathlib_OrderHom_curry___closed__2));
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_curry___boxed(lean_object* v_00_u03b1_528_, lean_object* v_00_u03b2_529_, lean_object* v_00_u03b3_530_, lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_OrderHom_curry(v_00_u03b1_528_, v_00_u03b2_529_, v_00_u03b3_530_, v_inst_531_, v_inst_532_, v_inst_533_);
lean_dec_ref(v_inst_533_);
lean_dec_ref(v_inst_532_);
lean_dec_ref(v_inst_531_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___redArg___lam__0(lean_object* v_g_535_, lean_object* v___y_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lean_apply_1(v_g_535_, v___y_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___redArg(lean_object* v_g_538_, lean_object* v_f_539_){
_start:
{
lean_object* v___f_540_; lean_object* v___f_541_; lean_object* v___x_542_; 
v___f_540_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_540_, 0, v_g_538_);
v___f_541_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_541_, 0, v_f_539_);
v___x_542_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_542_, 0, lean_box(0));
lean_closure_set(v___x_542_, 1, lean_box(0));
lean_closure_set(v___x_542_, 2, lean_box(0));
lean_closure_set(v___x_542_, 3, v___f_540_);
lean_closure_set(v___x_542_, 4, v___f_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp(lean_object* v_00_u03b1_543_, lean_object* v_00_u03b2_544_, lean_object* v_00_u03b3_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_g_549_, lean_object* v_f_550_){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib_OrderHom_comp___redArg(v_g_549_, v_f_550_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp___boxed(lean_object* v_00_u03b1_552_, lean_object* v_00_u03b2_553_, lean_object* v_00_u03b3_554_, lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_g_558_, lean_object* v_f_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_OrderHom_comp(v_00_u03b1_552_, v_00_u03b2_553_, v_00_u03b3_554_, v_inst_555_, v_inst_556_, v_inst_557_, v_g_558_, v_f_559_);
lean_dec_ref(v_inst_557_);
lean_dec_ref(v_inst_556_);
lean_dec_ref(v_inst_555_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg___lam__0(lean_object* v_f_561_, lean_object* v___y_562_){
_start:
{
lean_object* v_fst_563_; lean_object* v_snd_564_; lean_object* v___x_51__overap_565_; lean_object* v___x_566_; 
v_fst_563_ = lean_ctor_get(v_f_561_, 0);
lean_inc(v_fst_563_);
v_snd_564_ = lean_ctor_get(v_f_561_, 1);
lean_inc(v_snd_564_);
lean_dec_ref(v_f_561_);
v___x_51__overap_565_ = lp_mathlib_OrderHom_comp___redArg(v_fst_563_, v_snd_564_);
v___x_566_ = lean_apply_1(v___x_51__overap_565_, v___y_562_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg(lean_object* v_inst_568_, lean_object* v_inst_569_, lean_object* v_inst_570_){
_start:
{
lean_object* v___f_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___f_571_ = ((lean_object*)(lp_mathlib_OrderHom_comp_u2098___redArg___closed__0));
v___x_572_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_569_, v_inst_570_);
v___x_573_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_568_, v_inst_569_);
v___x_574_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_568_, v_inst_570_);
v___x_575_ = lp_mathlib_OrderHom_curry(lean_box(0), lean_box(0), lean_box(0), v___x_572_, v___x_573_, v___x_574_);
lean_dec_ref(v___x_574_);
lean_dec_ref(v___x_573_);
lean_dec_ref(v___x_572_);
v___x_576_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_575_, v___f_571_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___redArg___boxed(lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_mathlib_OrderHom_comp_u2098___redArg(v_inst_577_, v_inst_578_, v_inst_579_);
lean_dec_ref(v_inst_579_);
lean_dec_ref(v_inst_578_);
lean_dec_ref(v_inst_577_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098(lean_object* v_00_u03b1_581_, lean_object* v_00_u03b2_582_, lean_object* v_00_u03b3_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = lp_mathlib_OrderHom_comp_u2098___redArg(v_inst_584_, v_inst_585_, v_inst_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_comp_u2098___boxed(lean_object* v_00_u03b1_588_, lean_object* v_00_u03b2_589_, lean_object* v_00_u03b3_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib_OrderHom_comp_u2098(v_00_u03b1_588_, v_00_u03b2_589_, v_00_u03b3_590_, v_inst_591_, v_inst_592_, v_inst_593_);
lean_dec_ref(v_inst_593_);
lean_dec_ref(v_inst_592_);
lean_dec_ref(v_inst_591_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___lam__0(lean_object* v_b_595_, lean_object* v___y_596_){
_start:
{
lean_inc(v_b_595_);
return v_b_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___lam__0___boxed(lean_object* v_b_597_, lean_object* v___y_598_){
_start:
{
lean_object* v_res_599_; 
v_res_599_ = lp_mathlib_OrderHom_const___lam__0(v_b_597_, v___y_598_);
lean_dec(v___y_598_);
lean_dec(v_b_597_);
return v_res_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const(lean_object* v_00_u03b1_601_, lean_object* v_inst_602_, lean_object* v_00_u03b2_603_, lean_object* v_inst_604_){
_start:
{
lean_object* v___f_605_; 
v___f_605_ = ((lean_object*)(lp_mathlib_OrderHom_const___closed__0));
return v___f_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_const___boxed(lean_object* v_00_u03b1_606_, lean_object* v_inst_607_, lean_object* v_00_u03b2_608_, lean_object* v_inst_609_){
_start:
{
lean_object* v_res_610_; 
v_res_610_ = lp_mathlib_OrderHom_const(v_00_u03b1_606_, v_inst_607_, v_00_u03b2_608_, v_inst_609_);
lean_dec_ref(v_inst_609_);
lean_dec_ref(v_inst_607_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___redArg___lam__0(lean_object* v_f_611_, lean_object* v_g_612_, lean_object* v_x_613_){
_start:
{
lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
lean_inc(v_x_613_);
v___x_614_ = lean_apply_1(v_f_611_, v_x_613_);
v___x_615_ = lean_apply_1(v_g_612_, v_x_613_);
v___x_616_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_614_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___redArg(lean_object* v_f_617_, lean_object* v_g_618_){
_start:
{
lean_object* v___f_619_; 
v___f_619_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_619_, 0, v_f_617_);
lean_closure_set(v___f_619_, 1, v_g_618_);
return v___f_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod(lean_object* v_00_u03b1_620_, lean_object* v_00_u03b2_621_, lean_object* v_00_u03b3_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_f_626_, lean_object* v_g_627_){
_start:
{
lean_object* v___f_628_; 
v___f_628_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_628_, 0, v_f_626_);
lean_closure_set(v___f_628_, 1, v_g_627_);
return v___f_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod___boxed(lean_object* v_00_u03b1_629_, lean_object* v_00_u03b2_630_, lean_object* v_00_u03b3_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_f_635_, lean_object* v_g_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_OrderHom_prod(v_00_u03b1_629_, v_00_u03b2_630_, v_00_u03b3_631_, v_inst_632_, v_inst_633_, v_inst_634_, v_f_635_, v_g_636_);
lean_dec_ref(v_inst_634_);
lean_dec_ref(v_inst_633_);
lean_dec_ref(v_inst_632_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg___lam__0(lean_object* v_f_638_, lean_object* v___y_639_){
_start:
{
lean_object* v_fst_640_; lean_object* v_snd_641_; lean_object* v___x_642_; 
v_fst_640_ = lean_ctor_get(v_f_638_, 0);
lean_inc(v_fst_640_);
v_snd_641_ = lean_ctor_get(v_f_638_, 1);
lean_inc(v_snd_641_);
lean_dec_ref(v_f_638_);
v___x_642_ = lp_mathlib_OrderHom_prod___redArg___lam__0(v_fst_640_, v_snd_641_, v___y_639_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg(lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_inst_646_){
_start:
{
lean_object* v___f_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___f_647_ = ((lean_object*)(lp_mathlib_OrderHom_prod_u2098___redArg___closed__0));
v___x_648_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_644_, v_inst_645_);
v___x_649_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_644_, v_inst_646_);
v___x_650_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_inst_645_, v_inst_646_);
v___x_651_ = lp_mathlib_OrderHom_instPreorder(lean_box(0), lean_box(0), v_inst_644_, v___x_650_);
lean_dec_ref(v___x_650_);
v___x_652_ = lp_mathlib_OrderHom_curry(lean_box(0), lean_box(0), lean_box(0), v___x_648_, v___x_649_, v___x_651_);
lean_dec_ref(v___x_651_);
lean_dec_ref(v___x_649_);
lean_dec_ref(v___x_648_);
v___x_653_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_652_, v___f_647_);
return v___x_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___redArg___boxed(lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_){
_start:
{
lean_object* v_res_657_; 
v_res_657_ = lp_mathlib_OrderHom_prod_u2098___redArg(v_inst_654_, v_inst_655_, v_inst_656_);
lean_dec_ref(v_inst_656_);
lean_dec_ref(v_inst_655_);
lean_dec_ref(v_inst_654_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098(lean_object* v_00_u03b1_658_, lean_object* v_00_u03b2_659_, lean_object* v_00_u03b3_660_, lean_object* v_inst_661_, lean_object* v_inst_662_, lean_object* v_inst_663_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lp_mathlib_OrderHom_prod_u2098___redArg(v_inst_661_, v_inst_662_, v_inst_663_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prod_u2098___boxed(lean_object* v_00_u03b1_665_, lean_object* v_00_u03b2_666_, lean_object* v_00_u03b3_667_, lean_object* v_inst_668_, lean_object* v_inst_669_, lean_object* v_inst_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib_OrderHom_prod_u2098(v_00_u03b1_665_, v_00_u03b2_666_, v_00_u03b3_667_, v_inst_668_, v_inst_669_, v_inst_670_);
lean_dec_ref(v_inst_670_);
lean_dec_ref(v_inst_669_);
lean_dec_ref(v_inst_668_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_diag(lean_object* v_00_u03b1_674_, lean_object* v_inst_675_){
_start:
{
lean_object* v___f_676_; 
v___f_676_ = ((lean_object*)(lp_mathlib_OrderHom_diag___closed__0));
return v___f_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_diag___boxed(lean_object* v_00_u03b1_677_, lean_object* v_inst_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_OrderHom_diag(v_00_u03b1_677_, v_inst_678_);
lean_dec_ref(v_inst_678_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___redArg(lean_object* v_inst_680_, lean_object* v_inst_681_, lean_object* v_f_682_){
_start:
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_683_ = lp_mathlib_OrderHom_curry(lean_box(0), lean_box(0), lean_box(0), v_inst_680_, v_inst_680_, v_inst_681_);
v___x_684_ = lp_mathlib_Equiv_symm___redArg(v___x_683_);
v___x_685_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_684_, v_f_682_);
v___x_686_ = lp_mathlib_OrderHom_diag(lean_box(0), v_inst_680_);
v___x_687_ = lp_mathlib_OrderHom_comp___redArg(v___x_685_, v___x_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___redArg___boxed(lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_f_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib_OrderHom_onDiag___redArg(v_inst_688_, v_inst_689_, v_f_690_);
lean_dec_ref(v_inst_689_);
lean_dec_ref(v_inst_688_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag(lean_object* v_00_u03b1_692_, lean_object* v_00_u03b2_693_, lean_object* v_inst_694_, lean_object* v_inst_695_, lean_object* v_f_696_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lp_mathlib_OrderHom_onDiag___redArg(v_inst_694_, v_inst_695_, v_f_696_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_onDiag___boxed(lean_object* v_00_u03b1_698_, lean_object* v_00_u03b2_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_f_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_mathlib_OrderHom_onDiag(v_00_u03b1_698_, v_00_u03b2_699_, v_inst_700_, v_inst_701_, v_f_702_);
lean_dec_ref(v_inst_701_);
lean_dec_ref(v_inst_700_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___lam__0(lean_object* v_self_704_){
_start:
{
lean_object* v_fst_705_; 
v_fst_705_ = lean_ctor_get(v_self_704_, 0);
lean_inc(v_fst_705_);
return v_fst_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___lam__0___boxed(lean_object* v_self_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_OrderHom_fst___lam__0(v_self_706_);
lean_dec_ref(v_self_706_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst(lean_object* v_00_u03b1_709_, lean_object* v_00_u03b2_710_, lean_object* v_inst_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v___f_713_; 
v___f_713_ = ((lean_object*)(lp_mathlib_OrderHom_fst___closed__0));
return v___f_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_fst___boxed(lean_object* v_00_u03b1_714_, lean_object* v_00_u03b2_715_, lean_object* v_inst_716_, lean_object* v_inst_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_mathlib_OrderHom_fst(v_00_u03b1_714_, v_00_u03b2_715_, v_inst_716_, v_inst_717_);
lean_dec_ref(v_inst_717_);
lean_dec_ref(v_inst_716_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___lam__0(lean_object* v_self_719_){
_start:
{
lean_object* v_snd_720_; 
v_snd_720_ = lean_ctor_get(v_self_719_, 1);
lean_inc(v_snd_720_);
return v_snd_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___lam__0___boxed(lean_object* v_self_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_OrderHom_snd___lam__0(v_self_721_);
lean_dec_ref(v_self_721_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd(lean_object* v_00_u03b1_724_, lean_object* v_00_u03b2_725_, lean_object* v_inst_726_, lean_object* v_inst_727_){
_start:
{
lean_object* v___f_728_; 
v___f_728_ = ((lean_object*)(lp_mathlib_OrderHom_snd___closed__0));
return v___f_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_snd___boxed(lean_object* v_00_u03b1_729_, lean_object* v_00_u03b2_730_, lean_object* v_inst_731_, lean_object* v_inst_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_mathlib_OrderHom_snd(v_00_u03b1_729_, v_00_u03b2_730_, v_inst_731_, v_inst_732_);
lean_dec_ref(v_inst_732_);
lean_dec_ref(v_inst_731_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso___lam__1(lean_object* v_f_734_){
_start:
{
lean_object* v___f_735_; lean_object* v___x_736_; lean_object* v___f_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___f_735_ = ((lean_object*)(lp_mathlib_OrderHom_fst___closed__0));
lean_inc_ref(v_f_734_);
v___x_736_ = lp_mathlib_OrderHom_comp___redArg(v___f_735_, v_f_734_);
v___f_737_ = ((lean_object*)(lp_mathlib_OrderHom_snd___closed__0));
v___x_738_ = lp_mathlib_OrderHom_comp___redArg(v___f_737_, v_f_734_);
v___x_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_739_, 0, v___x_736_);
lean_ctor_set(v___x_739_, 1, v___x_738_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso(lean_object* v_00_u03b1_744_, lean_object* v_00_u03b2_745_, lean_object* v_00_u03b3_746_, lean_object* v_inst_747_, lean_object* v_inst_748_, lean_object* v_inst_749_){
_start:
{
lean_object* v___x_750_; 
v___x_750_ = ((lean_object*)(lp_mathlib_OrderHom_prodIso___closed__1));
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodIso___boxed(lean_object* v_00_u03b1_751_, lean_object* v_00_u03b2_752_, lean_object* v_00_u03b3_753_, lean_object* v_inst_754_, lean_object* v_inst_755_, lean_object* v_inst_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_mathlib_OrderHom_prodIso(v_00_u03b1_751_, v_00_u03b2_752_, v_00_u03b3_753_, v_inst_754_, v_inst_755_, v_inst_756_);
lean_dec_ref(v_inst_756_);
lean_dec_ref(v_inst_755_);
lean_dec_ref(v_inst_754_);
return v_res_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap___redArg(lean_object* v_f_758_, lean_object* v_g_759_){
_start:
{
lean_object* v___f_760_; lean_object* v___f_761_; lean_object* v___x_762_; 
v___f_760_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_760_, 0, v_f_758_);
v___f_761_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_761_, 0, v_g_759_);
v___x_762_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_762_, 0, lean_box(0));
lean_closure_set(v___x_762_, 1, lean_box(0));
lean_closure_set(v___x_762_, 2, lean_box(0));
lean_closure_set(v___x_762_, 3, lean_box(0));
lean_closure_set(v___x_762_, 4, v___f_760_);
lean_closure_set(v___x_762_, 5, v___f_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap(lean_object* v_00_u03b1_763_, lean_object* v_00_u03b2_764_, lean_object* v_00_u03b3_765_, lean_object* v_00_u03b4_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_f_771_, lean_object* v_g_772_){
_start:
{
lean_object* v___x_773_; 
v___x_773_ = lp_mathlib_OrderHom_prodMap___redArg(v_f_771_, v_g_772_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_prodMap___boxed(lean_object* v_00_u03b1_774_, lean_object* v_00_u03b2_775_, lean_object* v_00_u03b3_776_, lean_object* v_00_u03b4_777_, lean_object* v_inst_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_f_782_, lean_object* v_g_783_){
_start:
{
lean_object* v_res_784_; 
v_res_784_ = lp_mathlib_OrderHom_prodMap(v_00_u03b1_774_, v_00_u03b2_775_, v_00_u03b3_776_, v_00_u03b4_777_, v_inst_778_, v_inst_779_, v_inst_780_, v_inst_781_, v_f_782_, v_g_783_);
lean_dec_ref(v_inst_781_);
lean_dec_ref(v_inst_780_);
lean_dec_ref(v_inst_779_);
lean_dec_ref(v_inst_778_);
return v_res_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom___redArg(lean_object* v_i_785_){
_start:
{
lean_object* v___x_786_; 
v___x_786_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_786_, 0, lean_box(0));
lean_closure_set(v___x_786_, 1, lean_box(0));
lean_closure_set(v___x_786_, 2, v_i_785_);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom(lean_object* v_00_u03b9_787_, lean_object* v_00_u03c0_788_, lean_object* v_inst_789_, lean_object* v_i_790_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_791_, 0, lean_box(0));
lean_closure_set(v___x_791_, 1, lean_box(0));
lean_closure_set(v___x_791_, 2, v_i_790_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalOrderHom___boxed(lean_object* v_00_u03b9_792_, lean_object* v_00_u03c0_793_, lean_object* v_inst_794_, lean_object* v_i_795_){
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_mathlib_Pi_evalOrderHom(v_00_u03b9_792_, v_00_u03c0_793_, v_inst_794_, v_i_795_);
lean_dec_ref(v_inst_794_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_coeFnHom(lean_object* v_00_u03b1_797_, lean_object* v_00_u03b2_798_, lean_object* v_inst_799_, lean_object* v_inst_800_){
_start:
{
lean_object* v___f_801_; 
v___f_801_ = ((lean_object*)(lp_mathlib_OrderHom_equivRelHom___closed__0));
return v___f_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_coeFnHom___boxed(lean_object* v_00_u03b1_802_, lean_object* v_00_u03b2_803_, lean_object* v_inst_804_, lean_object* v_inst_805_){
_start:
{
lean_object* v_res_806_; 
v_res_806_ = lp_mathlib_OrderHom_coeFnHom(v_00_u03b1_802_, v_00_u03b2_803_, v_inst_804_, v_inst_805_);
lean_dec_ref(v_inst_805_);
lean_dec_ref(v_inst_804_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply___redArg(lean_object* v_x_807_){
_start:
{
lean_object* v___x_808_; lean_object* v___f_809_; lean_object* v___x_810_; 
v___x_808_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_808_, 0, lean_box(0));
lean_closure_set(v___x_808_, 1, lean_box(0));
lean_closure_set(v___x_808_, 2, v_x_807_);
v___f_809_ = ((lean_object*)(lp_mathlib_OrderHom_equivRelHom___closed__0));
v___x_810_ = lp_mathlib_OrderHom_comp___redArg(v___x_808_, v___f_809_);
return v___x_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply(lean_object* v_00_u03b1_811_, lean_object* v_00_u03b2_812_, lean_object* v_inst_813_, lean_object* v_inst_814_, lean_object* v_x_815_){
_start:
{
lean_object* v___x_816_; 
v___x_816_ = lp_mathlib_OrderHom_apply___redArg(v_x_815_);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_apply___boxed(lean_object* v_00_u03b1_817_, lean_object* v_00_u03b2_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_x_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_OrderHom_apply(v_00_u03b1_817_, v_00_u03b2_818_, v_inst_819_, v_inst_820_, v_x_821_);
lean_dec_ref(v_inst_820_);
lean_dec_ref(v_inst_819_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___redArg___lam__0(lean_object* v_f_823_, lean_object* v_x_824_, lean_object* v_i_825_){
_start:
{
lean_object* v___x_826_; 
v___x_826_ = lean_apply_2(v_f_823_, v_i_825_, v_x_824_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___redArg(lean_object* v_f_827_){
_start:
{
lean_object* v___f_828_; 
v___f_828_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_828_, 0, v_f_827_);
return v___f_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi(lean_object* v_00_u03b1_829_, lean_object* v_inst_830_, lean_object* v_00_u03b9_831_, lean_object* v_00_u03c0_832_, lean_object* v_inst_833_, lean_object* v_f_834_){
_start:
{
lean_object* v___f_835_; 
v___f_835_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_pi___redArg___lam__0), 3, 1);
lean_closure_set(v___f_835_, 0, v_f_834_);
return v___f_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_pi___boxed(lean_object* v_00_u03b1_836_, lean_object* v_inst_837_, lean_object* v_00_u03b9_838_, lean_object* v_00_u03c0_839_, lean_object* v_inst_840_, lean_object* v_f_841_){
_start:
{
lean_object* v_res_842_; 
v_res_842_ = lp_mathlib_OrderHom_pi(v_00_u03b1_836_, v_inst_837_, v_00_u03b9_838_, v_00_u03c0_839_, v_inst_840_, v_f_841_);
lean_dec_ref(v_inst_840_);
lean_dec_ref(v_inst_837_);
return v_res_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso___redArg___lam__0(lean_object* v_f_843_, lean_object* v_i_844_, lean_object* v___y_845_){
_start:
{
lean_object* v___x_846_; lean_object* v___x_20__overap_847_; lean_object* v___x_848_; 
v___x_846_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_846_, 0, lean_box(0));
lean_closure_set(v___x_846_, 1, lean_box(0));
lean_closure_set(v___x_846_, 2, v_i_844_);
v___x_20__overap_847_ = lp_mathlib_OrderHom_comp___redArg(v___x_846_, v_f_843_);
v___x_848_ = lean_apply_1(v___x_20__overap_847_, v___y_845_);
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso___redArg(lean_object* v_inst_850_, lean_object* v_inst_851_){
_start:
{
lean_object* v___f_852_; lean_object* v___x_853_; lean_object* v___x_854_; 
v___f_852_ = ((lean_object*)(lp_mathlib_OrderHom_piIso___redArg___closed__0));
v___x_853_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_pi___boxed), 6, 5);
lean_closure_set(v___x_853_, 0, lean_box(0));
lean_closure_set(v___x_853_, 1, v_inst_850_);
lean_closure_set(v___x_853_, 2, lean_box(0));
lean_closure_set(v___x_853_, 3, lean_box(0));
lean_closure_set(v___x_853_, 4, v_inst_851_);
v___x_854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_854_, 0, v___f_852_);
lean_ctor_set(v___x_854_, 1, v___x_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_piIso(lean_object* v_00_u03b1_855_, lean_object* v_inst_856_, lean_object* v_00_u03b9_857_, lean_object* v_00_u03c0_858_, lean_object* v_inst_859_){
_start:
{
lean_object* v___x_860_; 
v___x_860_ = lp_mathlib_OrderHom_piIso___redArg(v_inst_856_, v_inst_859_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___lam__0(lean_object* v_self_861_){
_start:
{
lean_inc(v_self_861_);
return v_self_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___lam__0___boxed(lean_object* v_self_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_mathlib_OrderHom_Subtype_val___lam__0(v_self_862_);
lean_dec(v_self_862_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val(lean_object* v_00_u03b1_865_, lean_object* v_inst_866_, lean_object* v_p_867_){
_start:
{
lean_object* v___f_868_; 
v___f_868_ = ((lean_object*)(lp_mathlib_OrderHom_Subtype_val___closed__0));
return v___f_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_Subtype_val___boxed(lean_object* v_00_u03b1_869_, lean_object* v_inst_870_, lean_object* v_p_871_){
_start:
{
lean_object* v_res_872_; 
v_res_872_ = lp_mathlib_OrderHom_Subtype_val(v_00_u03b1_869_, v_inst_870_, v_p_871_);
lean_dec_ref(v_inst_870_);
return v_res_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderEmbedding(lean_object* v_00_u03b1_874_, lean_object* v_inst_875_, lean_object* v_p_876_, lean_object* v_q_877_, lean_object* v_h_878_){
_start:
{
lean_object* v___f_879_; 
v___f_879_ = ((lean_object*)(lp_mathlib_Subtype_orderEmbedding___closed__0));
return v___f_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_orderEmbedding___boxed(lean_object* v_00_u03b1_880_, lean_object* v_inst_881_, lean_object* v_p_882_, lean_object* v_q_883_, lean_object* v_h_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib_Subtype_orderEmbedding(v_00_u03b1_880_, v_inst_881_, v_p_882_, v_q_883_, v_h_884_);
lean_dec_ref(v_inst_881_);
return v_res_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_unique(lean_object* v_00_u03b1_886_, lean_object* v_inst_887_, lean_object* v_inst_888_){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = ((lean_object*)(lp_mathlib_OrderHom_id___closed__0));
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_unique___boxed(lean_object* v_00_u03b1_890_, lean_object* v_inst_891_, lean_object* v_inst_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_mathlib_OrderHom_unique(v_00_u03b1_890_, v_inst_891_, v_inst_892_);
lean_dec_ref(v_inst_891_);
return v_res_893_;
}
}
static lean_object* _init_lp_mathlib_OrderHom_dual___lam__0___closed__0(void){
_start:
{
lean_object* v___x_894_; 
v___x_894_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual___lam__0(lean_object* v_f_895_, lean_object* v___y_896_){
_start:
{
lean_object* v___x_897_; lean_object* v_toFun_898_; lean_object* v_toFun_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_897_ = lean_obj_once(&lp_mathlib_OrderHom_dual___lam__0___closed__0, &lp_mathlib_OrderHom_dual___lam__0___closed__0_once, _init_lp_mathlib_OrderHom_dual___lam__0___closed__0);
v_toFun_898_ = lean_ctor_get(v___x_897_, 0);
v_toFun_899_ = lean_ctor_get(v___x_897_, 0);
lean_inc(v_toFun_898_);
v___x_900_ = lean_apply_1(v_toFun_898_, v___y_896_);
v___x_901_ = lean_apply_1(v_f_895_, v___x_900_);
lean_inc(v_toFun_899_);
v___x_902_ = lean_apply_1(v_toFun_899_, v___x_901_);
return v___x_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual(lean_object* v_00_u03b1_906_, lean_object* v_00_u03b2_907_, lean_object* v_inst_908_, lean_object* v_inst_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = ((lean_object*)(lp_mathlib_OrderHom_dual___closed__1));
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dual___boxed(lean_object* v_00_u03b1_911_, lean_object* v_00_u03b2_912_, lean_object* v_inst_913_, lean_object* v_inst_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_OrderHom_dual(v_00_u03b1_911_, v_00_u03b2_912_, v_inst_913_, v_inst_914_);
lean_dec_ref(v_inst_914_);
lean_dec_ref(v_inst_913_);
return v_res_915_;
}
}
static lean_object* _init_lp_mathlib_OrderHom_dualIso___redArg___closed__0(void){
_start:
{
lean_object* v___x_916_; 
v___x_916_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___redArg(lean_object* v_inst_917_, lean_object* v_inst_918_){
_start:
{
lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_919_ = lp_mathlib_OrderHom_dual(lean_box(0), lean_box(0), v_inst_917_, v_inst_918_);
v___x_920_ = lean_obj_once(&lp_mathlib_OrderHom_dualIso___redArg___closed__0, &lp_mathlib_OrderHom_dualIso___redArg___closed__0_once, _init_lp_mathlib_OrderHom_dualIso___redArg___closed__0);
v___x_921_ = lp_mathlib_Equiv_trans___redArg(v___x_919_, v___x_920_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___redArg___boxed(lean_object* v_inst_922_, lean_object* v_inst_923_){
_start:
{
lean_object* v_res_924_; 
v_res_924_ = lp_mathlib_OrderHom_dualIso___redArg(v_inst_922_, v_inst_923_);
lean_dec_ref(v_inst_923_);
lean_dec_ref(v_inst_922_);
return v_res_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso(lean_object* v_00_u03b1_925_, lean_object* v_00_u03b2_926_, lean_object* v_inst_927_, lean_object* v_inst_928_){
_start:
{
lean_object* v___x_929_; 
v___x_929_ = lp_mathlib_OrderHom_dualIso___redArg(v_inst_927_, v_inst_928_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_dualIso___boxed(lean_object* v_00_u03b1_930_, lean_object* v_00_u03b2_931_, lean_object* v_inst_932_, lean_object* v_inst_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_mathlib_OrderHom_dualIso(v_00_u03b1_930_, v_00_u03b2_931_, v_inst_932_, v_inst_933_);
lean_dec_ref(v_inst_933_);
lean_dec_ref(v_inst_932_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___redArg___lam__0(lean_object* v_f_935_, lean_object* v_i_936_){
_start:
{
lean_object* v___x_937_; 
v___x_937_ = lean_apply_1(v_f_935_, v_i_936_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___redArg(lean_object* v_f_938_){
_start:
{
lean_object* v___f_939_; 
v___f_939_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_939_, 0, v_f_938_);
return v___f_939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap(lean_object* v_00_u03b1_940_, lean_object* v_00_u03b2_941_, lean_object* v_inst_942_, lean_object* v_inst_943_, lean_object* v_f_944_){
_start:
{
lean_object* v___f_945_; 
v___f_945_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_945_, 0, v_f_944_);
return v___f_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftMap___boxed(lean_object* v_00_u03b1_946_, lean_object* v_00_u03b2_947_, lean_object* v_inst_948_, lean_object* v_inst_949_, lean_object* v_f_950_){
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_mathlib_OrderHom_uliftMap(v_00_u03b1_946_, v_00_u03b2_947_, v_inst_948_, v_inst_949_, v_f_950_);
lean_dec_ref(v_inst_949_);
lean_dec_ref(v_inst_948_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap___redArg(lean_object* v_f_952_){
_start:
{
lean_object* v___f_953_; 
v___f_953_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_953_, 0, v_f_952_);
return v___f_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap(lean_object* v_00_u03b1_954_, lean_object* v_00_u03b2_955_, lean_object* v_inst_956_, lean_object* v_inst_957_, lean_object* v_f_958_){
_start:
{
lean_object* v___f_959_; 
v___f_959_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_959_, 0, v_f_958_);
return v___f_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftRightMap___boxed(lean_object* v_00_u03b1_960_, lean_object* v_00_u03b2_961_, lean_object* v_inst_962_, lean_object* v_inst_963_, lean_object* v_f_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_mathlib_OrderHom_uliftRightMap(v_00_u03b1_960_, v_00_u03b2_961_, v_inst_962_, v_inst_963_, v_f_964_);
lean_dec_ref(v_inst_963_);
lean_dec_ref(v_inst_962_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap___redArg(lean_object* v_f_966_){
_start:
{
lean_object* v___f_967_; 
v___f_967_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_967_, 0, v_f_966_);
return v___f_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap(lean_object* v_00_u03b1_968_, lean_object* v_00_u03b2_969_, lean_object* v_inst_970_, lean_object* v_inst_971_, lean_object* v_f_972_){
_start:
{
lean_object* v___f_973_; 
v___f_973_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_uliftMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_973_, 0, v_f_972_);
return v___f_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_uliftLeftMap___boxed(lean_object* v_00_u03b1_974_, lean_object* v_00_u03b2_975_, lean_object* v_inst_976_, lean_object* v_inst_977_, lean_object* v_f_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_OrderHom_uliftLeftMap(v_00_u03b1_974_, v_00_u03b2_975_, v_inst_976_, v_inst_977_, v_f_978_);
lean_dec_ref(v_inst_977_);
lean_dec_ref(v_inst_976_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___redArg(lean_object* v_f_980_){
_start:
{
lean_inc(v_f_980_);
return v_f_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___redArg___boxed(lean_object* v_f_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___redArg(v_f_981_);
lean_dec(v_f_981_);
return v_res_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding(lean_object* v_00_u03b1_983_, lean_object* v_00_u03b2_984_, lean_object* v_inst_985_, lean_object* v_inst_986_, lean_object* v_f_987_){
_start:
{
lean_inc(v_f_987_);
return v_f_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding___boxed(lean_object* v_00_u03b1_988_, lean_object* v_00_u03b2_989_, lean_object* v_inst_990_, lean_object* v_inst_991_, lean_object* v_f_992_){
_start:
{
lean_object* v_res_993_; 
v_res_993_ = lp_mathlib_RelEmbedding_orderEmbeddingOfLTEmbedding(v_00_u03b1_988_, v_00_u03b2_989_, v_inst_990_, v_inst_991_, v_f_992_);
lean_dec(v_f_992_);
lean_dec_ref(v_inst_991_);
lean_dec_ref(v_inst_990_);
return v_res_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_id(lean_object* v_00_u03b1_995_, lean_object* v_inst_996_){
_start:
{
lean_object* v___f_997_; 
v___f_997_ = ((lean_object*)(lp_mathlib_OrderEmbedding_id___closed__0));
return v___f_997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_comp___redArg(lean_object* v_f_998_, lean_object* v_g_999_){
_start:
{
lean_object* v___f_1000_; 
v___f_1000_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1000_, 0, v_f_998_);
lean_closure_set(v___f_1000_, 1, v_g_999_);
return v___f_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_comp(lean_object* v_00_u03b1_1001_, lean_object* v_00_u03b2_1002_, lean_object* v_00_u03b3_1003_, lean_object* v_inst_1004_, lean_object* v_inst_1005_, lean_object* v_inst_1006_, lean_object* v_f_1007_, lean_object* v_g_1008_){
_start:
{
lean_object* v___f_1009_; 
v___f_1009_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1009_, 0, v_f_1007_);
lean_closure_set(v___f_1009_, 1, v_g_1008_);
return v___f_1009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___redArg(lean_object* v_f_1010_){
_start:
{
lean_inc(v_f_1010_);
return v_f_1010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___redArg___boxed(lean_object* v_f_1011_){
_start:
{
lean_object* v_res_1012_; 
v_res_1012_ = lp_mathlib_OrderEmbedding_ltEmbedding___redArg(v_f_1011_);
lean_dec(v_f_1011_);
return v_res_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding(lean_object* v_00_u03b1_1013_, lean_object* v_00_u03b2_1014_, lean_object* v_inst_1015_, lean_object* v_inst_1016_, lean_object* v_f_1017_){
_start:
{
lean_inc(v_f_1017_);
return v_f_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___boxed(lean_object* v_00_u03b1_1018_, lean_object* v_00_u03b2_1019_, lean_object* v_inst_1020_, lean_object* v_inst_1021_, lean_object* v_f_1022_){
_start:
{
lean_object* v_res_1023_; 
v_res_1023_ = lp_mathlib_OrderEmbedding_ltEmbedding(v_00_u03b1_1018_, v_00_u03b2_1019_, v_inst_1020_, v_inst_1021_, v_f_1022_);
lean_dec(v_f_1022_);
lean_dec_ref(v_inst_1021_);
lean_dec_ref(v_inst_1020_);
return v_res_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___redArg(lean_object* v_f_1024_){
_start:
{
lean_inc(v_f_1024_);
return v_f_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___redArg___boxed(lean_object* v_f_1025_){
_start:
{
lean_object* v_res_1026_; 
v_res_1026_ = lp_mathlib_OrderEmbedding_gtEmbedding___redArg(v_f_1025_);
lean_dec(v_f_1025_);
return v_res_1026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding(lean_object* v_00_u03b1_1027_, lean_object* v_00_u03b2_1028_, lean_object* v_inst_1029_, lean_object* v_inst_1030_, lean_object* v_f_1031_){
_start:
{
lean_inc(v_f_1031_);
return v_f_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_gtEmbedding___boxed(lean_object* v_00_u03b1_1032_, lean_object* v_00_u03b2_1033_, lean_object* v_inst_1034_, lean_object* v_inst_1035_, lean_object* v_f_1036_){
_start:
{
lean_object* v_res_1037_; 
v_res_1037_ = lp_mathlib_OrderEmbedding_gtEmbedding(v_00_u03b1_1032_, v_00_u03b2_1033_, v_inst_1034_, v_inst_1035_, v_f_1036_);
lean_dec(v_f_1036_);
lean_dec_ref(v_inst_1035_);
lean_dec_ref(v_inst_1034_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___redArg(lean_object* v_f_1038_){
_start:
{
lean_inc(v_f_1038_);
return v_f_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___redArg___boxed(lean_object* v_f_1039_){
_start:
{
lean_object* v_res_1040_; 
v_res_1040_ = lp_mathlib_OrderEmbedding_dual___redArg(v_f_1039_);
lean_dec(v_f_1039_);
return v_res_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual(lean_object* v_00_u03b1_1041_, lean_object* v_00_u03b2_1042_, lean_object* v_inst_1043_, lean_object* v_inst_1044_, lean_object* v_f_1045_){
_start:
{
lean_inc(v_f_1045_);
return v_f_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_dual___boxed(lean_object* v_00_u03b1_1046_, lean_object* v_00_u03b2_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_, lean_object* v_f_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib_OrderEmbedding_dual(v_00_u03b1_1046_, v_00_u03b2_1047_, v_inst_1048_, v_inst_1049_, v_f_1050_);
lean_dec(v_f_1050_);
lean_dec_ref(v_inst_1049_);
lean_dec_ref(v_inst_1048_);
return v_res_1051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___redArg(lean_object* v_f_1052_){
_start:
{
lean_inc(v_f_1052_);
return v_f_1052_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___redArg___boxed(lean_object* v_f_1053_){
_start:
{
lean_object* v_res_1054_; 
v_res_1054_ = lp_mathlib_OrderEmbedding_ofMapLEIff___redArg(v_f_1053_);
lean_dec(v_f_1053_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff(lean_object* v_00_u03b1_1055_, lean_object* v_00_u03b2_1056_, lean_object* v_inst_1057_, lean_object* v_inst_1058_, lean_object* v_f_1059_, lean_object* v_hf_1060_){
_start:
{
lean_inc(v_f_1059_);
return v_f_1059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofMapLEIff___boxed(lean_object* v_00_u03b1_1061_, lean_object* v_00_u03b2_1062_, lean_object* v_inst_1063_, lean_object* v_inst_1064_, lean_object* v_f_1065_, lean_object* v_hf_1066_){
_start:
{
lean_object* v_res_1067_; 
v_res_1067_ = lp_mathlib_OrderEmbedding_ofMapLEIff(v_00_u03b1_1061_, v_00_u03b2_1062_, v_inst_1063_, v_inst_1064_, v_f_1065_, v_hf_1066_);
lean_dec(v_f_1065_);
lean_dec_ref(v_inst_1064_);
lean_dec_ref(v_inst_1063_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___redArg(lean_object* v_f_1068_){
_start:
{
lean_inc(v_f_1068_);
return v_f_1068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___redArg___boxed(lean_object* v_f_1069_){
_start:
{
lean_object* v_res_1070_; 
v_res_1070_ = lp_mathlib_OrderEmbedding_ofStrictMono___redArg(v_f_1069_);
lean_dec(v_f_1069_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono(lean_object* v_00_u03b1_1071_, lean_object* v_00_u03b2_1072_, lean_object* v_inst_1073_, lean_object* v_inst_1074_, lean_object* v_f_1075_, lean_object* v_h_1076_){
_start:
{
lean_inc(v_f_1075_);
return v_f_1075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofStrictMono___boxed(lean_object* v_00_u03b1_1077_, lean_object* v_00_u03b2_1078_, lean_object* v_inst_1079_, lean_object* v_inst_1080_, lean_object* v_f_1081_, lean_object* v_h_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_mathlib_OrderEmbedding_ofStrictMono(v_00_u03b1_1077_, v_00_u03b2_1078_, v_inst_1079_, v_inst_1080_, v_f_1081_, v_h_1082_);
lean_dec(v_f_1081_);
lean_dec_ref(v_inst_1080_);
lean_dec_ref(v_inst_1079_);
return v_res_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_subtype(lean_object* v_00_u03b1_1085_, lean_object* v_inst_1086_, lean_object* v_p_1087_){
_start:
{
lean_object* v___f_1088_; 
v___f_1088_ = ((lean_object*)(lp_mathlib_OrderEmbedding_subtype___closed__0));
return v___f_1088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_subtype___boxed(lean_object* v_00_u03b1_1089_, lean_object* v_inst_1090_, lean_object* v_p_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_OrderEmbedding_subtype(v_00_u03b1_1089_, v_inst_1090_, v_p_1091_);
lean_dec_ref(v_inst_1090_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___redArg(lean_object* v_f_1093_){
_start:
{
lean_object* v___f_1094_; 
v___f_1094_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_1094_, 0, v_f_1093_);
return v___f_1094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom(lean_object* v_X_1095_, lean_object* v_Y_1096_, lean_object* v_inst_1097_, lean_object* v_inst_1098_, lean_object* v_f_1099_){
_start:
{
lean_object* v___f_1100_; 
v___f_1100_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_1100_, 0, v_f_1099_);
return v___f_1100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_toOrderHom___boxed(lean_object* v_X_1101_, lean_object* v_Y_1102_, lean_object* v_inst_1103_, lean_object* v_inst_1104_, lean_object* v_f_1105_){
_start:
{
lean_object* v_res_1106_; 
v_res_1106_ = lp_mathlib_OrderEmbedding_toOrderHom(v_X_1101_, v_Y_1102_, v_inst_1103_, v_inst_1104_, v_f_1105_);
lean_dec_ref(v_inst_1104_);
lean_dec_ref(v_inst_1103_);
return v_res_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0(lean_object* v_a_1107_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0___boxed(lean_object* v_a_1108_){
_start:
{
lean_object* v_res_1109_; 
v_res_1109_ = lp_mathlib_OrderEmbedding_ofIsEmpty___lam__0(v_a_1108_);
lean_dec(v_a_1108_);
return v_res_1109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty(lean_object* v_00_u03b1_1111_, lean_object* v_00_u03b2_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_, lean_object* v_inst_1115_){
_start:
{
lean_object* v___f_1116_; 
v___f_1116_ = ((lean_object*)(lp_mathlib_OrderEmbedding_ofIsEmpty___closed__0));
return v___f_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ofIsEmpty___boxed(lean_object* v_00_u03b1_1117_, lean_object* v_00_u03b2_1118_, lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_inst_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_mathlib_OrderEmbedding_ofIsEmpty(v_00_u03b1_1117_, v_00_u03b2_1118_, v_inst_1119_, v_inst_1120_, v_inst_1121_);
lean_dec_ref(v_inst_1120_);
lean_dec_ref(v_inst_1119_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom___redArg(lean_object* v_f_1123_){
_start:
{
lean_object* v___f_1124_; 
v___f_1124_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_1124_, 0, v_f_1123_);
return v___f_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom(lean_object* v_00_u03b1_1125_, lean_object* v_00_u03b2_1126_, lean_object* v_inst_1127_, lean_object* v_inst_1128_, lean_object* v_f_1129_){
_start:
{
lean_object* v___f_1130_; 
v___f_1130_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_1130_, 0, v_f_1129_);
return v___f_1130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_toOrderHom___boxed(lean_object* v_00_u03b1_1131_, lean_object* v_00_u03b2_1132_, lean_object* v_inst_1133_, lean_object* v_inst_1134_, lean_object* v_f_1135_){
_start:
{
lean_object* v_res_1136_; 
v_res_1136_ = lp_mathlib_RelHom_toOrderHom(v_00_u03b1_1131_, v_00_u03b2_1132_, v_inst_1133_, v_inst_1134_, v_f_1135_);
lean_dec_ref(v_inst_1134_);
lean_dec_ref(v_inst_1133_);
return v_res_1136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toOrderEmbedding___redArg(lean_object* v_e_1137_){
_start:
{
lean_object* v___f_1138_; 
v___f_1138_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1138_, 0, v_e_1137_);
return v___f_1138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toOrderEmbedding(lean_object* v_00_u03b1_1139_, lean_object* v_00_u03b2_1140_, lean_object* v_inst_1141_, lean_object* v_inst_1142_, lean_object* v_e_1143_){
_start:
{
lean_object* v___f_1144_; 
v___f_1144_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1144_, 0, v_e_1143_);
return v___f_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_refl(lean_object* v_00_u03b1_1145_, lean_object* v_inst_1146_){
_start:
{
lean_object* v___x_1147_; 
v___x_1147_ = lean_obj_once(&lp_mathlib_OrderHom_dual___lam__0___closed__0, &lp_mathlib_OrderHom_dual___lam__0___closed__0_once, _init_lp_mathlib_OrderHom_dual___lam__0___closed__0);
return v___x_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_symm___redArg(lean_object* v_e_1148_){
_start:
{
lean_object* v___x_1149_; 
v___x_1149_ = lp_mathlib_Equiv_symm___redArg(v_e_1148_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_symm(lean_object* v_00_u03b1_1150_, lean_object* v_00_u03b2_1151_, lean_object* v_inst_1152_, lean_object* v_inst_1153_, lean_object* v_e_1154_){
_start:
{
lean_object* v___x_1155_; 
v___x_1155_ = lp_mathlib_Equiv_symm___redArg(v_e_1154_);
return v___x_1155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_trans___redArg(lean_object* v_e_1156_, lean_object* v_e_x27_1157_){
_start:
{
lean_object* v___x_1158_; 
v___x_1158_ = lp_mathlib_Equiv_trans___redArg(v_e_1156_, v_e_x27_1157_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_trans(lean_object* v_00_u03b1_1159_, lean_object* v_00_u03b2_1160_, lean_object* v_00_u03b3_1161_, lean_object* v_inst_1162_, lean_object* v_inst_1163_, lean_object* v_inst_1164_, lean_object* v_e_1165_, lean_object* v_e_x27_1166_){
_start:
{
lean_object* v___x_1167_; 
v___x_1167_ = lp_mathlib_Equiv_trans___redArg(v_e_1165_, v_e_x27_1166_);
return v___x_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__0(lean_object* v_g_1168_, lean_object* v___y_1169_){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_g_1168_, v___y_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__1(lean_object* v___x_1171_, lean_object* v___y_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_1171_, v___y_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__2(lean_object* v_f_1174_, lean_object* v___f_1175_, lean_object* v_p_1176_, lean_object* v___y_1177_){
_start:
{
lean_object* v___x_1178_; lean_object* v___f_1179_; lean_object* v___x_1180_; lean_object* v___x_135__overap_1181_; lean_object* v___x_1182_; 
v___x_1178_ = lp_mathlib_Equiv_symm___redArg(v_f_1174_);
v___f_1179_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1179_, 0, v___x_1178_);
v___x_1180_ = lp_mathlib_OrderHom_comp___redArg(v_p_1176_, v___f_1179_);
v___x_135__overap_1181_ = lp_mathlib_OrderHom_comp___redArg(v___f_1175_, v___x_1180_);
v___x_1182_ = lean_apply_1(v___x_135__overap_1181_, v___y_1177_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__3(lean_object* v_f_1183_, lean_object* v___y_1184_){
_start:
{
lean_object* v___x_1185_; 
v___x_1185_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1183_, v___y_1184_);
return v___x_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg___lam__5(lean_object* v_g_1186_, lean_object* v___f_1187_, lean_object* v_p_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v___x_1190_; lean_object* v___f_1191_; lean_object* v___x_1192_; lean_object* v___x_146__overap_1193_; lean_object* v___x_1194_; 
v___x_1190_ = lp_mathlib_Equiv_symm___redArg(v_g_1186_);
v___f_1191_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1191_, 0, v___x_1190_);
v___x_1192_ = lp_mathlib_OrderHom_comp___redArg(v_p_1188_, v___f_1187_);
v___x_146__overap_1193_ = lp_mathlib_OrderHom_comp___redArg(v___f_1191_, v___x_1192_);
v___x_1194_ = lean_apply_1(v___x_146__overap_1193_, v___y_1189_);
return v___x_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___redArg(lean_object* v_f_1195_, lean_object* v_g_1196_){
_start:
{
lean_object* v___f_1197_; lean_object* v___f_1198_; lean_object* v___f_1199_; lean_object* v___f_1200_; lean_object* v___x_1201_; 
lean_inc_ref(v_g_1196_);
v___f_1197_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1197_, 0, v_g_1196_);
lean_inc_ref(v_f_1195_);
v___f_1198_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1198_, 0, v_f_1195_);
lean_closure_set(v___f_1198_, 1, v___f_1197_);
v___f_1199_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__3), 2, 1);
lean_closure_set(v___f_1199_, 0, v_f_1195_);
v___f_1200_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_arrowCongr___redArg___lam__5), 4, 2);
lean_closure_set(v___f_1200_, 0, v_g_1196_);
lean_closure_set(v___f_1200_, 1, v___f_1199_);
v___x_1201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___f_1198_);
lean_ctor_set(v___x_1201_, 1, v___f_1200_);
return v___x_1201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr(lean_object* v_00_u03b1_1202_, lean_object* v_00_u03b2_1203_, lean_object* v_00_u03b3_1204_, lean_object* v_00_u03b4_1205_, lean_object* v_inst_1206_, lean_object* v_inst_1207_, lean_object* v_inst_1208_, lean_object* v_inst_1209_, lean_object* v_f_1210_, lean_object* v_g_1211_){
_start:
{
lean_object* v___x_1212_; 
v___x_1212_ = lp_mathlib_OrderIso_arrowCongr___redArg(v_f_1210_, v_g_1211_);
return v___x_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_arrowCongr___boxed(lean_object* v_00_u03b1_1213_, lean_object* v_00_u03b2_1214_, lean_object* v_00_u03b3_1215_, lean_object* v_00_u03b4_1216_, lean_object* v_inst_1217_, lean_object* v_inst_1218_, lean_object* v_inst_1219_, lean_object* v_inst_1220_, lean_object* v_f_1221_, lean_object* v_g_1222_){
_start:
{
lean_object* v_res_1223_; 
v_res_1223_ = lp_mathlib_OrderIso_arrowCongr(v_00_u03b1_1213_, v_00_u03b2_1214_, v_00_u03b3_1215_, v_00_u03b4_1216_, v_inst_1217_, v_inst_1218_, v_inst_1219_, v_inst_1220_, v_f_1221_, v_g_1222_);
lean_dec_ref(v_inst_1220_);
lean_dec_ref(v_inst_1219_);
lean_dec_ref(v_inst_1218_);
lean_dec_ref(v_inst_1217_);
return v_res_1223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___redArg___lam__0(lean_object* v_f_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_){
_start:
{
lean_object* v___x_1227_; lean_object* v_toFun_1228_; lean_object* v___x_1229_; 
v___x_1227_ = lp_mathlib_Equiv_symm___redArg(v_f_1224_);
v_toFun_1228_ = lean_ctor_get(v___x_1227_, 0);
lean_inc(v_toFun_1228_);
lean_dec_ref(v___x_1227_);
v___x_1229_ = lean_apply_2(v_toFun_1228_, v___y_1225_, v___y_1226_);
return v___x_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___redArg(lean_object* v_f_1235_){
_start:
{
lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1236_ = ((lean_object*)(lp_mathlib_OrderIso_conj___redArg___closed__2));
lean_inc_ref(v_f_1235_);
v___x_1237_ = lp_mathlib_OrderIso_arrowCongr___redArg(v_f_1235_, v_f_1235_);
v___x_1238_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_1236_, v___x_1237_);
return v___x_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj(lean_object* v_00_u03b1_1239_, lean_object* v_00_u03b2_1240_, lean_object* v_inst_1241_, lean_object* v_inst_1242_, lean_object* v_f_1243_){
_start:
{
lean_object* v___x_1244_; 
v___x_1244_ = lp_mathlib_OrderIso_conj___redArg(v_f_1243_);
return v___x_1244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_conj___boxed(lean_object* v_00_u03b1_1245_, lean_object* v_00_u03b2_1246_, lean_object* v_inst_1247_, lean_object* v_inst_1248_, lean_object* v_f_1249_){
_start:
{
lean_object* v_res_1250_; 
v_res_1250_ = lp_mathlib_OrderIso_conj(v_00_u03b1_1245_, v_00_u03b2_1246_, v_inst_1247_, v_inst_1248_, v_f_1249_);
lean_dec_ref(v_inst_1248_);
lean_dec_ref(v_inst_1247_);
return v_res_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderEmbeddingCongr___redArg(lean_object* v_f_1251_, lean_object* v_g_1252_){
_start:
{
lean_object* v___x_1253_; 
v___x_1253_ = lp_mathlib_RelIso_relEmbeddingCongr___redArg(v_f_1251_, v_g_1252_);
return v___x_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderEmbeddingCongr(lean_object* v_00_u03b1_1254_, lean_object* v_00_u03b2_1255_, lean_object* v_00_u03b3_1256_, lean_object* v_00_u03b4_1257_, lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_inst_1260_, lean_object* v_inst_1261_, lean_object* v_f_1262_, lean_object* v_g_1263_){
_start:
{
lean_object* v___x_1264_; 
v___x_1264_ = lp_mathlib_RelIso_relEmbeddingCongr___redArg(v_f_1262_, v_g_1263_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderIsoCongr___redArg(lean_object* v_f_1265_, lean_object* v_g_1266_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = lp_mathlib_RelIso_relIsoCongr___redArg(v_f_1265_, v_g_1266_);
return v___x_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_orderIsoCongr(lean_object* v_00_u03b1_1268_, lean_object* v_00_u03b2_1269_, lean_object* v_00_u03b3_1270_, lean_object* v_00_u03b4_1271_, lean_object* v_inst_1272_, lean_object* v_inst_1273_, lean_object* v_inst_1274_, lean_object* v_inst_1275_, lean_object* v_f_1276_, lean_object* v_g_1277_){
_start:
{
lean_object* v___x_1278_; 
v___x_1278_ = lp_mathlib_RelIso_relIsoCongr___redArg(v_f_1276_, v_g_1277_);
return v___x_1278_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_prodComm___closed__0(void){
_start:
{
lean_object* v___x_1279_; 
v___x_1279_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_prodComm(lean_object* v_00_u03b1_1280_, lean_object* v_00_u03b2_1281_, lean_object* v_inst_1282_, lean_object* v_inst_1283_){
_start:
{
lean_object* v___x_1284_; 
v___x_1284_ = lean_obj_once(&lp_mathlib_OrderIso_prodComm___closed__0, &lp_mathlib_OrderIso_prodComm___closed__0_once, _init_lp_mathlib_OrderIso_prodComm___closed__0);
return v___x_1284_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_prodAssoc___closed__0(void){
_start:
{
lean_object* v___x_1285_; 
v___x_1285_ = lp_mathlib_Equiv_prodAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_prodAssoc(lean_object* v_00_u03b1_1286_, lean_object* v_00_u03b2_1287_, lean_object* v_00_u03b3_1288_, lean_object* v_inst_1289_, lean_object* v_inst_1290_, lean_object* v_inst_1291_){
_start:
{
lean_object* v___x_1292_; 
v___x_1292_ = lean_obj_once(&lp_mathlib_OrderIso_prodAssoc___closed__0, &lp_mathlib_OrderIso_prodAssoc___closed__0_once, _init_lp_mathlib_OrderIso_prodAssoc___closed__0);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualDual(lean_object* v_00_u03b1_1293_, lean_object* v_inst_1294_){
_start:
{
lean_object* v___x_1295_; 
v___x_1295_ = lean_obj_once(&lp_mathlib_OrderHom_dual___lam__0___closed__0, &lp_mathlib_OrderHom_dual___lam__0___closed__0_once, _init_lp_mathlib_OrderHom_dual___lam__0___closed__0);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___redArg(lean_object* v_e_1296_){
_start:
{
lean_inc_ref(v_e_1296_);
return v_e_1296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___redArg___boxed(lean_object* v_e_1297_){
_start:
{
lean_object* v_res_1298_; 
v_res_1298_ = lp_mathlib_OrderIso_toRelIsoLT___redArg(v_e_1297_);
lean_dec_ref(v_e_1297_);
return v_res_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT(lean_object* v_00_u03b1_1299_, lean_object* v_00_u03b2_1300_, lean_object* v_inst_1301_, lean_object* v_inst_1302_, lean_object* v_e_1303_){
_start:
{
lean_inc_ref(v_e_1303_);
return v_e_1303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoLT___boxed(lean_object* v_00_u03b1_1304_, lean_object* v_00_u03b2_1305_, lean_object* v_inst_1306_, lean_object* v_inst_1307_, lean_object* v_e_1308_){
_start:
{
lean_object* v_res_1309_; 
v_res_1309_ = lp_mathlib_OrderIso_toRelIsoLT(v_00_u03b1_1304_, v_00_u03b2_1305_, v_inst_1306_, v_inst_1307_, v_e_1308_);
lean_dec_ref(v_e_1308_);
lean_dec_ref(v_inst_1307_);
lean_dec_ref(v_inst_1306_);
return v_res_1309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___redArg(lean_object* v_e_1310_){
_start:
{
lean_inc_ref(v_e_1310_);
return v_e_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___redArg___boxed(lean_object* v_e_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_mathlib_OrderIso_toRelIsoGT___redArg(v_e_1311_);
lean_dec_ref(v_e_1311_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT(lean_object* v_00_u03b1_1313_, lean_object* v_00_u03b2_1314_, lean_object* v_inst_1315_, lean_object* v_inst_1316_, lean_object* v_e_1317_){
_start:
{
lean_inc_ref(v_e_1317_);
return v_e_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_toRelIsoGT___boxed(lean_object* v_00_u03b1_1318_, lean_object* v_00_u03b2_1319_, lean_object* v_inst_1320_, lean_object* v_inst_1321_, lean_object* v_e_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_mathlib_OrderIso_toRelIsoGT(v_00_u03b1_1318_, v_00_u03b2_1319_, v_inst_1320_, v_inst_1321_, v_e_1322_);
lean_dec_ref(v_e_1322_);
lean_dec_ref(v_inst_1321_);
lean_dec_ref(v_inst_1320_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___redArg(lean_object* v_e_1324_){
_start:
{
lean_inc_ref(v_e_1324_);
return v_e_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___redArg___boxed(lean_object* v_e_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_mathlib_OrderIso_ofRelIsoLT___redArg(v_e_1325_);
lean_dec_ref(v_e_1325_);
return v_res_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT(lean_object* v_00_u03b1_1327_, lean_object* v_00_u03b2_1328_, lean_object* v_inst_1329_, lean_object* v_inst_1330_, lean_object* v_e_1331_){
_start:
{
lean_inc_ref(v_e_1331_);
return v_e_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofRelIsoLT___boxed(lean_object* v_00_u03b1_1332_, lean_object* v_00_u03b2_1333_, lean_object* v_inst_1334_, lean_object* v_inst_1335_, lean_object* v_e_1336_){
_start:
{
lean_object* v_res_1337_; 
v_res_1337_ = lp_mathlib_OrderIso_ofRelIsoLT(v_00_u03b1_1332_, v_00_u03b2_1333_, v_inst_1334_, v_inst_1335_, v_e_1336_);
lean_dec_ref(v_e_1336_);
lean_dec_ref(v_inst_1335_);
lean_dec_ref(v_inst_1334_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp___redArg(lean_object* v_f_1338_, lean_object* v_g_1339_){
_start:
{
lean_object* v___x_1340_; 
v___x_1340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1340_, 0, v_f_1338_);
lean_ctor_set(v___x_1340_, 1, v_g_1339_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp(lean_object* v_00_u03b1_1341_, lean_object* v_00_u03b2_1342_, lean_object* v_inst_1343_, lean_object* v_inst_1344_, lean_object* v_f_1345_, lean_object* v_g_1346_, lean_object* v_h_1347_){
_start:
{
lean_object* v___x_1348_; 
v___x_1348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1348_, 0, v_f_1345_);
lean_ctor_set(v___x_1348_, 1, v_g_1346_);
return v___x_1348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofCmpEqCmp___boxed(lean_object* v_00_u03b1_1349_, lean_object* v_00_u03b2_1350_, lean_object* v_inst_1351_, lean_object* v_inst_1352_, lean_object* v_f_1353_, lean_object* v_g_1354_, lean_object* v_h_1355_){
_start:
{
lean_object* v_res_1356_; 
v_res_1356_ = lp_mathlib_OrderIso_ofCmpEqCmp(v_00_u03b1_1349_, v_00_u03b2_1350_, v_inst_1351_, v_inst_1352_, v_f_1353_, v_g_1354_, v_h_1355_);
lean_dec_ref(v_inst_1352_);
lean_dec_ref(v_inst_1351_);
return v_res_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv___redArg(lean_object* v_f_1357_, lean_object* v_g_1358_){
_start:
{
lean_object* v___f_1359_; lean_object* v___f_1360_; lean_object* v___x_1361_; 
v___f_1359_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_equivRelHom___lam__0), 2, 1);
lean_closure_set(v___f_1359_, 0, v_f_1357_);
v___f_1360_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1360_, 0, v_g_1358_);
v___x_1361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1361_, 0, v___f_1359_);
lean_ctor_set(v___x_1361_, 1, v___f_1360_);
return v___x_1361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv(lean_object* v_00_u03b1_1362_, lean_object* v_00_u03b2_1363_, lean_object* v_inst_1364_, lean_object* v_inst_1365_, lean_object* v_f_1366_, lean_object* v_g_1367_, lean_object* v_h_u2081_1368_, lean_object* v_h_u2082_1369_){
_start:
{
lean_object* v___x_1370_; 
v___x_1370_ = lp_mathlib_OrderIso_ofHomInv___redArg(v_f_1366_, v_g_1367_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofHomInv___boxed(lean_object* v_00_u03b1_1371_, lean_object* v_00_u03b2_1372_, lean_object* v_inst_1373_, lean_object* v_inst_1374_, lean_object* v_f_1375_, lean_object* v_g_1376_, lean_object* v_h_u2081_1377_, lean_object* v_h_u2082_1378_){
_start:
{
lean_object* v_res_1379_; 
v_res_1379_ = lp_mathlib_OrderIso_ofHomInv(v_00_u03b1_1371_, v_00_u03b2_1372_, v_inst_1373_, v_inst_1374_, v_f_1375_, v_g_1376_, v_h_u2081_1377_, v_h_u2082_1378_);
lean_dec_ref(v_inst_1374_);
lean_dec_ref(v_inst_1373_);
return v_res_1379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique___redArg(lean_object* v_inst_1380_){
_start:
{
lean_object* v___x_1381_; 
v___x_1381_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique(lean_object* v_00_u03b1_1382_, lean_object* v_00_u03b2_1383_, lean_object* v_inst_1384_, lean_object* v_inst_1385_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = lp_mathlib_Equiv_piUnique___redArg(v_inst_1384_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_funUnique___boxed(lean_object* v_00_u03b1_1387_, lean_object* v_00_u03b2_1388_, lean_object* v_inst_1389_, lean_object* v_inst_1390_){
_start:
{
lean_object* v_res_1391_; 
v_res_1391_ = lp_mathlib_OrderIso_funUnique(v_00_u03b1_1387_, v_00_u03b2_1388_, v_inst_1389_, v_inst_1390_);
lean_dec_ref(v_inst_1390_);
return v_res_1391_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_ofIsEmpty___closed__0(void){
_start:
{
lean_object* v___x_1392_; 
v___x_1392_ = lp_mathlib_Equiv_equivOfIsEmpty(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofIsEmpty(lean_object* v_00_u03b1_1393_, lean_object* v_00_u03b2_1394_, lean_object* v_inst_1395_, lean_object* v_inst_1396_, lean_object* v_inst_1397_, lean_object* v_inst_1398_){
_start:
{
lean_object* v___x_1399_; 
v___x_1399_ = lean_obj_once(&lp_mathlib_OrderIso_ofIsEmpty___closed__0, &lp_mathlib_OrderIso_ofIsEmpty___closed__0_once, _init_lp_mathlib_OrderIso_ofIsEmpty___closed__0);
return v___x_1399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_ofIsEmpty___boxed(lean_object* v_00_u03b1_1400_, lean_object* v_00_u03b2_1401_, lean_object* v_inst_1402_, lean_object* v_inst_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_){
_start:
{
lean_object* v_res_1406_; 
v_res_1406_ = lp_mathlib_OrderIso_ofIsEmpty(v_00_u03b1_1400_, v_00_u03b2_1401_, v_inst_1402_, v_inst_1403_, v_inst_1404_, v_inst_1405_);
lean_dec_ref(v_inst_1403_);
lean_dec_ref(v_inst_1402_);
return v_res_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___redArg(lean_object* v_e_1407_){
_start:
{
lean_inc_ref(v_e_1407_);
return v_e_1407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___redArg___boxed(lean_object* v_e_1408_){
_start:
{
lean_object* v_res_1409_; 
v_res_1409_ = lp_mathlib_Equiv_toOrderIso___redArg(v_e_1408_);
lean_dec_ref(v_e_1408_);
return v_res_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso(lean_object* v_00_u03b1_1410_, lean_object* v_00_u03b2_1411_, lean_object* v_inst_1412_, lean_object* v_inst_1413_, lean_object* v_e_1414_, lean_object* v_h_u2081_1415_, lean_object* v_h_u2082_1416_){
_start:
{
lean_inc_ref(v_e_1414_);
return v_e_1414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIso___boxed(lean_object* v_00_u03b1_1417_, lean_object* v_00_u03b2_1418_, lean_object* v_inst_1419_, lean_object* v_inst_1420_, lean_object* v_e_1421_, lean_object* v_h_u2081_1422_, lean_object* v_h_u2082_1423_){
_start:
{
lean_object* v_res_1424_; 
v_res_1424_ = lp_mathlib_Equiv_toOrderIso(v_00_u03b1_1417_, v_00_u03b2_1418_, v_inst_1419_, v_inst_1420_, v_e_1421_, v_h_u2081_1422_, v_h_u2082_1423_);
lean_dec_ref(v_e_1421_);
lean_dec_ref(v_inst_1420_);
lean_dec_ref(v_inst_1419_);
return v_res_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse___redArg(lean_object* v_f_1425_, lean_object* v_g_1426_){
_start:
{
lean_object* v___x_1427_; 
v___x_1427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1427_, 0, v_f_1425_);
lean_ctor_set(v___x_1427_, 1, v_g_1426_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse(lean_object* v_00_u03b1_1428_, lean_object* v_00_u03b2_1429_, lean_object* v_inst_1430_, lean_object* v_inst_1431_, lean_object* v_f_1432_, lean_object* v_h__mono_1433_, lean_object* v_g_1434_, lean_object* v_hg_1435_){
_start:
{
lean_object* v___x_1436_; 
v___x_1436_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1436_, 0, v_f_1432_);
lean_ctor_set(v___x_1436_, 1, v_g_1434_);
return v___x_1436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_StrictMono_orderIsoOfRightInverse___boxed(lean_object* v_00_u03b1_1437_, lean_object* v_00_u03b2_1438_, lean_object* v_inst_1439_, lean_object* v_inst_1440_, lean_object* v_f_1441_, lean_object* v_h__mono_1442_, lean_object* v_g_1443_, lean_object* v_hg_1444_){
_start:
{
lean_object* v_res_1445_; 
v_res_1445_ = lp_mathlib_StrictMono_orderIsoOfRightInverse(v_00_u03b1_1437_, v_00_u03b2_1438_, v_inst_1439_, v_inst_1440_, v_f_1441_, v_h__mono_1442_, v_g_1443_, v_hg_1444_);
lean_dec_ref(v_inst_1440_);
lean_dec_ref(v_inst_1439_);
return v_res_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___redArg(lean_object* v_f_1446_){
_start:
{
lean_inc_ref(v_f_1446_);
return v_f_1446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___redArg___boxed(lean_object* v_f_1447_){
_start:
{
lean_object* v_res_1448_; 
v_res_1448_ = lp_mathlib_OrderIso_dual___redArg(v_f_1447_);
lean_dec_ref(v_f_1447_);
return v_res_1448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual(lean_object* v_00_u03b1_1449_, lean_object* v_00_u03b2_1450_, lean_object* v_inst_1451_, lean_object* v_inst_1452_, lean_object* v_f_1453_){
_start:
{
lean_inc_ref(v_f_1453_);
return v_f_1453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dual___boxed(lean_object* v_00_u03b1_1454_, lean_object* v_00_u03b2_1455_, lean_object* v_inst_1456_, lean_object* v_inst_1457_, lean_object* v_f_1458_){
_start:
{
lean_object* v_res_1459_; 
v_res_1459_ = lp_mathlib_OrderIso_dual(v_00_u03b1_1454_, v_00_u03b2_1455_, v_inst_1456_, v_inst_1457_, v_f_1458_);
lean_dec_ref(v_f_1458_);
return v_res_1459_;
}
}
static lean_object* _init_lp_mathlib_ULift_orderIso___closed__0(void){
_start:
{
lean_object* v___x_1460_; 
v___x_1460_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_1460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_orderIso(lean_object* v_00_u03b1_1461_, lean_object* v_inst_1462_){
_start:
{
lean_object* v___x_1463_; 
v___x_1463_ = lean_obj_once(&lp_mathlib_ULift_orderIso___closed__0, &lp_mathlib_ULift_orderIso___closed__0_once, _init_lp_mathlib_ULift_orderIso___closed__0);
return v___x_1463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_orderIso___boxed(lean_object* v_00_u03b1_1464_, lean_object* v_inst_1465_){
_start:
{
lean_object* v_res_1466_; 
v_res_1466_ = lp_mathlib_ULift_orderIso(v_00_u03b1_1464_, v_inst_1465_);
lean_dec_ref(v_inst_1465_);
return v_res_1466_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Disjoint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_PPWithUniv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_PPWithUniv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
