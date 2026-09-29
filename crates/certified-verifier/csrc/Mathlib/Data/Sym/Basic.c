// Lean compiler output
// Module: Mathlib.Data.Sym.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Multiset public import Mathlib.Data.Setoid.Basic public import Mathlib.Data.Vector.Basic public import Mathlib.Tactic.ApplyFun
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
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_erase___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_some___lam__0(lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_count___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_hasCoe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_hasCoe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_Perm_isSetoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_Perm_isSetoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelVectorEquivOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_nil(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Sym"};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value;
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term_::ₛ_"};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__1 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 137, 8, 139, 198, 210, 169)}};
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(95, 223, 235, 94, 133, 124, 203, 76)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2_value;
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__3 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__4 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__4_value;
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " ::ₛ "};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__5 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__5_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__6 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__6_value;
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__7 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__8 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__8_value),((lean_object*)(((size_t)(67) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__9 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__4_value),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__6_value),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__9_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__10 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2_value),((lean_object*)(((size_t)(67) << 1) | 1)),((lean_object*)(((size_t)(68) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__10_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__11 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Sym_term___x3a_x3a_u209b__ = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__11_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__0 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__0_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__1 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__1_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__2 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__2_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__3 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(164, 29, 181, 108, 63, 0, 109, 66)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__7 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 137, 8, 139, 198, 210, 169)}};
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(77, 117, 84, 87, 107, 186, 196, 38)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__9 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__10 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__10_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__11 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__12 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__0 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__1 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Sym_instCoeVector___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sym_instCoeVector___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sym_instCoeVector___closed__0 = (const lean_object*)&lp_mathlib_Sym_instCoeVector___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instMembership(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instMembership___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sym_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_decidableMem___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sym_decidableMem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_decidableMem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_::_"};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__0 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 137, 8, 139, 198, 210, 169)}};
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 238, 101, 189, 137, 99, 55, 86)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__1 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__1_value;
static const lean_string_object lp_mathlib_Sym_term___x3a_x3a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " :: "};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__2 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__2_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__3 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__4 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__4_value),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__3_value),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__4_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__5 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Sym_term___x3a_x3a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__5_value)}};
static const lean_object* lp_mathlib_Sym_term___x3a_x3a___00__closed__6 = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Sym_term___x3a_x3a__ = (const lean_object*)&lp_mathlib_Sym_term___x3a_x3a___00__closed__6_value;
static const lean_string_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cons'"};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(209, 39, 239, 97, 72, 139, 30, 148)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__2 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 137, 8, 139, 198, 210, 169)}};
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(128, 56, 125, 52, 179, 78, 136, 99)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__4 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__3_value)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__5 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__6 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__6_value;
static const lean_ctor_object lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__4_value),((lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__6_value)}};
static const lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__7 = (const lean_object*)&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons_x27__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons_x27__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instZeroSym(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instEmptyCollectionOfNatNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_uniqueZero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_replicate___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_replicate(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instUnique___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instUnique(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Sym_cast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sym_cast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sym_cast___closed__0 = (const lean_object*)&lp_mathlib_Sym_cast___closed__0_value;
static const lean_ctor_object lp_mathlib_Sym_cast___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sym_cast___closed__0_value),((lean_object*)&lp_mathlib_Sym_cast___closed__0_value)}};
static const lean_object* lp_mathlib_Sym_cast___closed__1 = (const lean_object*)&lp_mathlib_Sym_cast___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_append(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_append___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_oneEquiv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv___lam__0(lean_object*);
static lean_once_cell_t lp_mathlib_Sym_oneEquiv___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Sym_oneEquiv___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Sym_oneEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sym_oneEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sym_oneEquiv___closed__0 = (const lean_object*)&lp_mathlib_Sym_oneEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Sym_oneEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Sym_oneEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Sym_oneEquiv___closed__1 = (const lean_object*)&lp_mathlib_Sym_oneEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Sym_oneEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Sym_oneEquiv___closed__0_value),((lean_object*)&lp_mathlib_Sym_oneEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Sym_oneEquiv___closed__2 = (const lean_object*)&lp_mathlib_Sym_oneEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sym_filterNe___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SymOptionSuccEquiv_encode___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___closed__0 = (const lean_object*)&lp_mathlib_SymOptionSuccEquiv_encode___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SymOptionSuccEquiv_decode___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_some___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SymOptionSuccEquiv_decode___redArg___closed__0 = (const lean_object*)&lp_mathlib_SymOptionSuccEquiv_decode___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symOptionSuccEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_symOptionSuccEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___aux__1___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_List_decidablePerm___redArg(v_inst_1_, v_a_2_, v_b_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___aux__1___redArg___boxed(lean_object* v_inst_5_, lean_object* v_a_6_, lean_object* v_b_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_mathlib_instDecidableEqSym___aux__1___redArg(v_inst_5_, v_a_6_, v_b_7_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___aux__1(lean_object* v_00_u03b1_10_, lean_object* v_n_11_, lean_object* v_inst_12_, lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
uint8_t v___x_15_; 
v___x_15_ = l_List_decidablePerm___redArg(v_inst_12_, v_a_13_, v_b_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___aux__1___boxed(lean_object* v_00_u03b1_16_, lean_object* v_n_17_, lean_object* v_inst_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_mathlib_instDecidableEqSym___aux__1(v_00_u03b1_16_, v_n_17_, v_inst_18_, v_a_19_, v_b_20_);
lean_dec(v_n_17_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym___redArg(lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = l_List_decidablePerm___redArg(v_inst_23_, v_a_24_, v_b_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___redArg___boxed(lean_object* v_inst_27_, lean_object* v_a_28_, lean_object* v_b_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_instDecidableEqSym___redArg(v_inst_27_, v_a_28_, v_b_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqSym(lean_object* v_00_u03b1_32_, lean_object* v_n_33_, lean_object* v_inst_34_, lean_object* v_a_35_, lean_object* v_b_36_){
_start:
{
uint8_t v___x_37_; 
v___x_37_ = l_List_decidablePerm___redArg(v_inst_34_, v_a_35_, v_b_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqSym___boxed(lean_object* v_00_u03b1_38_, lean_object* v_n_39_, lean_object* v_inst_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
uint8_t v_res_43_; lean_object* v_r_44_; 
v_res_43_ = lp_mathlib_instDecidableEqSym(v_00_u03b1_38_, v_n_39_, v_inst_40_, v_a_41_, v_b_42_);
lean_dec(v_n_39_);
v_r_44_ = lean_box(v_res_43_);
return v_r_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___redArg(lean_object* v_s_45_){
_start:
{
lean_inc(v_s_45_);
return v_s_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___redArg___boxed(lean_object* v_s_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Sym_toMultiset___redArg(v_s_46_);
lean_dec(v_s_46_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset(lean_object* v_00_u03b1_48_, lean_object* v_n_49_, lean_object* v_s_50_){
_start:
{
lean_inc(v_s_50_);
return v_s_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_toMultiset___boxed(lean_object* v_00_u03b1_51_, lean_object* v_n_52_, lean_object* v_s_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Sym_toMultiset(v_00_u03b1_51_, v_n_52_, v_s_53_);
lean_dec(v_s_53_);
lean_dec(v_n_52_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_hasCoe___redArg(lean_object* v_n_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_alloc_closure((void*)(lp_mathlib_Sym_toMultiset___boxed), 3, 2);
lean_closure_set(v___x_56_, 0, lean_box(0));
lean_closure_set(v___x_56_, 1, v_n_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_hasCoe(lean_object* v_00_u03b1_57_, lean_object* v_n_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_alloc_closure((void*)(lp_mathlib_Sym_toMultiset___boxed), 3, 2);
lean_closure_set(v___x_59_, 0, lean_box(0));
lean_closure_set(v___x_59_, 1, v_n_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_Perm_isSetoid(lean_object* v_00_u03b1_60_, lean_object* v_n_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lean_box(0);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_Perm_isSetoid___boxed(lean_object* v_00_u03b1_63_, lean_object* v_n_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_List_Vector_Perm_isSetoid(v_00_u03b1_63_, v_n_64_);
lean_dec(v_n_64_);
return v_res_65_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___redArg(lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
uint8_t v___x_69_; 
v___x_69_ = l_List_decidablePerm___redArg(v_inst_66_, v_x_67_, v_x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___redArg___boxed(lean_object* v_inst_70_, lean_object* v_x_71_, lean_object* v_x_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___redArg(v_inst_70_, v_x_71_, v_x_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableRelVectorEquivOfDecidableEq(lean_object* v_00_u03b1_75_, lean_object* v_n_76_, lean_object* v_inst_77_, lean_object* v_x_78_, lean_object* v_x_79_){
_start:
{
uint8_t v___x_80_; 
v___x_80_ = l_List_decidablePerm___redArg(v_inst_77_, v_x_78_, v_x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableRelVectorEquivOfDecidableEq___boxed(lean_object* v_00_u03b1_81_, lean_object* v_n_82_, lean_object* v_inst_83_, lean_object* v_x_84_, lean_object* v_x_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_instDecidableRelVectorEquivOfDecidableEq(v_00_u03b1_81_, v_n_82_, v_inst_83_, v_x_84_, v_x_85_);
lean_dec(v_n_82_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___redArg(lean_object* v_m_88_){
_start:
{
lean_inc(v_m_88_);
return v_m_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___redArg___boxed(lean_object* v_m_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Sym_mk___redArg(v_m_89_);
lean_dec(v_m_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk(lean_object* v_00_u03b1_91_, lean_object* v_n_92_, lean_object* v_m_93_, lean_object* v_h_94_){
_start:
{
lean_inc(v_m_93_);
return v_m_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_mk___boxed(lean_object* v_00_u03b1_95_, lean_object* v_n_96_, lean_object* v_m_97_, lean_object* v_h_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Sym_mk(v_00_u03b1_95_, v_n_96_, v_m_97_, v_h_98_);
lean_dec(v_m_97_);
lean_dec(v_n_96_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_nil(lean_object* v_00_u03b1_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons___redArg(lean_object* v_a_102_, lean_object* v_s_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v_a_102_);
lean_ctor_set(v___x_104_, 1, v_s_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons(lean_object* v_00_u03b1_105_, lean_object* v_n_106_, lean_object* v_a_107_, lean_object* v_s_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_109_, 0, v_a_107_);
lean_ctor_set(v___x_109_, 1, v_s_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons___boxed(lean_object* v_00_u03b1_110_, lean_object* v_n_111_, lean_object* v_a_112_, lean_object* v_s_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Sym_cons(v_00_u03b1_110_, v_n_111_, v_a_112_, v_s_113_);
lean_dec(v_n_111_);
return v_res_114_;
}
}
static lean_object* _init_lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__5));
v___x_153_ = l_String_toRawSubstring_x27(v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1(lean_object* v_x_168_, lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_171_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2));
lean_inc(v_x_168_);
v___x_172_ = l_Lean_Syntax_isOfKind(v_x_168_, v___x_171_);
if (v___x_172_ == 0)
{
lean_object* v___x_173_; lean_object* v___x_174_; 
lean_dec(v_x_168_);
v___x_173_ = lean_box(1);
v___x_174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v_a_170_);
return v___x_174_;
}
else
{
lean_object* v_quotContext_175_; lean_object* v_currMacroScope_176_; lean_object* v_ref_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; uint8_t v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_quotContext_175_ = lean_ctor_get(v_a_169_, 1);
v_currMacroScope_176_ = lean_ctor_get(v_a_169_, 2);
v_ref_177_ = lean_ctor_get(v_a_169_, 5);
v___x_178_ = lean_unsigned_to_nat(0u);
v___x_179_ = l_Lean_Syntax_getArg(v_x_168_, v___x_178_);
v___x_180_ = lean_unsigned_to_nat(2u);
v___x_181_ = l_Lean_Syntax_getArg(v_x_168_, v___x_180_);
lean_dec(v_x_168_);
v___x_182_ = 0;
v___x_183_ = l_Lean_SourceInfo_fromRef(v_ref_177_, v___x_182_);
v___x_184_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4));
v___x_185_ = lean_obj_once(&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6, &lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6_once, _init_lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__6);
v___x_186_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__7));
lean_inc(v_currMacroScope_176_);
lean_inc(v_quotContext_175_);
v___x_187_ = l_Lean_addMacroScope(v_quotContext_175_, v___x_186_, v_currMacroScope_176_);
v___x_188_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__10));
lean_inc_n(v___x_183_, 2);
v___x_189_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_189_, 0, v___x_183_);
lean_ctor_set(v___x_189_, 1, v___x_185_);
lean_ctor_set(v___x_189_, 2, v___x_187_);
lean_ctor_set(v___x_189_, 3, v___x_188_);
v___x_190_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__12));
v___x_191_ = l_Lean_Syntax_node2(v___x_183_, v___x_190_, v___x_179_, v___x_181_);
v___x_192_ = l_Lean_Syntax_node2(v___x_183_, v___x_184_, v___x_189_, v___x_191_);
v___x_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_192_);
lean_ctor_set(v___x_193_, 1, v_a_170_);
return v___x_193_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___boxed(lean_object* v_x_194_, lean_object* v_a_195_, lean_object* v_a_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1(v_x_194_, v_a_195_, v_a_196_);
lean_dec_ref(v_a_195_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1(lean_object* v_x_201_, lean_object* v_a_202_, lean_object* v_a_203_){
_start:
{
lean_object* v___x_204_; uint8_t v___x_205_; 
v___x_204_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4));
lean_inc(v_x_201_);
v___x_205_ = l_Lean_Syntax_isOfKind(v_x_201_, v___x_204_);
if (v___x_205_ == 0)
{
lean_object* v___x_206_; lean_object* v___x_207_; 
lean_dec(v_x_201_);
v___x_206_ = lean_box(0);
v___x_207_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v_a_203_);
return v___x_207_;
}
else
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; uint8_t v___x_211_; 
v___x_208_ = lean_unsigned_to_nat(0u);
v___x_209_ = l_Lean_Syntax_getArg(v_x_201_, v___x_208_);
v___x_210_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__1));
lean_inc(v___x_209_);
v___x_211_ = l_Lean_Syntax_isOfKind(v___x_209_, v___x_210_);
if (v___x_211_ == 0)
{
lean_object* v___x_212_; lean_object* v___x_213_; 
lean_dec(v___x_209_);
lean_dec(v_x_201_);
v___x_212_ = lean_box(0);
v___x_213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
lean_ctor_set(v___x_213_, 1, v_a_203_);
return v___x_213_;
}
else
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; uint8_t v___x_217_; 
v___x_214_ = lean_unsigned_to_nat(1u);
v___x_215_ = l_Lean_Syntax_getArg(v_x_201_, v___x_214_);
lean_dec(v_x_201_);
v___x_216_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_215_);
v___x_217_ = l_Lean_Syntax_matchesNull(v___x_215_, v___x_216_);
if (v___x_217_ == 0)
{
lean_object* v___x_218_; lean_object* v___x_219_; 
lean_dec(v___x_215_);
lean_dec(v___x_209_);
v___x_218_ = lean_box(0);
v___x_219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_218_);
lean_ctor_set(v___x_219_, 1, v_a_203_);
return v___x_219_;
}
else
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v_ref_222_; uint8_t v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_220_ = l_Lean_Syntax_getArg(v___x_215_, v___x_208_);
v___x_221_ = l_Lean_Syntax_getArg(v___x_215_, v___x_214_);
lean_dec(v___x_215_);
v_ref_222_ = l_Lean_replaceRef(v___x_209_, v_a_202_);
lean_dec(v___x_209_);
v___x_223_ = 0;
v___x_224_ = l_Lean_SourceInfo_fromRef(v_ref_222_, v___x_223_);
lean_dec(v_ref_222_);
v___x_225_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__2));
v___x_226_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a_u209b___00__closed__5));
lean_inc(v___x_224_);
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_224_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v___x_228_ = l_Lean_Syntax_node3(v___x_224_, v___x_225_, v___x_220_, v___x_227_, v___x_221_);
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v_a_203_);
return v___x_229_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___boxed(lean_object* v_x_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1(v_x_230_, v_a_231_, v_a_232_);
lean_dec(v_a_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___redArg(lean_object* v_x_234_){
_start:
{
lean_inc(v_x_234_);
return v_x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___redArg___boxed(lean_object* v_x_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_Sym_ofVector___redArg(v_x_235_);
lean_dec(v_x_235_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector(lean_object* v_00_u03b1_237_, lean_object* v_n_238_, lean_object* v_x_239_){
_start:
{
lean_inc(v_x_239_);
return v_x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_ofVector___boxed(lean_object* v_00_u03b1_240_, lean_object* v_n_241_, lean_object* v_x_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_Sym_ofVector(v_00_u03b1_240_, v_n_241_, v_x_242_);
lean_dec(v_x_242_);
lean_dec(v_n_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___lam__0(lean_object* v_x_244_){
_start:
{
lean_inc(v_x_244_);
return v_x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___lam__0___boxed(lean_object* v_x_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Sym_instCoeVector___lam__0(v_x_245_);
lean_dec(v_x_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector(lean_object* v_00_u03b1_248_, lean_object* v_n_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = ((lean_object*)(lp_mathlib_Sym_instCoeVector___closed__0));
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instCoeVector___boxed(lean_object* v_00_u03b1_251_, lean_object* v_n_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_Sym_instCoeVector(v_00_u03b1_251_, v_n_252_);
lean_dec(v_n_252_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instMembership(lean_object* v_00_u03b1_254_, lean_object* v_n_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lean_box(0);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instMembership___boxed(lean_object* v_00_u03b1_257_, lean_object* v_n_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_mathlib_Sym_instMembership(v_00_u03b1_257_, v_n_258_);
lean_dec(v_n_258_);
return v_res_259_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sym_decidableMem___redArg(lean_object* v_inst_260_, lean_object* v_a_261_, lean_object* v_s_262_){
_start:
{
uint8_t v___x_263_; 
v___x_263_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_260_, v_a_261_, v_s_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_decidableMem___redArg___boxed(lean_object* v_inst_264_, lean_object* v_a_265_, lean_object* v_s_266_){
_start:
{
uint8_t v_res_267_; lean_object* v_r_268_; 
v_res_267_ = lp_mathlib_Sym_decidableMem___redArg(v_inst_264_, v_a_265_, v_s_266_);
v_r_268_ = lean_box(v_res_267_);
return v_r_268_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sym_decidableMem(lean_object* v_00_u03b1_269_, lean_object* v_n_270_, lean_object* v_inst_271_, lean_object* v_a_272_, lean_object* v_s_273_){
_start:
{
uint8_t v___x_274_; 
v___x_274_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_271_, v_a_272_, v_s_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_decidableMem___boxed(lean_object* v_00_u03b1_275_, lean_object* v_n_276_, lean_object* v_inst_277_, lean_object* v_a_278_, lean_object* v_s_279_){
_start:
{
uint8_t v_res_280_; lean_object* v_r_281_; 
v_res_280_ = lp_mathlib_Sym_decidableMem(v_00_u03b1_275_, v_n_276_, v_inst_277_, v_a_278_, v_s_279_);
lean_dec(v_n_276_);
v_r_281_ = lean_box(v_res_280_);
return v_r_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase___redArg(lean_object* v_inst_282_, lean_object* v_s_283_, lean_object* v_a_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_mathlib_Multiset_erase___redArg(v_inst_282_, v_s_283_, v_a_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase(lean_object* v_00_u03b1_286_, lean_object* v_n_287_, lean_object* v_inst_288_, lean_object* v_s_289_, lean_object* v_a_290_, lean_object* v_h_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Multiset_erase___redArg(v_inst_288_, v_s_289_, v_a_290_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_erase___boxed(lean_object* v_00_u03b1_293_, lean_object* v_n_294_, lean_object* v_inst_295_, lean_object* v_s_296_, lean_object* v_a_297_, lean_object* v_h_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Sym_erase(v_00_u03b1_293_, v_n_294_, v_inst_295_, v_s_296_, v_a_297_, v_h_298_);
lean_dec(v_n_294_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27___redArg(lean_object* v_a_300_, lean_object* v_a_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_302_, 0, v_a_300_);
lean_ctor_set(v___x_302_, 1, v_a_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27(lean_object* v_00_u03b1_303_, lean_object* v_n_304_, lean_object* v_a_305_, lean_object* v_a_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_307_, 0, v_a_305_);
lean_ctor_set(v___x_307_, 1, v_a_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cons_x27___boxed(lean_object* v_00_u03b1_308_, lean_object* v_n_309_, lean_object* v_a_310_, lean_object* v_a_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_Sym_cons_x27(v_00_u03b1_308_, v_n_309_, v_a_310_, v_a_311_);
lean_dec(v_n_309_);
return v_res_312_;
}
}
static lean_object* _init_lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1(void){
_start:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__0));
v___x_335_ = l_String_toRawSubstring_x27(v___x_334_);
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1(lean_object* v_x_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_355_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a___00__closed__1));
lean_inc(v_x_352_);
v___x_356_ = l_Lean_Syntax_isOfKind(v_x_352_, v___x_355_);
if (v___x_356_ == 0)
{
lean_object* v___x_357_; lean_object* v___x_358_; 
lean_dec(v_x_352_);
v___x_357_ = lean_box(1);
v___x_358_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_a_354_);
return v___x_358_;
}
else
{
lean_object* v_quotContext_359_; lean_object* v_currMacroScope_360_; lean_object* v_ref_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v_quotContext_359_ = lean_ctor_get(v_a_353_, 1);
v_currMacroScope_360_ = lean_ctor_get(v_a_353_, 2);
v_ref_361_ = lean_ctor_get(v_a_353_, 5);
v___x_362_ = lean_unsigned_to_nat(0u);
v___x_363_ = l_Lean_Syntax_getArg(v_x_352_, v___x_362_);
v___x_364_ = lean_unsigned_to_nat(2u);
v___x_365_ = l_Lean_Syntax_getArg(v_x_352_, v___x_364_);
lean_dec(v_x_352_);
v___x_366_ = 0;
v___x_367_ = l_Lean_SourceInfo_fromRef(v_ref_361_, v___x_366_);
v___x_368_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4));
v___x_369_ = lean_obj_once(&lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1, &lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1_once, _init_lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__1);
v___x_370_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__2));
lean_inc(v_currMacroScope_360_);
lean_inc(v_quotContext_359_);
v___x_371_ = l_Lean_addMacroScope(v_quotContext_359_, v___x_370_, v_currMacroScope_360_);
v___x_372_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___closed__7));
lean_inc_n(v___x_367_, 2);
v___x_373_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_373_, 0, v___x_367_);
lean_ctor_set(v___x_373_, 1, v___x_369_);
lean_ctor_set(v___x_373_, 2, v___x_371_);
lean_ctor_set(v___x_373_, 3, v___x_372_);
v___x_374_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__12));
v___x_375_ = l_Lean_Syntax_node2(v___x_367_, v___x_374_, v___x_363_, v___x_365_);
v___x_376_ = l_Lean_Syntax_node2(v___x_367_, v___x_368_, v___x_373_, v___x_375_);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v_a_354_);
return v___x_377_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1___boxed(lean_object* v_x_378_, lean_object* v_a_379_, lean_object* v_a_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a____1(v_x_378_, v_a_379_, v_a_380_);
lean_dec_ref(v_a_379_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons_x27__1(lean_object* v_x_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v___x_385_; uint8_t v___x_386_; 
v___x_385_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______macroRules__Sym__term___x3a_x3a_u209b____1___closed__4));
lean_inc(v_x_382_);
v___x_386_ = l_Lean_Syntax_isOfKind(v_x_382_, v___x_385_);
if (v___x_386_ == 0)
{
lean_object* v___x_387_; lean_object* v___x_388_; 
lean_dec(v_x_382_);
v___x_387_ = lean_box(0);
v___x_388_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_387_);
lean_ctor_set(v___x_388_, 1, v_a_384_);
return v___x_388_;
}
else
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; uint8_t v___x_392_; 
v___x_389_ = lean_unsigned_to_nat(0u);
v___x_390_ = l_Lean_Syntax_getArg(v_x_382_, v___x_389_);
v___x_391_ = ((lean_object*)(lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons__1___closed__1));
lean_inc(v___x_390_);
v___x_392_ = l_Lean_Syntax_isOfKind(v___x_390_, v___x_391_);
if (v___x_392_ == 0)
{
lean_object* v___x_393_; lean_object* v___x_394_; 
lean_dec(v___x_390_);
lean_dec(v_x_382_);
v___x_393_ = lean_box(0);
v___x_394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_393_);
lean_ctor_set(v___x_394_, 1, v_a_384_);
return v___x_394_;
}
else
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_395_ = lean_unsigned_to_nat(1u);
v___x_396_ = l_Lean_Syntax_getArg(v_x_382_, v___x_395_);
lean_dec(v_x_382_);
v___x_397_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_396_);
v___x_398_ = l_Lean_Syntax_matchesNull(v___x_396_, v___x_397_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; lean_object* v___x_400_; 
lean_dec(v___x_396_);
lean_dec(v___x_390_);
v___x_399_ = lean_box(0);
v___x_400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v_a_384_);
return v___x_400_;
}
else
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v_ref_403_; uint8_t v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_401_ = l_Lean_Syntax_getArg(v___x_396_, v___x_389_);
v___x_402_ = l_Lean_Syntax_getArg(v___x_396_, v___x_395_);
lean_dec(v___x_396_);
v_ref_403_ = l_Lean_replaceRef(v___x_390_, v_a_383_);
lean_dec(v___x_390_);
v___x_404_ = 0;
v___x_405_ = l_Lean_SourceInfo_fromRef(v_ref_403_, v___x_404_);
lean_dec(v_ref_403_);
v___x_406_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a___00__closed__1));
v___x_407_ = ((lean_object*)(lp_mathlib_Sym_term___x3a_x3a___00__closed__2));
lean_inc(v___x_405_);
v___x_408_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_405_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = l_Lean_Syntax_node3(v___x_405_, v___x_406_, v___x_401_, v___x_408_, v___x_402_);
v___x_410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_409_);
lean_ctor_set(v___x_410_, 1, v_a_384_);
return v___x_410_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons_x27__1___boxed(lean_object* v_x_411_, lean_object* v_a_412_, lean_object* v_a_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_Sym___aux__Mathlib__Data__Sym__Basic______unexpand__Sym__cons_x27__1(v_x_411_, v_a_412_, v_a_413_);
lean_dec(v_a_412_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0(lean_object* v_a_415_){
_start:
{
lean_inc(v_a_415_);
return v_a_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0___boxed(lean_object* v_a_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___lam__0(v_a_416_);
lean_dec(v_a_416_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_object* v_00_u03b1_421_, lean_object* v_n_422_, lean_object* v_p_u2081_423_, lean_object* v_p_u2082_424_, lean_object* v_hp_u2082_425_, lean_object* v_h_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = ((lean_object*)(lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__1));
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___boxed(lean_object* v_00_u03b1_428_, lean_object* v_n_429_, lean_object* v_p_u2081_430_, lean_object* v_p_u2082_431_, lean_object* v_hp_u2082_432_, lean_object* v_h_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(v_00_u03b1_428_, v_n_429_, v_p_u2081_430_, v_p_u2082_431_, v_hp_u2082_432_, v_h_433_);
lean_dec(v_n_429_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___redArg(lean_object* v_n_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_box(0), v_n_435_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___redArg___boxed(lean_object* v_n_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_mathlib_Sym_symEquivSym_x27___redArg(v_n_437_);
lean_dec(v_n_437_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27(lean_object* v_00_u03b1_439_, lean_object* v_n_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_box(0), v_n_440_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_symEquivSym_x27___boxed(lean_object* v_00_u03b1_442_, lean_object* v_n_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_Sym_symEquivSym_x27(v_00_u03b1_442_, v_n_443_);
lean_dec(v_n_443_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instZeroSym(lean_object* v_00_u03b1_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lean_box(0);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instEmptyCollectionOfNatNat(lean_object* v_00_u03b1_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lean_box(0);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_uniqueZero(lean_object* v_00_u03b1_449_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lean_box(0);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_replicate___redArg(lean_object* v_n_451_, lean_object* v_a_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = l_List_replicateTR___redArg(v_n_451_, v_a_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_replicate(lean_object* v_00_u03b1_454_, lean_object* v_n_455_, lean_object* v_a_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = l_List_replicateTR___redArg(v_n_455_, v_a_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym___redArg(lean_object* v_inst_458_, lean_object* v_n_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = l_List_replicateTR___redArg(v_n_459_, v_inst_458_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym(lean_object* v_00_u03b1_461_, lean_object* v_inst_462_, lean_object* v_n_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = l_List_replicateTR___redArg(v_n_463_, v_inst_462_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym_x27___redArg(lean_object* v_inst_465_, lean_object* v_n_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = l_List_replicateTR___redArg(v_n_466_, v_inst_465_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_inhabitedSym_x27(lean_object* v_00_u03b1_468_, lean_object* v_inst_469_, lean_object* v_n_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = l_List_replicateTR___redArg(v_n_470_, v_inst_469_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instUnique___redArg(lean_object* v_n_472_, lean_object* v_inst_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = l_List_replicateTR___redArg(v_n_472_, v_inst_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instUnique(lean_object* v_00_u03b1_475_, lean_object* v_n_476_, lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = l_List_replicateTR___redArg(v_n_476_, v_inst_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_map___redArg(lean_object* v_f_479_, lean_object* v_x_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_mathlib_Multiset_map___redArg(v_f_479_, v_x_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_map(lean_object* v_00_u03b1_482_, lean_object* v_00_u03b2_483_, lean_object* v_n_484_, lean_object* v_f_485_, lean_object* v_x_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_Multiset_map___redArg(v_f_485_, v_x_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_map___boxed(lean_object* v_00_u03b1_488_, lean_object* v_00_u03b2_489_, lean_object* v_n_490_, lean_object* v_f_491_, lean_object* v_x_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib_Sym_map(v_00_u03b1_488_, v_00_u03b2_489_, v_n_490_, v_f_491_, v_x_492_);
lean_dec(v_n_490_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg___lam__0(lean_object* v_e_494_, lean_object* v___y_495_){
_start:
{
lean_object* v_toFun_496_; lean_object* v___x_497_; 
v_toFun_496_ = lean_ctor_get(v_e_494_, 0);
lean_inc(v_toFun_496_);
lean_dec_ref(v_e_494_);
v___x_497_ = lean_apply_1(v_toFun_496_, v___y_495_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg___lam__1(lean_object* v___x_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_toFun_500_; lean_object* v___x_501_; 
v_toFun_500_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_toFun_500_);
lean_dec_ref(v___x_498_);
v___x_501_ = lean_apply_1(v_toFun_500_, v___y_499_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr___redArg(lean_object* v_n_502_, lean_object* v_e_503_){
_start:
{
lean_object* v___f_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___f_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
lean_inc_ref(v_e_503_);
v___f_504_ = lean_alloc_closure((void*)(lp_mathlib_Sym_equivCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_504_, 0, v_e_503_);
lean_inc(v_n_502_);
v___x_505_ = lean_alloc_closure((void*)(lp_mathlib_Sym_map___boxed), 5, 4);
lean_closure_set(v___x_505_, 0, lean_box(0));
lean_closure_set(v___x_505_, 1, lean_box(0));
lean_closure_set(v___x_505_, 2, v_n_502_);
lean_closure_set(v___x_505_, 3, v___f_504_);
v___x_506_ = lp_mathlib_Equiv_symm___redArg(v_e_503_);
v___f_507_ = lean_alloc_closure((void*)(lp_mathlib_Sym_equivCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_507_, 0, v___x_506_);
v___x_508_ = lean_alloc_closure((void*)(lp_mathlib_Sym_map___boxed), 5, 4);
lean_closure_set(v___x_508_, 0, lean_box(0));
lean_closure_set(v___x_508_, 1, lean_box(0));
lean_closure_set(v___x_508_, 2, v_n_502_);
lean_closure_set(v___x_508_, 3, v___f_507_);
v___x_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_505_);
lean_ctor_set(v___x_509_, 1, v___x_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_equivCongr(lean_object* v_00_u03b1_510_, lean_object* v_00_u03b2_511_, lean_object* v_n_512_, lean_object* v_e_513_){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = lp_mathlib_Sym_equivCongr___redArg(v_n_512_, v_e_513_);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach___redArg(lean_object* v_s_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_mathlib_Multiset_attach___redArg(v_s_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach(lean_object* v_00_u03b1_517_, lean_object* v_n_518_, lean_object* v_s_519_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_mathlib_Multiset_attach___redArg(v_s_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_attach___boxed(lean_object* v_00_u03b1_521_, lean_object* v_n_522_, lean_object* v_s_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_Sym_attach(v_00_u03b1_521_, v_n_522_, v_s_523_);
lean_dec(v_n_522_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___lam__0(lean_object* v_s_525_){
_start:
{
lean_inc(v_s_525_);
return v_s_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___lam__0___boxed(lean_object* v_s_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_Sym_cast___lam__0(v_s_526_);
lean_dec(v_s_526_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast(lean_object* v_00_u03b1_531_, lean_object* v_n_532_, lean_object* v_m_533_, lean_object* v_h_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = ((lean_object*)(lp_mathlib_Sym_cast___closed__1));
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_cast___boxed(lean_object* v_00_u03b1_536_, lean_object* v_n_537_, lean_object* v_m_538_, lean_object* v_h_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_Sym_cast(v_00_u03b1_536_, v_n_537_, v_m_538_, v_h_539_);
lean_dec(v_m_538_);
lean_dec(v_n_537_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_append___redArg(lean_object* v_s_541_, lean_object* v_s_x27_542_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = l_List_appendTR___redArg(v_s_541_, v_s_x27_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_append(lean_object* v_00_u03b1_544_, lean_object* v_n_545_, lean_object* v_n_x27_546_, lean_object* v_s_547_, lean_object* v_s_x27_548_){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = l_List_appendTR___redArg(v_s_547_, v_s_x27_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_append___boxed(lean_object* v_00_u03b1_550_, lean_object* v_n_551_, lean_object* v_n_x27_552_, lean_object* v_s_553_, lean_object* v_s_x27_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_Sym_append(v_00_u03b1_550_, v_n_551_, v_n_x27_552_, v_s_553_, v_s_x27_554_);
lean_dec(v_n_x27_552_);
lean_dec(v_n_551_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_oneEquiv_spec__0(lean_object* v_00_u03b1_556_, lean_object* v_p_u2081_557_, lean_object* v_p_u2082_558_, lean_object* v_hp_u2082_559_, lean_object* v_h_560_){
_start:
{
lean_object* v___x_561_; 
v___x_561_ = ((lean_object*)(lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0___closed__1));
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv___lam__0(lean_object* v_a_562_){
_start:
{
lean_object* v___x_563_; lean_object* v___x_564_; 
v___x_563_ = lean_box(0);
v___x_564_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_564_, 0, v_a_562_);
lean_ctor_set(v___x_564_, 1, v___x_563_);
return v___x_564_;
}
}
static lean_object* _init_lp_mathlib_Sym_oneEquiv___lam__1___closed__0(void){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_oneEquiv_spec__0(lean_box(0), lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv___lam__1(lean_object* v_s_566_){
_start:
{
lean_object* v___x_567_; lean_object* v_toFun_568_; lean_object* v___x_569_; lean_object* v_head_570_; 
v___x_567_ = lean_obj_once(&lp_mathlib_Sym_oneEquiv___lam__1___closed__0, &lp_mathlib_Sym_oneEquiv___lam__1___closed__0_once, _init_lp_mathlib_Sym_oneEquiv___lam__1___closed__0);
v_toFun_568_ = lean_ctor_get(v___x_567_, 0);
lean_inc(v_toFun_568_);
v___x_569_ = lean_apply_1(v_toFun_568_, v_s_566_);
v_head_570_ = lean_ctor_get(v___x_569_, 0);
lean_inc(v_head_570_);
lean_dec(v___x_569_);
return v_head_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_oneEquiv(lean_object* v_00_u03b1_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = ((lean_object*)(lp_mathlib_Sym_oneEquiv___closed__2));
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___redArg(lean_object* v_n_578_, lean_object* v_a_579_, lean_object* v_i_580_, lean_object* v_m_581_){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v_toFun_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_582_ = lean_nat_sub(v_n_578_, v_i_580_);
v___x_583_ = lean_nat_add(v___x_582_, v_i_580_);
lean_dec(v___x_582_);
v___x_584_ = lp_mathlib_Sym_cast(lean_box(0), v___x_583_, v_n_578_, lean_box(0));
lean_dec(v___x_583_);
v_toFun_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_toFun_585_);
lean_dec_ref(v___x_584_);
v___x_586_ = l_List_replicateTR___redArg(v_i_580_, v_a_579_);
v___x_587_ = l_List_appendTR___redArg(v_m_581_, v___x_586_);
v___x_588_ = lean_apply_1(v_toFun_585_, v___x_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___redArg___boxed(lean_object* v_n_589_, lean_object* v_a_590_, lean_object* v_i_591_, lean_object* v_m_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_Sym_fill___redArg(v_n_589_, v_a_590_, v_i_591_, v_m_592_);
lean_dec(v_n_589_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill(lean_object* v_00_u03b1_594_, lean_object* v_n_595_, lean_object* v_a_596_, lean_object* v_i_597_, lean_object* v_m_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_mathlib_Sym_fill___redArg(v_n_595_, v_a_596_, v_i_597_, v_m_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_fill___boxed(lean_object* v_00_u03b1_600_, lean_object* v_n_601_, lean_object* v_a_602_, lean_object* v_i_603_, lean_object* v_m_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_Sym_fill(v_00_u03b1_600_, v_n_601_, v_a_602_, v_i_603_, v_m_604_);
lean_dec(v_n_601_);
return v_res_605_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sym_filterNe___redArg___lam__0(lean_object* v_inst_606_, lean_object* v_a_607_, lean_object* v_a_608_){
_start:
{
lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_609_ = lean_apply_2(v_inst_606_, v_a_607_, v_a_608_);
v___x_610_ = lean_unbox(v___x_609_);
if (v___x_610_ == 0)
{
uint8_t v___x_611_; 
v___x_611_ = 1;
return v___x_611_;
}
else
{
uint8_t v___x_612_; 
v___x_612_ = 0;
return v___x_612_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___redArg___lam__0___boxed(lean_object* v_inst_613_, lean_object* v_a_614_, lean_object* v_a_615_){
_start:
{
uint8_t v_res_616_; lean_object* v_r_617_; 
v_res_616_ = lp_mathlib_Sym_filterNe___redArg___lam__0(v_inst_613_, v_a_614_, v_a_615_);
v_r_617_ = lean_box(v_res_616_);
return v_r_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___redArg(lean_object* v_inst_618_, lean_object* v_a_619_, lean_object* v_m_620_){
_start:
{
lean_object* v___f_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
lean_inc(v_a_619_);
lean_inc_ref(v_inst_618_);
v___f_621_ = lean_alloc_closure((void*)(lp_mathlib_Sym_filterNe___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_621_, 0, v_inst_618_);
lean_closure_set(v___f_621_, 1, v_a_619_);
lean_inc(v_m_620_);
v___x_622_ = lp_mathlib_Multiset_count___redArg(v_inst_618_, v_a_619_, v_m_620_);
v___x_623_ = lp_mathlib_Multiset_filter___redArg(v___f_621_, v_m_620_);
v___x_624_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_624_, 0, v___x_622_);
lean_ctor_set(v___x_624_, 1, v___x_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe(lean_object* v_00_u03b1_625_, lean_object* v_n_626_, lean_object* v_inst_627_, lean_object* v_a_628_, lean_object* v_m_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_Sym_filterNe___redArg(v_inst_627_, v_a_628_, v_m_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_filterNe___boxed(lean_object* v_00_u03b1_631_, lean_object* v_n_632_, lean_object* v_inst_633_, lean_object* v_a_634_, lean_object* v_m_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib_Sym_filterNe(v_00_u03b1_631_, v_n_632_, v_inst_633_, v_a_634_, v_m_635_);
lean_dec(v_n_632_);
return v_res_636_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0(lean_object* v_inst_637_, lean_object* v_a_638_, lean_object* v_b_639_){
_start:
{
uint8_t v___x_640_; 
v___x_640_ = l_Option_instDecidableEq___redArg(v_inst_637_, v_a_638_, v_b_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0___boxed(lean_object* v_inst_641_, lean_object* v_a_642_, lean_object* v_b_643_){
_start:
{
uint8_t v_res_644_; lean_object* v_r_645_; 
v_res_644_ = lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0(v_inst_641_, v_a_642_, v_b_643_);
v_r_645_ = lean_box(v_res_644_);
return v_r_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1(lean_object* v_o_646_){
_start:
{
lean_object* v_val_647_; 
v_val_647_ = lean_ctor_get(v_o_646_, 0);
lean_inc(v_val_647_);
return v_val_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1___boxed(lean_object* v_o_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__1(v_o_648_);
lean_dec(v_o_648_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___redArg(lean_object* v_inst_651_, lean_object* v_s_652_){
_start:
{
lean_object* v___f_653_; lean_object* v___x_654_; uint8_t v___x_655_; 
v___f_653_ = lean_alloc_closure((void*)(lp_mathlib_SymOptionSuccEquiv_encode___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_653_, 0, v_inst_651_);
v___x_654_ = lean_box(0);
lean_inc(v_s_652_);
lean_inc_ref(v___f_653_);
v___x_655_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v___f_653_, v___x_654_, v_s_652_);
if (v___x_655_ == 0)
{
lean_object* v___f_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; 
lean_dec_ref(v___f_653_);
v___f_656_ = ((lean_object*)(lp_mathlib_SymOptionSuccEquiv_encode___redArg___closed__0));
v___x_657_ = lp_mathlib_Multiset_attach___redArg(v_s_652_);
v___x_658_ = lp_mathlib_Multiset_map___redArg(v___f_656_, v___x_657_);
v___x_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_659_, 0, v___x_658_);
return v___x_659_;
}
else
{
lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_660_ = lp_mathlib_Multiset_erase___redArg(v___f_653_, v_s_652_, v___x_654_);
v___x_661_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_661_, 0, v___x_660_);
return v___x_661_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode(lean_object* v_00_u03b1_662_, lean_object* v_n_663_, lean_object* v_inst_664_, lean_object* v_s_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lp_mathlib_SymOptionSuccEquiv_encode___redArg(v_inst_664_, v_s_665_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_encode___boxed(lean_object* v_00_u03b1_667_, lean_object* v_n_668_, lean_object* v_inst_669_, lean_object* v_s_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib_SymOptionSuccEquiv_encode(v_00_u03b1_667_, v_n_668_, v_inst_669_, v_s_670_);
lean_dec(v_n_668_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode___redArg(lean_object* v_x_673_){
_start:
{
if (lean_obj_tag(v_x_673_) == 0)
{
lean_object* v_val_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v_val_674_ = lean_ctor_get(v_x_673_, 0);
lean_inc(v_val_674_);
lean_dec_ref_known(v_x_673_, 1);
v___x_675_ = lean_box(0);
v___x_676_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
lean_ctor_set(v___x_676_, 1, v_val_674_);
return v___x_676_;
}
else
{
lean_object* v_val_677_; lean_object* v___f_678_; lean_object* v___x_679_; 
v_val_677_ = lean_ctor_get(v_x_673_, 0);
lean_inc(v_val_677_);
lean_dec_ref_known(v_x_673_, 1);
v___f_678_ = ((lean_object*)(lp_mathlib_SymOptionSuccEquiv_decode___redArg___closed__0));
v___x_679_ = lp_mathlib_Multiset_map___redArg(v___f_678_, v_val_677_);
return v___x_679_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode(lean_object* v_00_u03b1_680_, lean_object* v_n_681_, lean_object* v_x_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib_SymOptionSuccEquiv_decode___redArg(v_x_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SymOptionSuccEquiv_decode___boxed(lean_object* v_00_u03b1_684_, lean_object* v_n_685_, lean_object* v_x_686_){
_start:
{
lean_object* v_res_687_; 
v_res_687_ = lp_mathlib_SymOptionSuccEquiv_decode(v_00_u03b1_684_, v_n_685_, v_x_686_);
lean_dec(v_n_685_);
return v_res_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter___redArg(lean_object* v_x_688_, lean_object* v_h__1_689_, lean_object* v_h__2_690_){
_start:
{
if (lean_obj_tag(v_x_688_) == 0)
{
lean_object* v_val_691_; lean_object* v___x_692_; 
lean_dec(v_h__2_690_);
v_val_691_ = lean_ctor_get(v_x_688_, 0);
lean_inc(v_val_691_);
lean_dec_ref_known(v_x_688_, 1);
v___x_692_ = lean_apply_1(v_h__1_689_, v_val_691_);
return v___x_692_;
}
else
{
lean_object* v_val_693_; lean_object* v___x_694_; 
lean_dec(v_h__1_689_);
v_val_693_ = lean_ctor_get(v_x_688_, 0);
lean_inc(v_val_693_);
lean_dec_ref_known(v_x_688_, 1);
v___x_694_ = lean_apply_1(v_h__2_690_, v_val_693_);
return v___x_694_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter(lean_object* v_00_u03b1_695_, lean_object* v_n_696_, lean_object* v_motive_697_, lean_object* v_x_698_, lean_object* v_h__1_699_, lean_object* v_h__2_700_){
_start:
{
if (lean_obj_tag(v_x_698_) == 0)
{
lean_object* v_val_701_; lean_object* v___x_702_; 
lean_dec(v_h__2_700_);
v_val_701_ = lean_ctor_get(v_x_698_, 0);
lean_inc(v_val_701_);
lean_dec_ref_known(v_x_698_, 1);
v___x_702_ = lean_apply_1(v_h__1_699_, v_val_701_);
return v___x_702_;
}
else
{
lean_object* v_val_703_; lean_object* v___x_704_; 
lean_dec(v_h__1_699_);
v_val_703_ = lean_ctor_get(v_x_698_, 0);
lean_inc(v_val_703_);
lean_dec_ref_known(v_x_698_, 1);
v___x_704_ = lean_apply_1(v_h__2_700_, v_val_703_);
return v___x_704_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter___boxed(lean_object* v_00_u03b1_705_, lean_object* v_n_706_, lean_object* v_motive_707_, lean_object* v_x_708_, lean_object* v_h__1_709_, lean_object* v_h__2_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib___private_Mathlib_Data_Sym_Basic_0__SymOptionSuccEquiv_decode_match__1_splitter(v_00_u03b1_705_, v_n_706_, v_motive_707_, v_x_708_, v_h__1_709_, v_h__2_710_);
lean_dec(v_n_706_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symOptionSuccEquiv___redArg(lean_object* v_n_712_, lean_object* v_inst_713_){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
lean_inc(v_n_712_);
v___x_714_ = lean_alloc_closure((void*)(lp_mathlib_SymOptionSuccEquiv_encode___boxed), 4, 3);
lean_closure_set(v___x_714_, 0, lean_box(0));
lean_closure_set(v___x_714_, 1, v_n_712_);
lean_closure_set(v___x_714_, 2, v_inst_713_);
v___x_715_ = lean_alloc_closure((void*)(lp_mathlib_SymOptionSuccEquiv_decode___boxed), 3, 2);
lean_closure_set(v___x_715_, 0, lean_box(0));
lean_closure_set(v___x_715_, 1, v_n_712_);
v___x_716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_716_, 0, v___x_714_);
lean_ctor_set(v___x_716_, 1, v___x_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_symOptionSuccEquiv(lean_object* v_00_u03b1_717_, lean_object* v_n_718_, lean_object* v_inst_719_){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = lp_mathlib_symOptionSuccEquiv___redArg(v_n_718_, v_inst_719_);
return v___x_720_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
