// Lean compiler output
// Module: Mathlib.Order.OrderDual
// Imports: public import Init public meta import Init public import Mathlib.Logic.Equiv.Defs public import Mathlib.Order.Basic
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d52_u1d48___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term_ᵒᵈ"};
static const lean_object* lp_mathlib_term___u1d52_u1d48___closed__0 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d52_u1d48___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__0_value),LEAN_SCALAR_PTR_LITERAL(86, 68, 120, 226, 215, 90, 210, 84)}};
static const lean_object* lp_mathlib_term___u1d52_u1d48___closed__1 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d52_u1d48___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ᵒᵈ"};
static const lean_object* lp_mathlib_term___u1d52_u1d48___closed__2 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d52_u1d48___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d52_u1d48___closed__3 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d52_u1d48___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d52_u1d48___closed__4 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d52_u1d48 = (const lean_object*)&lp_mathlib_term___u1d52_u1d48___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "OrderDual"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 229, 252, 204, 94, 197, 88, 206)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLT(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instOrd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMinOfMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMinOfMax(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_OrderDual_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OrderDual_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_OrderDual_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__16___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__16(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__20___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__20(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderDual_toDual___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderDual_toDual___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_toDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_ofDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_rec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_top___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_top(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_bot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_bot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_sdiff___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_sdiff___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_sdiff___redArg___closed__0 = (const lean_object*)&lp_mathlib_Equiv_sdiff___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_himp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_himp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_hnot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_hnot(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6(void){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__5));
v___x_24_ = l_String_toRawSubstring_x27(v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1(lean_object* v_x_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___x_39_; uint8_t v___x_40_; 
v___x_39_ = ((lean_object*)(lp_mathlib_term___u1d52_u1d48___closed__1));
lean_inc(v_x_36_);
v___x_40_ = l_Lean_Syntax_isOfKind(v_x_36_, v___x_39_);
if (v___x_40_ == 0)
{
lean_object* v___x_41_; lean_object* v___x_42_; 
lean_dec(v_x_36_);
v___x_41_ = lean_box(1);
v___x_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v_a_38_);
return v___x_42_;
}
else
{
lean_object* v_quotContext_43_; lean_object* v_currMacroScope_44_; lean_object* v_ref_45_; lean_object* v___x_46_; lean_object* v___x_47_; uint8_t v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v_quotContext_43_ = lean_ctor_get(v_a_37_, 1);
v_currMacroScope_44_ = lean_ctor_get(v_a_37_, 2);
v_ref_45_ = lean_ctor_get(v_a_37_, 5);
v___x_46_ = lean_unsigned_to_nat(0u);
v___x_47_ = l_Lean_Syntax_getArg(v_x_36_, v___x_46_);
lean_dec(v_x_36_);
v___x_48_ = 0;
v___x_49_ = l_Lean_SourceInfo_fromRef(v_ref_45_, v___x_48_);
v___x_50_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4));
v___x_51_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6, &lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__6);
v___x_52_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__7));
lean_inc(v_currMacroScope_44_);
lean_inc(v_quotContext_43_);
v___x_53_ = l_Lean_addMacroScope(v_quotContext_43_, v___x_52_, v_currMacroScope_44_);
v___x_54_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__9));
lean_inc_n(v___x_49_, 2);
v___x_55_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_55_, 0, v___x_49_);
lean_ctor_set(v___x_55_, 1, v___x_51_);
lean_ctor_set(v___x_55_, 2, v___x_53_);
lean_ctor_set(v___x_55_, 3, v___x_54_);
v___x_56_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__11));
v___x_57_ = l_Lean_Syntax_node1(v___x_49_, v___x_56_, v___x_47_);
v___x_58_ = l_Lean_Syntax_node2(v___x_49_, v___x_50_, v___x_55_, v___x_57_);
v___x_59_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_38_);
return v___x_59_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___boxed(lean_object* v_x_60_, lean_object* v_a_61_, lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1(v_x_60_, v_a_61_, v_a_62_);
lean_dec_ref(v_a_61_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1(lean_object* v_x_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v___x_70_; uint8_t v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______macroRules__term___u1d52_u1d48__1___closed__4));
lean_inc(v_x_67_);
v___x_71_ = l_Lean_Syntax_isOfKind(v_x_67_, v___x_70_);
if (v___x_71_ == 0)
{
lean_object* v___x_72_; lean_object* v___x_73_; 
lean_dec(v_x_67_);
v___x_72_ = lean_box(0);
v___x_73_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v_a_69_);
return v___x_73_;
}
else
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = l_Lean_Syntax_getArg(v_x_67_, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___closed__1));
lean_inc(v___x_75_);
v___x_77_ = l_Lean_Syntax_isOfKind(v___x_75_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; lean_object* v___x_79_; 
lean_dec(v___x_75_);
lean_dec(v_x_67_);
v___x_78_ = lean_box(0);
v___x_79_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_69_);
return v___x_79_;
}
else
{
lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_80_ = lean_unsigned_to_nat(1u);
v___x_81_ = l_Lean_Syntax_getArg(v_x_67_, v___x_80_);
lean_dec(v_x_67_);
lean_inc(v___x_81_);
v___x_82_ = l_Lean_Syntax_matchesNull(v___x_81_, v___x_80_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v___x_81_);
lean_dec(v___x_75_);
v___x_83_ = lean_box(0);
v___x_84_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_69_);
return v___x_84_;
}
else
{
lean_object* v___x_85_; lean_object* v_ref_86_; uint8_t v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_85_ = l_Lean_Syntax_getArg(v___x_81_, v___x_74_);
lean_dec(v___x_81_);
v_ref_86_ = l_Lean_replaceRef(v___x_75_, v_a_68_);
lean_dec(v___x_75_);
v___x_87_ = 0;
v___x_88_ = l_Lean_SourceInfo_fromRef(v_ref_86_, v___x_87_);
lean_dec(v_ref_86_);
v___x_89_ = ((lean_object*)(lp_mathlib_term___u1d52_u1d48___closed__1));
v___x_90_ = ((lean_object*)(lp_mathlib_term___u1d52_u1d48___closed__2));
lean_inc(v___x_88_);
v___x_91_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_88_);
lean_ctor_set(v___x_91_, 1, v___x_90_);
v___x_92_ = l_Lean_Syntax_node2(v___x_88_, v___x_89_, v___x_85_, v___x_91_);
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_69_);
return v___x_93_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1___boxed(lean_object* v_x_94_, lean_object* v_a_95_, lean_object* v_a_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib___aux__Mathlib__Order__OrderDual______unexpand__OrderDual__1(v_x_94_, v_a_95_, v_a_96_);
lean_dec(v_a_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLE(lean_object* v_00_u03b1_98_, lean_object* v_h_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_box(0);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLT(lean_object* v_00_u03b1_101_, lean_object* v_h_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_box(0);
return v___x_103_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instOrd___redArg___lam__0(lean_object* v_h_104_, lean_object* v_a_105_, lean_object* v_b_106_){
_start:
{
lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_107_ = lean_apply_2(v_h_104_, v_b_106_, v_a_105_);
v___x_108_ = lean_unbox(v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd___redArg___lam__0___boxed(lean_object* v_h_109_, lean_object* v_a_110_, lean_object* v_b_111_){
_start:
{
uint8_t v_res_112_; lean_object* v_r_113_; 
v_res_112_ = lp_mathlib_OrderDual_instOrd___redArg___lam__0(v_h_109_, v_a_110_, v_b_111_);
v_r_113_ = lean_box(v_res_112_);
return v_r_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd___redArg(lean_object* v_h_114_){
_start:
{
lean_object* v___f_115_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_115_, 0, v_h_114_);
return v___f_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instOrd(lean_object* v_00_u03b1_116_, lean_object* v_h_117_){
_start:
{
lean_object* v___f_118_; 
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_118_, 0, v_h_117_);
return v___f_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0(lean_object* v_h_119_, lean_object* v_a_120_, lean_object* v_b_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_apply_2(v_h_119_, v_a_120_, v_b_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin___redArg(lean_object* v_h_123_){
_start:
{
lean_object* v___f_124_; 
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_124_, 0, v_h_123_);
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMaxOfMin(lean_object* v_00_u03b1_125_, lean_object* v_h_126_){
_start:
{
lean_object* v___f_127_; 
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_127_, 0, v_h_126_);
return v___f_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMinOfMax___redArg(lean_object* v_h_128_){
_start:
{
lean_object* v___f_129_; 
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_129_, 0, v_h_128_);
return v___f_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instMinOfMax(lean_object* v_00_u03b1_130_, lean_object* v_h_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_132_, 0, v_h_131_);
return v___f_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPreorder(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = ((lean_object*)(lp_mathlib_OrderDual_instPreorder___closed__0));
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPreorder___boxed(lean_object* v_00_u03b1_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_OrderDual_instPreorder(v_00_u03b1_139_, v_inst_140_);
lean_dec_ref(v_inst_140_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___redArg(lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_inst_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___redArg___boxed(lean_object* v_inst_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_OrderDual_instPartialOrder___redArg(v_inst_144_);
lean_dec_ref(v_inst_144_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instPartialOrder___boxed(lean_object* v_00_u03b1_149_, lean_object* v_inst_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_OrderDual_instPartialOrder(v_00_u03b1_149_, v_inst_150_);
lean_dec_ref(v_inst_150_);
return v_res_151_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableEq___redArg(lean_object* v_inst_152_, lean_object* v_a_153_, lean_object* v_b_154_){
_start:
{
lean_object* v___x_155_; uint8_t v___x_156_; 
v___x_155_ = lean_apply_2(v_inst_152_, v_a_153_, v_b_154_);
v___x_156_ = lean_unbox(v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableEq___redArg___boxed(lean_object* v_inst_157_, lean_object* v_a_158_, lean_object* v_b_159_){
_start:
{
uint8_t v_res_160_; lean_object* v_r_161_; 
v_res_160_ = lp_mathlib_OrderDual_instDecidableEq___redArg(v_inst_157_, v_a_158_, v_b_159_);
v_r_161_ = lean_box(v_res_160_);
return v_r_161_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableEq(lean_object* v_00_u03b1_162_, lean_object* v_inst_163_, lean_object* v_a_164_, lean_object* v_b_165_){
_start:
{
lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_166_ = lean_apply_2(v_inst_163_, v_a_164_, v_b_165_);
v___x_167_ = lean_unbox(v___x_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableEq___boxed(lean_object* v_00_u03b1_168_, lean_object* v_inst_169_, lean_object* v_a_170_, lean_object* v_b_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_mathlib_OrderDual_instDecidableEq(v_00_u03b1_168_, v_inst_169_, v_a_170_, v_b_171_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLT___redArg(lean_object* v_h_174_, lean_object* v_a_175_, lean_object* v_b_176_){
_start:
{
lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_177_ = lean_apply_2(v_h_174_, v_b_176_, v_a_175_);
v___x_178_ = lean_unbox(v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLT___redArg___boxed(lean_object* v_h_179_, lean_object* v_a_180_, lean_object* v_b_181_){
_start:
{
uint8_t v_res_182_; lean_object* v_r_183_; 
v_res_182_ = lp_mathlib_OrderDual_instDecidableLT___redArg(v_h_179_, v_a_180_, v_b_181_);
v_r_183_ = lean_box(v_res_182_);
return v_r_183_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLT(lean_object* v_00_u03b1_184_, lean_object* v_inst_185_, lean_object* v_h_186_, lean_object* v_a_187_, lean_object* v_b_188_){
_start:
{
lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_189_ = lean_apply_2(v_h_186_, v_b_188_, v_a_187_);
v___x_190_ = lean_unbox(v___x_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLT___boxed(lean_object* v_00_u03b1_191_, lean_object* v_inst_192_, lean_object* v_h_193_, lean_object* v_a_194_, lean_object* v_b_195_){
_start:
{
uint8_t v_res_196_; lean_object* v_r_197_; 
v_res_196_ = lp_mathlib_OrderDual_instDecidableLT(v_00_u03b1_191_, v_inst_192_, v_h_193_, v_a_194_, v_b_195_);
v_r_197_ = lean_box(v_res_196_);
return v_r_197_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLE___redArg(lean_object* v_h_198_, lean_object* v_a_199_, lean_object* v_b_200_){
_start:
{
lean_object* v___x_201_; uint8_t v___x_202_; 
v___x_201_ = lean_apply_2(v_h_198_, v_b_200_, v_a_199_);
v___x_202_ = lean_unbox(v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLE___redArg___boxed(lean_object* v_h_203_, lean_object* v_a_204_, lean_object* v_b_205_){
_start:
{
uint8_t v_res_206_; lean_object* v_r_207_; 
v_res_206_ = lp_mathlib_OrderDual_instDecidableLE___redArg(v_h_203_, v_a_204_, v_b_205_);
v_r_207_ = lean_box(v_res_206_);
return v_r_207_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instDecidableLE(lean_object* v_00_u03b1_208_, lean_object* v_inst_209_, lean_object* v_h_210_, lean_object* v_a_211_, lean_object* v_b_212_){
_start:
{
lean_object* v___x_213_; uint8_t v___x_214_; 
v___x_213_ = lean_apply_2(v_h_210_, v_b_212_, v_a_211_);
v___x_214_ = lean_unbox(v___x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDecidableLE___boxed(lean_object* v_00_u03b1_215_, lean_object* v_inst_216_, lean_object* v_h_217_, lean_object* v_a_218_, lean_object* v_b_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_mathlib_OrderDual_instDecidableLE(v_00_u03b1_215_, v_inst_216_, v_h_217_, v_a_218_, v_b_219_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0(lean_object* v_toDecidableEq_222_, lean_object* v_a_223_, lean_object* v_b_224_){
_start:
{
lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_225_ = lean_apply_2(v_toDecidableEq_222_, v_a_223_, v_b_224_);
v___x_226_ = lean_unbox(v___x_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0___boxed(lean_object* v_toDecidableEq_227_, lean_object* v_a_228_, lean_object* v_b_229_){
_start:
{
uint8_t v_res_230_; lean_object* v_r_231_; 
v_res_230_ = lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0(v_toDecidableEq_227_, v_a_228_, v_b_229_);
v_r_231_ = lean_box(v_res_230_);
return v_r_231_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1(lean_object* v_toDecidableLT_232_, lean_object* v_a_233_, lean_object* v_b_234_){
_start:
{
lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_235_ = lean_apply_2(v_toDecidableLT_232_, v_b_234_, v_a_233_);
v___x_236_ = lean_unbox(v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1___boxed(lean_object* v_toDecidableLT_237_, lean_object* v_a_238_, lean_object* v_b_239_){
_start:
{
uint8_t v_res_240_; lean_object* v_r_241_; 
v_res_240_ = lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1(v_toDecidableLT_237_, v_a_238_, v_b_239_);
v_r_241_ = lean_box(v_res_240_);
return v_r_241_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2(lean_object* v_toDecidableLE_242_, lean_object* v_a_243_, lean_object* v_b_244_){
_start:
{
lean_object* v___x_245_; uint8_t v___x_246_; 
v___x_245_ = lean_apply_2(v_toDecidableLE_242_, v_b_244_, v_a_243_);
v___x_246_ = lean_unbox(v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2___boxed(lean_object* v_toDecidableLE_247_, lean_object* v_a_248_, lean_object* v_b_249_){
_start:
{
uint8_t v_res_250_; lean_object* v_r_251_; 
v_res_250_ = lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2(v_toDecidableLE_247_, v_a_248_, v_b_249_);
v_r_251_ = lean_box(v_res_250_);
return v_r_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object* v_inst_252_){
_start:
{
lean_object* v_toPartialOrder_253_; lean_object* v_toMin_254_; lean_object* v_toMax_255_; lean_object* v_toOrd_256_; lean_object* v_toDecidableLE_257_; lean_object* v_toDecidableEq_258_; lean_object* v_toDecidableLT_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_273_; 
v_toPartialOrder_253_ = lean_ctor_get(v_inst_252_, 0);
v_toMin_254_ = lean_ctor_get(v_inst_252_, 1);
v_toMax_255_ = lean_ctor_get(v_inst_252_, 2);
v_toOrd_256_ = lean_ctor_get(v_inst_252_, 3);
v_toDecidableLE_257_ = lean_ctor_get(v_inst_252_, 4);
v_toDecidableEq_258_ = lean_ctor_get(v_inst_252_, 5);
v_toDecidableLT_259_ = lean_ctor_get(v_inst_252_, 6);
v_isSharedCheck_273_ = !lean_is_exclusive(v_inst_252_);
if (v_isSharedCheck_273_ == 0)
{
v___x_261_ = v_inst_252_;
v_isShared_262_ = v_isSharedCheck_273_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_toDecidableLT_259_);
lean_inc(v_toDecidableEq_258_);
lean_inc(v_toDecidableLE_257_);
lean_inc(v_toOrd_256_);
lean_inc(v_toMax_255_);
lean_inc(v_toMin_254_);
lean_inc(v_toPartialOrder_253_);
lean_dec(v_inst_252_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_273_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___f_263_; lean_object* v___f_264_; lean_object* v___f_265_; lean_object* v___x_266_; lean_object* v___f_267_; lean_object* v___f_268_; lean_object* v___f_269_; lean_object* v___x_271_; 
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_263_, 0, v_toDecidableEq_258_);
v___f_264_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_264_, 0, v_toDecidableLT_259_);
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_265_, 0, v_toDecidableLE_257_);
v___x_266_ = lp_mathlib_OrderDual_instPreorder(lean_box(0), v_toPartialOrder_253_);
lean_dec_ref(v_toPartialOrder_253_);
v___f_267_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_267_, 0, v_toMax_255_);
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instMaxOfMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_268_, 0, v_toMin_254_);
v___f_269_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_269_, 0, v_toOrd_256_);
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 6, v___f_264_);
lean_ctor_set(v___x_261_, 5, v___f_263_);
lean_ctor_set(v___x_261_, 4, v___f_265_);
lean_ctor_set(v___x_261_, 3, v___f_269_);
lean_ctor_set(v___x_261_, 2, v___f_268_);
lean_ctor_set(v___x_261_, 1, v___f_267_);
lean_ctor_set(v___x_261_, 0, v___x_266_);
v___x_271_ = v___x_261_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_266_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v___f_267_);
lean_ctor_set(v_reuseFailAlloc_272_, 2, v___f_268_);
lean_ctor_set(v_reuseFailAlloc_272_, 3, v___f_269_);
lean_ctor_set(v_reuseFailAlloc_272_, 4, v___f_265_);
lean_ctor_set(v_reuseFailAlloc_272_, 5, v___f_263_);
lean_ctor_set(v_reuseFailAlloc_272_, 6, v___f_264_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
return v___x_271_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLinearOrder(lean_object* v_00_u03b1_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_OrderDual_instLinearOrder___redArg(v_inst_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__9___redArg(lean_object* v_x_277_, lean_object* v_a_278_, lean_object* v_b_279_){
_start:
{
lean_object* v_toMax_280_; lean_object* v___x_281_; 
v_toMax_280_ = lean_ctor_get(v_x_277_, 2);
lean_inc(v_toMax_280_);
lean_dec_ref(v_x_277_);
v___x_281_ = lean_apply_2(v_toMax_280_, v_a_278_, v_b_279_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__9(lean_object* v_00_u03b1_282_, lean_object* v_x_283_, lean_object* v_a_284_, lean_object* v_b_285_){
_start:
{
lean_object* v_toMax_286_; lean_object* v___x_287_; 
v_toMax_286_ = lean_ctor_get(v_x_283_, 2);
lean_inc(v_toMax_286_);
lean_dec_ref(v_x_283_);
v___x_287_ = lean_apply_2(v_toMax_286_, v_a_284_, v_b_285_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__11___redArg(lean_object* v_x_288_, lean_object* v_a_289_, lean_object* v_b_290_){
_start:
{
lean_object* v_toMin_291_; lean_object* v___x_292_; 
v_toMin_291_ = lean_ctor_get(v_x_288_, 1);
lean_inc(v_toMin_291_);
lean_dec_ref(v_x_288_);
v___x_292_ = lean_apply_2(v_toMin_291_, v_a_289_, v_b_290_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__11(lean_object* v_00_u03b1_293_, lean_object* v_x_294_, lean_object* v_a_295_, lean_object* v_b_296_){
_start:
{
lean_object* v_toMin_297_; lean_object* v___x_298_; 
v_toMin_297_ = lean_ctor_get(v_x_294_, 1);
lean_inc(v_toMin_297_);
lean_dec_ref(v_x_294_);
v___x_298_ = lean_apply_2(v_toMin_297_, v_a_295_, v_b_296_);
return v___x_298_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__13___redArg(lean_object* v_x_299_, lean_object* v_a_300_, lean_object* v_b_301_){
_start:
{
lean_object* v_toOrd_302_; lean_object* v___x_303_; uint8_t v___x_304_; 
v_toOrd_302_ = lean_ctor_get(v_x_299_, 3);
lean_inc_ref(v_toOrd_302_);
lean_dec_ref(v_x_299_);
v___x_303_ = lean_apply_2(v_toOrd_302_, v_b_301_, v_a_300_);
v___x_304_ = lean_unbox(v___x_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__13___redArg___boxed(lean_object* v_x_305_, lean_object* v_a_306_, lean_object* v_b_307_){
_start:
{
uint8_t v_res_308_; lean_object* v_r_309_; 
v_res_308_ = lp_mathlib_LinearOrder_swap___aux__13___redArg(v_x_305_, v_a_306_, v_b_307_);
v_r_309_ = lean_box(v_res_308_);
return v_r_309_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__13(lean_object* v_00_u03b1_310_, lean_object* v_x_311_, lean_object* v_a_312_, lean_object* v_b_313_){
_start:
{
lean_object* v_toOrd_314_; lean_object* v___x_315_; uint8_t v___x_316_; 
v_toOrd_314_ = lean_ctor_get(v_x_311_, 3);
lean_inc_ref(v_toOrd_314_);
lean_dec_ref(v_x_311_);
v___x_315_ = lean_apply_2(v_toOrd_314_, v_b_313_, v_a_312_);
v___x_316_ = lean_unbox(v___x_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__13___boxed(lean_object* v_00_u03b1_317_, lean_object* v_x_318_, lean_object* v_a_319_, lean_object* v_b_320_){
_start:
{
uint8_t v_res_321_; lean_object* v_r_322_; 
v_res_321_ = lp_mathlib_LinearOrder_swap___aux__13(v_00_u03b1_317_, v_x_318_, v_a_319_, v_b_320_);
v_r_322_ = lean_box(v_res_321_);
return v_r_322_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__16___redArg(lean_object* v_x_323_, lean_object* v_a_324_, lean_object* v_b_325_){
_start:
{
lean_object* v_toDecidableLE_326_; lean_object* v___x_327_; uint8_t v___x_328_; 
v_toDecidableLE_326_ = lean_ctor_get(v_x_323_, 4);
lean_inc_ref(v_toDecidableLE_326_);
lean_dec_ref(v_x_323_);
v___x_327_ = lean_apply_2(v_toDecidableLE_326_, v_b_325_, v_a_324_);
v___x_328_ = lean_unbox(v___x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__16___redArg___boxed(lean_object* v_x_329_, lean_object* v_a_330_, lean_object* v_b_331_){
_start:
{
uint8_t v_res_332_; lean_object* v_r_333_; 
v_res_332_ = lp_mathlib_LinearOrder_swap___aux__16___redArg(v_x_329_, v_a_330_, v_b_331_);
v_r_333_ = lean_box(v_res_332_);
return v_r_333_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__16(lean_object* v_00_u03b1_334_, lean_object* v_x_335_, lean_object* v_a_336_, lean_object* v_b_337_){
_start:
{
uint8_t v___x_338_; 
v___x_338_ = lp_mathlib_LinearOrder_swap___aux__16___redArg(v_x_335_, v_a_336_, v_b_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__16___boxed(lean_object* v_00_u03b1_339_, lean_object* v_x_340_, lean_object* v_a_341_, lean_object* v_b_342_){
_start:
{
uint8_t v_res_343_; lean_object* v_r_344_; 
v_res_343_ = lp_mathlib_LinearOrder_swap___aux__16(v_00_u03b1_339_, v_x_340_, v_a_341_, v_b_342_);
v_r_344_ = lean_box(v_res_343_);
return v_r_344_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__18___redArg(lean_object* v_x_345_, lean_object* v_a_346_, lean_object* v_b_347_){
_start:
{
lean_object* v_toDecidableEq_348_; lean_object* v___x_349_; uint8_t v___x_350_; 
v_toDecidableEq_348_ = lean_ctor_get(v_x_345_, 5);
lean_inc_ref(v_toDecidableEq_348_);
lean_dec_ref(v_x_345_);
v___x_349_ = lean_apply_2(v_toDecidableEq_348_, v_a_346_, v_b_347_);
v___x_350_ = lean_unbox(v___x_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__18___redArg___boxed(lean_object* v_x_351_, lean_object* v_a_352_, lean_object* v_b_353_){
_start:
{
uint8_t v_res_354_; lean_object* v_r_355_; 
v_res_354_ = lp_mathlib_LinearOrder_swap___aux__18___redArg(v_x_351_, v_a_352_, v_b_353_);
v_r_355_ = lean_box(v_res_354_);
return v_r_355_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__18(lean_object* v_00_u03b1_356_, lean_object* v_x_357_, lean_object* v_a_358_, lean_object* v_b_359_){
_start:
{
uint8_t v___x_360_; 
v___x_360_ = lp_mathlib_LinearOrder_swap___aux__18___redArg(v_x_357_, v_a_358_, v_b_359_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__18___boxed(lean_object* v_00_u03b1_361_, lean_object* v_x_362_, lean_object* v_a_363_, lean_object* v_b_364_){
_start:
{
uint8_t v_res_365_; lean_object* v_r_366_; 
v_res_365_ = lp_mathlib_LinearOrder_swap___aux__18(v_00_u03b1_361_, v_x_362_, v_a_363_, v_b_364_);
v_r_366_ = lean_box(v_res_365_);
return v_r_366_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__20___redArg(lean_object* v_x_367_, lean_object* v_a_368_, lean_object* v_b_369_){
_start:
{
lean_object* v_toDecidableLT_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v_toDecidableLT_370_ = lean_ctor_get(v_x_367_, 6);
lean_inc_ref(v_toDecidableLT_370_);
lean_dec_ref(v_x_367_);
v___x_371_ = lean_apply_2(v_toDecidableLT_370_, v_b_369_, v_a_368_);
v___x_372_ = lean_unbox(v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__20___redArg___boxed(lean_object* v_x_373_, lean_object* v_a_374_, lean_object* v_b_375_){
_start:
{
uint8_t v_res_376_; lean_object* v_r_377_; 
v_res_376_ = lp_mathlib_LinearOrder_swap___aux__20___redArg(v_x_373_, v_a_374_, v_b_375_);
v_r_377_ = lean_box(v_res_376_);
return v_r_377_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LinearOrder_swap___aux__20(lean_object* v_00_u03b1_378_, lean_object* v_x_379_, lean_object* v_a_380_, lean_object* v_b_381_){
_start:
{
uint8_t v___x_382_; 
v___x_382_ = lp_mathlib_LinearOrder_swap___aux__20___redArg(v_x_379_, v_a_380_, v_b_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___aux__20___boxed(lean_object* v_00_u03b1_383_, lean_object* v_x_384_, lean_object* v_a_385_, lean_object* v_b_386_){
_start:
{
uint8_t v_res_387_; lean_object* v_r_388_; 
v_res_387_ = lp_mathlib_LinearOrder_swap___aux__20(v_00_u03b1_383_, v_x_384_, v_a_385_, v_b_386_);
v_r_388_ = lean_box(v_res_387_);
return v_r_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap___redArg(lean_object* v_x_389_){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_390_ = ((lean_object*)(lp_mathlib_OrderDual_instPreorder___closed__0));
lean_inc_ref_n(v_x_389_, 5);
v___x_391_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__9), 4, 2);
lean_closure_set(v___x_391_, 0, lean_box(0));
lean_closure_set(v___x_391_, 1, v_x_389_);
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__11), 4, 2);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, v_x_389_);
v___x_393_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__13___boxed), 4, 2);
lean_closure_set(v___x_393_, 0, lean_box(0));
lean_closure_set(v___x_393_, 1, v_x_389_);
v___x_394_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__16___boxed), 4, 2);
lean_closure_set(v___x_394_, 0, lean_box(0));
lean_closure_set(v___x_394_, 1, v_x_389_);
v___x_395_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__18___boxed), 4, 2);
lean_closure_set(v___x_395_, 0, lean_box(0));
lean_closure_set(v___x_395_, 1, v_x_389_);
v___x_396_ = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_swap___aux__20___boxed), 4, 2);
lean_closure_set(v___x_396_, 0, lean_box(0));
lean_closure_set(v___x_396_, 1, v_x_389_);
v___x_397_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_397_, 0, v___x_390_);
lean_ctor_set(v___x_397_, 1, v___x_391_);
lean_ctor_set(v___x_397_, 2, v___x_392_);
lean_ctor_set(v___x_397_, 3, v___x_393_);
lean_ctor_set(v___x_397_, 4, v___x_394_);
lean_ctor_set(v___x_397_, 5, v___x_395_);
lean_ctor_set(v___x_397_, 6, v___x_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_swap(lean_object* v_00_u03b1_398_, lean_object* v_x_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_mathlib_LinearOrder_swap___redArg(v_x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___redArg(lean_object* v_h_401_){
_start:
{
lean_inc(v_h_401_);
return v_h_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___redArg___boxed(lean_object* v_h_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_OrderDual_instInhabited___redArg(v_h_402_);
lean_dec(v_h_402_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited(lean_object* v_00_u03b1_404_, lean_object* v_h_405_){
_start:
{
lean_inc(v_h_405_);
return v_h_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instInhabited___boxed(lean_object* v_00_u03b1_406_, lean_object* v_h_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_OrderDual_instInhabited(v_00_u03b1_406_, v_h_407_);
lean_dec(v_h_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___redArg(lean_object* v_h_409_){
_start:
{
lean_inc(v_h_409_);
return v_h_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___redArg___boxed(lean_object* v_h_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_OrderDual_instUnique___redArg(v_h_410_);
lean_dec(v_h_410_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique(lean_object* v_00_u03b1_412_, lean_object* v_h_413_){
_start:
{
lean_inc(v_h_413_);
return v_h_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instUnique___boxed(lean_object* v_00_u03b1_414_, lean_object* v_h_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_OrderDual_instUnique(v_00_u03b1_414_, v_h_415_);
lean_dec(v_h_415_);
return v_res_416_;
}
}
static lean_object* _init_lp_mathlib_OrderDual_toDual___closed__0(void){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_toDual(lean_object* v___y_418_){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lean_obj_once(&lp_mathlib_OrderDual_toDual___closed__0, &lp_mathlib_OrderDual_toDual___closed__0_once, _init_lp_mathlib_OrderDual_toDual___closed__0);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_ofDual(lean_object* v_00_u03b1_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lean_obj_once(&lp_mathlib_OrderDual_toDual___closed__0, &lp_mathlib_OrderDual_toDual___closed__0_once, _init_lp_mathlib_OrderDual_toDual___closed__0);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_rec___redArg(lean_object* v_toDual_422_, lean_object* v_a_423_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lean_apply_1(v_toDual_422_, v_a_423_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_rec(lean_object* v_00_u03b1_425_, lean_object* v_motive_426_, lean_object* v_toDual_427_, lean_object* v_a_428_){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lean_apply_1(v_toDual_427_, v_a_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_top___redArg(lean_object* v_e_430_, lean_object* v_inst_431_){
_start:
{
lean_object* v___x_432_; lean_object* v_toFun_433_; lean_object* v___x_434_; 
v___x_432_ = lp_mathlib_Equiv_symm___redArg(v_e_430_);
v_toFun_433_ = lean_ctor_get(v___x_432_, 0);
lean_inc(v_toFun_433_);
lean_dec_ref(v___x_432_);
v___x_434_ = lean_apply_1(v_toFun_433_, v_inst_431_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_top(lean_object* v_00_u03b1_435_, lean_object* v_00_u03b2_436_, lean_object* v_e_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; lean_object* v_toFun_440_; lean_object* v___x_441_; 
v___x_439_ = lp_mathlib_Equiv_symm___redArg(v_e_437_);
v_toFun_440_ = lean_ctor_get(v___x_439_, 0);
lean_inc(v_toFun_440_);
lean_dec_ref(v___x_439_);
v___x_441_ = lean_apply_1(v_toFun_440_, v_inst_438_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_bot___redArg(lean_object* v_e_442_, lean_object* v_inst_443_){
_start:
{
lean_object* v___x_444_; lean_object* v_toFun_445_; lean_object* v___x_446_; 
v___x_444_ = lp_mathlib_Equiv_symm___redArg(v_e_442_);
v_toFun_445_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_toFun_445_);
lean_dec_ref(v___x_444_);
v___x_446_ = lean_apply_1(v_toFun_445_, v_inst_443_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_bot(lean_object* v_00_u03b1_447_, lean_object* v_00_u03b2_448_, lean_object* v_e_449_, lean_object* v_inst_450_){
_start:
{
lean_object* v___x_451_; lean_object* v_toFun_452_; lean_object* v___x_453_; 
v___x_451_ = lp_mathlib_Equiv_symm___redArg(v_e_449_);
v_toFun_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_toFun_452_);
lean_dec_ref(v___x_451_);
v___x_453_ = lean_apply_1(v_toFun_452_, v_inst_450_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl___redArg___lam__0(lean_object* v_e_454_, lean_object* v_inst_455_, lean_object* v_a_456_){
_start:
{
lean_object* v_toFun_457_; lean_object* v___x_458_; lean_object* v_toFun_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v_toFun_457_ = lean_ctor_get(v_e_454_, 0);
lean_inc(v_toFun_457_);
v___x_458_ = lp_mathlib_Equiv_symm___redArg(v_e_454_);
v_toFun_459_ = lean_ctor_get(v___x_458_, 0);
lean_inc(v_toFun_459_);
lean_dec_ref(v___x_458_);
v___x_460_ = lean_apply_1(v_toFun_457_, v_a_456_);
v___x_461_ = lean_apply_1(v_inst_455_, v___x_460_);
v___x_462_ = lean_apply_1(v_toFun_459_, v___x_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl___redArg(lean_object* v_e_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_compl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_465_, 0, v_e_463_);
lean_closure_set(v___f_465_, 1, v_inst_464_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_compl(lean_object* v_00_u03b1_466_, lean_object* v_00_u03b2_467_, lean_object* v_e_468_, lean_object* v_inst_469_){
_start:
{
lean_object* v___f_470_; 
v___f_470_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_compl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_470_, 0, v_e_468_);
lean_closure_set(v___f_470_, 1, v_inst_469_);
return v___f_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg___lam__0(lean_object* v_self_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_toFun_473_; lean_object* v___x_474_; 
v_toFun_473_ = lean_ctor_get(v_self_471_, 0);
lean_inc(v_toFun_473_);
lean_dec_ref(v_self_471_);
v___x_474_ = lean_apply_1(v_toFun_473_, v___y_472_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg___lam__1(lean_object* v_e_475_, lean_object* v___f_476_, lean_object* v_inst_477_, lean_object* v_a_478_, lean_object* v_b_479_){
_start:
{
lean_object* v___x_480_; lean_object* v_toFun_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
lean_inc_ref_n(v_e_475_, 2);
v___x_480_ = lp_mathlib_Equiv_symm___redArg(v_e_475_);
v_toFun_481_ = lean_ctor_get(v___x_480_, 0);
lean_inc(v_toFun_481_);
lean_dec_ref(v___x_480_);
lean_inc(v___f_476_);
v___x_482_ = lean_apply_2(v___f_476_, v_e_475_, v_a_478_);
v___x_483_ = lean_apply_2(v___f_476_, v_e_475_, v_b_479_);
v___x_484_ = lean_apply_2(v_inst_477_, v___x_482_, v___x_483_);
v___x_485_ = lean_apply_1(v_toFun_481_, v___x_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff___redArg(lean_object* v_e_487_, lean_object* v_inst_488_){
_start:
{
lean_object* v___f_489_; lean_object* v___f_490_; 
v___f_489_ = ((lean_object*)(lp_mathlib_Equiv_sdiff___redArg___closed__0));
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sdiff___redArg___lam__1), 5, 3);
lean_closure_set(v___f_490_, 0, v_e_487_);
lean_closure_set(v___f_490_, 1, v___f_489_);
lean_closure_set(v___f_490_, 2, v_inst_488_);
return v___f_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sdiff(lean_object* v_00_u03b1_491_, lean_object* v_00_u03b2_492_, lean_object* v_e_493_, lean_object* v_inst_494_){
_start:
{
lean_object* v___f_495_; lean_object* v___f_496_; 
v___f_495_ = ((lean_object*)(lp_mathlib_Equiv_sdiff___redArg___closed__0));
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sdiff___redArg___lam__1), 5, 3);
lean_closure_set(v___f_496_, 0, v_e_493_);
lean_closure_set(v___f_496_, 1, v___f_495_);
lean_closure_set(v___f_496_, 2, v_inst_494_);
return v___f_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_himp___redArg(lean_object* v_e_497_, lean_object* v_inst_498_){
_start:
{
lean_object* v___f_499_; lean_object* v___f_500_; 
v___f_499_ = ((lean_object*)(lp_mathlib_Equiv_sdiff___redArg___closed__0));
v___f_500_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sdiff___redArg___lam__1), 5, 3);
lean_closure_set(v___f_500_, 0, v_e_497_);
lean_closure_set(v___f_500_, 1, v___f_499_);
lean_closure_set(v___f_500_, 2, v_inst_498_);
return v___f_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_himp(lean_object* v_00_u03b1_501_, lean_object* v_00_u03b2_502_, lean_object* v_e_503_, lean_object* v_inst_504_){
_start:
{
lean_object* v___f_505_; lean_object* v___f_506_; 
v___f_505_ = ((lean_object*)(lp_mathlib_Equiv_sdiff___redArg___closed__0));
v___f_506_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_sdiff___redArg___lam__1), 5, 3);
lean_closure_set(v___f_506_, 0, v_e_503_);
lean_closure_set(v___f_506_, 1, v___f_505_);
lean_closure_set(v___f_506_, 2, v_inst_504_);
return v___f_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_hnot___redArg(lean_object* v_e_507_, lean_object* v_inst_508_){
_start:
{
lean_object* v___f_509_; 
v___f_509_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_compl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_509_, 0, v_e_507_);
lean_closure_set(v___f_509_, 1, v_inst_508_);
return v___f_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_hnot(lean_object* v_00_u03b1_510_, lean_object* v_00_u03b2_511_, lean_object* v_e_512_, lean_object* v_inst_513_){
_start:
{
lean_object* v___f_514_; 
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_compl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_514_, 0, v_e_512_);
lean_closure_set(v___f_514_, 1, v_inst_513_);
return v___f_514_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_OrderDual(builtin);
}
#ifdef __cplusplus
}
#endif
