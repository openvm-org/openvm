// Lean compiler output
// Module: Mathlib.Algebra.Group.Equiv.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Logic.Equiv.Defs
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2a___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≃*_"};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(3, 252, 97, 197, 172, 144, 171, 102)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2a___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2243_x2a___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≃* "};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2243_x2a___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__7_value),((lean_object*)(((size_t)(26) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2a___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2243_x2a___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2a__ = (const lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "MulEquiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(234, 83, 148, 217, 3, 205, 94, 186)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__8_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__10_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2243_x2b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_≃+_"};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 13, 39, 194, 183, 10, 128, 102)}};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2243_x2b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≃+ "};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2243_x2a___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2243_x2b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__1_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2243_x2b___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2243_x2b__ = (const lean_object*)&lp_mathlib_term___u2243_x2b___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "AddEquiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 174, 201, 198, 168, 181, 98, 183)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__AddEquiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__AddEquiv__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulEquivOfMulEquivClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulEquivOfMulEquivClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddEquivOfAddEquivClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddEquivOfAddEquivClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_instEquivLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_instEquivLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_instEquivLike___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_instEquivLike___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_instEquivLike___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_instEquivLike___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_instEquivLike___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_instEquivLike___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_instEquivLike___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_instEquivLike___closed__0_value),((lean_object*)&lp_mathlib_MulEquiv_instEquivLike___closed__1_value)}};
static const lean_object* lp_mathlib_MulEquiv_instEquivLike___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_instEquivLike___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instEquivLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instCoeFunForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instCoeFunForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instCoeFunForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instCoeFunForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_refl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_refl___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_refl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_refl___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symmEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symmEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symmEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symmEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_cast___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_cast___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Monoid_End_equiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Monoid_End_equiv___closed__0 = (const lean_object*)&lp_mathlib_Monoid_End_equiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Monoid_End_equiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Monoid_End_equiv___closed__0_value),((lean_object*)&lp_mathlib_Monoid_End_equiv___closed__0_value)}};
static const lean_object* lp_mathlib_Monoid_End_equiv___closed__1 = (const lean_object*)&lp_mathlib_Monoid_End_equiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_equiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_equiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toFun_2_; 
v_toFun_2_ = lean_ctor_get(v_self_1_, 0);
lean_inc(v_toFun_2_);
return v_toFun_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___redArg___boxed(lean_object* v_self_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_AddEquiv_toAddHom___redArg(v_self_3_);
lean_dec_ref(v_self_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom(lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_self_9_){
_start:
{
lean_object* v_toFun_10_; 
v_toFun_10_ = lean_ctor_get(v_self_9_, 0);
lean_inc(v_toFun_10_);
return v_toFun_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddHom___boxed(lean_object* v_A_11_, lean_object* v_B_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_self_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_AddEquiv_toAddHom(v_A_11_, v_B_12_, v_inst_13_, v_inst_14_, v_self_15_);
lean_dec_ref(v_self_15_);
lean_dec(v_inst_14_);
lean_dec(v_inst_13_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___redArg(lean_object* v_self_17_){
_start:
{
lean_object* v_toFun_18_; 
v_toFun_18_ = lean_ctor_get(v_self_17_, 0);
lean_inc(v_toFun_18_);
return v_toFun_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___redArg___boxed(lean_object* v_self_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_MulEquiv_toMulHom___redArg(v_self_19_);
lean_dec_ref(v_self_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom(lean_object* v_M_21_, lean_object* v_N_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_self_25_){
_start:
{
lean_object* v_toFun_26_; 
v_toFun_26_ = lean_ctor_get(v_self_25_, 0);
lean_inc(v_toFun_26_);
return v_toFun_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMulHom___boxed(lean_object* v_M_27_, lean_object* v_N_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_self_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_MulEquiv_toMulHom(v_M_27_, v_N_28_, v_inst_29_, v_inst_30_, v_self_31_);
lean_dec_ref(v_self_31_);
lean_dec(v_inst_30_);
lean_dec(v_inst_29_);
return v_res_32_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__5));
v___x_68_ = l_String_toRawSubstring_x27(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1(lean_object* v_x_85_, lean_object* v_a_86_, lean_object* v_a_87_){
_start:
{
lean_object* v___x_88_; uint8_t v___x_89_; 
v___x_88_ = ((lean_object*)(lp_mathlib_term___u2243_x2a___00__closed__1));
lean_inc(v_x_85_);
v___x_89_ = l_Lean_Syntax_isOfKind(v_x_85_, v___x_88_);
if (v___x_89_ == 0)
{
lean_object* v___x_90_; lean_object* v___x_91_; 
lean_dec(v_x_85_);
v___x_90_ = lean_box(1);
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v_a_87_);
return v___x_91_;
}
else
{
lean_object* v_quotContext_92_; lean_object* v_currMacroScope_93_; lean_object* v_ref_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; uint8_t v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v_quotContext_92_ = lean_ctor_get(v_a_86_, 1);
v_currMacroScope_93_ = lean_ctor_get(v_a_86_, 2);
v_ref_94_ = lean_ctor_get(v_a_86_, 5);
v___x_95_ = lean_unsigned_to_nat(0u);
v___x_96_ = l_Lean_Syntax_getArg(v_x_85_, v___x_95_);
v___x_97_ = lean_unsigned_to_nat(2u);
v___x_98_ = l_Lean_Syntax_getArg(v_x_85_, v___x_97_);
lean_dec(v_x_85_);
v___x_99_ = 0;
v___x_100_ = l_Lean_SourceInfo_fromRef(v_ref_94_, v___x_99_);
v___x_101_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4));
v___x_102_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__6);
v___x_103_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__7));
lean_inc(v_currMacroScope_93_);
lean_inc(v_quotContext_92_);
v___x_104_ = l_Lean_addMacroScope(v_quotContext_92_, v___x_103_, v_currMacroScope_93_);
v___x_105_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__11));
lean_inc_n(v___x_100_, 2);
v___x_106_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_106_, 0, v___x_100_);
lean_ctor_set(v___x_106_, 1, v___x_102_);
lean_ctor_set(v___x_106_, 2, v___x_104_);
lean_ctor_set(v___x_106_, 3, v___x_105_);
v___x_107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__13));
v___x_108_ = l_Lean_Syntax_node2(v___x_100_, v___x_107_, v___x_96_, v___x_98_);
v___x_109_ = l_Lean_Syntax_node2(v___x_100_, v___x_101_, v___x_106_, v___x_108_);
v___x_110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_87_);
return v___x_110_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___boxed(lean_object* v_x_111_, lean_object* v_a_112_, lean_object* v_a_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1(v_x_111_, v_a_112_, v_a_113_);
lean_dec_ref(v_a_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1(lean_object* v_x_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_121_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4));
lean_inc(v_x_118_);
v___x_122_ = l_Lean_Syntax_isOfKind(v_x_118_, v___x_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v___x_124_; 
lean_dec(v_x_118_);
v___x_123_ = lean_box(0);
v___x_124_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
lean_ctor_set(v___x_124_, 1, v_a_120_);
return v___x_124_;
}
else
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v___x_125_ = lean_unsigned_to_nat(0u);
v___x_126_ = l_Lean_Syntax_getArg(v_x_118_, v___x_125_);
v___x_127_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__1));
lean_inc(v___x_126_);
v___x_128_ = l_Lean_Syntax_isOfKind(v___x_126_, v___x_127_);
if (v___x_128_ == 0)
{
lean_object* v___x_129_; lean_object* v___x_130_; 
lean_dec(v___x_126_);
lean_dec(v_x_118_);
v___x_129_ = lean_box(0);
v___x_130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v_a_120_);
return v___x_130_;
}
else
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; uint8_t v___x_134_; 
v___x_131_ = lean_unsigned_to_nat(1u);
v___x_132_ = l_Lean_Syntax_getArg(v_x_118_, v___x_131_);
lean_dec(v_x_118_);
v___x_133_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_132_);
v___x_134_ = l_Lean_Syntax_matchesNull(v___x_132_, v___x_133_);
if (v___x_134_ == 0)
{
lean_object* v___x_135_; lean_object* v___x_136_; 
lean_dec(v___x_132_);
lean_dec(v___x_126_);
v___x_135_ = lean_box(0);
v___x_136_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
lean_ctor_set(v___x_136_, 1, v_a_120_);
return v___x_136_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v_ref_139_; uint8_t v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_137_ = l_Lean_Syntax_getArg(v___x_132_, v___x_125_);
v___x_138_ = l_Lean_Syntax_getArg(v___x_132_, v___x_131_);
lean_dec(v___x_132_);
v_ref_139_ = l_Lean_replaceRef(v___x_126_, v_a_119_);
lean_dec(v___x_126_);
v___x_140_ = 0;
v___x_141_ = l_Lean_SourceInfo_fromRef(v_ref_139_, v___x_140_);
lean_dec(v_ref_139_);
v___x_142_ = ((lean_object*)(lp_mathlib_term___u2243_x2a___00__closed__1));
v___x_143_ = ((lean_object*)(lp_mathlib_term___u2243_x2a___00__closed__4));
lean_inc(v___x_141_);
v___x_144_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_141_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_145_ = l_Lean_Syntax_node3(v___x_141_, v___x_142_, v___x_137_, v___x_144_, v___x_138_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v_a_120_);
return v___x_146_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___boxed(lean_object* v_x_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1(v_x_147_, v_a_148_, v_a_149_);
lean_dec(v_a_148_);
return v_res_150_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_167_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__0));
v___x_168_ = l_String_toRawSubstring_x27(v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1(lean_object* v_x_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v___x_185_; uint8_t v___x_186_; 
v___x_185_ = ((lean_object*)(lp_mathlib_term___u2243_x2b___00__closed__1));
lean_inc(v_x_182_);
v___x_186_ = l_Lean_Syntax_isOfKind(v_x_182_, v___x_185_);
if (v___x_186_ == 0)
{
lean_object* v___x_187_; lean_object* v___x_188_; 
lean_dec(v_x_182_);
v___x_187_ = lean_box(1);
v___x_188_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v_a_184_);
return v___x_188_;
}
else
{
lean_object* v_quotContext_189_; lean_object* v_currMacroScope_190_; lean_object* v_ref_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; uint8_t v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v_quotContext_189_ = lean_ctor_get(v_a_183_, 1);
v_currMacroScope_190_ = lean_ctor_get(v_a_183_, 2);
v_ref_191_ = lean_ctor_get(v_a_183_, 5);
v___x_192_ = lean_unsigned_to_nat(0u);
v___x_193_ = l_Lean_Syntax_getArg(v_x_182_, v___x_192_);
v___x_194_ = lean_unsigned_to_nat(2u);
v___x_195_ = l_Lean_Syntax_getArg(v_x_182_, v___x_194_);
lean_dec(v_x_182_);
v___x_196_ = 0;
v___x_197_ = l_Lean_SourceInfo_fromRef(v_ref_191_, v___x_196_);
v___x_198_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4));
v___x_199_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__1);
v___x_200_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__2));
lean_inc(v_currMacroScope_190_);
lean_inc(v_quotContext_189_);
v___x_201_ = l_Lean_addMacroScope(v_quotContext_189_, v___x_200_, v_currMacroScope_190_);
v___x_202_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___closed__6));
lean_inc_n(v___x_197_, 2);
v___x_203_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_203_, 0, v___x_197_);
lean_ctor_set(v___x_203_, 1, v___x_199_);
lean_ctor_set(v___x_203_, 2, v___x_201_);
lean_ctor_set(v___x_203_, 3, v___x_202_);
v___x_204_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__13));
v___x_205_ = l_Lean_Syntax_node2(v___x_197_, v___x_204_, v___x_193_, v___x_195_);
v___x_206_ = l_Lean_Syntax_node2(v___x_197_, v___x_198_, v___x_203_, v___x_205_);
v___x_207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v_a_184_);
return v___x_207_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1___boxed(lean_object* v_x_208_, lean_object* v_a_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2b____1(v_x_208_, v_a_209_, v_a_210_);
lean_dec_ref(v_a_209_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__AddEquiv__1(lean_object* v_x_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v___x_215_; uint8_t v___x_216_; 
v___x_215_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______macroRules__term___u2243_x2a____1___closed__4));
lean_inc(v_x_212_);
v___x_216_ = l_Lean_Syntax_isOfKind(v_x_212_, v___x_215_);
if (v___x_216_ == 0)
{
lean_object* v___x_217_; lean_object* v___x_218_; 
lean_dec(v_x_212_);
v___x_217_ = lean_box(0);
v___x_218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v_a_214_);
return v___x_218_;
}
else
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; uint8_t v___x_222_; 
v___x_219_ = lean_unsigned_to_nat(0u);
v___x_220_ = l_Lean_Syntax_getArg(v_x_212_, v___x_219_);
v___x_221_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__MulEquiv__1___closed__1));
lean_inc(v___x_220_);
v___x_222_ = l_Lean_Syntax_isOfKind(v___x_220_, v___x_221_);
if (v___x_222_ == 0)
{
lean_object* v___x_223_; lean_object* v___x_224_; 
lean_dec(v___x_220_);
lean_dec(v_x_212_);
v___x_223_ = lean_box(0);
v___x_224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
lean_ctor_set(v___x_224_, 1, v_a_214_);
return v___x_224_;
}
else
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_225_ = lean_unsigned_to_nat(1u);
v___x_226_ = l_Lean_Syntax_getArg(v_x_212_, v___x_225_);
lean_dec(v_x_212_);
v___x_227_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_226_);
v___x_228_ = l_Lean_Syntax_matchesNull(v___x_226_, v___x_227_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; 
lean_dec(v___x_226_);
lean_dec(v___x_220_);
v___x_229_ = lean_box(0);
v___x_230_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v_a_214_);
return v___x_230_;
}
else
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v_ref_233_; uint8_t v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_231_ = l_Lean_Syntax_getArg(v___x_226_, v___x_219_);
v___x_232_ = l_Lean_Syntax_getArg(v___x_226_, v___x_225_);
lean_dec(v___x_226_);
v_ref_233_ = l_Lean_replaceRef(v___x_220_, v_a_213_);
lean_dec(v___x_220_);
v___x_234_ = 0;
v___x_235_ = l_Lean_SourceInfo_fromRef(v_ref_233_, v___x_234_);
lean_dec(v_ref_233_);
v___x_236_ = ((lean_object*)(lp_mathlib_term___u2243_x2b___00__closed__1));
v___x_237_ = ((lean_object*)(lp_mathlib_term___u2243_x2b___00__closed__2));
lean_inc(v___x_235_);
v___x_238_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_235_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = l_Lean_Syntax_node3(v___x_235_, v___x_236_, v___x_231_, v___x_238_, v___x_232_);
v___x_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
lean_ctor_set(v___x_240_, 1, v_a_214_);
return v___x_240_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__AddEquiv__1___boxed(lean_object* v_x_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib___aux__Mathlib__Algebra__Group__Equiv__Defs______unexpand__AddEquiv__1(v_x_241_, v_a_242_, v_a_243_);
lean_dec(v_a_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv___redArg(lean_object* v_inst_245_, lean_object* v_f_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_245_, v_f_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv(lean_object* v_F_248_, lean_object* v_00_u03b1_249_, lean_object* v_00_u03b2_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_f_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_251_, v_f_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquivClass_toMulEquiv___boxed(lean_object* v_F_257_, lean_object* v_00_u03b1_258_, lean_object* v_00_u03b2_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_f_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_MulEquivClass_toMulEquiv(v_F_257_, v_00_u03b1_258_, v_00_u03b2_259_, v_inst_260_, v_inst_261_, v_inst_262_, v_inst_263_, v_f_264_);
lean_dec(v_inst_262_);
lean_dec(v_inst_261_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv___redArg(lean_object* v_inst_266_, lean_object* v_f_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_266_, v_f_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv(lean_object* v_F_269_, lean_object* v_00_u03b1_270_, lean_object* v_00_u03b2_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_f_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_EquivLike_toEquiv___redArg(v_inst_272_, v_f_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquivClass_toAddEquiv___boxed(lean_object* v_F_278_, lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_f_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_AddEquivClass_toAddEquiv(v_F_278_, v_00_u03b1_279_, v_00_u03b2_280_, v_inst_281_, v_inst_282_, v_inst_283_, v_inst_284_, v_f_285_);
lean_dec(v_inst_283_);
lean_dec(v_inst_282_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulEquivOfMulEquivClass___redArg(lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lean_alloc_closure((void*)(lp_mathlib_MulEquivClass_toMulEquiv___boxed), 8, 7);
lean_closure_set(v___x_290_, 0, lean_box(0));
lean_closure_set(v___x_290_, 1, lean_box(0));
lean_closure_set(v___x_290_, 2, lean_box(0));
lean_closure_set(v___x_290_, 3, v_inst_287_);
lean_closure_set(v___x_290_, 4, v_inst_288_);
lean_closure_set(v___x_290_, 5, v_inst_289_);
lean_closure_set(v___x_290_, 6, lean_box(0));
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCMulEquivOfMulEquivClass(lean_object* v_F_291_, lean_object* v_00_u03b1_292_, lean_object* v_00_u03b2_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lean_alloc_closure((void*)(lp_mathlib_MulEquivClass_toMulEquiv___boxed), 8, 7);
lean_closure_set(v___x_298_, 0, lean_box(0));
lean_closure_set(v___x_298_, 1, lean_box(0));
lean_closure_set(v___x_298_, 2, lean_box(0));
lean_closure_set(v___x_298_, 3, v_inst_294_);
lean_closure_set(v___x_298_, 4, v_inst_295_);
lean_closure_set(v___x_298_, 5, v_inst_296_);
lean_closure_set(v___x_298_, 6, lean_box(0));
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddEquivOfAddEquivClass___redArg(lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_inst_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lean_alloc_closure((void*)(lp_mathlib_AddEquivClass_toAddEquiv___boxed), 8, 7);
lean_closure_set(v___x_302_, 0, lean_box(0));
lean_closure_set(v___x_302_, 1, lean_box(0));
lean_closure_set(v___x_302_, 2, lean_box(0));
lean_closure_set(v___x_302_, 3, v_inst_299_);
lean_closure_set(v___x_302_, 4, v_inst_300_);
lean_closure_set(v___x_302_, 5, v_inst_301_);
lean_closure_set(v___x_302_, 6, lean_box(0));
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCAddEquivOfAddEquivClass(lean_object* v_F_303_, lean_object* v_00_u03b1_304_, lean_object* v_00_u03b2_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lean_alloc_closure((void*)(lp_mathlib_AddEquivClass_toAddEquiv___boxed), 8, 7);
lean_closure_set(v___x_310_, 0, lean_box(0));
lean_closure_set(v___x_310_, 1, lean_box(0));
lean_closure_set(v___x_310_, 2, lean_box(0));
lean_closure_set(v___x_310_, 3, v_inst_306_);
lean_closure_set(v___x_310_, 4, v_inst_307_);
lean_closure_set(v___x_310_, 5, v_inst_308_);
lean_closure_set(v___x_310_, 6, lean_box(0));
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___lam__0(lean_object* v_f_311_, lean_object* v___y_312_){
_start:
{
lean_object* v_toFun_313_; lean_object* v___x_314_; 
v_toFun_313_ = lean_ctor_get(v_f_311_, 0);
lean_inc(v_toFun_313_);
lean_dec_ref(v_f_311_);
v___x_314_ = lean_apply_1(v_toFun_313_, v___y_312_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___lam__1(lean_object* v_f_315_, lean_object* v___y_316_){
_start:
{
lean_object* v_invFun_317_; lean_object* v___x_318_; 
v_invFun_317_ = lean_ctor_get(v_f_315_, 1);
lean_inc(v_invFun_317_);
lean_dec_ref(v_f_315_);
v___x_318_ = lean_apply_1(v_invFun_317_, v___y_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike(lean_object* v_M_324_, lean_object* v_N_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = ((lean_object*)(lp_mathlib_MulEquiv_instEquivLike___closed__2));
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instEquivLike___boxed(lean_object* v_M_329_, lean_object* v_N_330_, lean_object* v_inst_331_, lean_object* v_inst_332_){
_start:
{
lean_object* v_res_333_; 
v_res_333_ = lp_mathlib_MulEquiv_instEquivLike(v_M_329_, v_N_330_, v_inst_331_, v_inst_332_);
lean_dec(v_inst_332_);
lean_dec(v_inst_331_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instEquivLike(lean_object* v_M_334_, lean_object* v_N_335_, lean_object* v_inst_336_, lean_object* v_inst_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = ((lean_object*)(lp_mathlib_MulEquiv_instEquivLike___closed__2));
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instEquivLike___boxed(lean_object* v_M_339_, lean_object* v_N_340_, lean_object* v_inst_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_AddEquiv_instEquivLike(v_M_339_, v_N_340_, v_inst_341_, v_inst_342_);
lean_dec(v_inst_342_);
lean_dec(v_inst_341_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instCoeFunForall(lean_object* v_M_344_, lean_object* v_N_345_, lean_object* v_inst_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v___f_348_; 
v___f_348_ = ((lean_object*)(lp_mathlib_MulEquiv_instEquivLike___closed__0));
return v___f_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instCoeFunForall___boxed(lean_object* v_M_349_, lean_object* v_N_350_, lean_object* v_inst_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_MulEquiv_instCoeFunForall(v_M_349_, v_N_350_, v_inst_351_, v_inst_352_);
lean_dec(v_inst_352_);
lean_dec(v_inst_351_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instCoeFunForall(lean_object* v_M_354_, lean_object* v_N_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = ((lean_object*)(lp_mathlib_MulEquiv_instEquivLike___closed__0));
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instCoeFunForall___boxed(lean_object* v_M_359_, lean_object* v_N_360_, lean_object* v_inst_361_, lean_object* v_inst_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_AddEquiv_instCoeFunForall(v_M_359_, v_N_360_, v_inst_361_, v_inst_362_);
lean_dec(v_inst_362_);
lean_dec(v_inst_361_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___redArg(lean_object* v_f_364_){
_start:
{
lean_inc_ref(v_f_364_);
return v_f_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___redArg___boxed(lean_object* v_f_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_MulEquiv_mk_x27___redArg(v_f_365_);
lean_dec_ref(v_f_365_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27(lean_object* v_M_367_, lean_object* v_N_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_f_371_, lean_object* v_h_372_){
_start:
{
lean_inc_ref(v_f_371_);
return v_f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_mk_x27___boxed(lean_object* v_M_373_, lean_object* v_N_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_f_377_, lean_object* v_h_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_MulEquiv_mk_x27(v_M_373_, v_N_374_, v_inst_375_, v_inst_376_, v_f_377_, v_h_378_);
lean_dec_ref(v_f_377_);
lean_dec(v_inst_376_);
lean_dec(v_inst_375_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___redArg(lean_object* v_f_380_){
_start:
{
lean_inc_ref(v_f_380_);
return v_f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___redArg___boxed(lean_object* v_f_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_AddEquiv_mk_x27___redArg(v_f_381_);
lean_dec_ref(v_f_381_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27(lean_object* v_M_383_, lean_object* v_N_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_f_387_, lean_object* v_h_388_){
_start:
{
lean_inc_ref(v_f_387_);
return v_f_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mk_x27___boxed(lean_object* v_M_389_, lean_object* v_N_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_f_393_, lean_object* v_h_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_AddEquiv_mk_x27(v_M_389_, v_N_390_, v_inst_391_, v_inst_392_, v_f_393_, v_h_394_);
lean_dec_ref(v_f_393_);
lean_dec(v_inst_392_);
lean_dec(v_inst_391_);
return v_res_395_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_refl___closed__0(void){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl(lean_object* v_M_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___closed__0, &lp_mathlib_MulEquiv_refl___closed__0_once, _init_lp_mathlib_MulEquiv_refl___closed__0);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_refl___boxed(lean_object* v_M_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_mathlib_MulEquiv_refl(v_M_400_, v_inst_401_);
lean_dec(v_inst_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_refl(lean_object* v_M_403_, lean_object* v_inst_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___closed__0, &lp_mathlib_MulEquiv_refl___closed__0_once, _init_lp_mathlib_MulEquiv_refl___closed__0);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_refl___boxed(lean_object* v_M_406_, lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_AddEquiv_refl(v_M_406_, v_inst_407_);
lean_dec(v_inst_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instInhabited(lean_object* v_M_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___closed__0, &lp_mathlib_MulEquiv_refl___closed__0_once, _init_lp_mathlib_MulEquiv_refl___closed__0);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instInhabited___boxed(lean_object* v_M_412_, lean_object* v_inst_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_MulEquiv_instInhabited(v_M_412_, v_inst_413_);
lean_dec(v_inst_413_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instInhabited(lean_object* v_M_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lean_obj_once(&lp_mathlib_MulEquiv_refl___closed__0, &lp_mathlib_MulEquiv_refl___closed__0_once, _init_lp_mathlib_MulEquiv_refl___closed__0);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instInhabited___boxed(lean_object* v_M_418_, lean_object* v_inst_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_AddEquiv_instInhabited(v_M_418_, v_inst_419_);
lean_dec(v_inst_419_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___redArg(lean_object* v_h_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_Equiv_symm___redArg(v_h_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm(lean_object* v_M_423_, lean_object* v_N_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_h_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_Equiv_symm___redArg(v_h_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___boxed(lean_object* v_M_429_, lean_object* v_N_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_h_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_MulEquiv_symm(v_M_429_, v_N_430_, v_inst_431_, v_inst_432_, v_h_433_);
lean_dec(v_inst_432_);
lean_dec(v_inst_431_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___redArg(lean_object* v_h_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_Equiv_symm___redArg(v_h_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm(lean_object* v_M_437_, lean_object* v_N_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_h_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lp_mathlib_Equiv_symm___redArg(v_h_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___boxed(lean_object* v_M_443_, lean_object* v_N_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_h_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_mathlib_AddEquiv_symm(v_M_443_, v_N_444_, v_inst_445_, v_inst_446_, v_h_447_);
lean_dec(v_inst_446_);
lean_dec(v_inst_445_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply___redArg(lean_object* v_e_449_, lean_object* v_a_450_){
_start:
{
lean_object* v___x_451_; lean_object* v_toFun_452_; lean_object* v___x_453_; 
v___x_451_ = lp_mathlib_Equiv_symm___redArg(v_e_449_);
v_toFun_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_toFun_452_);
lean_dec_ref(v___x_451_);
v___x_453_ = lean_apply_1(v_toFun_452_, v_a_450_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply(lean_object* v_M_454_, lean_object* v_N_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_e_458_, lean_object* v_a_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_MulEquiv_Simps_symm__apply___redArg(v_e_458_, v_a_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Simps_symm__apply___boxed(lean_object* v_M_461_, lean_object* v_N_462_, lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_e_465_, lean_object* v_a_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_MulEquiv_Simps_symm__apply(v_M_461_, v_N_462_, v_inst_463_, v_inst_464_, v_e_465_, v_a_466_);
lean_dec(v_inst_464_);
lean_dec(v_inst_463_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply___redArg(lean_object* v_e_468_, lean_object* v_a_469_){
_start:
{
lean_object* v___x_470_; lean_object* v_toFun_471_; lean_object* v___x_472_; 
v___x_470_ = lp_mathlib_Equiv_symm___redArg(v_e_468_);
v_toFun_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_toFun_471_);
lean_dec_ref(v___x_470_);
v___x_472_ = lean_apply_1(v_toFun_471_, v_a_469_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply(lean_object* v_M_473_, lean_object* v_N_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_e_477_, lean_object* v_a_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_AddEquiv_Simps_symm__apply___redArg(v_e_477_, v_a_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_Simps_symm__apply___boxed(lean_object* v_M_480_, lean_object* v_N_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_e_484_, lean_object* v_a_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_mathlib_AddEquiv_Simps_symm__apply(v_M_480_, v_N_481_, v_inst_482_, v_inst_483_, v_e_484_, v_a_485_);
lean_dec(v_inst_483_);
lean_dec(v_inst_482_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans___redArg(lean_object* v_h1_487_, lean_object* v_h2_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_mathlib_Equiv_trans___redArg(v_h1_487_, v_h2_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans(lean_object* v_M_490_, lean_object* v_N_491_, lean_object* v_P_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_h1_496_, lean_object* v_h2_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_Equiv_trans___redArg(v_h1_496_, v_h2_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_trans___boxed(lean_object* v_M_499_, lean_object* v_N_500_, lean_object* v_P_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_h1_505_, lean_object* v_h2_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_MulEquiv_trans(v_M_499_, v_N_500_, v_P_501_, v_inst_502_, v_inst_503_, v_inst_504_, v_h1_505_, v_h2_506_);
lean_dec(v_inst_504_);
lean_dec(v_inst_503_);
lean_dec(v_inst_502_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans___redArg(lean_object* v_h1_508_, lean_object* v_h2_509_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lp_mathlib_Equiv_trans___redArg(v_h1_508_, v_h2_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans(lean_object* v_M_511_, lean_object* v_N_512_, lean_object* v_P_513_, lean_object* v_inst_514_, lean_object* v_inst_515_, lean_object* v_inst_516_, lean_object* v_h1_517_, lean_object* v_h2_518_){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lp_mathlib_Equiv_trans___redArg(v_h1_517_, v_h2_518_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_trans___boxed(lean_object* v_M_520_, lean_object* v_N_521_, lean_object* v_P_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_h1_526_, lean_object* v_h2_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_AddEquiv_trans(v_M_520_, v_N_521_, v_P_522_, v_inst_523_, v_inst_524_, v_inst_525_, v_h1_526_, v_h2_527_);
lean_dec(v_inst_525_);
lean_dec(v_inst_524_);
lean_dec(v_inst_523_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symmEquiv___redArg(lean_object* v_inst_529_, lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; 
lean_inc(v_inst_530_);
lean_inc(v_inst_529_);
v___x_531_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_531_, 0, lean_box(0));
lean_closure_set(v___x_531_, 1, lean_box(0));
lean_closure_set(v___x_531_, 2, v_inst_529_);
lean_closure_set(v___x_531_, 3, v_inst_530_);
v___x_532_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_532_, 0, lean_box(0));
lean_closure_set(v___x_532_, 1, lean_box(0));
lean_closure_set(v___x_532_, 2, v_inst_530_);
lean_closure_set(v___x_532_, 3, v_inst_529_);
v___x_533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_533_, 0, v___x_531_);
lean_ctor_set(v___x_533_, 1, v___x_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symmEquiv(lean_object* v_P_534_, lean_object* v_Q_535_, lean_object* v_inst_536_, lean_object* v_inst_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_mathlib_MulEquiv_symmEquiv___redArg(v_inst_536_, v_inst_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symmEquiv___redArg(lean_object* v_inst_539_, lean_object* v_inst_540_){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; 
lean_inc(v_inst_540_);
lean_inc(v_inst_539_);
v___x_541_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_541_, 0, lean_box(0));
lean_closure_set(v___x_541_, 1, lean_box(0));
lean_closure_set(v___x_541_, 2, v_inst_539_);
lean_closure_set(v___x_541_, 3, v_inst_540_);
v___x_542_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_symm___boxed), 5, 4);
lean_closure_set(v___x_542_, 0, lean_box(0));
lean_closure_set(v___x_542_, 1, lean_box(0));
lean_closure_set(v___x_542_, 2, v_inst_540_);
lean_closure_set(v___x_542_, 3, v_inst_539_);
v___x_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symmEquiv(lean_object* v_P_544_, lean_object* v_Q_545_, lean_object* v_inst_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_mathlib_AddEquiv_symmEquiv___redArg(v_inst_546_, v_inst_547_);
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_cast___closed__0(void){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_cast(lean_object* v_00_u03b9_550_, lean_object* v_M_551_, lean_object* v_inst_552_, lean_object* v_i_553_, lean_object* v_j_554_, lean_object* v_h_555_){
_start:
{
lean_object* v___x_556_; 
v___x_556_ = lean_obj_once(&lp_mathlib_MulEquiv_cast___closed__0, &lp_mathlib_MulEquiv_cast___closed__0_once, _init_lp_mathlib_MulEquiv_cast___closed__0);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_cast___boxed(lean_object* v_00_u03b9_557_, lean_object* v_M_558_, lean_object* v_inst_559_, lean_object* v_i_560_, lean_object* v_j_561_, lean_object* v_h_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_MulEquiv_cast(v_00_u03b9_557_, v_M_558_, v_inst_559_, v_i_560_, v_j_561_, v_h_562_);
lean_dec(v_j_561_);
lean_dec(v_i_560_);
lean_dec(v_inst_559_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_cast(lean_object* v_00_u03b9_564_, lean_object* v_M_565_, lean_object* v_inst_566_, lean_object* v_i_567_, lean_object* v_j_568_, lean_object* v_h_569_){
_start:
{
lean_object* v___x_570_; 
v___x_570_ = lean_obj_once(&lp_mathlib_MulEquiv_cast___closed__0, &lp_mathlib_MulEquiv_cast___closed__0_once, _init_lp_mathlib_MulEquiv_cast___closed__0);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_cast___boxed(lean_object* v_00_u03b9_571_, lean_object* v_M_572_, lean_object* v_inst_573_, lean_object* v_i_574_, lean_object* v_j_575_, lean_object* v_h_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_AddEquiv_cast(v_00_u03b9_571_, v_M_572_, v_inst_573_, v_i_574_, v_j_575_, v_h_576_);
lean_dec(v_j_575_);
lean_dec(v_i_574_);
lean_dec(v_inst_573_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___redArg(lean_object* v_h_578_){
_start:
{
lean_object* v_toFun_579_; 
v_toFun_579_ = lean_ctor_get(v_h_578_, 0);
lean_inc(v_toFun_579_);
return v_toFun_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___redArg___boxed(lean_object* v_h_580_){
_start:
{
lean_object* v_res_581_; 
v_res_581_ = lp_mathlib_MulEquiv_toMonoidHom___redArg(v_h_580_);
lean_dec_ref(v_h_580_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom(lean_object* v_M_582_, lean_object* v_N_583_, lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_h_586_){
_start:
{
lean_object* v_toFun_587_; 
v_toFun_587_ = lean_ctor_get(v_h_586_, 0);
lean_inc(v_toFun_587_);
return v_toFun_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___boxed(lean_object* v_M_588_, lean_object* v_N_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_h_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_mathlib_MulEquiv_toMonoidHom(v_M_588_, v_N_589_, v_inst_590_, v_inst_591_, v_h_592_);
lean_dec_ref(v_h_592_);
lean_dec_ref(v_inst_591_);
lean_dec_ref(v_inst_590_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___redArg(lean_object* v_h_594_){
_start:
{
lean_object* v_toFun_595_; 
v_toFun_595_ = lean_ctor_get(v_h_594_, 0);
lean_inc(v_toFun_595_);
return v_toFun_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___redArg___boxed(lean_object* v_h_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_AddEquiv_toAddMonoidHom___redArg(v_h_596_);
lean_dec_ref(v_h_596_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom(lean_object* v_M_598_, lean_object* v_N_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_h_602_){
_start:
{
lean_object* v_toFun_603_; 
v_toFun_603_ = lean_ctor_get(v_h_602_, 0);
lean_inc(v_toFun_603_);
return v_toFun_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___boxed(lean_object* v_M_604_, lean_object* v_N_605_, lean_object* v_inst_606_, lean_object* v_inst_607_, lean_object* v_h_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_AddEquiv_toAddMonoidHom(v_M_604_, v_N_605_, v_inst_606_, v_inst_607_, v_h_608_);
lean_dec_ref(v_h_608_);
lean_dec_ref(v_inst_607_);
lean_dec_ref(v_inst_606_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg___lam__0(lean_object* v_f_610_, lean_object* v___y_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lean_apply_1(v_f_610_, v___y_611_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg___lam__1(lean_object* v_g_613_, lean_object* v___y_614_){
_start:
{
lean_object* v___x_615_; 
v___x_615_ = lean_apply_1(v_g_613_, v___y_614_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___redArg(lean_object* v_f_616_, lean_object* v_g_617_){
_start:
{
lean_object* v___f_618_; lean_object* v___f_619_; lean_object* v___x_620_; 
v___f_618_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_618_, 0, v_f_616_);
v___f_619_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_619_, 0, v_g_617_);
v___x_620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_620_, 0, v___f_618_);
lean_ctor_set(v___x_620_, 1, v___f_619_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv(lean_object* v_M_621_, lean_object* v_N_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_f_625_, lean_object* v_g_626_, lean_object* v_h_u2081_627_, lean_object* v_h_u2082_628_){
_start:
{
lean_object* v___x_629_; 
v___x_629_ = lp_mathlib_MulHom_toMulEquiv___redArg(v_f_625_, v_g_626_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toMulEquiv___boxed(lean_object* v_M_630_, lean_object* v_N_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_f_634_, lean_object* v_g_635_, lean_object* v_h_u2081_636_, lean_object* v_h_u2082_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_mathlib_MulHom_toMulEquiv(v_M_630_, v_N_631_, v_inst_632_, v_inst_633_, v_f_634_, v_g_635_, v_h_u2081_636_, v_h_u2082_637_);
lean_dec(v_inst_633_);
lean_dec(v_inst_632_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv___redArg(lean_object* v_f_639_, lean_object* v_g_640_){
_start:
{
lean_object* v___f_641_; lean_object* v___f_642_; lean_object* v___x_643_; 
v___f_641_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_641_, 0, v_f_639_);
v___f_642_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_642_, 0, v_g_640_);
v___x_643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_643_, 0, v___f_641_);
lean_ctor_set(v___x_643_, 1, v___f_642_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv(lean_object* v_M_644_, lean_object* v_N_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_f_648_, lean_object* v_g_649_, lean_object* v_h_u2081_650_, lean_object* v_h_u2082_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_mathlib_AddHom_toAddEquiv___redArg(v_f_648_, v_g_649_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toAddEquiv___boxed(lean_object* v_M_653_, lean_object* v_N_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_f_657_, lean_object* v_g_658_, lean_object* v_h_u2081_659_, lean_object* v_h_u2082_660_){
_start:
{
lean_object* v_res_661_; 
v_res_661_ = lp_mathlib_AddHom_toAddEquiv(v_M_653_, v_N_654_, v_inst_655_, v_inst_656_, v_f_657_, v_g_658_, v_h_u2081_659_, v_h_u2082_660_);
lean_dec(v_inst_656_);
lean_dec(v_inst_655_);
return v_res_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv___redArg(lean_object* v_f_662_, lean_object* v_g_663_){
_start:
{
lean_object* v___f_664_; lean_object* v___f_665_; lean_object* v___x_666_; 
v___f_664_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_664_, 0, v_f_662_);
v___f_665_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_665_, 0, v_g_663_);
v___x_666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_666_, 0, v___f_664_);
lean_ctor_set(v___x_666_, 1, v___f_665_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv(lean_object* v_M_667_, lean_object* v_N_668_, lean_object* v_inst_669_, lean_object* v_inst_670_, lean_object* v_f_671_, lean_object* v_g_672_, lean_object* v_h_u2081_673_, lean_object* v_h_u2082_674_){
_start:
{
lean_object* v___x_675_; 
v___x_675_ = lp_mathlib_MonoidHom_toMulEquiv___redArg(v_f_671_, v_g_672_);
return v___x_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toMulEquiv___boxed(lean_object* v_M_676_, lean_object* v_N_677_, lean_object* v_inst_678_, lean_object* v_inst_679_, lean_object* v_f_680_, lean_object* v_g_681_, lean_object* v_h_u2081_682_, lean_object* v_h_u2082_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_MonoidHom_toMulEquiv(v_M_676_, v_N_677_, v_inst_678_, v_inst_679_, v_f_680_, v_g_681_, v_h_u2081_682_, v_h_u2082_683_);
lean_dec_ref(v_inst_679_);
lean_dec_ref(v_inst_678_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv___redArg(lean_object* v_f_685_, lean_object* v_g_686_){
_start:
{
lean_object* v___f_687_; lean_object* v___f_688_; lean_object* v___x_689_; 
v___f_687_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_687_, 0, v_f_685_);
v___f_688_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toMulEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_688_, 0, v_g_686_);
v___x_689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_689_, 0, v___f_687_);
lean_ctor_set(v___x_689_, 1, v___f_688_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv(lean_object* v_M_690_, lean_object* v_N_691_, lean_object* v_inst_692_, lean_object* v_inst_693_, lean_object* v_f_694_, lean_object* v_g_695_, lean_object* v_h_u2081_696_, lean_object* v_h_u2082_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_AddMonoidHom_toAddEquiv___redArg(v_f_694_, v_g_695_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toAddEquiv___boxed(lean_object* v_M_699_, lean_object* v_N_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_f_703_, lean_object* v_g_704_, lean_object* v_h_u2081_705_, lean_object* v_h_u2082_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_AddMonoidHom_toAddEquiv(v_M_699_, v_N_700_, v_inst_701_, v_inst_702_, v_f_703_, v_g_704_, v_h_u2081_705_, v_h_u2082_706_);
lean_dec_ref(v_inst_702_);
lean_dec_ref(v_inst_701_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_equiv(lean_object* v_M_711_, lean_object* v_inst_712_){
_start:
{
lean_object* v___x_713_; 
v___x_713_ = ((lean_object*)(lp_mathlib_Monoid_End_equiv___closed__1));
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_End_equiv___boxed(lean_object* v_M_714_, lean_object* v_inst_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_Monoid_End_equiv(v_M_714_, v_inst_715_);
lean_dec_ref(v_inst_715_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_equiv(lean_object* v_M_717_, lean_object* v_inst_718_){
_start:
{
lean_object* v___x_719_; 
v___x_719_ = ((lean_object*)(lp_mathlib_Monoid_End_equiv___closed__1));
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_equiv___boxed(lean_object* v_M_720_, lean_object* v_inst_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_AddMonoid_End_equiv(v_M_720_, v_inst_721_);
lean_dec_ref(v_inst_721_);
return v_res_722_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
