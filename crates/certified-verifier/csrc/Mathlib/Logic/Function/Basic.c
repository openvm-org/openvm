// Lean compiler output
// Module: Mathlib.Logic.Function.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.ExistsUnique public import Mathlib.Basic.Logic.Basic public import Mathlib.Basic.Nonempty public import Mathlib.Basic.Nontrivial.Defs public import Mathlib.Data.Set.Defs public import Mathlib.Logic.Function.Defs public import Batteries.Tactic.Init public import Mathlib.Order.Defs.Unbundled
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
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
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
LEAN_EXPORT lean_object* lp_mathlib_Function_eval___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_decidableEqPFun___redArg(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_decidableEqPFun___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_decidableEqPFun(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_decidableEqPFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_update___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function_term_u21bf___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__0 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__0_value;
static const lean_string_object lp_mathlib_Function_term_u21bf___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term↿_"};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__1 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(91, 94, 209, 254, 195, 16, 168, 100)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__2 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__2_value;
static const lean_string_object lp_mathlib_Function_term_u21bf___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__3 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__4 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__4_value;
static const lean_string_object lp_mathlib_Function_term_u21bf___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↿"};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__5 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__5_value)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__6 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__6_value;
static const lean_string_object lp_mathlib_Function_term_u21bf___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__7 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__8 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__8_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__9 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__4_value),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__6_value),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__9_value)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__10 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Function_term_u21bf___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__10_value)}};
static const lean_object* lp_mathlib_Function_term_u21bf___00__closed__11 = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Function_term_u21bf__ = (const lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__11_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__0_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__1_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__2 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__2_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__3 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "HasUncurry.uncurry"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__5 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "HasUncurry"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__7 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__7_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "uncurry"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__8 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(48, 207, 180, 212, 82, 23, 245, 168)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(25, 241, 47, 44, 74, 156, 14, 121)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term_u21bf___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(114, 26, 20, 28, 208, 159, 153, 136)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(3, 56, 213, 201, 95, 118, 112, 151)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__11 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__12 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__12_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__13 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__14 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Function_hasUncurryBase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Function_hasUncurryBase___closed__0 = (const lean_object*)&lp_mathlib_Function_hasUncurryBase___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryBase(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_piecewise___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_piecewise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableUncurryOfFstSnd__mathlib(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableCurryOfMk__mathlib___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableCurryOfMk__mathlib___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableCurryOfMk__mathlib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableCurryOfMk__mathlib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_eval___redArg(lean_object* v_x_1_, lean_object* v_f_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_2_, v_x_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_eval(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_x_6_, lean_object* v_f_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_apply_1(v_f_7_, v_x_6_);
return v___x_8_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_decidableEq___redArg(lean_object* v_f_9_, lean_object* v_inst_10_, lean_object* v_x_11_, lean_object* v_x_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
lean_inc(v_f_9_);
v___x_13_ = lean_apply_1(v_f_9_, v_x_11_);
v___x_14_ = lean_apply_1(v_f_9_, v_x_12_);
v___x_15_ = lean_apply_2(v_inst_10_, v___x_13_, v___x_14_);
v___x_16_ = lean_unbox(v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_decidableEq___redArg___boxed(lean_object* v_f_17_, lean_object* v_inst_18_, lean_object* v_x_19_, lean_object* v_x_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_mathlib_Function_Injective_decidableEq___redArg(v_f_17_, v_inst_18_, v_x_19_, v_x_20_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_Injective_decidableEq(lean_object* v_00_u03b1_23_, lean_object* v_00_u03b2_24_, lean_object* v_f_25_, lean_object* v_inst_26_, lean_object* v_I_27_, lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
uint8_t v___x_30_; 
v___x_30_ = lp_mathlib_Function_Injective_decidableEq___redArg(v_f_25_, v_inst_26_, v_x_28_, v_x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_decidableEq___boxed(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_f_33_, lean_object* v_inst_34_, lean_object* v_I_35_, lean_object* v_x_36_, lean_object* v_x_37_){
_start:
{
uint8_t v_res_38_; lean_object* v_r_39_; 
v_res_38_ = lp_mathlib_Function_Injective_decidableEq(v_00_u03b1_31_, v_00_u03b2_32_, v_f_33_, v_inst_34_, v_I_35_, v_x_36_, v_x_37_);
v_r_39_ = lean_box(v_res_38_);
return v_r_39_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_decidableEqPFun___redArg(uint8_t v_inst_40_, lean_object* v_inst_41_, lean_object* v_x_42_, lean_object* v_x_43_){
_start:
{
if (v_inst_40_ == 0)
{
uint8_t v___x_44_; 
lean_dec(v_x_43_);
lean_dec(v_x_42_);
lean_dec_ref(v_inst_41_);
v___x_44_ = 1;
return v___x_44_;
}
else
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; uint8_t v___x_48_; 
v___x_45_ = lean_apply_1(v_x_42_, lean_box(0));
v___x_46_ = lean_apply_1(v_x_43_, lean_box(0));
v___x_47_ = lean_apply_3(v_inst_41_, lean_box(0), v___x_45_, v___x_46_);
v___x_48_ = lean_unbox(v___x_47_);
return v___x_48_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_decidableEqPFun___redArg___boxed(lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
uint8_t v_inst_36__boxed_53_; uint8_t v_res_54_; lean_object* v_r_55_; 
v_inst_36__boxed_53_ = lean_unbox(v_inst_49_);
v_res_54_ = lp_mathlib_Function_decidableEqPFun___redArg(v_inst_36__boxed_53_, v_inst_50_, v_x_51_, v_x_52_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_decidableEqPFun(lean_object* v_p_56_, uint8_t v_inst_57_, lean_object* v_00_u03b1_58_, lean_object* v_inst_59_, lean_object* v_x_60_, lean_object* v_x_61_){
_start:
{
uint8_t v___x_62_; 
v___x_62_ = lp_mathlib_Function_decidableEqPFun___redArg(v_inst_57_, v_inst_59_, v_x_60_, v_x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_decidableEqPFun___boxed(lean_object* v_p_63_, lean_object* v_inst_64_, lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
uint8_t v_inst_58__boxed_69_; uint8_t v_res_70_; lean_object* v_r_71_; 
v_inst_58__boxed_69_ = lean_unbox(v_inst_64_);
v_res_70_ = lp_mathlib_Function_decidableEqPFun(v_p_63_, v_inst_58__boxed_69_, v_00_u03b1_65_, v_inst_66_, v_x_67_, v_x_68_);
v_r_71_ = lean_box(v_res_70_);
return v_r_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___redArg(lean_object* v_inst_72_, lean_object* v_f_73_, lean_object* v_a_x27_74_, lean_object* v_v_75_, lean_object* v_a_76_){
_start:
{
lean_object* v___x_77_; uint8_t v___x_78_; 
lean_inc(v_a_76_);
v___x_77_ = lean_apply_2(v_inst_72_, v_a_76_, v_a_x27_74_);
v___x_78_ = lean_unbox(v___x_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; 
v___x_79_ = lean_apply_1(v_f_73_, v_a_76_);
return v___x_79_;
}
else
{
lean_dec(v_a_76_);
lean_dec(v_f_73_);
lean_inc(v_v_75_);
return v_v_75_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___redArg___boxed(lean_object* v_inst_80_, lean_object* v_f_81_, lean_object* v_a_x27_82_, lean_object* v_v_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Function_update___redArg(v_inst_80_, v_f_81_, v_a_x27_82_, v_v_83_, v_a_84_);
lean_dec(v_v_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update(lean_object* v_00_u03b1_86_, lean_object* v_00_u03b2_87_, lean_object* v_inst_88_, lean_object* v_f_89_, lean_object* v_a_x27_90_, lean_object* v_v_91_, lean_object* v_a_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Function_update___redArg(v_inst_88_, v_f_89_, v_a_x27_90_, v_v_91_, v_a_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_update___boxed(lean_object* v_00_u03b1_94_, lean_object* v_00_u03b2_95_, lean_object* v_inst_96_, lean_object* v_f_97_, lean_object* v_a_x27_98_, lean_object* v_v_99_, lean_object* v_a_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Function_update(v_00_u03b1_94_, v_00_u03b2_95_, v_inst_96_, v_f_97_, v_a_x27_98_, v_v_99_, v_a_100_);
lean_dec(v_v_99_);
return v_res_101_;
}
}
static lean_object* _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__5));
v___x_139_ = l_String_toRawSubstring_x27(v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1(lean_object* v_x_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_161_ = ((lean_object*)(lp_mathlib_Function_term_u21bf___00__closed__2));
lean_inc(v_x_158_);
v___x_162_ = l_Lean_Syntax_isOfKind(v_x_158_, v___x_161_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; lean_object* v___x_164_; 
lean_dec(v_x_158_);
v___x_163_ = lean_box(1);
v___x_164_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v_a_160_);
return v___x_164_;
}
else
{
lean_object* v_quotContext_165_; lean_object* v_currMacroScope_166_; lean_object* v_ref_167_; lean_object* v___x_168_; lean_object* v___x_169_; uint8_t v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v_quotContext_165_ = lean_ctor_get(v_a_159_, 1);
v_currMacroScope_166_ = lean_ctor_get(v_a_159_, 2);
v_ref_167_ = lean_ctor_get(v_a_159_, 5);
v___x_168_ = lean_unsigned_to_nat(1u);
v___x_169_ = l_Lean_Syntax_getArg(v_x_158_, v___x_168_);
lean_dec(v_x_158_);
v___x_170_ = 0;
v___x_171_ = l_Lean_SourceInfo_fromRef(v_ref_167_, v___x_170_);
v___x_172_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4));
v___x_173_ = lean_obj_once(&lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6, &lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6_once, _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__6);
v___x_174_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__9));
lean_inc(v_currMacroScope_166_);
lean_inc(v_quotContext_165_);
v___x_175_ = l_Lean_addMacroScope(v_quotContext_165_, v___x_174_, v_currMacroScope_166_);
v___x_176_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__12));
lean_inc_n(v___x_171_, 2);
v___x_177_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_177_, 0, v___x_171_);
lean_ctor_set(v___x_177_, 1, v___x_173_);
lean_ctor_set(v___x_177_, 2, v___x_175_);
lean_ctor_set(v___x_177_, 3, v___x_176_);
v___x_178_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__14));
v___x_179_ = l_Lean_Syntax_node1(v___x_171_, v___x_178_, v___x_169_);
v___x_180_ = l_Lean_Syntax_node2(v___x_171_, v___x_172_, v___x_177_, v___x_179_);
v___x_181_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
lean_ctor_set(v___x_181_, 1, v_a_160_);
return v___x_181_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___boxed(lean_object* v_x_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1(v_x_182_, v_a_183_, v_a_184_);
lean_dec_ref(v_a_183_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1(lean_object* v_x_189_, lean_object* v_a_190_, lean_object* v_a_191_){
_start:
{
lean_object* v___x_192_; uint8_t v___x_193_; 
v___x_192_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______macroRules__Function__term_u21bf____1___closed__4));
lean_inc(v_x_189_);
v___x_193_ = l_Lean_Syntax_isOfKind(v_x_189_, v___x_192_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec(v_x_189_);
v___x_194_ = lean_box(0);
v___x_195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_a_191_);
return v___x_195_;
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_196_ = lean_unsigned_to_nat(0u);
v___x_197_ = l_Lean_Syntax_getArg(v_x_189_, v___x_196_);
v___x_198_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___closed__1));
lean_inc(v___x_197_);
v___x_199_ = l_Lean_Syntax_isOfKind(v___x_197_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec(v___x_197_);
lean_dec(v_x_189_);
v___x_200_ = lean_box(0);
v___x_201_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v_a_191_);
return v___x_201_;
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; uint8_t v___x_204_; 
v___x_202_ = lean_unsigned_to_nat(1u);
v___x_203_ = l_Lean_Syntax_getArg(v_x_189_, v___x_202_);
lean_dec(v_x_189_);
lean_inc(v___x_203_);
v___x_204_ = l_Lean_Syntax_matchesNull(v___x_203_, v___x_202_);
if (v___x_204_ == 0)
{
lean_object* v___x_205_; lean_object* v___x_206_; 
lean_dec(v___x_203_);
lean_dec(v___x_197_);
v___x_205_ = lean_box(0);
v___x_206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
lean_ctor_set(v___x_206_, 1, v_a_191_);
return v___x_206_;
}
else
{
lean_object* v___x_207_; lean_object* v_ref_208_; uint8_t v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_207_ = l_Lean_Syntax_getArg(v___x_203_, v___x_196_);
lean_dec(v___x_203_);
v_ref_208_ = l_Lean_replaceRef(v___x_197_, v_a_190_);
lean_dec(v___x_197_);
v___x_209_ = 0;
v___x_210_ = l_Lean_SourceInfo_fromRef(v_ref_208_, v___x_209_);
lean_dec(v_ref_208_);
v___x_211_ = ((lean_object*)(lp_mathlib_Function_term_u21bf___00__closed__2));
v___x_212_ = ((lean_object*)(lp_mathlib_Function_term_u21bf___00__closed__5));
lean_inc(v___x_210_);
v___x_213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_210_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = l_Lean_Syntax_node2(v___x_210_, v___x_211_, v___x_213_, v___x_207_);
v___x_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_191_);
return v___x_215_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1___boxed(lean_object* v_x_216_, lean_object* v_a_217_, lean_object* v_a_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Basic______unexpand__Function__HasUncurry__uncurry__1(v_x_216_, v_a_217_, v_a_218_);
lean_dec(v_a_217_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryBase(lean_object* v_00_u03b1_221_, lean_object* v_00_u03b2_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = ((lean_object*)(lp_mathlib_Function_hasUncurryBase___closed__0));
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction___redArg___lam__0(lean_object* v_inst_224_, lean_object* v_f_225_, lean_object* v_p_226_){
_start:
{
lean_object* v_fst_227_; lean_object* v_snd_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_fst_227_ = lean_ctor_get(v_p_226_, 0);
lean_inc(v_fst_227_);
v_snd_228_ = lean_ctor_get(v_p_226_, 1);
lean_inc(v_snd_228_);
lean_dec_ref(v_p_226_);
v___x_229_ = lean_apply_1(v_f_225_, v_fst_227_);
v___x_230_ = lean_apply_2(v_inst_224_, v___x_229_, v_snd_228_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction___redArg(lean_object* v_inst_231_){
_start:
{
lean_object* v___f_232_; 
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_Function_hasUncurryInduction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_232_, 0, v_inst_231_);
return v___f_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_hasUncurryInduction(lean_object* v_00_u03b1_233_, lean_object* v_00_u03b2_234_, lean_object* v_00_u03b3_235_, lean_object* v_00_u03b4_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v___f_238_; 
v___f_238_ = lean_alloc_closure((void*)(lp_mathlib_Function_hasUncurryInduction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_238_, 0, v_inst_237_);
return v___f_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_piecewise___redArg(lean_object* v_f_239_, lean_object* v_g_240_, lean_object* v_inst_241_, lean_object* v_i_242_){
_start:
{
lean_object* v___x_243_; uint8_t v___x_244_; 
lean_inc(v_i_242_);
v___x_243_ = lean_apply_1(v_inst_241_, v_i_242_);
v___x_244_ = lean_unbox(v___x_243_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; 
lean_dec(v_f_239_);
v___x_245_ = lean_apply_1(v_g_240_, v_i_242_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; 
lean_dec(v_g_240_);
v___x_246_ = lean_apply_1(v_f_239_, v_i_242_);
return v___x_246_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_piecewise(lean_object* v_00_u03b1_247_, lean_object* v_00_u03b2_248_, lean_object* v_s_249_, lean_object* v_f_250_, lean_object* v_g_251_, lean_object* v_inst_252_, lean_object* v_i_253_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_Set_piecewise___redArg(v_f_250_, v_g_251_, v_inst_252_, v_i_253_);
return v___x_254_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___redArg(uint8_t v_inst_255_){
_start:
{
return v_inst_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___redArg___boxed(lean_object* v_inst_256_){
_start:
{
uint8_t v_inst_5__boxed_257_; uint8_t v_res_258_; lean_object* v_r_259_; 
v_inst_5__boxed_257_ = lean_unbox(v_inst_256_);
v_res_258_ = lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___redArg(v_inst_5__boxed_257_);
v_r_259_ = lean_box(v_res_258_);
return v_r_259_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableUncurryOfFstSnd__mathlib(lean_object* v_00_u03b1_260_, lean_object* v_00_u03b2_261_, lean_object* v_r_262_, lean_object* v_x_263_, uint8_t v_inst_264_){
_start:
{
return v_inst_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableUncurryOfFstSnd__mathlib___boxed(lean_object* v_00_u03b1_265_, lean_object* v_00_u03b2_266_, lean_object* v_r_267_, lean_object* v_x_268_, lean_object* v_inst_269_){
_start:
{
uint8_t v_inst_8__boxed_270_; uint8_t v_res_271_; lean_object* v_r_272_; 
v_inst_8__boxed_270_ = lean_unbox(v_inst_269_);
v_res_271_ = lp_mathlib_instDecidableUncurryOfFstSnd__mathlib(v_00_u03b1_265_, v_00_u03b2_266_, v_r_267_, v_x_268_, v_inst_8__boxed_270_);
lean_dec_ref(v_x_268_);
v_r_272_ = lean_box(v_res_271_);
return v_r_272_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableCurryOfMk__mathlib___redArg(uint8_t v_inst_273_){
_start:
{
return v_inst_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableCurryOfMk__mathlib___redArg___boxed(lean_object* v_inst_274_){
_start:
{
uint8_t v_inst_5__boxed_275_; uint8_t v_res_276_; lean_object* v_r_277_; 
v_inst_5__boxed_275_ = lean_unbox(v_inst_274_);
v_res_276_ = lp_mathlib_instDecidableCurryOfMk__mathlib___redArg(v_inst_5__boxed_275_);
v_r_277_ = lean_box(v_res_276_);
return v_r_277_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableCurryOfMk__mathlib(lean_object* v_00_u03b1_278_, lean_object* v_00_u03b2_279_, lean_object* v_r_280_, lean_object* v_a_281_, lean_object* v_b_282_, uint8_t v_inst_283_){
_start:
{
return v_inst_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableCurryOfMk__mathlib___boxed(lean_object* v_00_u03b1_284_, lean_object* v_00_u03b2_285_, lean_object* v_r_286_, lean_object* v_a_287_, lean_object* v_b_288_, lean_object* v_inst_289_){
_start:
{
uint8_t v_inst_8__boxed_290_; uint8_t v_res_291_; lean_object* v_r_292_; 
v_inst_8__boxed_290_ = lean_unbox(v_inst_289_);
v_res_291_ = lp_mathlib_instDecidableCurryOfMk__mathlib(v_00_u03b1_284_, v_00_u03b2_285_, v_r_286_, v_a_287_, v_b_288_, v_inst_8__boxed_290_);
lean_dec(v_b_288_);
lean_dec(v_a_287_);
v_r_292_ = lean_box(v_res_291_);
return v_r_292_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_ExistsUnique(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nonempty(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_ExistsUnique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nonempty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_ExistsUnique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nonempty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_Unbundled(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_ExistsUnique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nonempty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_Unbundled(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
