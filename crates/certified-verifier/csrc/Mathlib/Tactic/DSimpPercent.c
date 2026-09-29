// Lean compiler output
// Module: Mathlib.Tactic.DSimpPercent
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.Elab.Tactic.Simp
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpLemma;
extern lean_object* l_Lean_Parser_Tactic_simpErase;
extern lean_object* l_Lean_Parser_Tactic_discharger;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_dsimp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpContext(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "dsimpPercent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(5, 30, 218, 196, 240, 58, 207, 43)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "dsimp%"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__13_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "withoutPosition"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__19_value),LEAN_SCALAR_PTR_LITERAL(69, 6, 27, 142, 141, 165, 41, 16)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__21_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__26_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__35_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__37_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__39_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__40_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__41_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercent;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "`dsimp%` made no progress"};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_getSimpTheorems___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_14_ = l_Lean_Parser_Tactic_optConfig;
v___x_15_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__7));
v___x_16_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_17_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
lean_ctor_set(v___x_17_, 1, v___x_15_);
lean_ctor_set(v___x_17_, 2, v___x_14_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_21_ = l_Lean_Parser_Tactic_discharger;
v___x_22_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10));
v___x_23_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_23_, 0, v___x_22_);
lean_ctor_set(v___x_23_, 1, v___x_21_);
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12(void){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_24_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__11);
v___x_25_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__8);
v___x_26_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_27_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v___x_25_);
lean_ctor_set(v___x_27_, 2, v___x_24_);
return v___x_27_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_35_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__15));
v___x_36_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__12);
v___x_37_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_38_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
lean_ctor_set(v___x_38_, 1, v___x_36_);
lean_ctor_set(v___x_38_, 2, v___x_35_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_48_ = l_Lean_Parser_Tactic_simpLemma;
v___x_49_ = l_Lean_Parser_Tactic_simpErase;
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__22));
v___x_51_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_51_, 0, v___x_50_);
lean_ctor_set(v___x_51_, 1, v___x_49_);
lean_ctor_set(v___x_51_, 2, v___x_48_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27(void){
_start:
{
uint8_t v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_56_ = 1;
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__26));
v___x_58_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__24));
v___x_59_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__23);
v___x_60_ = lean_alloc_ctor(10, 3, 1);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v___x_58_);
lean_ctor_set(v___x_60_, 2, v___x_57_);
lean_ctor_set_uint8(v___x_60_, sizeof(void*)*3, v___x_56_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__27);
v___x_62_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__20));
v___x_63_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_64_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__28);
v___x_65_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__18));
v___x_66_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_67_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_65_);
lean_ctor_set(v___x_67_, 2, v___x_64_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_71_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__31));
v___x_72_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__29);
v___x_73_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_74_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___x_72_);
lean_ctor_set(v___x_74_, 2, v___x_71_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__32);
v___x_76_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__10));
v___x_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__33);
v___x_79_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__16);
v___x_80_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_81_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
lean_ctor_set(v___x_81_, 2, v___x_78_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_87_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__37));
v___x_88_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__34);
v___x_89_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_90_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v___x_88_);
lean_ctor_set(v___x_90_, 2, v___x_87_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_97_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__41));
v___x_98_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__38);
v___x_99_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__5));
v___x_100_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
lean_ctor_set(v___x_100_, 1, v___x_98_);
lean_ctor_set(v___x_100_, 2, v___x_97_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_101_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__42);
v___x_102_ = lean_unsigned_to_nat(1022u);
v___x_103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__3));
v___x_104_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v___x_102_);
lean_ctor_set(v___x_104_, 2, v___x_101_);
return v___x_104_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercent(void){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43, &lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercent___closed__43);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg(lean_object* v_e_106_, lean_object* v___y_107_){
_start:
{
uint8_t v___x_109_; 
v___x_109_ = l_Lean_Expr_hasMVar(v_e_106_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; 
v___x_110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_110_, 0, v_e_106_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v_mctx_112_; lean_object* v___x_113_; lean_object* v_fst_114_; lean_object* v_snd_115_; lean_object* v___x_116_; lean_object* v_cache_117_; lean_object* v_zetaDeltaFVarIds_118_; lean_object* v_postponed_119_; lean_object* v_diag_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_129_; 
v___x_111_ = lean_st_ref_get(v___y_107_);
v_mctx_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc_ref(v_mctx_112_);
lean_dec(v___x_111_);
v___x_113_ = l_Lean_instantiateMVarsCore(v_mctx_112_, v_e_106_);
v_fst_114_ = lean_ctor_get(v___x_113_, 0);
lean_inc(v_fst_114_);
v_snd_115_ = lean_ctor_get(v___x_113_, 1);
lean_inc(v_snd_115_);
lean_dec_ref(v___x_113_);
v___x_116_ = lean_st_ref_take(v___y_107_);
v_cache_117_ = lean_ctor_get(v___x_116_, 1);
v_zetaDeltaFVarIds_118_ = lean_ctor_get(v___x_116_, 2);
v_postponed_119_ = lean_ctor_get(v___x_116_, 3);
v_diag_120_ = lean_ctor_get(v___x_116_, 4);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_129_ == 0)
{
lean_object* v_unused_130_; 
v_unused_130_ = lean_ctor_get(v___x_116_, 0);
lean_dec(v_unused_130_);
v___x_122_ = v___x_116_;
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_diag_120_);
lean_inc(v_postponed_119_);
lean_inc(v_zetaDeltaFVarIds_118_);
lean_inc(v_cache_117_);
lean_dec(v___x_116_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_129_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_125_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 0, v_snd_115_);
v___x_125_ = v___x_122_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v_snd_115_);
lean_ctor_set(v_reuseFailAlloc_128_, 1, v_cache_117_);
lean_ctor_set(v_reuseFailAlloc_128_, 2, v_zetaDeltaFVarIds_118_);
lean_ctor_set(v_reuseFailAlloc_128_, 3, v_postponed_119_);
lean_ctor_set(v_reuseFailAlloc_128_, 4, v_diag_120_);
v___x_125_ = v_reuseFailAlloc_128_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_st_ref_set(v___y_107_, v___x_125_);
v___x_127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_127_, 0, v_fst_114_);
return v___x_127_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg___boxed(lean_object* v_e_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg(v_e_131_, v___y_132_);
lean_dec(v___y_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0(lean_object* v_e_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg(v_e_135_, v___y_137_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___boxed(lean_object* v_e_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0(v_e_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1(lean_object* v_msgData_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v___x_155_; lean_object* v_env_156_; lean_object* v___x_157_; lean_object* v_mctx_158_; lean_object* v_lctx_159_; lean_object* v_options_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_155_ = lean_st_ref_get(v___y_153_);
v_env_156_ = lean_ctor_get(v___x_155_, 0);
lean_inc_ref(v_env_156_);
lean_dec(v___x_155_);
v___x_157_ = lean_st_ref_get(v___y_151_);
v_mctx_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc_ref(v_mctx_158_);
lean_dec(v___x_157_);
v_lctx_159_ = lean_ctor_get(v___y_150_, 2);
v_options_160_ = lean_ctor_get(v___y_152_, 2);
lean_inc_ref(v_options_160_);
lean_inc_ref(v_lctx_159_);
v___x_161_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_161_, 0, v_env_156_);
lean_ctor_set(v___x_161_, 1, v_mctx_158_);
lean_ctor_set(v___x_161_, 2, v_lctx_159_);
lean_ctor_set(v___x_161_, 3, v_options_160_);
v___x_162_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
lean_ctor_set(v___x_162_, 1, v_msgData_149_);
v___x_163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1___boxed(lean_object* v_msgData_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1(v_msgData_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg(lean_object* v_msg_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_){
_start:
{
lean_object* v_ref_177_; lean_object* v___x_178_; lean_object* v_a_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_187_; 
v_ref_177_ = lean_ctor_get(v___y_174_, 5);
v___x_178_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1_spec__1(v_msg_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
v_a_179_ = lean_ctor_get(v___x_178_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v___x_178_);
if (v_isSharedCheck_187_ == 0)
{
v___x_181_ = v___x_178_;
v_isShared_182_ = v_isSharedCheck_187_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_a_179_);
lean_dec(v___x_178_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_187_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_183_; lean_object* v___x_185_; 
lean_inc(v_ref_177_);
v___x_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_183_, 0, v_ref_177_);
lean_ctor_set(v___x_183_, 1, v_a_179_);
if (v_isShared_182_ == 0)
{
lean_ctor_set_tag(v___x_181_, 1);
lean_ctor_set(v___x_181_, 0, v___x_183_);
v___x_185_ = v___x_181_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_183_);
v___x_185_ = v_reuseFailAlloc_186_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
return v___x_185_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg___boxed(lean_object* v_msg_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg(v_msg_188_, v___y_189_, v___y_190_, v___y_191_, v___y_192_);
lean_dec(v___y_192_);
lean_dec_ref(v___y_191_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
return v_res_194_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0(void){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_195_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__0);
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
return v___x_197_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_198_ = lean_unsigned_to_nat(0u);
v___x_199_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1);
v___x_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v___x_198_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_201_ = lean_unsigned_to_nat(32u);
v___x_202_ = lean_mk_empty_array_with_capacity(v___x_201_);
v___x_203_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
return v___x_203_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4(void){
_start:
{
size_t v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_204_ = ((size_t)5ULL);
v___x_205_ = lean_unsigned_to_nat(0u);
v___x_206_ = lean_unsigned_to_nat(32u);
v___x_207_ = lean_mk_empty_array_with_capacity(v___x_206_);
v___x_208_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__3);
v___x_209_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v___x_207_);
lean_ctor_set(v___x_209_, 2, v___x_205_);
lean_ctor_set(v___x_209_, 3, v___x_205_);
lean_ctor_set_usize(v___x_209_, 4, v___x_204_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_210_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__4);
v___x_211_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__1);
v___x_212_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___x_211_);
lean_ctor_set(v___x_212_, 2, v___x_211_);
lean_ctor_set(v___x_212_, 3, v___x_210_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_213_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__5);
v___x_214_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__2);
v___x_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___x_213_);
return v___x_215_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8(void){
_start:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__7));
v___x_218_ = l_Lean_stringToMessageData(v___x_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0(lean_object* v_ctx_219_, lean_object* v_simprocs_220_, lean_object* v_e_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_227_; lean_object* v_a_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_227_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__0___redArg(v_e_221_, v___y_223_);
v_a_228_ = lean_ctor_get(v___x_227_, 0);
lean_inc_n(v_a_228_, 2);
lean_dec_ref(v___x_227_);
v___x_229_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__6);
v___x_230_ = l_Lean_Meta_dsimp(v_a_228_, v_ctx_219_, v_simprocs_220_, v___x_229_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
if (lean_obj_tag(v___x_230_) == 0)
{
lean_object* v_a_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_250_; 
v_a_231_ = lean_ctor_get(v___x_230_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_250_ == 0)
{
v___x_233_ = v___x_230_;
v_isShared_234_ = v_isSharedCheck_250_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_a_231_);
lean_dec(v___x_230_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_250_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v_fst_235_; uint8_t v___x_236_; 
v_fst_235_ = lean_ctor_get(v_a_231_, 0);
lean_inc(v_fst_235_);
lean_dec(v_a_231_);
v___x_236_ = lean_expr_eqv(v_fst_235_, v_a_228_);
lean_dec(v_a_228_);
if (v___x_236_ == 0)
{
lean_object* v___x_238_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 0, v_fst_235_);
v___x_238_ = v___x_233_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v_fst_235_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
else
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_249_; 
lean_dec(v_fst_235_);
lean_del_object(v___x_233_);
v___x_240_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___closed__8);
v___x_241_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg(v___x_240_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
v_a_242_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_249_ == 0)
{
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_249_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_247_; 
if (v_isShared_245_ == 0)
{
v___x_247_ = v___x_244_;
goto v_reusejp_246_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v_a_242_);
v___x_247_ = v_reuseFailAlloc_248_;
goto v_reusejp_246_;
}
v_reusejp_246_:
{
return v___x_247_;
}
}
}
}
}
else
{
lean_object* v_a_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
lean_dec(v_a_228_);
v_a_251_ = lean_ctor_get(v___x_230_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v___x_230_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_a_251_);
lean_dec(v___x_230_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___x_256_; 
if (v_isShared_254_ == 0)
{
v___x_256_ = v___x_253_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_a_251_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0___boxed(lean_object* v_ctx_259_, lean_object* v_simprocs_260_, lean_object* v_e_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0(v_ctx_259_, v_simprocs_260_, v_e_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec_ref(v___y_264_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator(lean_object* v_stx_272_, lean_object* v_expectedType_273_, lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_){
_start:
{
lean_object* v___x_281_; uint8_t v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_281_ = lean_box(0);
v___x_282_ = 0;
v___x_283_ = lean_box(0);
v___x_284_ = l_Lean_Meta_mkFreshExprMVar(v___x_281_, v___x_282_, v___x_283_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___y_291_; lean_object* v___x_301_; lean_object* v___x_302_; uint8_t v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
v___x_286_ = l_Lean_Expr_mvarId_x21(v_a_285_);
lean_dec(v_a_285_);
v___x_287_ = lean_box(0);
v___x_288_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_288_, 0, v___x_286_);
lean_ctor_set(v___x_288_, 1, v___x_287_);
v___x_289_ = lean_st_mk_ref(v___x_288_);
v___x_301_ = lean_unsigned_to_nat(5u);
v___x_302_ = l_Lean_Syntax_getArg(v_stx_272_, v___x_301_);
v___x_303_ = 1;
v___x_304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__0));
v___x_305_ = l_Lean_Elab_Term_elabTerm(v___x_302_, v_expectedType_273_, v___x_303_, v___x_303_, v_a_274_, v_a_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v_a_306_; uint8_t v___x_307_; uint8_t v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v_a_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v___x_305_, 1);
v___x_307_ = 0;
v___x_308_ = 2;
v___x_309_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___closed__1));
v___x_310_ = l_Lean_Elab_Tactic_mkSimpContext(v_stx_272_, v___x_307_, v___x_308_, v___x_307_, v___x_309_, v___x_304_, v___x_289_, v_a_274_, v_a_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_310_) == 0)
{
lean_object* v_a_311_; lean_object* v_ctx_312_; lean_object* v_simprocs_313_; lean_object* v___x_314_; 
v_a_311_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_a_311_);
lean_dec_ref_known(v___x_310_, 1);
v_ctx_312_ = lean_ctor_get(v_a_311_, 0);
lean_inc_ref(v_ctx_312_);
v_simprocs_313_ = lean_ctor_get(v_a_311_, 1);
lean_inc_ref(v_simprocs_313_);
lean_dec(v_a_311_);
lean_inc(v_a_306_);
v___x_314_ = l_Lean_Meta_isProof(v_a_306_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_314_) == 0)
{
lean_object* v_a_315_; uint8_t v___x_316_; 
v_a_315_ = lean_ctor_get(v___x_314_, 0);
lean_inc(v_a_315_);
lean_dec_ref_known(v___x_314_, 1);
v___x_316_ = lean_unbox(v_a_315_);
lean_dec(v_a_315_);
if (v___x_316_ == 0)
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0(v_ctx_312_, v_simprocs_313_, v_a_306_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
v___y_291_ = v___x_317_;
goto v___jp_290_;
}
else
{
lean_object* v___x_318_; 
lean_inc(v_a_279_);
lean_inc_ref(v_a_278_);
lean_inc(v_a_277_);
lean_inc_ref(v_a_276_);
lean_inc(v_a_306_);
v___x_318_ = lean_infer_type(v_a_306_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_318_) == 0)
{
lean_object* v_a_319_; lean_object* v___x_320_; 
v_a_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_a_319_);
lean_dec_ref_known(v___x_318_, 1);
v___x_320_ = lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___lam__0(v_ctx_312_, v_simprocs_313_, v_a_319_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
if (lean_obj_tag(v___x_320_) == 0)
{
lean_object* v_a_321_; lean_object* v___x_322_; 
v_a_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc(v_a_321_);
lean_dec_ref_known(v___x_320_, 1);
v___x_322_ = l_Lean_Meta_mkExpectedTypeHint(v_a_306_, v_a_321_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
v___y_291_ = v___x_322_;
goto v___jp_290_;
}
else
{
lean_dec(v_a_306_);
lean_dec(v___x_289_);
return v___x_320_;
}
}
else
{
lean_dec_ref(v_simprocs_313_);
lean_dec_ref(v_ctx_312_);
lean_dec(v_a_306_);
v___y_291_ = v___x_318_;
goto v___jp_290_;
}
}
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_dec_ref(v_simprocs_313_);
lean_dec_ref(v_ctx_312_);
lean_dec(v_a_306_);
lean_dec(v___x_289_);
v_a_323_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_314_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_314_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
else
{
lean_object* v_a_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_338_; 
lean_dec(v_a_306_);
lean_dec(v___x_289_);
v_a_331_ = lean_ctor_get(v___x_310_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v___x_310_);
if (v_isSharedCheck_338_ == 0)
{
v___x_333_ = v___x_310_;
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
else
{
lean_inc(v_a_331_);
lean_dec(v___x_310_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_336_; 
if (v_isShared_334_ == 0)
{
v___x_336_ = v___x_333_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v_a_331_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
}
else
{
v___y_291_ = v___x_305_;
goto v___jp_290_;
}
v___jp_290_:
{
if (lean_obj_tag(v___y_291_) == 0)
{
lean_object* v_a_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_300_; 
v_a_292_ = lean_ctor_get(v___y_291_, 0);
v_isSharedCheck_300_ = !lean_is_exclusive(v___y_291_);
if (v_isSharedCheck_300_ == 0)
{
v___x_294_ = v___y_291_;
v_isShared_295_ = v_isSharedCheck_300_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_a_292_);
lean_dec(v___y_291_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_300_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_296_; lean_object* v___x_298_; 
v___x_296_ = lean_st_ref_get(v___x_289_);
lean_dec(v___x_289_);
lean_dec(v___x_296_);
if (v_isShared_295_ == 0)
{
v___x_298_ = v___x_294_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_292_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
else
{
lean_dec(v___x_289_);
return v___y_291_;
}
}
}
else
{
lean_dec(v_expectedType_273_);
return v___x_284_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator___boxed(lean_object* v_stx_339_, lean_object* v_expectedType_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Mathlib_Tactic_dsimpPercentElaborator(v_stx_339_, v_expectedType_340_, v_a_341_, v_a_342_, v_a_343_, v_a_344_, v_a_345_, v_a_346_);
lean_dec(v_a_346_);
lean_dec_ref(v_a_345_);
lean_dec(v_a_344_);
lean_dec_ref(v_a_343_);
lean_dec(v_a_342_);
lean_dec_ref(v_a_341_);
lean_dec(v_stx_339_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1(lean_object* v_00_u03b1_349_, lean_object* v_msg_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___redArg(v_msg_350_, v___y_351_, v___y_352_, v___y_353_, v___y_354_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1___boxed(lean_object* v_00_u03b1_357_, lean_object* v_msg_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_dsimpPercentElaborator_spec__1(v_00_u03b1_357_, v_msg_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
lean_dec(v___y_362_);
lean_dec_ref(v___y_361_);
lean_dec(v___y_360_);
lean_dec_ref(v___y_359_);
return v_res_364_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_dsimpPercent = _init_lp_mathlib_Mathlib_Tactic_dsimpPercent();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_dsimpPercent);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_DSimpPercent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_DSimpPercent(builtin);
}
#ifdef __cplusplus
}
#endif
