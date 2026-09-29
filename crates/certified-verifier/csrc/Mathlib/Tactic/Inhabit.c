// Lean compiler output
// Module: Mathlib.Tactic.Inhabit
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.ElabTerm public import Mathlib.Init
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
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_assert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_nonempty__prop__to__inhabited(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "inhabit"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__3_value),LEAN_SCALAR_PTR_LITERAL(222, 101, 59, 46, 93, 24, 196, 11)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "inhabit "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__11_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__17 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__15_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__17_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__12_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__10_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__20 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__8_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__20_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__21_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__22 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__22_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__22_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__23 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__23_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__24 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__24_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__6_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__21_value),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__24_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__25 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__25_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_inhabit___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__25_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit___closed__26 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__26_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Tactic_inhabit = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__26_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Inhabited"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(164, 88, 86, 106, 191, 136, 33, 185)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Nonempty"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(142, 191, 110, 220, 210, 100, 152, 183)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "nonempty_to_inhabited"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(225, 230, 174, 97, 163, 173, 173, 57)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "nonempty_prop_to_inhabited"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_inhabit___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(111, 196, 60, 127, 189, 14, 135, 234)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "inhabited_h"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(144, 85, 182, 174, 210, 204, 233, 34)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_nonempty__prop__to__inhabited(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b1__nonempty_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0(lean_object* v_x_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
lean_inc(v___y_67_);
lean_inc_ref(v___y_66_);
lean_inc(v___y_65_);
lean_inc_ref(v___y_64_);
v___x_73_ = lean_apply_9(v_x_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_, lean_box(0));
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0___boxed(lean_object* v_x_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0(v_x_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg(lean_object* v_mvarId_85_, lean_object* v_x_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v___f_96_; lean_object* v___x_97_; 
lean_inc(v___y_90_);
lean_inc_ref(v___y_89_);
lean_inc(v___y_88_);
lean_inc_ref(v___y_87_);
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_96_, 0, v_x_86_);
lean_closure_set(v___f_96_, 1, v___y_87_);
lean_closure_set(v___f_96_, 2, v___y_88_);
lean_closure_set(v___f_96_, 3, v___y_89_);
lean_closure_set(v___f_96_, 4, v___y_90_);
v___x_97_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_85_, v___f_96_, v___y_91_, v___y_92_, v___y_93_, v___y_94_);
if (lean_obj_tag(v___x_97_) == 0)
{
return v___x_97_;
}
else
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
v_a_98_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_97_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_97_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg___boxed(lean_object* v_mvarId_106_, lean_object* v_x_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg(v_mvarId_106_, v_x_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0(lean_object* v_00_u03b1_118_, lean_object* v_mvarId_119_, lean_object* v_x_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg(v_mvarId_119_, v_x_120_, v___y_121_, v___y_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___boxed(lean_object* v_00_u03b1_131_, lean_object* v_mvarId_132_, lean_object* v_x_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0(v_00_u03b1_131_, v_mvarId_132_, v_x_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_);
lean_dec(v___y_141_);
lean_dec_ref(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0(lean_object* v_term_165_, lean_object* v___x_166_, uint8_t v___x_167_, lean_object* v_goal_168_, lean_object* v_h__name_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = l_Lean_Elab_Tactic_elabTerm(v_term_165_, v___x_166_, v___x_167_, v___y_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_179_) == 0)
{
lean_object* v_a_180_; lean_object* v___x_181_; 
v_a_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc_n(v_a_180_, 2);
lean_dec_ref_known(v___x_179_, 1);
v___x_181_ = l_Lean_Meta_getLevel(v_a_180_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_181_) == 0)
{
lean_object* v_a_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___y_189_; lean_object* v_pf_190_; lean_object* v___y_191_; lean_object* v___y_192_; lean_object* v___y_193_; lean_object* v___y_194_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v_a_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_a_182_);
lean_dec_ref_known(v___x_181_, 1);
v___x_183_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__1));
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_185_, 0, v_a_182_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
lean_inc_ref(v___x_185_);
v___x_186_ = l_Lean_mkConst(v___x_183_, v___x_185_);
lean_inc_n(v_a_180_, 2);
v___x_187_ = l_Lean_Expr_app___override(v___x_186_, v_a_180_);
v___x_216_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__3));
v___x_217_ = l_Lean_mkConst(v___x_216_, v___x_185_);
v___x_218_ = l_Lean_Expr_app___override(v___x_217_, v_a_180_);
v___x_219_ = lean_box(0);
v___x_220_ = l_Lean_Meta_synthInstance(v___x_218_, v___x_219_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_220_) == 0)
{
lean_object* v_a_221_; lean_object* v___y_223_; 
v_a_221_ = lean_ctor_get(v___x_220_, 0);
lean_inc(v_a_221_);
lean_dec_ref_known(v___x_220_, 1);
if (lean_obj_tag(v_h__name_169_) == 0)
{
lean_object* v___x_265_; 
v___x_265_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__9));
v___y_223_ = v___x_265_;
goto v___jp_222_;
}
else
{
lean_object* v_val_266_; lean_object* v___x_267_; 
v_val_266_ = lean_ctor_get(v_h__name_169_, 0);
v___x_267_ = l_Lean_TSyntax_getId(v_val_266_);
v___y_223_ = v___x_267_;
goto v___jp_222_;
}
v___jp_222_:
{
lean_object* v___x_224_; 
lean_inc(v_a_180_);
v___x_224_ = l_Lean_Meta_isProp(v_a_180_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_224_) == 0)
{
lean_object* v_a_225_; uint8_t v___x_226_; 
v_a_225_ = lean_ctor_get(v___x_224_, 0);
lean_inc(v_a_225_);
lean_dec_ref_known(v___x_224_, 1);
v___x_226_ = lean_unbox(v_a_225_);
lean_dec(v_a_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_227_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__5));
v___x_228_ = lean_unsigned_to_nat(2u);
v___x_229_ = lean_mk_empty_array_with_capacity(v___x_228_);
v___x_230_ = lean_array_push(v___x_229_, v_a_180_);
v___x_231_ = lean_array_push(v___x_230_, v_a_221_);
v___x_232_ = l_Lean_Meta_mkAppM(v___x_227_, v___x_231_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v_a_233_; 
v_a_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_a_233_);
lean_dec_ref_known(v___x_232_, 1);
v___y_189_ = v___y_223_;
v_pf_190_ = v_a_233_;
v___y_191_ = v___y_174_;
v___y_192_ = v___y_175_;
v___y_193_ = v___y_176_;
v___y_194_ = v___y_177_;
goto v___jp_188_;
}
else
{
lean_object* v_a_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_241_; 
lean_dec(v___y_223_);
lean_dec_ref(v___x_187_);
lean_dec(v_goal_168_);
v_a_234_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_241_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_241_ == 0)
{
v___x_236_ = v___x_232_;
v_isShared_237_ = v_isSharedCheck_241_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_a_234_);
lean_dec(v___x_232_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_241_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v___x_239_; 
if (v_isShared_237_ == 0)
{
v___x_239_ = v___x_236_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v_a_234_);
v___x_239_ = v_reuseFailAlloc_240_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
return v___x_239_;
}
}
}
}
else
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_242_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___closed__7));
v___x_243_ = lean_unsigned_to_nat(2u);
v___x_244_ = lean_mk_empty_array_with_capacity(v___x_243_);
v___x_245_ = lean_array_push(v___x_244_, v_a_180_);
v___x_246_ = lean_array_push(v___x_245_, v_a_221_);
v___x_247_ = l_Lean_Meta_mkAppM(v___x_242_, v___x_246_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
if (lean_obj_tag(v___x_247_) == 0)
{
lean_object* v_a_248_; 
v_a_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_a_248_);
lean_dec_ref_known(v___x_247_, 1);
v___y_189_ = v___y_223_;
v_pf_190_ = v_a_248_;
v___y_191_ = v___y_174_;
v___y_192_ = v___y_175_;
v___y_193_ = v___y_176_;
v___y_194_ = v___y_177_;
goto v___jp_188_;
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
lean_dec(v___y_223_);
lean_dec_ref(v___x_187_);
lean_dec(v_goal_168_);
v_a_249_ = lean_ctor_get(v___x_247_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_247_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_247_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec(v___y_223_);
lean_dec(v_a_221_);
lean_dec_ref(v___x_187_);
lean_dec(v_a_180_);
lean_dec(v_goal_168_);
v_a_257_ = lean_ctor_get(v___x_224_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_224_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_224_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_a_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
}
else
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_dec_ref(v___x_187_);
lean_dec(v_a_180_);
lean_dec(v_goal_168_);
v_a_268_ = lean_ctor_get(v___x_220_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_220_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_220_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_273_; 
if (v_isShared_271_ == 0)
{
v___x_273_ = v___x_270_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_268_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
}
v___jp_188_:
{
lean_object* v___x_195_; 
v___x_195_ = l_Lean_MVarId_assert(v_goal_168_, v___y_189_, v___x_187_, v_pf_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
if (lean_obj_tag(v___x_195_) == 0)
{
lean_object* v_a_196_; uint8_t v___x_197_; lean_object* v___x_198_; 
v_a_196_ = lean_ctor_get(v___x_195_, 0);
lean_inc(v_a_196_);
lean_dec_ref_known(v___x_195_, 1);
v___x_197_ = 1;
v___x_198_ = l_Lean_Meta_intro1Core(v_a_196_, v___x_197_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
if (lean_obj_tag(v___x_198_) == 0)
{
lean_object* v_a_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_207_; 
v_a_199_ = lean_ctor_get(v___x_198_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_207_ == 0)
{
v___x_201_ = v___x_198_;
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_a_199_);
lean_dec(v___x_198_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v_snd_203_; lean_object* v___x_205_; 
v_snd_203_ = lean_ctor_get(v_a_199_, 1);
lean_inc(v_snd_203_);
lean_dec(v_a_199_);
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 0, v_snd_203_);
v___x_205_ = v___x_201_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_snd_203_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
else
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_215_; 
v_a_208_ = lean_ctor_get(v___x_198_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_215_ == 0)
{
v___x_210_ = v___x_198_;
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_198_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_215_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v___x_213_; 
if (v_isShared_211_ == 0)
{
v___x_213_ = v___x_210_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v_a_208_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
}
else
{
return v___x_195_;
}
}
}
else
{
lean_object* v_a_276_; lean_object* v___x_278_; uint8_t v_isShared_279_; uint8_t v_isSharedCheck_283_; 
lean_dec(v_a_180_);
lean_dec(v_goal_168_);
v_a_276_ = lean_ctor_get(v___x_181_, 0);
v_isSharedCheck_283_ = !lean_is_exclusive(v___x_181_);
if (v_isSharedCheck_283_ == 0)
{
v___x_278_ = v___x_181_;
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
else
{
lean_inc(v_a_276_);
lean_dec(v___x_181_);
v___x_278_ = lean_box(0);
v_isShared_279_ = v_isSharedCheck_283_;
goto v_resetjp_277_;
}
v_resetjp_277_:
{
lean_object* v___x_281_; 
if (v_isShared_279_ == 0)
{
v___x_281_ = v___x_278_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v_a_276_);
v___x_281_ = v_reuseFailAlloc_282_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
return v___x_281_;
}
}
}
}
else
{
lean_object* v_a_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec(v_goal_168_);
v_a_284_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_179_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_a_284_);
lean_dec(v___x_179_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_a_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___boxed(lean_object* v_term_292_, lean_object* v___x_293_, lean_object* v___x_294_, lean_object* v_goal_295_, lean_object* v_h__name_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
uint8_t v___x_3795__boxed_306_; lean_object* v_res_307_; 
v___x_3795__boxed_306_ = lean_unbox(v___x_294_);
v_res_307_ = lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0(v_term_292_, v___x_293_, v___x_3795__boxed_306_, v_goal_295_, v_h__name_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v_h__name_296_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit(lean_object* v_goal_308_, lean_object* v_h__name_309_, lean_object* v_term_310_, lean_object* v_a_311_, lean_object* v_a_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_, lean_object* v_a_318_){
_start:
{
lean_object* v___x_320_; uint8_t v___x_321_; lean_object* v___x_322_; lean_object* v___f_323_; lean_object* v___x_324_; 
v___x_320_ = lean_box(0);
v___x_321_ = 0;
v___x_322_ = lean_box(v___x_321_);
lean_inc(v_goal_308_);
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_evalInhabit___lam__0___boxed), 14, 5);
lean_closure_set(v___f_323_, 0, v_term_310_);
lean_closure_set(v___f_323_, 1, v___x_320_);
lean_closure_set(v___f_323_, 2, v___x_322_);
lean_closure_set(v___f_323_, 3, v_goal_308_);
lean_closure_set(v___f_323_, 4, v_h__name_309_);
v___x_324_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_evalInhabit_spec__0___redArg(v_goal_308_, v___f_323_, v_a_311_, v_a_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_, v_a_317_, v_a_318_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_evalInhabit___boxed(lean_object* v_goal_325_, lean_object* v_h__name_326_, lean_object* v_term_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Lean_Elab_Tactic_evalInhabit(v_goal_325_, v_h__name_326_, v_term_327_, v_a_328_, v_a_329_, v_a_330_, v_a_331_, v_a_332_, v_a_333_, v_a_334_, v_a_335_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
lean_dec(v_a_333_);
lean_dec_ref(v_a_332_);
lean_dec(v_a_331_);
lean_dec_ref(v_a_330_);
lean_dec(v_a_329_);
lean_dec_ref(v_a_328_);
return v_res_337_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_338_ = lean_box(0);
v___x_339_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_340_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
lean_ctor_set(v___x_340_, 1, v___x_338_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg(){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___closed__0);
v___x_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg___boxed(lean_object* v___y_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0(lean_object* v_00_u03b1_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___boxed(lean_object* v_00_u03b1_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0(v_00_u03b1_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_, v___y_363_, v___y_364_, v___y_365_);
lean_dec(v___y_365_);
lean_dec_ref(v___y_364_);
lean_dec(v___y_363_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1(lean_object* v_x_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_){
_start:
{
lean_object* v_h__name_379_; lean_object* v___y_380_; lean_object* v___y_381_; lean_object* v___y_382_; lean_object* v___y_383_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_386_; lean_object* v___y_387_; lean_object* v___x_413_; uint8_t v___x_414_; 
v___x_413_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_inhabit___closed__4));
lean_inc(v_x_368_);
v___x_414_ = l_Lean_Syntax_isOfKind(v_x_368_, v___x_413_);
if (v___x_414_ == 0)
{
lean_object* v___x_415_; 
lean_dec(v_x_368_);
v___x_415_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
return v___x_415_;
}
else
{
lean_object* v___x_416_; lean_object* v___x_417_; uint8_t v___x_418_; 
v___x_416_ = lean_unsigned_to_nat(1u);
v___x_417_ = l_Lean_Syntax_getArg(v_x_368_, v___x_416_);
v___x_418_ = l_Lean_Syntax_isNone(v___x_417_);
if (v___x_418_ == 0)
{
lean_object* v___x_419_; uint8_t v___x_420_; 
v___x_419_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_417_);
v___x_420_ = l_Lean_Syntax_matchesNull(v___x_417_, v___x_419_);
if (v___x_420_ == 0)
{
lean_object* v___x_421_; 
lean_dec(v___x_417_);
lean_dec(v_x_368_);
v___x_421_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
return v___x_421_;
}
else
{
lean_object* v___x_422_; lean_object* v_h__name_423_; lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_422_ = lean_unsigned_to_nat(0u);
v_h__name_423_ = l_Lean_Syntax_getArg(v___x_417_, v___x_422_);
lean_dec(v___x_417_);
v___x_424_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_inhabit___closed__14));
lean_inc(v_h__name_423_);
v___x_425_ = l_Lean_Syntax_isOfKind(v_h__name_423_, v___x_424_);
if (v___x_425_ == 0)
{
lean_object* v___x_426_; 
lean_dec(v_h__name_423_);
lean_dec(v_x_368_);
v___x_426_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1_spec__0___redArg();
return v___x_426_;
}
else
{
lean_object* v___x_427_; 
v___x_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_427_, 0, v_h__name_423_);
v_h__name_379_ = v___x_427_;
v___y_380_ = v_a_369_;
v___y_381_ = v_a_370_;
v___y_382_ = v_a_371_;
v___y_383_ = v_a_372_;
v___y_384_ = v_a_373_;
v___y_385_ = v_a_374_;
v___y_386_ = v_a_375_;
v___y_387_ = v_a_376_;
goto v___jp_378_;
}
}
}
else
{
lean_object* v___x_428_; 
lean_dec(v___x_417_);
v___x_428_ = lean_box(0);
v_h__name_379_ = v___x_428_;
v___y_380_ = v_a_369_;
v___y_381_ = v_a_370_;
v___y_382_ = v_a_371_;
v___y_383_ = v_a_372_;
v___y_384_ = v_a_373_;
v___y_385_ = v_a_374_;
v___y_386_ = v_a_375_;
v___y_387_ = v_a_376_;
goto v___jp_378_;
}
}
v___jp_378_:
{
lean_object* v___x_388_; 
v___x_388_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_381_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
if (lean_obj_tag(v___x_388_) == 0)
{
lean_object* v_a_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v_a_389_ = lean_ctor_get(v___x_388_, 0);
lean_inc(v_a_389_);
lean_dec_ref_known(v___x_388_, 1);
v___x_390_ = lean_unsigned_to_nat(2u);
v___x_391_ = l_Lean_Syntax_getArg(v_x_368_, v___x_390_);
lean_dec(v_x_368_);
v___x_392_ = lp_mathlib_Lean_Elab_Tactic_evalInhabit(v_a_389_, v_h__name_379_, v___x_391_, v___y_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
if (lean_obj_tag(v___x_392_) == 0)
{
lean_object* v_a_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v_a_393_ = lean_ctor_get(v___x_392_, 0);
lean_inc(v_a_393_);
lean_dec_ref_known(v___x_392_, 1);
v___x_394_ = lean_box(0);
v___x_395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_395_, 0, v_a_393_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_395_, v___y_381_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
return v___x_396_;
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
v_a_397_ = lean_ctor_get(v___x_392_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_392_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_392_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_392_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_a_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v_h__name_379_);
lean_dec(v_x_368_);
v_a_405_ = lean_ctor_get(v___x_388_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_388_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_388_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_388_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1___boxed(lean_object* v_x_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_, lean_object* v_a_437_, lean_object* v_a_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Inhabit______elabRules__Lean__Elab__Tactic__inhabit__1(v_x_429_, v_a_430_, v_a_431_, v_a_432_, v_a_433_, v_a_434_, v_a_435_, v_a_436_, v_a_437_);
lean_dec(v_a_437_);
lean_dec_ref(v_a_436_);
lean_dec(v_a_435_);
lean_dec_ref(v_a_434_);
lean_dec(v_a_433_);
lean_dec_ref(v_a_432_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_430_);
return v_res_439_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
}
#ifdef __cplusplus
}
#endif
