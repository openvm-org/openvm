// Lean compiler output
// Module: Mathlib.Tactic.ApplyCongr
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Conv.Basic public import Mathlib.Init
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
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_SimpCongrTheorems_get(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_getLhs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1;
static const lean_ctor_object lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__2 = (const lean_object*)&lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "No matching congr lemmas found"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Left-hand side must be an application of a constant."};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "applyCongr"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__3 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(131, 217, 8, 96, 215, 169, 189, 208)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__5 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "apply_congr"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__7 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__8 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__9 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__10 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__11 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__12 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__13 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__14 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__14_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__15 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__15_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__16 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__13_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__17 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__17_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__18 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__19 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__20 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__17_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__20_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__21 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__10_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__21_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__22 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__22_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__8_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__22_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__23 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__23_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__23_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__24 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Parser_Tactic_applyCongr = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__24_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg(v_e_30_, v___y_36_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___boxed(lean_object* v_e_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5(v_e_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
lean_dec(v___y_49_);
lean_dec_ref(v___y_48_);
lean_dec(v___y_47_);
lean_dec_ref(v___y_46_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0(lean_object* v_x_52_, lean_object* v_x_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
if (lean_obj_tag(v_x_52_) == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = l_List_reverse___redArg(v_x_53_);
v___x_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
return v___x_60_;
}
else
{
lean_object* v_head_61_; lean_object* v_tail_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_81_; 
v_head_61_ = lean_ctor_get(v_x_52_, 0);
v_tail_62_ = lean_ctor_get(v_x_52_, 1);
v_isSharedCheck_81_ = !lean_is_exclusive(v_x_52_);
if (v_isSharedCheck_81_ == 0)
{
v___x_64_ = v_x_52_;
v_isShared_65_ = v_isSharedCheck_81_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_tail_62_);
lean_inc(v_head_61_);
lean_dec(v_x_52_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_81_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_66_; 
v___x_66_ = l_Lean_MVarId_intros(v_head_61_, v___y_54_, v___y_55_, v___y_56_, v___y_57_);
if (lean_obj_tag(v___x_66_) == 0)
{
lean_object* v_a_67_; lean_object* v_snd_68_; lean_object* v___x_70_; 
v_a_67_ = lean_ctor_get(v___x_66_, 0);
lean_inc(v_a_67_);
lean_dec_ref_known(v___x_66_, 1);
v_snd_68_ = lean_ctor_get(v_a_67_, 1);
lean_inc(v_snd_68_);
lean_dec(v_a_67_);
if (v_isShared_65_ == 0)
{
lean_ctor_set(v___x_64_, 1, v_x_53_);
lean_ctor_set(v___x_64_, 0, v_snd_68_);
v___x_70_ = v___x_64_;
goto v_reusejp_69_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v_snd_68_);
lean_ctor_set(v_reuseFailAlloc_72_, 1, v_x_53_);
v___x_70_ = v_reuseFailAlloc_72_;
goto v_reusejp_69_;
}
v_reusejp_69_:
{
v_x_52_ = v_tail_62_;
v_x_53_ = v___x_70_;
goto _start;
}
}
else
{
lean_object* v_a_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_80_; 
lean_del_object(v___x_64_);
lean_dec(v_tail_62_);
lean_dec(v_x_53_);
v_a_73_ = lean_ctor_get(v___x_66_, 0);
v_isSharedCheck_80_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_80_ == 0)
{
v___x_75_ = v___x_66_;
v_isShared_76_ = v_isSharedCheck_80_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_a_73_);
lean_dec(v___x_66_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_80_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_78_; 
if (v_isShared_76_ == 0)
{
v___x_78_ = v___x_75_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_79_; 
v_reuseFailAlloc_79_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_79_, 0, v_a_73_);
v___x_78_ = v_reuseFailAlloc_79_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
return v___x_78_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0___boxed(lean_object* v_x_82_, lean_object* v_x_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0(v_x_82_, v_x_83_, v___y_84_, v___y_85_, v___y_86_, v___y_87_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4(lean_object* v_msgData_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v___x_96_; lean_object* v_env_97_; lean_object* v___x_98_; lean_object* v_mctx_99_; lean_object* v_lctx_100_; lean_object* v_options_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_96_ = lean_st_ref_get(v___y_94_);
v_env_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc_ref(v_env_97_);
lean_dec(v___x_96_);
v___x_98_ = lean_st_ref_get(v___y_92_);
v_mctx_99_ = lean_ctor_get(v___x_98_, 0);
lean_inc_ref(v_mctx_99_);
lean_dec(v___x_98_);
v_lctx_100_ = lean_ctor_get(v___y_91_, 2);
v_options_101_ = lean_ctor_get(v___y_93_, 2);
lean_inc_ref(v_options_101_);
lean_inc_ref(v_lctx_100_);
v___x_102_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_102_, 0, v_env_97_);
lean_ctor_set(v___x_102_, 1, v_mctx_99_);
lean_ctor_set(v___x_102_, 2, v_lctx_100_);
lean_ctor_set(v___x_102_, 3, v_options_101_);
v___x_103_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v_msgData_90_);
v___x_104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4___boxed(lean_object* v_msgData_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4(v_msgData_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
lean_dec(v___y_107_);
lean_dec_ref(v___y_106_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg(lean_object* v_msg_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_ref_118_; lean_object* v___x_119_; lean_object* v_a_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_128_; 
v_ref_118_ = lean_ctor_get(v___y_115_, 5);
v___x_119_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4(v_msg_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
v_a_120_ = lean_ctor_get(v___x_119_, 0);
v_isSharedCheck_128_ = !lean_is_exclusive(v___x_119_);
if (v_isSharedCheck_128_ == 0)
{
v___x_122_ = v___x_119_;
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_a_120_);
lean_dec(v___x_119_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_128_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_124_; lean_object* v___x_126_; 
lean_inc(v_ref_118_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v_ref_118_);
lean_ctor_set(v___x_124_, 1, v_a_120_);
if (v_isShared_123_ == 0)
{
lean_ctor_set_tag(v___x_122_, 1);
lean_ctor_set(v___x_122_, 0, v___x_124_);
v___x_126_ = v___x_122_;
goto v_reusejp_125_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v___x_124_);
v___x_126_ = v_reuseFailAlloc_127_;
goto v_reusejp_125_;
}
v_reusejp_125_:
{
return v___x_126_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg___boxed(lean_object* v_msg_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg(v_msg_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec(v___y_131_);
lean_dec_ref(v___y_130_);
return v_res_135_;
}
}
static lean_object* _init_lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = ((lean_object*)(lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__0));
v___x_138_ = l_Lean_stringToMessageData(v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1(lean_object* v_a_143_, lean_object* v_x_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
if (lean_obj_tag(v_x_144_) == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
lean_dec(v_a_143_);
v___x_150_ = lean_obj_once(&lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1, &lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1_once, _init_lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__1);
v___x_151_ = lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg(v___x_150_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
return v___x_151_;
}
else
{
lean_object* v_head_152_; lean_object* v_tail_153_; lean_object* v___x_154_; 
v_head_152_ = lean_ctor_get(v_x_144_, 0);
lean_inc(v_head_152_);
v_tail_153_ = lean_ctor_get(v_x_144_, 1);
lean_inc(v_tail_153_);
lean_dec_ref_known(v_x_144_, 2);
v___x_154_ = l_Lean_Meta_saveState___redArg(v___y_146_, v___y_148_);
if (lean_obj_tag(v___x_154_) == 0)
{
lean_object* v_a_155_; lean_object* v___y_157_; uint8_t v___y_158_; lean_object* v___y_170_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_a_155_ = lean_ctor_get(v___x_154_, 0);
lean_inc(v_a_155_);
lean_dec_ref_known(v___x_154_, 1);
v___x_174_ = ((lean_object*)(lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___closed__2));
v___x_175_ = lean_box(0);
lean_inc(v_a_143_);
v___x_176_ = l_Lean_MVarId_apply(v_a_143_, v_head_152_, v___x_174_, v___x_175_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
if (lean_obj_tag(v___x_176_) == 0)
{
lean_object* v_a_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v_a_177_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_a_177_);
lean_dec_ref_known(v___x_176_, 1);
v___x_178_ = lean_box(0);
v___x_179_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__0(v_a_177_, v___x_178_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
v___y_170_ = v___x_179_;
goto v___jp_169_;
}
else
{
v___y_170_ = v___x_176_;
goto v___jp_169_;
}
v___jp_156_:
{
if (v___y_158_ == 0)
{
lean_object* v___x_159_; 
lean_dec_ref(v___y_157_);
v___x_159_ = l_Lean_Meta_SavedState_restore___redArg(v_a_155_, v___y_146_, v___y_148_);
lean_dec(v_a_155_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_dec_ref_known(v___x_159_, 1);
v_x_144_ = v_tail_153_;
goto _start;
}
else
{
lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_168_; 
lean_dec(v_tail_153_);
lean_dec(v_a_143_);
v_a_161_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_168_ == 0)
{
v___x_163_ = v___x_159_;
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_159_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v___x_166_; 
if (v_isShared_164_ == 0)
{
v___x_166_ = v___x_163_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_a_161_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
else
{
lean_dec(v_a_155_);
lean_dec(v_tail_153_);
lean_dec(v_a_143_);
return v___y_157_;
}
}
v___jp_169_:
{
if (lean_obj_tag(v___y_170_) == 0)
{
lean_dec(v_a_155_);
lean_dec(v_tail_153_);
lean_dec(v_a_143_);
return v___y_170_;
}
else
{
lean_object* v_a_171_; uint8_t v___x_172_; 
v_a_171_ = lean_ctor_get(v___y_170_, 0);
v___x_172_ = l_Lean_Exception_isInterrupt(v_a_171_);
if (v___x_172_ == 0)
{
uint8_t v___x_173_; 
lean_inc(v_a_171_);
v___x_173_ = l_Lean_Exception_isRuntime(v_a_171_);
v___y_157_ = v___y_170_;
v___y_158_ = v___x_173_;
goto v___jp_156_;
}
else
{
v___y_157_ = v___y_170_;
v___y_158_ = v___x_172_;
goto v___jp_156_;
}
}
}
}
else
{
lean_object* v_a_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_187_; 
lean_dec(v_tail_153_);
lean_dec(v_head_152_);
lean_dec(v_a_143_);
v_a_180_ = lean_ctor_get(v___x_154_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v___x_154_);
if (v_isSharedCheck_187_ == 0)
{
v___x_182_ = v___x_154_;
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_a_180_);
lean_dec(v___x_154_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_185_; 
if (v_isShared_183_ == 0)
{
v___x_185_ = v___x_182_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_a_180_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1___boxed(lean_object* v_a_188_, lean_object* v_x_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1(v_a_188_, v_x_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0(lean_object* v_congrTheoremExprs_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_198_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_208_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc(v_a_207_);
lean_dec_ref_known(v___x_206_, 1);
v___x_208_ = lp_mathlib_List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1(v_a_207_, v_congrTheoremExprs_196_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
if (lean_obj_tag(v___x_208_) == 0)
{
lean_object* v_a_209_; lean_object* v___x_210_; 
v_a_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc(v_a_209_);
lean_dec_ref_known(v___x_208_, 1);
v___x_210_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_209_, v___y_198_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_218_; 
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v___x_210_, 0);
lean_dec(v_unused_219_);
v___x_212_ = v___x_210_;
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
else
{
lean_dec(v___x_210_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_214_; lean_object* v___x_216_; 
v___x_214_ = lean_box(0);
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v___x_214_);
v___x_216_ = v___x_212_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
else
{
return v___x_210_;
}
}
else
{
lean_object* v_a_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_227_; 
v_a_220_ = lean_ctor_get(v___x_208_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_227_ == 0)
{
v___x_222_ = v___x_208_;
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_a_220_);
lean_dec(v___x_208_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_225_; 
if (v_isShared_223_ == 0)
{
v___x_225_ = v___x_222_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v_a_220_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
else
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
lean_dec(v_congrTheoremExprs_196_);
v_a_228_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_206_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_206_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_228_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0___boxed(lean_object* v_congrTheoremExprs_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0(v_congrTheoremExprs_236_, v___y_237_, v___y_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg(lean_object* v_x_247_, lean_object* v_x_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
if (lean_obj_tag(v_x_247_) == 0)
{
lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_254_ = l_List_reverse___redArg(v_x_248_);
v___x_255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
return v___x_255_;
}
else
{
lean_object* v_head_256_; lean_object* v_tail_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_276_; 
v_head_256_ = lean_ctor_get(v_x_247_, 0);
v_tail_257_ = lean_ctor_get(v_x_247_, 1);
v_isSharedCheck_276_ = !lean_is_exclusive(v_x_247_);
if (v_isSharedCheck_276_ == 0)
{
v___x_259_ = v_x_247_;
v_isShared_260_ = v_isSharedCheck_276_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_tail_257_);
lean_inc(v_head_256_);
lean_dec(v_x_247_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_276_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v_theoremName_261_; lean_object* v___x_262_; 
v_theoremName_261_ = lean_ctor_get(v_head_256_, 0);
lean_inc(v_theoremName_261_);
lean_dec(v_head_256_);
v___x_262_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_theoremName_261_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v_a_263_; lean_object* v___x_265_; 
v_a_263_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v___x_262_, 1);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 1, v_x_248_);
lean_ctor_set(v___x_259_, 0, v_a_263_);
v___x_265_ = v___x_259_;
goto v_reusejp_264_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_a_263_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_x_248_);
v___x_265_ = v_reuseFailAlloc_267_;
goto v_reusejp_264_;
}
v_reusejp_264_:
{
v_x_247_ = v_tail_257_;
v_x_248_ = v___x_265_;
goto _start;
}
}
else
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_275_; 
lean_del_object(v___x_259_);
lean_dec(v_tail_257_);
lean_dec(v_x_248_);
v_a_268_ = lean_ctor_get(v___x_262_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_262_);
if (v_isSharedCheck_275_ == 0)
{
v___x_270_ = v___x_262_;
v_isShared_271_ = v_isSharedCheck_275_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_262_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg___boxed(lean_object* v_x_277_, lean_object* v_x_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg(v_x_277_, v_x_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
lean_dec(v___y_280_);
lean_dec_ref(v___y_279_);
return v_res_284_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2(lean_object* v_x_285_, lean_object* v_x_286_){
_start:
{
if (lean_obj_tag(v_x_285_) == 0)
{
if (lean_obj_tag(v_x_286_) == 0)
{
uint8_t v___x_287_; 
v___x_287_ = 1;
return v___x_287_;
}
else
{
uint8_t v___x_288_; 
v___x_288_ = 0;
return v___x_288_;
}
}
else
{
if (lean_obj_tag(v_x_286_) == 0)
{
uint8_t v___x_289_; 
v___x_289_ = 0;
return v___x_289_;
}
else
{
lean_object* v_head_290_; lean_object* v_tail_291_; lean_object* v_head_292_; lean_object* v_tail_293_; uint8_t v___x_294_; 
v_head_290_ = lean_ctor_get(v_x_285_, 0);
v_tail_291_ = lean_ctor_get(v_x_285_, 1);
v_head_292_ = lean_ctor_get(v_x_286_, 0);
v_tail_293_ = lean_ctor_get(v_x_286_, 1);
v___x_294_ = lean_expr_eqv(v_head_290_, v_head_292_);
if (v___x_294_ == 0)
{
return v___x_294_;
}
else
{
v_x_285_ = v_tail_291_;
v_x_286_ = v_tail_293_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2___boxed(lean_object* v_x_296_, lean_object* v_x_297_){
_start:
{
uint8_t v_res_298_; lean_object* v_r_299_; 
v_res_298_ = lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2(v_x_296_, v_x_297_);
lean_dec(v_x_297_);
lean_dec(v_x_296_);
v_r_299_ = lean_box(v_res_298_);
return v_r_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(lean_object* v_msg_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_ref_306_; lean_object* v___x_307_; lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_316_; 
v_ref_306_ = lean_ctor_get(v___y_303_, 5);
v___x_307_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3_spec__4(v_msg_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
v_a_308_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_316_ == 0)
{
v___x_310_ = v___x_307_;
v_isShared_311_ = v_isSharedCheck_316_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_316_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v___x_314_; 
lean_inc(v_ref_306_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v_ref_306_);
lean_ctor_set(v___x_312_, 1, v_a_308_);
if (v_isShared_311_ == 0)
{
lean_ctor_set_tag(v___x_310_, 1);
lean_ctor_set(v___x_310_, 0, v___x_312_);
v___x_314_ = v___x_310_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v___x_312_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg___boxed(lean_object* v_msg_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(v_msg_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
return v_res_323_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__0));
v___x_326_ = l_Lean_stringToMessageData(v___x_325_);
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__2));
v___x_329_ = l_Lean_stringToMessageData(v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr(lean_object* v_q_330_, lean_object* v_a_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_){
_start:
{
lean_object* v_congrTheoremExprs_341_; lean_object* v___y_342_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; lean_object* v_a_357_; lean_object* v___x_386_; 
v___x_386_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v_a_332_, v_a_335_, v_a_336_, v_a_337_, v_a_338_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_object* v_a_387_; lean_object* v___x_388_; lean_object* v_a_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v_a_387_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_a_387_);
lean_dec_ref_known(v___x_386_, 1);
v___x_388_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_Tactic_applyCongr_spec__5___redArg(v_a_387_, v_a_336_);
v_a_389_ = lean_ctor_get(v___x_388_, 0);
lean_inc(v_a_389_);
lean_dec_ref(v___x_388_);
v___x_390_ = l_Lean_Expr_cleanupAnnotations(v_a_389_);
v___x_391_ = l_Lean_Expr_getAppFn(v___x_390_);
lean_dec_ref(v___x_390_);
v_a_357_ = v___x_391_;
goto v___jp_356_;
}
else
{
lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_399_; 
v_a_392_ = lean_ctor_get(v___x_386_, 0);
v_isSharedCheck_399_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_399_ == 0)
{
v___x_394_ = v___x_386_;
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_386_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_397_; 
if (v_isShared_395_ == 0)
{
v___x_397_ = v___x_394_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_a_392_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
v___jp_340_:
{
lean_object* v___f_350_; lean_object* v___x_351_; uint8_t v___x_352_; 
lean_inc(v_congrTheoremExprs_341_);
v___f_350_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_applyCongr___lam__0___boxed), 10, 1);
lean_closure_set(v___f_350_, 0, v_congrTheoremExprs_341_);
v___x_351_ = lean_box(0);
v___x_352_ = lp_mathlib_List_beq___at___00Lean_Elab_Tactic_applyCongr_spec__2(v_congrTheoremExprs_341_, v___x_351_);
lean_dec(v_congrTheoremExprs_341_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; 
v___x_353_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_350_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_353_;
}
else
{
lean_object* v___x_354_; lean_object* v___x_355_; 
lean_dec_ref(v___f_350_);
v___x_354_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1, &lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__1);
v___x_355_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(v___x_354_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_355_;
}
}
v___jp_356_:
{
if (lean_obj_tag(v_a_357_) == 4)
{
if (lean_obj_tag(v_q_330_) == 0)
{
lean_object* v_declName_358_; lean_object* v___x_359_; 
v_declName_358_ = lean_ctor_get(v_a_357_, 0);
lean_inc(v_declName_358_);
lean_dec_ref_known(v_a_357_, 2);
v___x_359_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_338_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc(v_a_360_);
lean_dec_ref_known(v___x_359_, 1);
v___x_361_ = l_Lean_Meta_SimpCongrTheorems_get(v_a_360_, v_declName_358_);
lean_dec(v_declName_358_);
lean_dec(v_a_360_);
v___x_362_ = lean_box(0);
v___x_363_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg(v___x_361_, v___x_362_, v_a_335_, v_a_336_, v_a_337_, v_a_338_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_a_364_);
lean_dec_ref_known(v___x_363_, 1);
v_congrTheoremExprs_341_ = v_a_364_;
v___y_342_ = v_a_331_;
v___y_343_ = v_a_332_;
v___y_344_ = v_a_333_;
v___y_345_ = v_a_334_;
v___y_346_ = v_a_335_;
v___y_347_ = v_a_336_;
v___y_348_ = v_a_337_;
v___y_349_ = v_a_338_;
goto v___jp_340_;
}
else
{
lean_object* v_a_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_372_; 
v_a_365_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_372_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_372_ == 0)
{
v___x_367_ = v___x_363_;
v_isShared_368_ = v_isSharedCheck_372_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_a_365_);
lean_dec(v___x_363_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_372_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
lean_object* v___x_370_; 
if (v_isShared_368_ == 0)
{
v___x_370_ = v___x_367_;
goto v_reusejp_369_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v_a_365_);
v___x_370_ = v_reuseFailAlloc_371_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
return v___x_370_;
}
}
}
}
else
{
lean_object* v_a_373_; lean_object* v___x_375_; uint8_t v_isShared_376_; uint8_t v_isSharedCheck_380_; 
lean_dec(v_declName_358_);
v_a_373_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_380_ == 0)
{
v___x_375_ = v___x_359_;
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
else
{
lean_inc(v_a_373_);
lean_dec(v___x_359_);
v___x_375_ = lean_box(0);
v_isShared_376_ = v_isSharedCheck_380_;
goto v_resetjp_374_;
}
v_resetjp_374_:
{
lean_object* v___x_378_; 
if (v_isShared_376_ == 0)
{
v___x_378_ = v___x_375_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v_a_373_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
return v___x_378_;
}
}
}
}
else
{
lean_object* v_val_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
lean_dec_ref_known(v_a_357_, 2);
v_val_381_ = lean_ctor_get(v_q_330_, 0);
v___x_382_ = lean_box(0);
lean_inc(v_val_381_);
v___x_383_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_383_, 0, v_val_381_);
lean_ctor_set(v___x_383_, 1, v___x_382_);
v_congrTheoremExprs_341_ = v___x_383_;
v___y_342_ = v_a_331_;
v___y_343_ = v_a_332_;
v___y_344_ = v_a_333_;
v___y_345_ = v_a_334_;
v___y_346_ = v_a_335_;
v___y_347_ = v_a_336_;
v___y_348_ = v_a_337_;
v___y_349_ = v_a_338_;
goto v___jp_340_;
}
}
else
{
lean_object* v___x_384_; lean_object* v___x_385_; 
lean_dec_ref(v_a_357_);
v___x_384_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3, &lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3_once, _init_lp_mathlib_Lean_Elab_Tactic_applyCongr___closed__3);
v___x_385_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(v___x_384_, v_a_335_, v_a_336_, v_a_337_, v_a_338_);
return v___x_385_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_applyCongr___boxed(lean_object* v_q_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_, lean_object* v_a_406_, lean_object* v_a_407_, lean_object* v_a_408_, lean_object* v_a_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_Lean_Elab_Tactic_applyCongr(v_q_400_, v_a_401_, v_a_402_, v_a_403_, v_a_404_, v_a_405_, v_a_406_, v_a_407_, v_a_408_);
lean_dec(v_a_408_);
lean_dec_ref(v_a_407_);
lean_dec(v_a_406_);
lean_dec_ref(v_a_405_);
lean_dec(v_a_404_);
lean_dec_ref(v_a_403_);
lean_dec(v_a_402_);
lean_dec_ref(v_a_401_);
lean_dec(v_q_400_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3(lean_object* v_00_u03b1_411_, lean_object* v_msg_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___redArg(v_msg_412_, v___y_417_, v___y_418_, v___y_419_, v___y_420_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3___boxed(lean_object* v_00_u03b1_423_, lean_object* v_msg_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic_applyCongr_spec__3(v_00_u03b1_423_, v_msg_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_);
lean_dec(v___y_432_);
lean_dec_ref(v___y_431_);
lean_dec(v___y_430_);
lean_dec_ref(v___y_429_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4(lean_object* v_x_435_, lean_object* v_x_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___redArg(v_x_435_, v_x_436_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4___boxed(lean_object* v_x_447_, lean_object* v_x_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_List_mapM_loop___at___00Lean_Elab_Tactic_applyCongr_spec__4(v_x_447_, v_x_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_);
lean_dec(v___y_456_);
lean_dec_ref(v___y_455_);
lean_dec(v___y_454_);
lean_dec_ref(v___y_453_);
lean_dec(v___y_452_);
lean_dec_ref(v___y_451_);
lean_dec(v___y_450_);
lean_dec_ref(v___y_449_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1(lean_object* v_00_u03b1_459_, lean_object* v_msg_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___redArg(v_msg_460_, v___y_461_, v___y_462_, v___y_463_, v___y_464_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1___boxed(lean_object* v_00_u03b1_467_, lean_object* v_msg_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_mathlib_Lean_throwError___at___00List_firstM___at___00Lean_Elab_Tactic_applyCongr_spec__1_spec__1(v_00_u03b1_467_, v_msg_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
return v_res_474_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_530_ = lean_box(0);
v___x_531_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_532_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_531_);
lean_ctor_set(v___x_532_, 1, v___x_530_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg(){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_534_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___closed__0);
v___x_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_535_, 0, v___x_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg___boxed(lean_object* v___y_536_){
_start:
{
lean_object* v_res_537_; 
v_res_537_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg();
return v_res_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0(lean_object* v_00_u03b1_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg();
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___boxed(lean_object* v_00_u03b1_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0(v_00_u03b1_549_, v___y_550_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec(v___y_557_);
lean_dec_ref(v___y_556_);
lean_dec(v___y_555_);
lean_dec_ref(v___y_554_);
lean_dec(v___y_553_);
lean_dec_ref(v___y_552_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1(lean_object* v_x_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_){
_start:
{
lean_object* v___x_570_; uint8_t v___x_571_; 
v___x_570_ = ((lean_object*)(lp_mathlib_Lean_Parser_Tactic_applyCongr___closed__4));
lean_inc(v_x_560_);
v___x_571_ = l_Lean_Syntax_isOfKind(v_x_560_, v___x_570_);
if (v___x_571_ == 0)
{
lean_object* v___x_572_; 
lean_dec(v_x_560_);
v___x_572_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg();
return v___x_572_;
}
else
{
lean_object* v___x_573_; lean_object* v___x_574_; uint8_t v___x_575_; 
v___x_573_ = lean_unsigned_to_nat(1u);
v___x_574_ = l_Lean_Syntax_getArg(v_x_560_, v___x_573_);
lean_dec(v_x_560_);
v___x_575_ = l_Lean_Syntax_isNone(v___x_574_);
if (v___x_575_ == 0)
{
uint8_t v___x_576_; 
lean_inc(v___x_574_);
v___x_576_ = l_Lean_Syntax_matchesNull(v___x_574_, v___x_573_);
if (v___x_576_ == 0)
{
lean_object* v___x_577_; 
lean_dec(v___x_574_);
v___x_577_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1_spec__0___redArg();
return v___x_577_;
}
else
{
lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_578_ = lean_unsigned_to_nat(0u);
v___x_579_ = l_Lean_Syntax_getArg(v___x_574_, v___x_578_);
lean_dec(v___x_574_);
v___x_580_ = lean_box(0);
v___x_581_ = l_Lean_Elab_Tactic_elabTerm(v___x_579_, v___x_580_, v___x_575_, v_a_561_, v_a_562_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v_a_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v_a_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_a_582_);
lean_dec_ref_known(v___x_581_, 1);
v___x_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_583_, 0, v_a_582_);
v___x_584_ = lp_mathlib_Lean_Elab_Tactic_applyCongr(v___x_583_, v_a_561_, v_a_562_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_);
lean_dec_ref_known(v___x_583_, 1);
return v___x_584_;
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
v_a_585_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_581_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_581_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
}
else
{
lean_object* v___x_593_; lean_object* v___x_594_; 
lean_dec(v___x_574_);
v___x_593_ = lean_box(0);
v___x_594_ = lp_mathlib_Lean_Elab_Tactic_applyCongr(v___x_593_, v_a_561_, v_a_562_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_);
return v___x_594_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1___boxed(lean_object* v_x_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib___aux__Mathlib__Tactic__ApplyCongr______elabRules__Lean__Parser__Tactic__applyCongr__1(v_x_595_, v_a_596_, v_a_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_);
lean_dec(v_a_603_);
lean_dec_ref(v_a_602_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec_ref(v_a_598_);
lean_dec(v_a_597_);
lean_dec_ref(v_a_596_);
return v_res_605_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ApplyCongr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ApplyCongr(builtin);
}
#ifdef __cplusplus
}
#endif
