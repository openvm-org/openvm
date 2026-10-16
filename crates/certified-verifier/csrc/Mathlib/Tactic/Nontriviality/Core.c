// Lean compiler output
// Module: Mathlib.Tactic.Nontriviality.Core
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Meta public meta import Lean.Elab.Tactic.SolveByElim public import Qq.Macro public import Qq.Typ public meta import Qq.MetaM public import Mathlib.Basic.Nontrivial.Basic public import Mathlib.Tactic.Attr.Register
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
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getSepArgs(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isProp(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
extern lean_object* l_Lean_firstFrontendMacroScope;
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_runTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_dec(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_inferInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SolveByElim_processSyntax(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_assert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_instInhabitedTacticM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpArg;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Could not prove goal assuming `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "UnhygienicMain"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(124, 169, 242, 144, 140, 56, 85, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nontriviality"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(39, 78, 96, 91, 103, 133, 39, 103)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__24_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Nontrivial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(122, 234, 164, 90, 175, 175, 198, 2)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Nontriviality"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "subsingleton_or_nontrivial_elim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(141, 49, 118, 182, 214, 151, 120, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(221, 41, 173, 204, 121, 45, 138, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Subsingleton"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 130, 42, 228, 248, 162, 23, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "nontrivial_of_ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 153, 69, 3, 55, 144, 133, 89)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "nontrivial_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__7_value),LEAN_SCALAR_PTR_LITERAL(57, 185, 198, 64, 223, 49, 238, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__11_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(141, 49, 118, 182, 214, 151, 120, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(65, 198, 73, 103, 94, 100, 131, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__9_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__13_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Tactic_instInhabitedTacticM___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 128, .m_capacity = 128, .m_length = 125, .m_data = "The goal is not an (in)equality, so you'll need to specify the desired `Nontrivial α` instance by invoking `nontriviality α`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "inst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__0_value),LEAN_SCALAR_PTR_LITERAL(170, 188, 240, 205, 110, 63, 170, 91)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not a type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Mathlib.Tactic.Nontriviality.Core"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "Mathlib.Tactic.Nontriviality.elabNontriviality"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg(v_e_30_, v___y_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___boxed(lean_object* v_e_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0(v_e_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(lean_object* v_mvarId_44_, lean_object* v_x_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_44_, v_x_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
v_a_52_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_51_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_51_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
else
{
lean_object* v_a_60_; lean_object* v___x_62_; uint8_t v_isShared_63_; uint8_t v_isSharedCheck_67_; 
v_a_60_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_67_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_67_ == 0)
{
v___x_62_ = v___x_51_;
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
else
{
lean_inc(v_a_60_);
lean_dec(v___x_51_);
v___x_62_ = lean_box(0);
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
v_resetjp_61_:
{
lean_object* v___x_65_; 
if (v_isShared_63_ == 0)
{
v___x_65_ = v___x_62_;
goto v_reusejp_64_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v_a_60_);
v___x_65_ = v_reuseFailAlloc_66_;
goto v_reusejp_64_;
}
v_reusejp_64_:
{
return v___x_65_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg___boxed(lean_object* v_mvarId_68_, lean_object* v_x_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(v_mvarId_68_, v_x_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3(lean_object* v_00_u03b1_76_, lean_object* v_mvarId_77_, lean_object* v_x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(v_mvarId_77_, v_x_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___boxed(lean_object* v_00_u03b1_85_, lean_object* v_mvarId_86_, lean_object* v_x_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3(v_00_u03b1_85_, v_mvarId_86_, v_x_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
return v_res_93_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0(uint8_t v___y_94_, lean_object* v_x_95_){
_start:
{
return v___y_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0___boxed(lean_object* v___y_96_, lean_object* v_x_97_){
_start:
{
uint8_t v___y_13633__boxed_98_; uint8_t v_res_99_; lean_object* v_r_100_; 
v___y_13633__boxed_98_ = lean_unbox(v___y_96_);
v_res_99_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0(v___y_13633__boxed_98_, v_x_97_);
lean_dec(v_x_97_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1(lean_object* v_msgData_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v___x_107_; lean_object* v_env_108_; lean_object* v___x_109_; lean_object* v_mctx_110_; lean_object* v_lctx_111_; lean_object* v_options_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_107_ = lean_st_ref_get(v___y_105_);
v_env_108_ = lean_ctor_get(v___x_107_, 0);
lean_inc_ref(v_env_108_);
lean_dec(v___x_107_);
v___x_109_ = lean_st_ref_get(v___y_103_);
v_mctx_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc_ref(v_mctx_110_);
lean_dec(v___x_109_);
v_lctx_111_ = lean_ctor_get(v___y_102_, 2);
v_options_112_ = lean_ctor_get(v___y_104_, 2);
lean_inc_ref(v_options_112_);
lean_inc_ref(v_lctx_111_);
v___x_113_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_113_, 0, v_env_108_);
lean_ctor_set(v___x_113_, 1, v_mctx_110_);
lean_ctor_set(v___x_113_, 2, v_lctx_111_);
lean_ctor_set(v___x_113_, 3, v_options_112_);
v___x_114_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_msgData_101_);
v___x_115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1___boxed(lean_object* v_msgData_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1(v_msgData_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(lean_object* v_msg_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_){
_start:
{
lean_object* v_ref_129_; lean_object* v___x_130_; lean_object* v_a_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_139_; 
v_ref_129_ = lean_ctor_get(v___y_126_, 5);
v___x_130_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1(v_msg_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
v_a_131_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_139_ == 0)
{
v___x_133_ = v___x_130_;
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_a_131_);
lean_dec(v___x_130_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_135_; lean_object* v___x_137_; 
lean_inc(v_ref_129_);
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v_ref_129_);
lean_ctor_set(v___x_135_, 1, v_a_131_);
if (v_isShared_134_ == 0)
{
lean_ctor_set_tag(v___x_133_, 1);
lean_ctor_set(v___x_133_, 0, v___x_135_);
v___x_137_ = v___x_133_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_135_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg___boxed(lean_object* v_msg_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v_msg_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7___redArg(lean_object* v_x_147_, lean_object* v_x_148_, lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
lean_object* v_ks_151_; lean_object* v_vs_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_176_; 
v_ks_151_ = lean_ctor_get(v_x_147_, 0);
v_vs_152_ = lean_ctor_get(v_x_147_, 1);
v_isSharedCheck_176_ = !lean_is_exclusive(v_x_147_);
if (v_isSharedCheck_176_ == 0)
{
v___x_154_ = v_x_147_;
v_isShared_155_ = v_isSharedCheck_176_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_vs_152_);
lean_inc(v_ks_151_);
lean_dec(v_x_147_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_176_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_156_ = lean_array_get_size(v_ks_151_);
v___x_157_ = lean_nat_dec_lt(v_x_148_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_161_; 
lean_dec(v_x_148_);
v___x_158_ = lean_array_push(v_ks_151_, v_x_149_);
v___x_159_ = lean_array_push(v_vs_152_, v_x_150_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v___x_159_);
lean_ctor_set(v___x_154_, 0, v___x_158_);
v___x_161_ = v___x_154_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_158_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v___x_159_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
else
{
lean_object* v_k_x27_163_; uint8_t v___x_164_; 
v_k_x27_163_ = lean_array_fget_borrowed(v_ks_151_, v_x_148_);
v___x_164_ = l_Lean_instBEqMVarId_beq(v_x_149_, v_k_x27_163_);
if (v___x_164_ == 0)
{
lean_object* v___x_166_; 
if (v_isShared_155_ == 0)
{
v___x_166_ = v___x_154_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_170_; 
v_reuseFailAlloc_170_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_170_, 0, v_ks_151_);
lean_ctor_set(v_reuseFailAlloc_170_, 1, v_vs_152_);
v___x_166_ = v_reuseFailAlloc_170_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_167_ = lean_unsigned_to_nat(1u);
v___x_168_ = lean_nat_add(v_x_148_, v___x_167_);
lean_dec(v_x_148_);
v_x_147_ = v___x_166_;
v_x_148_ = v___x_168_;
goto _start;
}
}
else
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_174_; 
v___x_171_ = lean_array_fset(v_ks_151_, v_x_148_, v_x_149_);
v___x_172_ = lean_array_fset(v_vs_152_, v_x_148_, v_x_150_);
lean_dec(v_x_148_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v___x_172_);
lean_ctor_set(v___x_154_, 0, v___x_171_);
v___x_174_ = v___x_154_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v___x_171_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v___x_172_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6___redArg(lean_object* v_n_177_, lean_object* v_k_178_, lean_object* v_v_179_){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = lean_unsigned_to_nat(0u);
v___x_181_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7___redArg(v_n_177_, v___x_180_, v_k_178_, v_v_179_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(lean_object* v_x_183_, size_t v_x_184_, size_t v_x_185_, lean_object* v_x_186_, lean_object* v_x_187_){
_start:
{
if (lean_obj_tag(v_x_183_) == 0)
{
lean_object* v_es_188_; size_t v___x_189_; size_t v___x_190_; lean_object* v_j_191_; lean_object* v___x_192_; uint8_t v___x_193_; 
v_es_188_ = lean_ctor_get(v_x_183_, 0);
v___x_189_ = ((size_t)31ULL);
v___x_190_ = lean_usize_land(v_x_184_, v___x_189_);
v_j_191_ = lean_usize_to_nat(v___x_190_);
v___x_192_ = lean_array_get_size(v_es_188_);
v___x_193_ = lean_nat_dec_lt(v_j_191_, v___x_192_);
if (v___x_193_ == 0)
{
lean_dec(v_j_191_);
lean_dec(v_x_187_);
lean_dec(v_x_186_);
return v_x_183_;
}
else
{
lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_232_; 
lean_inc_ref(v_es_188_);
v_isSharedCheck_232_ = !lean_is_exclusive(v_x_183_);
if (v_isSharedCheck_232_ == 0)
{
lean_object* v_unused_233_; 
v_unused_233_ = lean_ctor_get(v_x_183_, 0);
lean_dec(v_unused_233_);
v___x_195_ = v_x_183_;
v_isShared_196_ = v_isSharedCheck_232_;
goto v_resetjp_194_;
}
else
{
lean_dec(v_x_183_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_232_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v_v_197_; lean_object* v___x_198_; lean_object* v_xs_x27_199_; lean_object* v___y_201_; 
v_v_197_ = lean_array_fget(v_es_188_, v_j_191_);
v___x_198_ = lean_box(0);
v_xs_x27_199_ = lean_array_fset(v_es_188_, v_j_191_, v___x_198_);
switch(lean_obj_tag(v_v_197_))
{
case 0:
{
lean_object* v_key_206_; lean_object* v_val_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_217_; 
v_key_206_ = lean_ctor_get(v_v_197_, 0);
v_val_207_ = lean_ctor_get(v_v_197_, 1);
v_isSharedCheck_217_ = !lean_is_exclusive(v_v_197_);
if (v_isSharedCheck_217_ == 0)
{
v___x_209_ = v_v_197_;
v_isShared_210_ = v_isSharedCheck_217_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_val_207_);
lean_inc(v_key_206_);
lean_dec(v_v_197_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_217_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
uint8_t v___x_211_; 
v___x_211_ = l_Lean_instBEqMVarId_beq(v_x_186_, v_key_206_);
if (v___x_211_ == 0)
{
lean_object* v___x_212_; lean_object* v___x_213_; 
lean_del_object(v___x_209_);
v___x_212_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_206_, v_val_207_, v_x_186_, v_x_187_);
v___x_213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_213_, 0, v___x_212_);
v___y_201_ = v___x_213_;
goto v___jp_200_;
}
else
{
lean_object* v___x_215_; 
lean_dec(v_val_207_);
lean_dec(v_key_206_);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 1, v_x_187_);
lean_ctor_set(v___x_209_, 0, v_x_186_);
v___x_215_ = v___x_209_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_x_186_);
lean_ctor_set(v_reuseFailAlloc_216_, 1, v_x_187_);
v___x_215_ = v_reuseFailAlloc_216_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
v___y_201_ = v___x_215_;
goto v___jp_200_;
}
}
}
}
case 1:
{
lean_object* v_node_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_230_; 
v_node_218_ = lean_ctor_get(v_v_197_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v_v_197_);
if (v_isSharedCheck_230_ == 0)
{
v___x_220_ = v_v_197_;
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_node_218_);
lean_dec(v_v_197_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
size_t v___x_222_; size_t v___x_223_; size_t v___x_224_; size_t v___x_225_; lean_object* v___x_226_; lean_object* v___x_228_; 
v___x_222_ = ((size_t)5ULL);
v___x_223_ = lean_usize_shift_right(v_x_184_, v___x_222_);
v___x_224_ = ((size_t)1ULL);
v___x_225_ = lean_usize_add(v_x_185_, v___x_224_);
v___x_226_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(v_node_218_, v___x_223_, v___x_225_, v_x_186_, v_x_187_);
if (v_isShared_221_ == 0)
{
lean_ctor_set(v___x_220_, 0, v___x_226_);
v___x_228_ = v___x_220_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v___x_226_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
v___y_201_ = v___x_228_;
goto v___jp_200_;
}
}
}
default: 
{
lean_object* v___x_231_; 
v___x_231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_231_, 0, v_x_186_);
lean_ctor_set(v___x_231_, 1, v_x_187_);
v___y_201_ = v___x_231_;
goto v___jp_200_;
}
}
v___jp_200_:
{
lean_object* v___x_202_; lean_object* v___x_204_; 
v___x_202_ = lean_array_fset(v_xs_x27_199_, v_j_191_, v___y_201_);
lean_dec(v_j_191_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 0, v___x_202_);
v___x_204_ = v___x_195_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v___x_202_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
}
}
else
{
lean_object* v_ks_234_; lean_object* v_vs_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_255_; 
v_ks_234_ = lean_ctor_get(v_x_183_, 0);
v_vs_235_ = lean_ctor_get(v_x_183_, 1);
v_isSharedCheck_255_ = !lean_is_exclusive(v_x_183_);
if (v_isSharedCheck_255_ == 0)
{
v___x_237_ = v_x_183_;
v_isShared_238_ = v_isSharedCheck_255_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_vs_235_);
lean_inc(v_ks_234_);
lean_dec(v_x_183_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_255_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_ks_234_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v_vs_235_);
v___x_240_ = v_reuseFailAlloc_254_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
lean_object* v_newNode_241_; uint8_t v___y_243_; size_t v___x_249_; uint8_t v___x_250_; 
v_newNode_241_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6___redArg(v___x_240_, v_x_186_, v_x_187_);
v___x_249_ = ((size_t)7ULL);
v___x_250_ = lean_usize_dec_le(v___x_249_, v_x_185_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_251_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_241_);
v___x_252_ = lean_unsigned_to_nat(4u);
v___x_253_ = lean_nat_dec_lt(v___x_251_, v___x_252_);
lean_dec(v___x_251_);
v___y_243_ = v___x_253_;
goto v___jp_242_;
}
else
{
v___y_243_ = v___x_250_;
goto v___jp_242_;
}
v___jp_242_:
{
if (v___y_243_ == 0)
{
lean_object* v_ks_244_; lean_object* v_vs_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v_ks_244_ = lean_ctor_get(v_newNode_241_, 0);
lean_inc_ref(v_ks_244_);
v_vs_245_ = lean_ctor_get(v_newNode_241_, 1);
lean_inc_ref(v_vs_245_);
lean_dec_ref(v_newNode_241_);
v___x_246_ = lean_unsigned_to_nat(0u);
v___x_247_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___closed__0);
v___x_248_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg(v_x_185_, v_ks_244_, v_vs_245_, v___x_246_, v___x_247_);
lean_dec_ref(v_vs_245_);
lean_dec_ref(v_ks_244_);
return v___x_248_;
}
else
{
return v_newNode_241_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg(size_t v_depth_256_, lean_object* v_keys_257_, lean_object* v_vals_258_, lean_object* v_i_259_, lean_object* v_entries_260_){
_start:
{
lean_object* v___x_261_; uint8_t v___x_262_; 
v___x_261_ = lean_array_get_size(v_keys_257_);
v___x_262_ = lean_nat_dec_lt(v_i_259_, v___x_261_);
if (v___x_262_ == 0)
{
lean_dec(v_i_259_);
return v_entries_260_;
}
else
{
lean_object* v_k_263_; lean_object* v_v_264_; uint64_t v___x_265_; size_t v_h_266_; size_t v___x_267_; lean_object* v___x_268_; size_t v___x_269_; size_t v___x_270_; size_t v___x_271_; size_t v_h_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v_k_263_ = lean_array_fget_borrowed(v_keys_257_, v_i_259_);
v_v_264_ = lean_array_fget_borrowed(v_vals_258_, v_i_259_);
v___x_265_ = l_Lean_instHashableMVarId_hash(v_k_263_);
v_h_266_ = lean_uint64_to_usize(v___x_265_);
v___x_267_ = ((size_t)5ULL);
v___x_268_ = lean_unsigned_to_nat(1u);
v___x_269_ = ((size_t)1ULL);
v___x_270_ = lean_usize_sub(v_depth_256_, v___x_269_);
v___x_271_ = lean_usize_mul(v___x_267_, v___x_270_);
v_h_272_ = lean_usize_shift_right(v_h_266_, v___x_271_);
v___x_273_ = lean_nat_add(v_i_259_, v___x_268_);
lean_dec(v_i_259_);
lean_inc(v_v_264_);
lean_inc(v_k_263_);
v___x_274_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(v_entries_260_, v_h_272_, v_depth_256_, v_k_263_, v_v_264_);
v_i_259_ = v___x_273_;
v_entries_260_ = v___x_274_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg___boxed(lean_object* v_depth_276_, lean_object* v_keys_277_, lean_object* v_vals_278_, lean_object* v_i_279_, lean_object* v_entries_280_){
_start:
{
size_t v_depth_boxed_281_; lean_object* v_res_282_; 
v_depth_boxed_281_ = lean_unbox_usize(v_depth_276_);
lean_dec(v_depth_276_);
v_res_282_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg(v_depth_boxed_281_, v_keys_277_, v_vals_278_, v_i_279_, v_entries_280_);
lean_dec_ref(v_vals_278_);
lean_dec_ref(v_keys_277_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_x_283_, lean_object* v_x_284_, lean_object* v_x_285_, lean_object* v_x_286_, lean_object* v_x_287_){
_start:
{
size_t v_x_13777__boxed_288_; size_t v_x_13778__boxed_289_; lean_object* v_res_290_; 
v_x_13777__boxed_288_ = lean_unbox_usize(v_x_284_);
lean_dec(v_x_284_);
v_x_13778__boxed_289_ = lean_unbox_usize(v_x_285_);
lean_dec(v_x_285_);
v_res_290_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(v_x_283_, v_x_13777__boxed_288_, v_x_13778__boxed_289_, v_x_286_, v_x_287_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3___redArg(lean_object* v_x_291_, lean_object* v_x_292_, lean_object* v_x_293_){
_start:
{
uint64_t v___x_294_; size_t v___x_295_; size_t v___x_296_; lean_object* v___x_297_; 
v___x_294_ = l_Lean_instHashableMVarId_hash(v_x_292_);
v___x_295_ = lean_uint64_to_usize(v___x_294_);
v___x_296_ = ((size_t)1ULL);
v___x_297_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(v_x_291_, v___x_295_, v___x_296_, v_x_292_, v_x_293_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(lean_object* v_mvarId_298_, lean_object* v_val_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___x_302_; lean_object* v_mctx_303_; lean_object* v_cache_304_; lean_object* v_zetaDeltaFVarIds_305_; lean_object* v_postponed_306_; lean_object* v_diag_307_; lean_object* v___x_309_; uint8_t v_isShared_310_; uint8_t v_isSharedCheck_335_; 
v___x_302_ = lean_st_ref_take(v___y_300_);
v_mctx_303_ = lean_ctor_get(v___x_302_, 0);
v_cache_304_ = lean_ctor_get(v___x_302_, 1);
v_zetaDeltaFVarIds_305_ = lean_ctor_get(v___x_302_, 2);
v_postponed_306_ = lean_ctor_get(v___x_302_, 3);
v_diag_307_ = lean_ctor_get(v___x_302_, 4);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_335_ == 0)
{
v___x_309_ = v___x_302_;
v_isShared_310_ = v_isSharedCheck_335_;
goto v_resetjp_308_;
}
else
{
lean_inc(v_diag_307_);
lean_inc(v_postponed_306_);
lean_inc(v_zetaDeltaFVarIds_305_);
lean_inc(v_cache_304_);
lean_inc(v_mctx_303_);
lean_dec(v___x_302_);
v___x_309_ = lean_box(0);
v_isShared_310_ = v_isSharedCheck_335_;
goto v_resetjp_308_;
}
v_resetjp_308_:
{
lean_object* v_depth_311_; lean_object* v_levelAssignDepth_312_; lean_object* v_lmvarCounter_313_; lean_object* v_mvarCounter_314_; lean_object* v_lDecls_315_; lean_object* v_decls_316_; lean_object* v_userNames_317_; lean_object* v_lAssignment_318_; lean_object* v_eAssignment_319_; lean_object* v_dAssignment_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_334_; 
v_depth_311_ = lean_ctor_get(v_mctx_303_, 0);
v_levelAssignDepth_312_ = lean_ctor_get(v_mctx_303_, 1);
v_lmvarCounter_313_ = lean_ctor_get(v_mctx_303_, 2);
v_mvarCounter_314_ = lean_ctor_get(v_mctx_303_, 3);
v_lDecls_315_ = lean_ctor_get(v_mctx_303_, 4);
v_decls_316_ = lean_ctor_get(v_mctx_303_, 5);
v_userNames_317_ = lean_ctor_get(v_mctx_303_, 6);
v_lAssignment_318_ = lean_ctor_get(v_mctx_303_, 7);
v_eAssignment_319_ = lean_ctor_get(v_mctx_303_, 8);
v_dAssignment_320_ = lean_ctor_get(v_mctx_303_, 9);
v_isSharedCheck_334_ = !lean_is_exclusive(v_mctx_303_);
if (v_isSharedCheck_334_ == 0)
{
v___x_322_ = v_mctx_303_;
v_isShared_323_ = v_isSharedCheck_334_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_dAssignment_320_);
lean_inc(v_eAssignment_319_);
lean_inc(v_lAssignment_318_);
lean_inc(v_userNames_317_);
lean_inc(v_decls_316_);
lean_inc(v_lDecls_315_);
lean_inc(v_mvarCounter_314_);
lean_inc(v_lmvarCounter_313_);
lean_inc(v_levelAssignDepth_312_);
lean_inc(v_depth_311_);
lean_dec(v_mctx_303_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_334_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_324_; lean_object* v___x_326_; 
v___x_324_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3___redArg(v_eAssignment_319_, v_mvarId_298_, v_val_299_);
if (v_isShared_323_ == 0)
{
lean_ctor_set(v___x_322_, 8, v___x_324_);
v___x_326_ = v___x_322_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_depth_311_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_levelAssignDepth_312_);
lean_ctor_set(v_reuseFailAlloc_333_, 2, v_lmvarCounter_313_);
lean_ctor_set(v_reuseFailAlloc_333_, 3, v_mvarCounter_314_);
lean_ctor_set(v_reuseFailAlloc_333_, 4, v_lDecls_315_);
lean_ctor_set(v_reuseFailAlloc_333_, 5, v_decls_316_);
lean_ctor_set(v_reuseFailAlloc_333_, 6, v_userNames_317_);
lean_ctor_set(v_reuseFailAlloc_333_, 7, v_lAssignment_318_);
lean_ctor_set(v_reuseFailAlloc_333_, 8, v___x_324_);
lean_ctor_set(v_reuseFailAlloc_333_, 9, v_dAssignment_320_);
v___x_326_ = v_reuseFailAlloc_333_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
lean_object* v___x_328_; 
if (v_isShared_310_ == 0)
{
lean_ctor_set(v___x_309_, 0, v___x_326_);
v___x_328_ = v___x_309_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___x_326_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v_cache_304_);
lean_ctor_set(v_reuseFailAlloc_332_, 2, v_zetaDeltaFVarIds_305_);
lean_ctor_set(v_reuseFailAlloc_332_, 3, v_postponed_306_);
lean_ctor_set(v_reuseFailAlloc_332_, 4, v_diag_307_);
v___x_328_ = v_reuseFailAlloc_332_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_329_ = lean_st_ref_set(v___y_300_, v___x_328_);
v___x_330_ = lean_box(0);
v___x_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
return v___x_331_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg___boxed(lean_object* v_mvarId_336_, lean_object* v_val_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(v_mvarId_336_, v_val_337_, v___y_338_);
lean_dec(v___y_338_);
return v_res_340_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__0));
v___x_343_ = l_Lean_stringToMessageData(v___x_342_);
return v___x_343_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__2));
v___x_346_ = l_Lean_stringToMessageData(v___x_345_);
return v___x_346_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13(void){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = l_Array_mkArray0(lean_box(0));
return v___x_362_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15(void){
_start:
{
lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__14));
v___x_365_ = l_String_toRawSubstring_x27(v___x_364_);
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_368_ = l_Lean_firstFrontendMacroScope;
v___x_369_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__16));
v___x_370_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__5));
v___x_371_ = l_Lean_addMacroScope(v___x_370_, v___x_369_, v___x_368_);
return v___x_371_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28(void){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_393_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__27));
v___x_394_ = l_Lean_stringToMessageData(v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1(lean_object* v___x_395_, lean_object* v_snd_396_, lean_object* v_simpArgs_397_, uint8_t v___x_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_){
_start:
{
lean_object* v___y_405_; uint8_t v___y_406_; lean_object* v___y_416_; lean_object* v_a_417_; lean_object* v___y_421_; lean_object* v___x_423_; 
v___x_423_ = l_Lean_Meta_saveState___redArg(v___y_400_, v___y_402_);
if (lean_obj_tag(v___x_423_) == 0)
{
lean_object* v_a_424_; lean_object* v___y_426_; lean_object* v___y_427_; uint8_t v___y_428_; lean_object* v___y_486_; lean_object* v_a_487_; lean_object* v___x_490_; 
v_a_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc(v_a_424_);
lean_dec_ref_known(v___x_423_, 1);
lean_inc(v_snd_396_);
v___x_490_ = l_Lean_MVarId_getType(v_snd_396_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_object* v_a_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_a_491_ = lean_ctor_get(v___x_490_, 0);
lean_inc(v_a_491_);
lean_dec_ref_known(v___x_490_, 1);
v___x_492_ = lean_box(0);
v___x_493_ = l_Lean_Meta_synthInstance(v_a_491_, v___x_492_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_495_; 
lean_dec(v_a_424_);
lean_dec_ref(v_simpArgs_397_);
lean_dec_ref(v___x_395_);
v_a_494_ = lean_ctor_get(v___x_493_, 0);
lean_inc(v_a_494_);
lean_dec_ref_known(v___x_493_, 1);
v___x_495_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(v_snd_396_, v_a_494_, v___y_400_);
return v___x_495_;
}
else
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_503_; 
v_a_496_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_503_ == 0)
{
v___x_498_ = v___x_493_;
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_493_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_501_; 
lean_inc(v_a_496_);
if (v_isShared_499_ == 0)
{
v___x_501_ = v___x_498_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v_a_496_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
v___y_486_ = v___x_501_;
v_a_487_ = v_a_496_;
goto v___jp_485_;
}
}
}
}
else
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_511_; 
v_a_504_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_511_ == 0)
{
v___x_506_ = v___x_490_;
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_490_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_511_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_509_; 
lean_inc(v_a_504_);
if (v_isShared_507_ == 0)
{
v___x_509_ = v___x_506_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_a_504_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
v___y_486_ = v___x_509_;
v_a_487_ = v_a_504_;
goto v___jp_485_;
}
}
}
v___jp_425_:
{
if (v___y_428_ == 0)
{
lean_object* v___x_429_; 
lean_dec_ref(v___y_427_);
lean_dec_ref(v___y_426_);
v___x_429_ = l_Lean_Meta_SavedState_restore___redArg(v_a_424_, v___y_400_, v___y_402_);
lean_dec(v_a_424_);
if (lean_obj_tag(v___x_429_) == 0)
{
lean_object* v___x_430_; lean_object* v___f_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
lean_dec_ref_known(v___x_429_, 1);
v___x_430_ = lean_box(v___y_428_);
v___f_431_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__0___boxed), 2, 1);
lean_closure_set(v___f_431_, 0, v___x_430_);
v___x_432_ = lean_box(0);
v___x_433_ = l_Lean_SourceInfo_fromRef(v___x_432_, v___y_428_);
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__10));
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__12));
v___x_436_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__13);
lean_inc_n(v___x_433_, 9);
v___x_437_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_437_, 0, v___x_433_);
lean_ctor_set(v___x_437_, 1, v___x_435_);
lean_ctor_set(v___x_437_, 2, v___x_436_);
v___x_438_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__15);
v___x_439_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__17);
v___x_440_ = lean_box(0);
v___x_441_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_441_, 0, v___x_433_);
lean_ctor_set(v___x_441_, 1, v___x_438_);
lean_ctor_set(v___x_441_, 2, v___x_439_);
lean_ctor_set(v___x_441_, 3, v___x_440_);
lean_inc_ref_n(v___x_437_, 5);
v___x_442_ = l_Lean_Syntax_node3(v___x_433_, v___x_434_, v___x_437_, v___x_437_, v___x_441_);
v___x_443_ = lean_array_push(v_simpArgs_397_, v___x_442_);
v___x_444_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__18));
v___x_445_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__19));
v___x_446_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_433_);
lean_ctor_set(v___x_446_, 1, v___x_444_);
v___x_447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__21));
v___x_448_ = l_Lean_Syntax_node1(v___x_433_, v___x_447_, v___x_437_);
v___x_449_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__22));
v___x_450_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_450_, 0, v___x_433_);
lean_ctor_set(v___x_450_, 1, v___x_449_);
v___x_451_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__23));
v___x_452_ = l_Lean_Syntax_SepArray_ofElems(v___x_451_, v___x_443_);
lean_dec_ref(v___x_443_);
v___x_453_ = l_Array_append___redArg(v___x_436_, v___x_452_);
lean_dec_ref(v___x_452_);
v___x_454_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_454_, 0, v___x_433_);
lean_ctor_set(v___x_454_, 1, v___x_435_);
lean_ctor_set(v___x_454_, 2, v___x_453_);
v___x_455_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__24));
v___x_456_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_433_);
lean_ctor_set(v___x_456_, 1, v___x_455_);
v___x_457_ = l_Lean_Syntax_node3(v___x_433_, v___x_435_, v___x_450_, v___x_454_, v___x_456_);
v___x_458_ = l_Lean_Syntax_node6(v___x_433_, v___x_445_, v___x_446_, v___x_448_, v___x_437_, v___x_437_, v___x_457_, v___x_437_);
v___x_459_ = lean_box(0);
v___x_460_ = lean_box(1);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__25));
v___x_462_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_462_, 0, v___x_459_);
lean_ctor_set(v___x_462_, 1, v___x_440_);
lean_ctor_set(v___x_462_, 2, v___x_459_);
lean_ctor_set(v___x_462_, 3, v___f_431_);
lean_ctor_set(v___x_462_, 4, v___x_460_);
lean_ctor_set(v___x_462_, 5, v___x_460_);
lean_ctor_set(v___x_462_, 6, v___x_459_);
lean_ctor_set(v___x_462_, 7, v___x_461_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8, v___x_398_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 1, v___x_398_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 2, v___x_398_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 3, v___x_398_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 4, v___y_428_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 5, v___y_428_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 6, v___y_428_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 7, v___y_428_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 8, v___x_398_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 9, v___y_428_);
lean_ctor_set_uint8(v___x_462_, sizeof(void*)*8 + 10, v___x_398_);
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__26));
lean_inc(v_snd_396_);
v___x_464_ = l_Lean_Elab_runTactic(v_snd_396_, v___x_458_, v___x_462_, v___x_463_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
if (lean_obj_tag(v___x_464_) == 0)
{
lean_object* v_a_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_476_; 
v_a_465_ = lean_ctor_get(v___x_464_, 0);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_476_ == 0)
{
v___x_467_ = v___x_464_;
v_isShared_468_ = v_isSharedCheck_476_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_a_465_);
lean_dec(v___x_464_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_476_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v_fst_469_; 
v_fst_469_ = lean_ctor_get(v_a_465_, 0);
lean_inc(v_fst_469_);
lean_dec(v_a_465_);
if (lean_obj_tag(v_fst_469_) == 0)
{
lean_object* v___x_470_; lean_object* v___x_472_; 
lean_dec(v_snd_396_);
lean_dec_ref(v___x_395_);
v___x_470_ = lean_box(0);
if (v_isShared_468_ == 0)
{
lean_ctor_set(v___x_467_, 0, v___x_470_);
v___x_472_ = v___x_467_;
goto v_reusejp_471_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v___x_470_);
v___x_472_ = v_reuseFailAlloc_473_;
goto v_reusejp_471_;
}
v_reusejp_471_:
{
return v___x_472_;
}
}
else
{
lean_object* v___x_474_; lean_object* v___x_475_; 
lean_dec(v_fst_469_);
lean_del_object(v___x_467_);
v___x_474_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28);
v___x_475_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v___x_474_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
v___y_421_ = v___x_475_;
goto v___jp_420_;
}
}
}
else
{
lean_object* v_a_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_484_; 
v_a_477_ = lean_ctor_get(v___x_464_, 0);
v_isSharedCheck_484_ = !lean_is_exclusive(v___x_464_);
if (v_isSharedCheck_484_ == 0)
{
v___x_479_ = v___x_464_;
v_isShared_480_ = v_isSharedCheck_484_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_a_477_);
lean_dec(v___x_464_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_484_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___x_482_; 
lean_inc(v_a_477_);
if (v_isShared_480_ == 0)
{
v___x_482_ = v___x_479_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v_a_477_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
v___y_416_ = v___x_482_;
v_a_417_ = v_a_477_;
goto v___jp_415_;
}
}
}
}
else
{
lean_dec_ref(v_simpArgs_397_);
v___y_421_ = v___x_429_;
goto v___jp_420_;
}
}
else
{
lean_dec(v_a_424_);
lean_dec_ref(v_simpArgs_397_);
v___y_416_ = v___y_427_;
v_a_417_ = v___y_426_;
goto v___jp_415_;
}
}
v___jp_485_:
{
uint8_t v___x_488_; 
v___x_488_ = l_Lean_Exception_isInterrupt(v_a_487_);
if (v___x_488_ == 0)
{
uint8_t v___x_489_; 
lean_inc_ref(v_a_487_);
v___x_489_ = l_Lean_Exception_isRuntime(v_a_487_);
v___y_426_ = v_a_487_;
v___y_427_ = v___y_486_;
v___y_428_ = v___x_489_;
goto v___jp_425_;
}
else
{
v___y_426_ = v_a_487_;
v___y_427_ = v___y_486_;
v___y_428_ = v___x_488_;
goto v___jp_425_;
}
}
}
else
{
lean_object* v_a_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_519_; 
lean_dec_ref(v_simpArgs_397_);
v_a_512_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_519_ == 0)
{
v___x_514_ = v___x_423_;
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_a_512_);
lean_dec(v___x_423_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_517_; 
lean_inc(v_a_512_);
if (v_isShared_515_ == 0)
{
v___x_517_ = v___x_514_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_a_512_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
v___y_416_ = v___x_517_;
v_a_417_ = v_a_512_;
goto v___jp_415_;
}
}
}
v___jp_404_:
{
if (v___y_406_ == 0)
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
lean_dec_ref(v___y_405_);
v___x_407_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__1);
v___x_408_ = l_Lean_MessageData_ofExpr(v___x_395_);
v___x_409_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_407_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
v___x_410_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__3);
v___x_411_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_409_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
v___x_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_412_, 0, v_snd_396_);
v___x_413_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_413_, 0, v___x_411_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
v___x_414_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v___x_413_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
return v___x_414_;
}
else
{
lean_dec(v_snd_396_);
lean_dec_ref(v___x_395_);
return v___y_405_;
}
}
v___jp_415_:
{
uint8_t v___x_418_; 
v___x_418_ = l_Lean_Exception_isInterrupt(v_a_417_);
if (v___x_418_ == 0)
{
uint8_t v___x_419_; 
v___x_419_ = l_Lean_Exception_isRuntime(v_a_417_);
v___y_405_ = v___y_416_;
v___y_406_ = v___x_419_;
goto v___jp_404_;
}
else
{
lean_dec_ref(v_a_417_);
v___y_405_ = v___y_416_;
v___y_406_ = v___x_418_;
goto v___jp_404_;
}
}
v___jp_420_:
{
if (lean_obj_tag(v___y_421_) == 0)
{
lean_dec(v_snd_396_);
lean_dec_ref(v___x_395_);
return v___y_421_;
}
else
{
lean_object* v_a_422_; 
v_a_422_ = lean_ctor_get(v___y_421_, 0);
lean_inc(v_a_422_);
v___y_416_ = v___y_421_;
v_a_417_ = v_a_422_;
goto v___jp_415_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___boxed(lean_object* v___x_520_, lean_object* v_snd_521_, lean_object* v_simpArgs_522_, lean_object* v___x_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
uint8_t v___x_14110__boxed_529_; lean_object* v_res_530_; 
v___x_14110__boxed_529_ = lean_unbox(v___x_523_);
v_res_530_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1(v___x_520_, v_snd_521_, v_simpArgs_522_, v___x_14110__boxed_529_, v___y_524_, v___y_525_, v___y_526_, v___y_527_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
return v_res_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2(lean_object* v___x_542_, uint8_t v___x_543_, lean_object* v___x_544_, lean_object* v___x_545_, lean_object* v_simpArgs_546_, uint8_t v___x_547_, lean_object* v_u_548_, lean_object* v___x_549_, lean_object* v_00_u03b1_550_, lean_object* v_a_551_, uint8_t v___x_552_, lean_object* v_g_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v___x_559_; 
lean_inc(v___x_544_);
v___x_559_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v___x_542_, v___x_543_, v___x_544_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_559_) == 0)
{
lean_object* v_a_560_; lean_object* v___x_561_; uint8_t v___x_562_; lean_object* v___x_563_; 
v_a_560_ = lean_ctor_get(v___x_559_, 0);
lean_inc(v_a_560_);
lean_dec_ref_known(v___x_559_, 1);
v___x_561_ = l_Lean_Expr_mvarId_x21(v_a_560_);
v___x_562_ = 0;
v___x_563_ = l_Lean_Meta_intro1Core(v___x_561_, v___x_562_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_563_) == 0)
{
lean_object* v_a_564_; lean_object* v_snd_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_613_; 
v_a_564_ = lean_ctor_get(v___x_563_, 0);
lean_inc(v_a_564_);
lean_dec_ref_known(v___x_563_, 1);
v_snd_565_ = lean_ctor_get(v_a_564_, 1);
v_isSharedCheck_613_ = !lean_is_exclusive(v_a_564_);
if (v_isSharedCheck_613_ == 0)
{
lean_object* v_unused_614_; 
v_unused_614_ = lean_ctor_get(v_a_564_, 0);
lean_dec(v_unused_614_);
v___x_567_ = v_a_564_;
v_isShared_568_ = v_isSharedCheck_613_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_snd_565_);
lean_dec(v_a_564_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_613_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v___x_569_; lean_object* v___f_570_; lean_object* v___x_571_; 
v___x_569_ = lean_box(v___x_547_);
lean_inc(v_snd_565_);
v___f_570_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___boxed), 9, 4);
lean_closure_set(v___f_570_, 0, v___x_545_);
lean_closure_set(v___f_570_, 1, v_snd_565_);
lean_closure_set(v___f_570_, 2, v_simpArgs_546_);
lean_closure_set(v___f_570_, 3, v___x_569_);
v___x_571_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(v_snd_565_, v___f_570_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_object* v___x_572_; lean_object* v___x_574_; 
lean_dec_ref_known(v___x_571_, 1);
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__1));
if (v_isShared_568_ == 0)
{
lean_ctor_set_tag(v___x_567_, 1);
lean_ctor_set(v___x_567_, 1, v___x_549_);
lean_ctor_set(v___x_567_, 0, v_u_548_);
v___x_574_ = v___x_567_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_u_548_);
lean_ctor_set(v_reuseFailAlloc_604_, 1, v___x_549_);
v___x_574_ = v_reuseFailAlloc_604_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; 
lean_inc_ref(v___x_574_);
v___x_575_ = l_Lean_Expr_const___override(v___x_572_, v___x_574_);
lean_inc_ref(v_00_u03b1_550_);
v___x_576_ = l_Lean_Expr_app___override(v___x_575_, v_00_u03b1_550_);
lean_inc_ref(v_a_551_);
lean_inc(v___x_544_);
v___x_577_ = l_Lean_Expr_forallE___override(v___x_544_, v___x_576_, v_a_551_, v___x_552_);
v___x_578_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v___x_577_, v___x_543_, v___x_544_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_578_) == 0)
{
lean_object* v_a_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_594_; 
v_a_579_ = lean_ctor_get(v___x_578_, 0);
lean_inc_n(v_a_579_, 2);
lean_dec_ref_known(v___x_578_, 1);
v___x_580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__5));
v___x_581_ = l_Lean_Expr_const___override(v___x_580_, v___x_574_);
v___x_582_ = l_Lean_Expr_app___override(v___x_581_, v_a_551_);
v___x_583_ = l_Lean_Expr_app___override(v___x_582_, v_00_u03b1_550_);
v___x_584_ = l_Lean_Expr_app___override(v___x_583_, v_a_560_);
v___x_585_ = l_Lean_Expr_app___override(v___x_584_, v_a_579_);
v___x_586_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(v_g_553_, v___x_585_, v___y_555_);
v_isSharedCheck_594_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_594_ == 0)
{
lean_object* v_unused_595_; 
v_unused_595_ = lean_ctor_get(v___x_586_, 0);
lean_dec(v_unused_595_);
v___x_588_ = v___x_586_;
v_isShared_589_ = v_isSharedCheck_594_;
goto v_resetjp_587_;
}
else
{
lean_dec(v___x_586_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_594_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v___x_590_; lean_object* v___x_592_; 
v___x_590_ = l_Lean_Expr_mvarId_x21(v_a_579_);
lean_dec(v_a_579_);
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v___x_590_);
v___x_592_ = v___x_588_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v___x_590_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
}
else
{
lean_object* v_a_596_; lean_object* v___x_598_; uint8_t v_isShared_599_; uint8_t v_isSharedCheck_603_; 
lean_dec_ref(v___x_574_);
lean_dec(v_a_560_);
lean_dec(v_g_553_);
lean_dec_ref(v_a_551_);
lean_dec_ref(v_00_u03b1_550_);
v_a_596_ = lean_ctor_get(v___x_578_, 0);
v_isSharedCheck_603_ = !lean_is_exclusive(v___x_578_);
if (v_isSharedCheck_603_ == 0)
{
v___x_598_ = v___x_578_;
v_isShared_599_ = v_isSharedCheck_603_;
goto v_resetjp_597_;
}
else
{
lean_inc(v_a_596_);
lean_dec(v___x_578_);
v___x_598_ = lean_box(0);
v_isShared_599_ = v_isSharedCheck_603_;
goto v_resetjp_597_;
}
v_resetjp_597_:
{
lean_object* v___x_601_; 
if (v_isShared_599_ == 0)
{
v___x_601_ = v___x_598_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_602_; 
v_reuseFailAlloc_602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_602_, 0, v_a_596_);
v___x_601_ = v_reuseFailAlloc_602_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
return v___x_601_;
}
}
}
}
}
else
{
lean_object* v_a_605_; lean_object* v___x_607_; uint8_t v_isShared_608_; uint8_t v_isSharedCheck_612_; 
lean_del_object(v___x_567_);
lean_dec(v_a_560_);
lean_dec(v_g_553_);
lean_dec_ref(v_a_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v___x_549_);
lean_dec(v_u_548_);
lean_dec(v___x_544_);
v_a_605_ = lean_ctor_get(v___x_571_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v___x_571_);
if (v_isSharedCheck_612_ == 0)
{
v___x_607_ = v___x_571_;
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
else
{
lean_inc(v_a_605_);
lean_dec(v___x_571_);
v___x_607_ = lean_box(0);
v_isShared_608_ = v_isSharedCheck_612_;
goto v_resetjp_606_;
}
v_resetjp_606_:
{
lean_object* v___x_610_; 
if (v_isShared_608_ == 0)
{
v___x_610_ = v___x_607_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v_a_605_);
v___x_610_ = v_reuseFailAlloc_611_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
return v___x_610_;
}
}
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_a_560_);
lean_dec(v_g_553_);
lean_dec_ref(v_a_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v___x_549_);
lean_dec(v_u_548_);
lean_dec_ref(v_simpArgs_546_);
lean_dec_ref(v___x_545_);
lean_dec(v___x_544_);
v_a_615_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_563_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_563_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec(v_g_553_);
lean_dec_ref(v_a_551_);
lean_dec_ref(v_00_u03b1_550_);
lean_dec(v___x_549_);
lean_dec(v_u_548_);
lean_dec_ref(v_simpArgs_546_);
lean_dec_ref(v___x_545_);
lean_dec(v___x_544_);
v_a_623_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_559_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_559_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___boxed(lean_object** _args){
lean_object* v___x_631_ = _args[0];
lean_object* v___x_632_ = _args[1];
lean_object* v___x_633_ = _args[2];
lean_object* v___x_634_ = _args[3];
lean_object* v_simpArgs_635_ = _args[4];
lean_object* v___x_636_ = _args[5];
lean_object* v_u_637_ = _args[6];
lean_object* v___x_638_ = _args[7];
lean_object* v_00_u03b1_639_ = _args[8];
lean_object* v_a_640_ = _args[9];
lean_object* v___x_641_ = _args[10];
lean_object* v_g_642_ = _args[11];
lean_object* v___y_643_ = _args[12];
lean_object* v___y_644_ = _args[13];
lean_object* v___y_645_ = _args[14];
lean_object* v___y_646_ = _args[15];
lean_object* v___y_647_ = _args[16];
_start:
{
uint8_t v___x_14447__boxed_648_; uint8_t v___x_14450__boxed_649_; uint8_t v___x_14453__boxed_650_; lean_object* v_res_651_; 
v___x_14447__boxed_648_ = lean_unbox(v___x_632_);
v___x_14450__boxed_649_ = lean_unbox(v___x_636_);
v___x_14453__boxed_650_ = lean_unbox(v___x_641_);
v_res_651_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2(v___x_631_, v___x_14447__boxed_648_, v___x_633_, v___x_634_, v_simpArgs_635_, v___x_14450__boxed_649_, v_u_637_, v___x_638_, v_00_u03b1_639_, v_a_640_, v___x_14453__boxed_650_, v_g_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim(lean_object* v_u_655_, lean_object* v_00_u03b1_656_, lean_object* v_g_657_, lean_object* v_simpArgs_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_){
_start:
{
lean_object* v___x_664_; 
lean_inc(v_g_657_);
v___x_664_ = l_Lean_MVarId_getType(v_g_657_, v_a_659_, v_a_660_, v_a_661_, v_a_662_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v_a_665_; lean_object* v___x_666_; 
v_a_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc_n(v_a_665_, 2);
lean_dec_ref_known(v___x_664_, 1);
lean_inc(v_a_662_);
lean_inc_ref(v_a_661_);
lean_inc(v_a_660_);
lean_inc_ref(v_a_659_);
v___x_666_ = lean_infer_type(v_a_665_, v_a_659_, v_a_660_, v_a_661_, v_a_662_);
if (lean_obj_tag(v___x_666_) == 0)
{
lean_object* v_a_667_; lean_object* v___x_668_; lean_object* v_a_669_; uint8_t v___x_670_; uint8_t v___x_671_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
lean_inc(v_a_667_);
lean_dec_ref_known(v___x_666_, 1);
v___x_668_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__0___redArg(v_a_667_, v_a_660_);
v_a_669_ = lean_ctor_get(v___x_668_, 0);
lean_inc(v_a_669_);
lean_dec_ref(v___x_668_);
v___x_670_ = l_Lean_Expr_isProp(v_a_669_);
lean_dec(v_a_669_);
v___x_671_ = 1;
if (v___x_670_ == 0)
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec(v_a_665_);
lean_dec_ref(v_simpArgs_658_);
lean_dec(v_g_657_);
lean_dec_ref(v_00_u03b1_656_);
lean_dec(v_u_655_);
v___x_688_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28);
v___x_689_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v___x_688_, v_a_659_, v_a_660_, v_a_661_, v_a_662_);
v_a_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_689_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_695_; 
if (v_isShared_693_ == 0)
{
v___x_695_ = v___x_692_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_a_690_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
else
{
goto v___jp_672_;
}
v___jp_672_:
{
lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; uint8_t v___x_680_; lean_object* v___x_681_; uint8_t v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___f_686_; lean_object* v___x_687_; 
v___x_673_ = lean_box(0);
v___x_674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___closed__1));
lean_inc(v_u_655_);
v___x_675_ = l_Lean_Level_succ___override(v_u_655_);
v___x_676_ = lean_box(0);
v___x_677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_675_);
lean_ctor_set(v___x_677_, 1, v___x_676_);
v___x_678_ = l_Lean_Expr_const___override(v___x_674_, v___x_677_);
lean_inc_ref(v_00_u03b1_656_);
v___x_679_ = l_Lean_Expr_app___override(v___x_678_, v_00_u03b1_656_);
v___x_680_ = 0;
lean_inc(v_a_665_);
lean_inc_ref(v___x_679_);
v___x_681_ = l_Lean_Expr_forallE___override(v___x_673_, v___x_679_, v_a_665_, v___x_680_);
v___x_682_ = 0;
v___x_683_ = lean_box(v___x_682_);
v___x_684_ = lean_box(v___x_671_);
v___x_685_ = lean_box(v___x_680_);
lean_inc(v_g_657_);
v___f_686_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___boxed), 17, 12);
lean_closure_set(v___f_686_, 0, v___x_681_);
lean_closure_set(v___f_686_, 1, v___x_683_);
lean_closure_set(v___f_686_, 2, v___x_673_);
lean_closure_set(v___f_686_, 3, v___x_679_);
lean_closure_set(v___f_686_, 4, v_simpArgs_658_);
lean_closure_set(v___f_686_, 5, v___x_684_);
lean_closure_set(v___f_686_, 6, v_u_655_);
lean_closure_set(v___f_686_, 7, v___x_676_);
lean_closure_set(v___f_686_, 8, v_00_u03b1_656_);
lean_closure_set(v___f_686_, 9, v_a_665_);
lean_closure_set(v___f_686_, 10, v___x_685_);
lean_closure_set(v___f_686_, 11, v_g_657_);
v___x_687_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__3___redArg(v_g_657_, v___f_686_, v_a_659_, v_a_660_, v_a_661_, v_a_662_);
return v___x_687_;
}
}
else
{
lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_705_; 
lean_dec(v_a_665_);
lean_dec_ref(v_simpArgs_658_);
lean_dec(v_g_657_);
lean_dec_ref(v_00_u03b1_656_);
lean_dec(v_u_655_);
v_a_698_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_705_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_705_ == 0)
{
v___x_700_ = v___x_666_;
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v___x_666_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_703_; 
if (v_isShared_701_ == 0)
{
v___x_703_ = v___x_700_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_a_698_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
return v___x_703_;
}
}
}
}
else
{
lean_object* v_a_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_713_; 
lean_dec_ref(v_simpArgs_658_);
lean_dec(v_g_657_);
lean_dec_ref(v_00_u03b1_656_);
lean_dec(v_u_655_);
v_a_706_ = lean_ctor_get(v___x_664_, 0);
v_isSharedCheck_713_ = !lean_is_exclusive(v___x_664_);
if (v_isSharedCheck_713_ == 0)
{
v___x_708_ = v___x_664_;
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_a_706_);
lean_dec(v___x_664_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_713_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v___x_711_; 
if (v_isShared_709_ == 0)
{
v___x_711_ = v___x_708_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v_a_706_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
return v___x_711_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___boxed(lean_object* v_u_714_, lean_object* v_00_u03b1_715_, lean_object* v_g_716_, lean_object* v_simpArgs_717_, lean_object* v_a_718_, lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_, lean_object* v_a_722_){
_start:
{
lean_object* v_res_723_; 
v_res_723_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim(v_u_714_, v_00_u03b1_715_, v_g_716_, v_simpArgs_717_, v_a_718_, v_a_719_, v_a_720_, v_a_721_);
lean_dec(v_a_721_);
lean_dec_ref(v_a_720_);
lean_dec(v_a_719_);
lean_dec_ref(v_a_718_);
return v_res_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1(lean_object* v_00_u03b1_724_, lean_object* v_msg_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
lean_object* v___x_731_; 
v___x_731_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v_msg_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___boxed(lean_object* v_00_u03b1_732_, lean_object* v_msg_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1(v_00_u03b1_732_, v_msg_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2(lean_object* v_mvarId_740_, lean_object* v_val_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___redArg(v_mvarId_740_, v_val_741_, v___y_743_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2___boxed(lean_object* v_mvarId_748_, lean_object* v_val_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2(v_mvarId_748_, v_val_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3(lean_object* v_00_u03b2_756_, lean_object* v_x_757_, lean_object* v_x_758_, lean_object* v_x_759_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3___redArg(v_x_757_, v_x_758_, v_x_759_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_761_, lean_object* v_x_762_, size_t v_x_763_, size_t v_x_764_, lean_object* v_x_765_, lean_object* v_x_766_){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___redArg(v_x_762_, v_x_763_, v_x_764_, v_x_765_, v_x_766_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_768_, lean_object* v_x_769_, lean_object* v_x_770_, lean_object* v_x_771_, lean_object* v_x_772_, lean_object* v_x_773_){
_start:
{
size_t v_x_14804__boxed_774_; size_t v_x_14805__boxed_775_; lean_object* v_res_776_; 
v_x_14804__boxed_774_ = lean_unbox_usize(v_x_770_);
lean_dec(v_x_770_);
v_x_14805__boxed_775_ = lean_unbox_usize(v_x_771_);
lean_dec(v_x_771_);
v_res_776_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5(v_00_u03b2_768_, v_x_769_, v_x_14804__boxed_774_, v_x_14805__boxed_775_, v_x_772_, v_x_773_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6(lean_object* v_00_u03b2_777_, lean_object* v_n_778_, lean_object* v_k_779_, lean_object* v_v_780_){
_start:
{
lean_object* v___x_781_; 
v___x_781_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6___redArg(v_n_778_, v_k_779_, v_v_780_);
return v___x_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7(lean_object* v_00_u03b2_782_, size_t v_depth_783_, lean_object* v_keys_784_, lean_object* v_vals_785_, lean_object* v_heq_786_, lean_object* v_i_787_, lean_object* v_entries_788_){
_start:
{
lean_object* v___x_789_; 
v___x_789_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___redArg(v_depth_783_, v_keys_784_, v_vals_785_, v_i_787_, v_entries_788_);
return v___x_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7___boxed(lean_object* v_00_u03b2_790_, lean_object* v_depth_791_, lean_object* v_keys_792_, lean_object* v_vals_793_, lean_object* v_heq_794_, lean_object* v_i_795_, lean_object* v_entries_796_){
_start:
{
size_t v_depth_boxed_797_; lean_object* v_res_798_; 
v_depth_boxed_797_ = lean_unbox_usize(v_depth_791_);
lean_dec(v_depth_791_);
v_res_798_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__7(v_00_u03b2_790_, v_depth_boxed_797_, v_keys_792_, v_vals_793_, v_heq_794_, v_i_795_, v_entries_796_);
lean_dec_ref(v_vals_793_);
lean_dec_ref(v_keys_792_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7(lean_object* v_00_u03b2_799_, lean_object* v_x_800_, lean_object* v_x_801_, lean_object* v_x_802_, lean_object* v_x_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__2_spec__3_spec__5_spec__6_spec__7___redArg(v_x_800_, v_x_801_, v_x_802_, v_x_803_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0(lean_object* v_x_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_){
_start:
{
lean_object* v___x_811_; lean_object* v___x_812_; 
v___x_811_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__28);
v___x_812_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1___redArg(v___x_811_, v___y_806_, v___y_807_, v___y_808_, v___y_809_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0___boxed(lean_object* v_x_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_){
_start:
{
lean_object* v_res_819_; 
v_res_819_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__0(v_x_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
lean_dec(v___y_815_);
lean_dec_ref(v___y_814_);
lean_dec(v_x_813_);
return v_res_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1(lean_object* v_x_820_, lean_object* v_x_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_827_; lean_object* v___x_828_; 
v___x_827_ = lean_box(0);
v___x_828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_828_, 0, v___x_827_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1___boxed(lean_object* v_x_829_, lean_object* v_x_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__1(v_x_829_, v_x_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
lean_dec(v___y_832_);
lean_dec_ref(v___y_831_);
lean_dec(v_x_830_);
lean_dec(v_x_829_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2(uint8_t v___y_837_, lean_object* v_x_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
lean_object* v___x_844_; lean_object* v___x_845_; 
v___x_844_ = lean_box(v___y_837_);
v___x_845_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_845_, 0, v___x_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2___boxed(lean_object* v___y_846_, lean_object* v_x_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_){
_start:
{
uint8_t v___y_3375__boxed_853_; lean_object* v_res_854_; 
v___y_3375__boxed_853_ = lean_unbox(v___y_846_);
v_res_854_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2(v___y_3375__boxed_853_, v_x_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v_x_847_);
return v_res_854_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__2));
v___x_859_ = l_String_toRawSubstring_x27(v___x_858_);
return v___x_859_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8(void){
_start:
{
lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__7));
v___x_870_ = l_String_toRawSubstring_x27(v___x_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption(lean_object* v_g_881_, lean_object* v_a_882_, lean_object* v_a_883_, lean_object* v_a_884_, lean_object* v_a_885_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = l_Lean_Meta_saveState___redArg(v_a_883_, v_a_885_);
if (lean_obj_tag(v___x_887_) == 0)
{
lean_object* v_a_888_; lean_object* v___x_889_; 
v_a_888_ = lean_ctor_get(v___x_887_, 0);
lean_inc(v_a_888_);
lean_dec_ref_known(v___x_887_, 1);
lean_inc(v_g_881_);
v___x_889_ = l_Lean_MVarId_inferInstance(v_g_881_, v_a_882_, v_a_883_, v_a_884_, v_a_885_);
if (lean_obj_tag(v___x_889_) == 0)
{
lean_dec(v_a_888_);
lean_dec(v_g_881_);
return v___x_889_;
}
else
{
lean_object* v_a_890_; lean_object* v___f_891_; lean_object* v___f_892_; uint8_t v___y_894_; uint8_t v___x_943_; 
v_a_890_ = lean_ctor_get(v___x_889_, 0);
lean_inc(v_a_890_);
v___f_891_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__0));
v___f_892_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__1));
v___x_943_ = l_Lean_Exception_isInterrupt(v_a_890_);
if (v___x_943_ == 0)
{
uint8_t v___x_944_; 
v___x_944_ = l_Lean_Exception_isRuntime(v_a_890_);
v___y_894_ = v___x_944_;
goto v___jp_893_;
}
else
{
lean_dec(v_a_890_);
v___y_894_ = v___x_943_;
goto v___jp_893_;
}
v___jp_893_:
{
if (v___y_894_ == 0)
{
lean_object* v___x_895_; 
lean_dec_ref_known(v___x_889_, 1);
v___x_895_ = l_Lean_Meta_SavedState_restore___redArg(v_a_888_, v_a_883_, v_a_885_);
lean_dec(v_a_888_);
if (lean_obj_tag(v___x_895_) == 0)
{
lean_object* v_ref_896_; lean_object* v_quotContext_897_; lean_object* v_currMacroScope_898_; lean_object* v___x_899_; lean_object* v___f_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; uint8_t v___x_915_; uint8_t v___x_916_; lean_object* v___x_917_; uint8_t v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
lean_dec_ref_known(v___x_895_, 1);
v_ref_896_ = lean_ctor_get(v_a_884_, 5);
v_quotContext_897_ = lean_ctor_get(v_a_884_, 10);
v_currMacroScope_898_ = lean_ctor_get(v_a_884_, 11);
v___x_899_ = lean_box(v___y_894_);
v___f_900_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___lam__2___boxed), 7, 1);
lean_closure_set(v___f_900_, 0, v___x_899_);
v___x_901_ = l_Lean_SourceInfo_fromRef(v_ref_896_, v___y_894_);
v___x_902_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__3);
v___x_903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__4));
lean_inc_n(v_currMacroScope_898_, 2);
lean_inc_n(v_quotContext_897_, 2);
v___x_904_ = l_Lean_addMacroScope(v_quotContext_897_, v___x_903_, v_currMacroScope_898_);
v___x_905_ = lean_box(0);
v___x_906_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__6));
lean_inc(v___x_901_);
v___x_907_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_907_, 0, v___x_901_);
lean_ctor_set(v___x_907_, 1, v___x_902_);
lean_ctor_set(v___x_907_, 2, v___x_904_);
lean_ctor_set(v___x_907_, 3, v___x_906_);
v___x_908_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__8);
v___x_909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__9));
v___x_910_ = l_Lean_addMacroScope(v_quotContext_897_, v___x_909_, v_currMacroScope_898_);
v___x_911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__11));
v___x_912_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_912_, 0, v___x_901_);
lean_ctor_set(v___x_912_, 1, v___x_908_);
lean_ctor_set(v___x_912_, 2, v___x_910_);
lean_ctor_set(v___x_912_, 3, v___x_911_);
v___x_913_ = lean_unsigned_to_nat(6u);
v___x_914_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_914_, 0, v___x_913_);
lean_ctor_set(v___x_914_, 1, v___f_892_);
lean_ctor_set(v___x_914_, 2, v___f_900_);
lean_ctor_set(v___x_914_, 3, v___f_891_);
lean_ctor_set_uint8(v___x_914_, sizeof(void*)*4, v___y_894_);
v___x_915_ = 0;
v___x_916_ = 1;
v___x_917_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_917_, 0, v___x_915_);
lean_ctor_set_uint8(v___x_917_, 1, v___x_916_);
lean_ctor_set_uint8(v___x_917_, 2, v___y_894_);
lean_ctor_set_uint8(v___x_917_, 3, v___x_916_);
v___x_918_ = 1;
v___x_919_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_919_, 0, v___x_914_);
lean_ctor_set(v___x_919_, 1, v___x_917_);
lean_ctor_set_uint8(v___x_919_, sizeof(void*)*2, v___x_918_);
lean_ctor_set_uint8(v___x_919_, sizeof(void*)*2 + 1, v___x_916_);
lean_ctor_set_uint8(v___x_919_, sizeof(void*)*2 + 2, v___x_916_);
v___x_920_ = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(v___x_920_, 0, v___x_919_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*1, v___x_916_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*1 + 1, v___x_916_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*1 + 2, v___x_916_);
lean_ctor_set_uint8(v___x_920_, sizeof(void*)*1 + 3, v___y_894_);
v___x_921_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_912_);
lean_ctor_set(v___x_921_, 1, v___x_905_);
v___x_922_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_922_, 0, v___x_907_);
lean_ctor_set(v___x_922_, 1, v___x_921_);
v___x_923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___closed__12));
v___x_924_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_924_, 0, v_g_881_);
lean_ctor_set(v___x_924_, 1, v___x_905_);
v___x_925_ = l_Lean_Elab_Tactic_SolveByElim_processSyntax(v___x_920_, v___y_894_, v___y_894_, v___x_922_, v___x_905_, v___x_923_, v___x_924_, v_a_882_, v_a_883_, v_a_884_, v_a_885_);
if (lean_obj_tag(v___x_925_) == 0)
{
lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_933_; 
v_isSharedCheck_933_ = !lean_is_exclusive(v___x_925_);
if (v_isSharedCheck_933_ == 0)
{
lean_object* v_unused_934_; 
v_unused_934_ = lean_ctor_get(v___x_925_, 0);
lean_dec(v_unused_934_);
v___x_927_ = v___x_925_;
v_isShared_928_ = v_isSharedCheck_933_;
goto v_resetjp_926_;
}
else
{
lean_dec(v___x_925_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_933_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v___x_929_; lean_object* v___x_931_; 
v___x_929_ = lean_box(0);
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 0, v___x_929_);
v___x_931_ = v___x_927_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v___x_929_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
v_a_935_ = lean_ctor_get(v___x_925_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_925_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_925_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_925_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
lean_object* v___x_940_; 
if (v_isShared_938_ == 0)
{
v___x_940_ = v___x_937_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_a_935_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
}
}
else
{
lean_dec(v_g_881_);
return v___x_895_;
}
}
else
{
lean_dec(v_a_888_);
lean_dec(v_g_881_);
return v___x_889_;
}
}
}
}
else
{
lean_object* v_a_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_952_; 
lean_dec(v_g_881_);
v_a_945_ = lean_ctor_get(v___x_887_, 0);
v_isSharedCheck_952_ = !lean_is_exclusive(v___x_887_);
if (v_isSharedCheck_952_ == 0)
{
v___x_947_ = v___x_887_;
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_a_945_);
lean_dec(v___x_887_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_952_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v___x_950_; 
if (v_isShared_948_ == 0)
{
v___x_950_ = v___x_947_;
goto v_reusejp_949_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v_a_945_);
v___x_950_ = v_reuseFailAlloc_951_;
goto v_reusejp_949_;
}
v_reusejp_949_:
{
return v___x_950_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption___boxed(lean_object* v_g_953_, lean_object* v_a_954_, lean_object* v_a_955_, lean_object* v_a_956_, lean_object* v_a_957_, lean_object* v_a_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption(v_g_953_, v_a_954_, v_a_955_, v_a_956_, v_a_957_);
lean_dec(v_a_957_);
lean_dec_ref(v_a_956_);
lean_dec(v_a_955_);
lean_dec_ref(v_a_954_);
return v_res_959_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23(void){
_start:
{
uint8_t v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1011_ = 0;
v___x_1012_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__22));
v___x_1013_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__1___closed__23));
v___x_1014_ = l_Lean_Parser_Tactic_simpArg;
v___x_1015_ = lean_alloc_ctor(11, 3, 1);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
lean_ctor_set(v___x_1015_, 1, v___x_1013_);
lean_ctor_set(v___x_1015_, 2, v___x_1012_);
lean_ctor_set_uint8(v___x_1015_, sizeof(void*)*3, v___x_1011_);
return v___x_1015_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; 
v___x_1016_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__23);
v___x_1017_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__20));
v___x_1018_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2));
v___x_1019_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1018_);
lean_ctor_set(v___x_1019_, 1, v___x_1017_);
lean_ctor_set(v___x_1019_, 2, v___x_1016_);
return v___x_1019_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25(void){
_start:
{
lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; 
v___x_1020_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__24);
v___x_1021_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__5));
v___x_1022_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1022_, 0, v___x_1021_);
lean_ctor_set(v___x_1022_, 1, v___x_1020_);
return v___x_1022_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26(void){
_start:
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1023_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__25);
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__18));
v___x_1025_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__2));
v___x_1026_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1026_, 0, v___x_1025_);
lean_ctor_set(v___x_1026_, 1, v___x_1024_);
lean_ctor_set(v___x_1026_, 2, v___x_1023_);
return v___x_1026_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27(void){
_start:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; 
v___x_1027_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__26);
v___x_1028_ = lean_unsigned_to_nat(1022u);
v___x_1029_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__0));
v___x_1030_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1029_);
lean_ctor_set(v___x_1030_, 1, v___x_1028_);
lean_ctor_set(v___x_1030_, 2, v___x_1027_);
return v___x_1030_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality(void){
_start:
{
lean_object* v___x_1031_; 
v___x_1031_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27, &lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality___closed__27);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1(lean_object* v_msg_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v___f_1043_; lean_object* v___x_9485__overap_1044_; lean_object* v___x_1045_; 
v___f_1043_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___closed__0));
v___x_9485__overap_1044_ = lean_panic_fn_borrowed(v___f_1043_, v_msg_1033_);
lean_inc(v___y_1041_);
lean_inc_ref(v___y_1040_);
lean_inc(v___y_1039_);
lean_inc_ref(v___y_1038_);
lean_inc(v___y_1037_);
lean_inc_ref(v___y_1036_);
lean_inc(v___y_1035_);
lean_inc_ref(v___y_1034_);
v___x_1045_ = lean_apply_9(v___x_9485__overap_1044_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_, lean_box(0));
return v___x_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1___boxed(lean_object* v_msg_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_){
_start:
{
lean_object* v_res_1056_; 
v_res_1056_ = lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1(v_msg_1046_, v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_);
lean_dec(v___y_1054_);
lean_dec_ref(v___y_1053_);
lean_dec(v___y_1052_);
lean_dec_ref(v___y_1051_);
lean_dec(v___y_1050_);
lean_dec_ref(v___y_1049_);
lean_dec(v___y_1048_);
lean_dec_ref(v___y_1047_);
return v_res_1056_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(lean_object* v_msg_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
lean_object* v_ref_1063_; lean_object* v___x_1064_; lean_object* v_a_1065_; lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1073_; 
v_ref_1063_ = lean_ctor_get(v___y_1060_, 5);
v___x_1064_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Nontriviality_nontrivialityByElim_spec__1_spec__1(v_msg_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
v_a_1065_ = lean_ctor_get(v___x_1064_, 0);
v_isSharedCheck_1073_ = !lean_is_exclusive(v___x_1064_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1067_ = v___x_1064_;
v_isShared_1068_ = v_isSharedCheck_1073_;
goto v_resetjp_1066_;
}
else
{
lean_inc(v_a_1065_);
lean_dec(v___x_1064_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1073_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v___x_1069_; lean_object* v___x_1071_; 
lean_inc(v_ref_1063_);
v___x_1069_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1069_, 0, v_ref_1063_);
lean_ctor_set(v___x_1069_, 1, v_a_1065_);
if (v_isShared_1068_ == 0)
{
lean_ctor_set_tag(v___x_1067_, 1);
lean_ctor_set(v___x_1067_, 0, v___x_1069_);
v___x_1071_ = v___x_1067_;
goto v_reusejp_1070_;
}
else
{
lean_object* v_reuseFailAlloc_1072_; 
v_reuseFailAlloc_1072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1072_, 0, v___x_1069_);
v___x_1071_ = v_reuseFailAlloc_1072_;
goto v_reusejp_1070_;
}
v_reusejp_1070_:
{
return v___x_1071_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg___boxed(lean_object* v_msg_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_){
_start:
{
lean_object* v_res_1080_; 
v_res_1080_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(v_msg_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
return v_res_1080_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9(void){
_start:
{
lean_object* v___x_1095_; lean_object* v___x_1096_; 
v___x_1095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__8));
v___x_1096_ = l_Lean_stringToMessageData(v___x_1095_);
return v___x_1096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0(lean_object* v_____r_1097_, lean_object* v_tgt_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; uint8_t v___x_1110_; 
v___x_1108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__1));
v___x_1109_ = lean_unsigned_to_nat(3u);
v___x_1110_ = l_Lean_Expr_isAppOfArity(v_tgt_1098_, v___x_1108_, v___x_1109_);
if (v___x_1110_ == 0)
{
lean_object* v___x_1111_; lean_object* v___x_1112_; uint8_t v___x_1113_; 
v___x_1111_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__4));
v___x_1112_ = lean_unsigned_to_nat(4u);
v___x_1113_ = l_Lean_Expr_isAppOfArity(v_tgt_1098_, v___x_1111_, v___x_1112_);
if (v___x_1113_ == 0)
{
lean_object* v___x_1114_; uint8_t v___x_1115_; 
v___x_1114_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__7));
v___x_1115_ = l_Lean_Expr_isAppOfArity(v_tgt_1098_, v___x_1114_, v___x_1112_);
if (v___x_1115_ == 0)
{
lean_object* v___x_1116_; lean_object* v___x_1117_; 
v___x_1116_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9, &lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___closed__9);
v___x_1117_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(v___x_1116_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
return v___x_1117_;
}
else
{
lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1118_ = l_Lean_Expr_appFn_x21(v_tgt_1098_);
v___x_1119_ = l_Lean_Expr_appFn_x21(v___x_1118_);
lean_dec_ref(v___x_1118_);
v___x_1120_ = l_Lean_Expr_appFn_x21(v___x_1119_);
lean_dec_ref(v___x_1119_);
v___x_1121_ = l_Lean_Expr_appArg_x21(v___x_1120_);
lean_dec_ref(v___x_1120_);
v___x_1122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1121_);
return v___x_1122_;
}
}
else
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1123_ = l_Lean_Expr_appFn_x21(v_tgt_1098_);
v___x_1124_ = l_Lean_Expr_appFn_x21(v___x_1123_);
lean_dec_ref(v___x_1123_);
v___x_1125_ = l_Lean_Expr_appFn_x21(v___x_1124_);
lean_dec_ref(v___x_1124_);
v___x_1126_ = l_Lean_Expr_appArg_x21(v___x_1125_);
lean_dec_ref(v___x_1125_);
v___x_1127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1126_);
return v___x_1127_;
}
}
else
{
lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1128_ = l_Lean_Expr_appFn_x21(v_tgt_1098_);
v___x_1129_ = l_Lean_Expr_appFn_x21(v___x_1128_);
lean_dec_ref(v___x_1128_);
v___x_1130_ = l_Lean_Expr_appArg_x21(v___x_1129_);
lean_dec_ref(v___x_1129_);
v___x_1131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1131_, 0, v___x_1130_);
return v___x_1131_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0___boxed(lean_object* v_____r_1132_, lean_object* v_tgt_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0(v_____r_1132_, v_tgt_1133_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
lean_dec(v___y_1137_);
lean_dec_ref(v___y_1136_);
lean_dec(v___y_1135_);
lean_dec_ref(v___y_1134_);
lean_dec_ref(v_tgt_1133_);
return v_res_1143_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3(void){
_start:
{
lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__2));
v___x_1149_ = l_Lean_stringToMessageData(v___x_1148_);
return v___x_1149_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7(void){
_start:
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; 
v___x_1153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__6));
v___x_1154_ = lean_unsigned_to_nat(41u);
v___x_1155_ = lean_unsigned_to_nat(123u);
v___x_1156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__5));
v___x_1157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__4));
v___x_1158_ = l_mkPanicMessageWithDecl(v___x_1157_, v___x_1156_, v___x_1155_, v___x_1154_, v___x_1153_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality(lean_object* v_stx_1162_, lean_object* v_a_1163_, lean_object* v_a_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_, lean_object* v_a_1170_){
_start:
{
lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v_a_1178_; lean_object* v___x_1201_; 
v___x_1201_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1164_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_);
if (lean_obj_tag(v___x_1201_) == 0)
{
lean_object* v_a_1202_; lean_object* v___x_1204_; uint8_t v_isShared_1205_; uint8_t v_isSharedCheck_1396_; 
v_a_1202_ = lean_ctor_get(v___x_1201_, 0);
v_isSharedCheck_1396_ = !lean_is_exclusive(v___x_1201_);
if (v_isSharedCheck_1396_ == 0)
{
v___x_1204_ = v___x_1201_;
v_isShared_1205_ = v_isSharedCheck_1396_;
goto v_resetjp_1203_;
}
else
{
lean_inc(v_a_1202_);
lean_dec(v___x_1201_);
v___x_1204_ = lean_box(0);
v_isShared_1205_ = v_isSharedCheck_1396_;
goto v_resetjp_1203_;
}
v_resetjp_1203_:
{
lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v___y_1212_; lean_object* v___y_1213_; lean_object* v___y_1214_; lean_object* v___y_1215_; uint8_t v___y_1216_; lean_object* v___y_1237_; lean_object* v___y_1238_; lean_object* v___y_1239_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1244_; lean_object* v_a_1245_; lean_object* v_00_u03b1_1249_; lean_object* v___y_1250_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___y_1253_; lean_object* v___y_1254_; lean_object* v___y_1255_; lean_object* v___y_1256_; lean_object* v___y_1257_; lean_object* v___y_1322_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1332_ = lean_unsigned_to_nat(1u);
v___x_1333_ = l_Lean_Syntax_getArg(v_stx_1162_, v___x_1332_);
v___x_1334_ = l_Lean_Syntax_getOptional_x3f(v___x_1333_);
lean_dec(v___x_1333_);
if (lean_obj_tag(v___x_1334_) == 0)
{
lean_object* v_keyedConfig_1335_; uint8_t v_trackZetaDelta_1336_; lean_object* v_zetaDeltaSet_1337_; lean_object* v_lctx_1338_; lean_object* v_localInstances_1339_; lean_object* v_defEqCtx_x3f_1340_; lean_object* v_synthPendingDepth_1341_; lean_object* v_customCanUnfoldPredicate_x3f_1342_; uint8_t v_univApprox_1343_; uint8_t v_inTypeClassResolution_1344_; uint8_t v_cacheInferType_1345_; lean_object* v_a_1347_; lean_object* v_a_1351_; uint8_t v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; 
v_keyedConfig_1335_ = lean_ctor_get(v_a_1167_, 0);
v_trackZetaDelta_1336_ = lean_ctor_get_uint8(v_a_1167_, sizeof(void*)*7);
v_zetaDeltaSet_1337_ = lean_ctor_get(v_a_1167_, 1);
v_lctx_1338_ = lean_ctor_get(v_a_1167_, 2);
v_localInstances_1339_ = lean_ctor_get(v_a_1167_, 3);
v_defEqCtx_x3f_1340_ = lean_ctor_get(v_a_1167_, 4);
v_synthPendingDepth_1341_ = lean_ctor_get(v_a_1167_, 5);
v_customCanUnfoldPredicate_x3f_1342_ = lean_ctor_get(v_a_1167_, 6);
v_univApprox_1343_ = lean_ctor_get_uint8(v_a_1167_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1344_ = lean_ctor_get_uint8(v_a_1167_, sizeof(void*)*7 + 2);
v_cacheInferType_1345_ = lean_ctor_get_uint8(v_a_1167_, sizeof(void*)*7 + 3);
v___x_1371_ = 2;
lean_inc_ref(v_keyedConfig_1335_);
v___x_1372_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1371_, v_keyedConfig_1335_);
lean_inc(v_customCanUnfoldPredicate_x3f_1342_);
lean_inc(v_synthPendingDepth_1341_);
lean_inc(v_defEqCtx_x3f_1340_);
lean_inc_ref(v_localInstances_1339_);
lean_inc_ref(v_lctx_1338_);
lean_inc(v_zetaDeltaSet_1337_);
v___x_1373_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
lean_ctor_set(v___x_1373_, 1, v_zetaDeltaSet_1337_);
lean_ctor_set(v___x_1373_, 2, v_lctx_1338_);
lean_ctor_set(v___x_1373_, 3, v_localInstances_1339_);
lean_ctor_set(v___x_1373_, 4, v_defEqCtx_x3f_1340_);
lean_ctor_set(v___x_1373_, 5, v_synthPendingDepth_1341_);
lean_ctor_set(v___x_1373_, 6, v_customCanUnfoldPredicate_x3f_1342_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*7, v_trackZetaDelta_1336_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*7 + 1, v_univApprox_1343_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1344_);
lean_ctor_set_uint8(v___x_1373_, sizeof(void*)*7 + 3, v_cacheInferType_1345_);
lean_inc(v_a_1202_);
v___x_1374_ = l_Lean_MVarId_getType_x27(v_a_1202_, v___x_1373_, v_a_1168_, v_a_1169_, v_a_1170_);
lean_dec_ref_known(v___x_1373_, 7);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1375_);
lean_dec_ref_known(v___x_1374_, 1);
v_a_1351_ = v_a_1375_;
goto v___jp_1350_;
}
else
{
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1376_; 
v_a_1376_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1376_);
lean_dec_ref_known(v___x_1374_, 1);
v_a_1351_ = v_a_1376_;
goto v___jp_1350_;
}
else
{
lean_object* v_a_1377_; lean_object* v___x_1379_; uint8_t v_isShared_1380_; uint8_t v_isSharedCheck_1384_; 
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1377_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1379_ = v___x_1374_;
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
else
{
lean_inc(v_a_1377_);
lean_dec(v___x_1374_);
v___x_1379_ = lean_box(0);
v_isShared_1380_ = v_isSharedCheck_1384_;
goto v_resetjp_1378_;
}
v_resetjp_1378_:
{
lean_object* v___x_1382_; 
if (v_isShared_1380_ == 0)
{
v___x_1382_ = v___x_1379_;
goto v_reusejp_1381_;
}
else
{
lean_object* v_reuseFailAlloc_1383_; 
v_reuseFailAlloc_1383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1383_, 0, v_a_1377_);
v___x_1382_ = v_reuseFailAlloc_1383_;
goto v_reusejp_1381_;
}
v_reusejp_1381_:
{
return v___x_1382_;
}
}
}
}
v___jp_1346_:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1348_ = lean_box(0);
v___x_1349_ = lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0(v___x_1348_, v_a_1347_, v_a_1163_, v_a_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_);
lean_dec_ref(v_a_1347_);
v___y_1322_ = v___x_1349_;
goto v___jp_1321_;
}
v___jp_1350_:
{
lean_object* v___x_1352_; uint8_t v___x_1353_; 
v___x_1352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__9));
v___x_1353_ = l_Lean_Expr_isAppOfArity(v_a_1351_, v___x_1352_, v___x_1332_);
if (v___x_1353_ == 0)
{
lean_object* v___x_1354_; lean_object* v___x_1355_; 
v___x_1354_ = lean_box(0);
v___x_1355_ = lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___lam__0(v___x_1354_, v_a_1351_, v_a_1163_, v_a_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_);
lean_dec_ref(v_a_1351_);
v___y_1322_ = v___x_1355_;
goto v___jp_1321_;
}
else
{
lean_object* v___x_1356_; uint8_t v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; 
v___x_1356_ = l_Lean_Expr_appArg_x21(v_a_1351_);
lean_dec_ref(v_a_1351_);
v___x_1357_ = 2;
lean_inc_ref(v_keyedConfig_1335_);
v___x_1358_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1357_, v_keyedConfig_1335_);
lean_inc(v_customCanUnfoldPredicate_x3f_1342_);
lean_inc(v_synthPendingDepth_1341_);
lean_inc(v_defEqCtx_x3f_1340_);
lean_inc_ref(v_localInstances_1339_);
lean_inc_ref(v_lctx_1338_);
lean_inc(v_zetaDeltaSet_1337_);
v___x_1359_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1359_, 0, v___x_1358_);
lean_ctor_set(v___x_1359_, 1, v_zetaDeltaSet_1337_);
lean_ctor_set(v___x_1359_, 2, v_lctx_1338_);
lean_ctor_set(v___x_1359_, 3, v_localInstances_1339_);
lean_ctor_set(v___x_1359_, 4, v_defEqCtx_x3f_1340_);
lean_ctor_set(v___x_1359_, 5, v_synthPendingDepth_1341_);
lean_ctor_set(v___x_1359_, 6, v_customCanUnfoldPredicate_x3f_1342_);
lean_ctor_set_uint8(v___x_1359_, sizeof(void*)*7, v_trackZetaDelta_1336_);
lean_ctor_set_uint8(v___x_1359_, sizeof(void*)*7 + 1, v_univApprox_1343_);
lean_ctor_set_uint8(v___x_1359_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1344_);
lean_ctor_set_uint8(v___x_1359_, sizeof(void*)*7 + 3, v_cacheInferType_1345_);
lean_inc(v_a_1170_);
lean_inc_ref(v_a_1169_);
lean_inc(v_a_1168_);
v___x_1360_ = lean_whnf(v___x_1356_, v___x_1359_, v_a_1168_, v_a_1169_, v_a_1170_);
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_object* v_a_1361_; 
v_a_1361_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_a_1361_);
lean_dec_ref_known(v___x_1360_, 1);
v_a_1347_ = v_a_1361_;
goto v___jp_1346_;
}
else
{
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_object* v_a_1362_; 
v_a_1362_ = lean_ctor_get(v___x_1360_, 0);
lean_inc(v_a_1362_);
lean_dec_ref_known(v___x_1360_, 1);
v_a_1347_ = v_a_1362_;
goto v___jp_1346_;
}
else
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1363_ = lean_ctor_get(v___x_1360_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1360_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1360_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1360_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_a_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
}
}
}
else
{
lean_object* v_val_1385_; lean_object* v___x_1386_; 
v_val_1385_ = lean_ctor_get(v___x_1334_, 0);
lean_inc(v_val_1385_);
lean_dec_ref_known(v___x_1334_, 1);
v___x_1386_ = l_Lean_Elab_Term_elabType(v_val_1385_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_);
if (lean_obj_tag(v___x_1386_) == 0)
{
lean_object* v_a_1387_; 
v_a_1387_ = lean_ctor_get(v___x_1386_, 0);
lean_inc(v_a_1387_);
lean_dec_ref_known(v___x_1386_, 1);
v_00_u03b1_1249_ = v_a_1387_;
v___y_1250_ = v_a_1163_;
v___y_1251_ = v_a_1164_;
v___y_1252_ = v_a_1165_;
v___y_1253_ = v_a_1166_;
v___y_1254_ = v_a_1167_;
v___y_1255_ = v_a_1168_;
v___y_1256_ = v_a_1169_;
v___y_1257_ = v_a_1170_;
goto v___jp_1248_;
}
else
{
lean_object* v_a_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1395_; 
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1388_ = lean_ctor_get(v___x_1386_, 0);
v_isSharedCheck_1395_ = !lean_is_exclusive(v___x_1386_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1390_ = v___x_1386_;
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_a_1388_);
lean_dec(v___x_1386_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v___x_1393_; 
if (v_isShared_1391_ == 0)
{
v___x_1393_ = v___x_1390_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v_a_1388_);
v___x_1393_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
return v___x_1393_;
}
}
}
}
v___jp_1206_:
{
if (v___y_1216_ == 0)
{
lean_object* v___x_1217_; 
lean_dec_ref(v___y_1209_);
lean_del_object(v___x_1204_);
v___x_1217_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1214_, v___y_1215_, v___y_1211_);
lean_dec_ref(v___y_1214_);
if (lean_obj_tag(v___x_1217_) == 0)
{
lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; 
lean_dec_ref_known(v___x_1217_, 1);
v___x_1218_ = lean_unsigned_to_nat(2u);
v___x_1219_ = l_Lean_Syntax_getArg(v_stx_1162_, v___x_1218_);
v___x_1220_ = lean_unsigned_to_nat(1u);
v___x_1221_ = l_Lean_Syntax_getArg(v___x_1219_, v___x_1220_);
lean_dec(v___x_1219_);
v___x_1222_ = l_Lean_Syntax_getSepArgs(v___x_1221_);
lean_dec(v___x_1221_);
v___x_1223_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim(v___y_1210_, v___y_1207_, v_a_1202_, v___x_1222_, v___y_1208_, v___y_1215_, v___y_1212_, v___y_1211_);
if (lean_obj_tag(v___x_1223_) == 0)
{
lean_object* v_a_1224_; 
v_a_1224_ = lean_ctor_get(v___x_1223_, 0);
lean_inc(v_a_1224_);
lean_dec_ref_known(v___x_1223_, 1);
v___y_1173_ = v___y_1208_;
v___y_1174_ = v___y_1211_;
v___y_1175_ = v___y_1212_;
v___y_1176_ = v___y_1213_;
v___y_1177_ = v___y_1215_;
v_a_1178_ = v_a_1224_;
goto v___jp_1172_;
}
else
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1232_; 
v_a_1225_ = lean_ctor_get(v___x_1223_, 0);
v_isSharedCheck_1232_ = !lean_is_exclusive(v___x_1223_);
if (v_isSharedCheck_1232_ == 0)
{
v___x_1227_ = v___x_1223_;
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1223_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1230_; 
if (v_isShared_1228_ == 0)
{
v___x_1230_ = v___x_1227_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1231_; 
v_reuseFailAlloc_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1231_, 0, v_a_1225_);
v___x_1230_ = v_reuseFailAlloc_1231_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
return v___x_1230_;
}
}
}
}
else
{
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1207_);
lean_dec(v_a_1202_);
return v___x_1217_;
}
}
else
{
lean_object* v___x_1234_; 
lean_dec_ref(v___y_1214_);
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1207_);
lean_dec(v_a_1202_);
if (v_isShared_1205_ == 0)
{
lean_ctor_set_tag(v___x_1204_, 1);
lean_ctor_set(v___x_1204_, 0, v___y_1209_);
v___x_1234_ = v___x_1204_;
goto v_reusejp_1233_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v___y_1209_);
v___x_1234_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1233_;
}
v_reusejp_1233_:
{
return v___x_1234_;
}
}
}
v___jp_1236_:
{
uint8_t v___x_1246_; 
v___x_1246_ = l_Lean_Exception_isInterrupt(v_a_1245_);
if (v___x_1246_ == 0)
{
uint8_t v___x_1247_; 
lean_inc_ref(v_a_1245_);
v___x_1247_ = l_Lean_Exception_isRuntime(v_a_1245_);
v___y_1207_ = v___y_1237_;
v___y_1208_ = v___y_1238_;
v___y_1209_ = v_a_1245_;
v___y_1210_ = v___y_1240_;
v___y_1211_ = v___y_1239_;
v___y_1212_ = v___y_1241_;
v___y_1213_ = v___y_1242_;
v___y_1214_ = v___y_1243_;
v___y_1215_ = v___y_1244_;
v___y_1216_ = v___x_1247_;
goto v___jp_1206_;
}
else
{
v___y_1207_ = v___y_1237_;
v___y_1208_ = v___y_1238_;
v___y_1209_ = v_a_1245_;
v___y_1210_ = v___y_1240_;
v___y_1211_ = v___y_1239_;
v___y_1212_ = v___y_1241_;
v___y_1213_ = v___y_1242_;
v___y_1214_ = v___y_1243_;
v___y_1215_ = v___y_1244_;
v___y_1216_ = v___x_1246_;
goto v___jp_1206_;
}
}
v___jp_1248_:
{
lean_object* v___x_1258_; 
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc(v___y_1255_);
lean_inc_ref(v___y_1254_);
lean_inc_ref(v_00_u03b1_1249_);
v___x_1258_ = lean_infer_type(v_00_u03b1_1249_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; lean_object* v___x_1260_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
lean_inc(v_a_1259_);
lean_dec_ref_known(v___x_1258_, 1);
lean_inc(v___y_1257_);
lean_inc_ref(v___y_1256_);
lean_inc(v___y_1255_);
lean_inc_ref(v___y_1254_);
v___x_1260_ = lean_whnf(v_a_1259_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1260_) == 0)
{
lean_object* v_a_1261_; 
v_a_1261_ = lean_ctor_get(v___x_1260_, 0);
lean_inc(v_a_1261_);
lean_dec_ref_known(v___x_1260_, 1);
if (lean_obj_tag(v_a_1261_) == 3)
{
lean_object* v_u_1262_; lean_object* v___x_1263_; 
v_u_1262_ = lean_ctor_get(v_a_1261_, 0);
lean_inc(v_u_1262_);
lean_dec_ref_known(v_a_1261_, 1);
v___x_1263_ = l_Lean_Level_dec(v_u_1262_);
lean_dec(v_u_1262_);
if (lean_obj_tag(v___x_1263_) == 1)
{
lean_object* v_val_1264_; lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1298_; 
v_val_1264_ = lean_ctor_get(v___x_1263_, 0);
v_isSharedCheck_1298_ = !lean_is_exclusive(v___x_1263_);
if (v_isSharedCheck_1298_ == 0)
{
v___x_1266_ = v___x_1263_;
v_isShared_1267_ = v_isSharedCheck_1298_;
goto v_resetjp_1265_;
}
else
{
lean_inc(v_val_1264_);
lean_dec(v___x_1263_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1298_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___x_1268_; 
v___x_1268_ = l_Lean_Meta_saveState___redArg(v___y_1255_, v___y_1257_);
if (lean_obj_tag(v___x_1268_) == 0)
{
lean_object* v_a_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1276_; 
v_a_1269_ = lean_ctor_get(v___x_1268_, 0);
lean_inc(v_a_1269_);
lean_dec_ref_known(v___x_1268_, 1);
v___x_1270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByElim___lam__2___closed__1));
v___x_1271_ = lean_box(0);
lean_inc(v_val_1264_);
v___x_1272_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1272_, 0, v_val_1264_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = l_Lean_Expr_const___override(v___x_1270_, v___x_1272_);
lean_inc_ref(v_00_u03b1_1249_);
v___x_1274_ = l_Lean_Expr_app___override(v___x_1273_, v_00_u03b1_1249_);
lean_inc_ref(v___x_1274_);
if (v_isShared_1267_ == 0)
{
lean_ctor_set(v___x_1266_, 0, v___x_1274_);
v___x_1276_ = v___x_1266_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v___x_1274_);
v___x_1276_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1275_;
}
v_reusejp_1275_:
{
uint8_t v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v___x_1277_ = 0;
v___x_1278_ = lean_box(0);
v___x_1279_ = l_Lean_Meta_mkFreshExprMVar(v___x_1276_, v___x_1277_, v___x_1278_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1279_) == 0)
{
lean_object* v_a_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; 
v_a_1280_ = lean_ctor_get(v___x_1279_, 0);
lean_inc(v_a_1280_);
lean_dec_ref_known(v___x_1279_, 1);
v___x_1281_ = l_Lean_Expr_mvarId_x21(v_a_1280_);
v___x_1282_ = lp_mathlib_Mathlib_Tactic_Nontriviality_nontrivialityByAssumption(v___x_1281_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1282_) == 0)
{
lean_object* v___x_1283_; lean_object* v___x_1284_; 
lean_dec_ref_known(v___x_1282_, 1);
v___x_1283_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__1));
lean_inc(v_a_1202_);
v___x_1284_ = l_Lean_MVarId_assert(v_a_1202_, v___x_1283_, v___x_1274_, v_a_1280_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
if (lean_obj_tag(v___x_1284_) == 0)
{
lean_object* v_a_1285_; 
lean_dec(v_a_1269_);
lean_dec(v_val_1264_);
lean_dec_ref(v_00_u03b1_1249_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1285_ = lean_ctor_get(v___x_1284_, 0);
lean_inc(v_a_1285_);
lean_dec_ref_known(v___x_1284_, 1);
v___y_1173_ = v___y_1254_;
v___y_1174_ = v___y_1257_;
v___y_1175_ = v___y_1256_;
v___y_1176_ = v___y_1251_;
v___y_1177_ = v___y_1255_;
v_a_1178_ = v_a_1285_;
goto v___jp_1172_;
}
else
{
lean_object* v_a_1286_; 
v_a_1286_ = lean_ctor_get(v___x_1284_, 0);
lean_inc(v_a_1286_);
lean_dec_ref_known(v___x_1284_, 1);
v___y_1237_ = v_00_u03b1_1249_;
v___y_1238_ = v___y_1254_;
v___y_1239_ = v___y_1257_;
v___y_1240_ = v_val_1264_;
v___y_1241_ = v___y_1256_;
v___y_1242_ = v___y_1251_;
v___y_1243_ = v_a_1269_;
v___y_1244_ = v___y_1255_;
v_a_1245_ = v_a_1286_;
goto v___jp_1236_;
}
}
else
{
lean_object* v_a_1287_; 
lean_dec(v_a_1280_);
lean_dec_ref(v___x_1274_);
v_a_1287_ = lean_ctor_get(v___x_1282_, 0);
lean_inc(v_a_1287_);
lean_dec_ref_known(v___x_1282_, 1);
v___y_1237_ = v_00_u03b1_1249_;
v___y_1238_ = v___y_1254_;
v___y_1239_ = v___y_1257_;
v___y_1240_ = v_val_1264_;
v___y_1241_ = v___y_1256_;
v___y_1242_ = v___y_1251_;
v___y_1243_ = v_a_1269_;
v___y_1244_ = v___y_1255_;
v_a_1245_ = v_a_1287_;
goto v___jp_1236_;
}
}
else
{
lean_object* v_a_1288_; 
lean_dec_ref(v___x_1274_);
v_a_1288_ = lean_ctor_get(v___x_1279_, 0);
lean_inc(v_a_1288_);
lean_dec_ref_known(v___x_1279_, 1);
v___y_1237_ = v_00_u03b1_1249_;
v___y_1238_ = v___y_1254_;
v___y_1239_ = v___y_1257_;
v___y_1240_ = v_val_1264_;
v___y_1241_ = v___y_1256_;
v___y_1242_ = v___y_1251_;
v___y_1243_ = v_a_1269_;
v___y_1244_ = v___y_1255_;
v_a_1245_ = v_a_1288_;
goto v___jp_1236_;
}
}
}
else
{
lean_object* v_a_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1297_; 
lean_del_object(v___x_1266_);
lean_dec(v_val_1264_);
lean_dec_ref(v_00_u03b1_1249_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1290_ = lean_ctor_get(v___x_1268_, 0);
v_isSharedCheck_1297_ = !lean_is_exclusive(v___x_1268_);
if (v_isSharedCheck_1297_ == 0)
{
v___x_1292_ = v___x_1268_;
v_isShared_1293_ = v_isSharedCheck_1297_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_a_1290_);
lean_dec(v___x_1268_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1297_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v___x_1295_; 
if (v_isShared_1293_ == 0)
{
v___x_1295_ = v___x_1292_;
goto v_reusejp_1294_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v_a_1290_);
v___x_1295_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1294_;
}
v_reusejp_1294_:
{
return v___x_1295_;
}
}
}
}
}
else
{
lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; 
lean_dec(v___x_1263_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v___x_1299_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3, &lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__3);
v___x_1300_ = l_Lean_indentExpr(v_00_u03b1_1249_);
v___x_1301_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1301_, 0, v___x_1299_);
lean_ctor_set(v___x_1301_, 1, v___x_1300_);
v___x_1302_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(v___x_1301_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
return v___x_1302_;
}
}
else
{
lean_object* v___x_1303_; lean_object* v___x_1304_; 
lean_dec(v_a_1261_);
lean_dec_ref(v_00_u03b1_1249_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v___x_1303_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7, &lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___closed__7);
v___x_1304_ = lp_mathlib_panic___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__1(v___x_1303_, v___y_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
return v___x_1304_;
}
}
else
{
lean_object* v_a_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1312_; 
lean_dec_ref(v_00_u03b1_1249_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1305_ = lean_ctor_get(v___x_1260_, 0);
v_isSharedCheck_1312_ = !lean_is_exclusive(v___x_1260_);
if (v_isSharedCheck_1312_ == 0)
{
v___x_1307_ = v___x_1260_;
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_a_1305_);
lean_dec(v___x_1260_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1312_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1310_; 
if (v_isShared_1308_ == 0)
{
v___x_1310_ = v___x_1307_;
goto v_reusejp_1309_;
}
else
{
lean_object* v_reuseFailAlloc_1311_; 
v_reuseFailAlloc_1311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1311_, 0, v_a_1305_);
v___x_1310_ = v_reuseFailAlloc_1311_;
goto v_reusejp_1309_;
}
v_reusejp_1309_:
{
return v___x_1310_;
}
}
}
}
else
{
lean_object* v_a_1313_; lean_object* v___x_1315_; uint8_t v_isShared_1316_; uint8_t v_isSharedCheck_1320_; 
lean_dec_ref(v_00_u03b1_1249_);
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1313_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1315_ = v___x_1258_;
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
else
{
lean_inc(v_a_1313_);
lean_dec(v___x_1258_);
v___x_1315_ = lean_box(0);
v_isShared_1316_ = v_isSharedCheck_1320_;
goto v_resetjp_1314_;
}
v_resetjp_1314_:
{
lean_object* v___x_1318_; 
if (v_isShared_1316_ == 0)
{
v___x_1318_ = v___x_1315_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v_a_1313_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
return v___x_1318_;
}
}
}
}
v___jp_1321_:
{
if (lean_obj_tag(v___y_1322_) == 0)
{
lean_object* v_a_1323_; 
v_a_1323_ = lean_ctor_get(v___y_1322_, 0);
lean_inc(v_a_1323_);
lean_dec_ref_known(v___y_1322_, 1);
v_00_u03b1_1249_ = v_a_1323_;
v___y_1250_ = v_a_1163_;
v___y_1251_ = v_a_1164_;
v___y_1252_ = v_a_1165_;
v___y_1253_ = v_a_1166_;
v___y_1254_ = v_a_1167_;
v___y_1255_ = v_a_1168_;
v___y_1256_ = v_a_1169_;
v___y_1257_ = v_a_1170_;
goto v___jp_1248_;
}
else
{
lean_object* v_a_1324_; lean_object* v___x_1326_; uint8_t v_isShared_1327_; uint8_t v_isSharedCheck_1331_; 
lean_del_object(v___x_1204_);
lean_dec(v_a_1202_);
v_a_1324_ = lean_ctor_get(v___y_1322_, 0);
v_isSharedCheck_1331_ = !lean_is_exclusive(v___y_1322_);
if (v_isSharedCheck_1331_ == 0)
{
v___x_1326_ = v___y_1322_;
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
else
{
lean_inc(v_a_1324_);
lean_dec(v___y_1322_);
v___x_1326_ = lean_box(0);
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
v_resetjp_1325_:
{
lean_object* v___x_1329_; 
if (v_isShared_1327_ == 0)
{
v___x_1329_ = v___x_1326_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1330_; 
v_reuseFailAlloc_1330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1330_, 0, v_a_1324_);
v___x_1329_ = v_reuseFailAlloc_1330_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
return v___x_1329_;
}
}
}
}
}
}
else
{
lean_object* v_a_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1404_; 
v_a_1397_ = lean_ctor_get(v___x_1201_, 0);
v_isSharedCheck_1404_ = !lean_is_exclusive(v___x_1201_);
if (v_isSharedCheck_1404_ == 0)
{
v___x_1399_ = v___x_1201_;
v_isShared_1400_ = v_isSharedCheck_1404_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_a_1397_);
lean_dec(v___x_1201_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1404_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v___x_1402_; 
if (v_isShared_1400_ == 0)
{
v___x_1402_ = v___x_1399_;
goto v_reusejp_1401_;
}
else
{
lean_object* v_reuseFailAlloc_1403_; 
v_reuseFailAlloc_1403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1403_, 0, v_a_1397_);
v___x_1402_ = v_reuseFailAlloc_1403_;
goto v_reusejp_1401_;
}
v_reusejp_1401_:
{
return v___x_1402_;
}
}
}
v___jp_1172_:
{
uint8_t v___x_1179_; lean_object* v___x_1180_; 
v___x_1179_ = 0;
v___x_1180_ = l_Lean_Meta_intro1Core(v_a_1178_, v___x_1179_, v___y_1173_, v___y_1177_, v___y_1175_, v___y_1174_);
if (lean_obj_tag(v___x_1180_) == 0)
{
lean_object* v_a_1181_; lean_object* v_snd_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1191_; 
v_a_1181_ = lean_ctor_get(v___x_1180_, 0);
lean_inc(v_a_1181_);
lean_dec_ref_known(v___x_1180_, 1);
v_snd_1182_ = lean_ctor_get(v_a_1181_, 1);
v_isSharedCheck_1191_ = !lean_is_exclusive(v_a_1181_);
if (v_isSharedCheck_1191_ == 0)
{
lean_object* v_unused_1192_; 
v_unused_1192_ = lean_ctor_get(v_a_1181_, 0);
lean_dec(v_unused_1192_);
v___x_1184_ = v_a_1181_;
v_isShared_1185_ = v_isSharedCheck_1191_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_snd_1182_);
lean_dec(v_a_1181_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1191_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v___x_1186_; lean_object* v___x_1188_; 
v___x_1186_ = lean_box(0);
if (v_isShared_1185_ == 0)
{
lean_ctor_set_tag(v___x_1184_, 1);
lean_ctor_set(v___x_1184_, 1, v___x_1186_);
lean_ctor_set(v___x_1184_, 0, v_snd_1182_);
v___x_1188_ = v___x_1184_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_snd_1182_);
lean_ctor_set(v_reuseFailAlloc_1190_, 1, v___x_1186_);
v___x_1188_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
lean_object* v___x_1189_; 
v___x_1189_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1188_, v___y_1176_, v___y_1173_, v___y_1177_, v___y_1175_, v___y_1174_);
return v___x_1189_;
}
}
}
else
{
lean_object* v_a_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1200_; 
v_a_1193_ = lean_ctor_get(v___x_1180_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1180_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1195_ = v___x_1180_;
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_a_1193_);
lean_dec(v___x_1180_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1198_; 
if (v_isShared_1196_ == 0)
{
v___x_1198_ = v___x_1195_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_a_1193_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality___boxed(lean_object* v_stx_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_, lean_object* v_a_1410_, lean_object* v_a_1411_, lean_object* v_a_1412_, lean_object* v_a_1413_, lean_object* v_a_1414_){
_start:
{
lean_object* v_res_1415_; 
v_res_1415_ = lp_mathlib_Mathlib_Tactic_Nontriviality_elabNontriviality(v_stx_1405_, v_a_1406_, v_a_1407_, v_a_1408_, v_a_1409_, v_a_1410_, v_a_1411_, v_a_1412_, v_a_1413_);
lean_dec(v_a_1413_);
lean_dec_ref(v_a_1412_);
lean_dec(v_a_1411_);
lean_dec_ref(v_a_1410_);
lean_dec(v_a_1409_);
lean_dec_ref(v_a_1408_);
lean_dec(v_a_1407_);
lean_dec_ref(v_a_1406_);
lean_dec(v_stx_1405_);
return v_res_1415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0(lean_object* v_00_u03b1_1416_, lean_object* v_msg_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_){
_start:
{
lean_object* v___x_1427_; 
v___x_1427_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___redArg(v_msg_1417_, v___y_1422_, v___y_1423_, v___y_1424_, v___y_1425_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0___boxed(lean_object* v_00_u03b1_1428_, lean_object* v_msg_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_){
_start:
{
lean_object* v_res_1439_; 
v_res_1439_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Nontriviality_elabNontriviality_spec__0(v_00_u03b1_1428_, v_msg_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_, v___y_1437_);
lean_dec(v___y_1437_);
lean_dec_ref(v___y_1436_);
lean_dec(v___y_1435_);
lean_dec_ref(v___y_1434_);
lean_dec(v___y_1433_);
lean_dec_ref(v___y_1432_);
lean_dec(v___y_1431_);
lean_dec_ref(v___y_1430_);
return v_res_1439_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Macro(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_SolveByElim(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_MetaM(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_SolveByElim(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality = _init_lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Nontriviality_nontriviality);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_SolveByElim(uint8_t builtin);
lean_object* initialize_Qq_Qq_Macro(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
lean_object* initialize_Qq_Qq_MetaM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_SolveByElim(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Nontriviality_Core(builtin);
}
#ifdef __cplusplus
}
#endif
