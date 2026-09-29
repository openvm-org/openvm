// Lean compiler output
// Module: Mathlib.Tactic.SplitIfs
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Location public meta import Lean.Meta.Tactic.SplitIf public meta import Lean.Elab.Tactic.Simp public import Mathlib.Tactic.Core
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SplitIf_getSimpContext(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setFailIfUnchanged(lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_Simp_SimprocsArray_add(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SplitIf_mkDischarge_x3f___redArg(uint8_t, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_simpLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getRevArg_x21(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
uint8_t l_Lean_Expr_isIte(lean_object*);
uint8_t l_Lean_Expr_isDIte(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_LocalContext_getFVarIds(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Tactic_getFVarId(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_find_expr(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_expandLocation(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_binderIdent;
extern lean_object* l_Lean_Parser_Tactic_location;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_target_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_target_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_hyp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_hyp_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___boxed(lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 152, 123, 219, 220, 182, 189, 250)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "reduceCtorEq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__1_value),LEAN_SCALAR_PTR_LITERAL(241, 230, 128, 19, 70, 224, 61, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticBy_cases_:_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 65, 31, 128, 134, 243, 21, 139)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "by_cases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "split_ifs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(44, 149, 114, 211, 22, 124, 131, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "no if-then-else conditions to split"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "splitIfs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(109, 48, 181, 174, 145, 245, 228, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__13_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__15_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__18_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_splitIfs___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_splitIfs___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_splitIfs___closed__27;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_splitIfs;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unused name: "};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_splitIfs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorIdx(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorIdx(v_x_4_);
lean_dec(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(lean_object* v_t_6_, lean_object* v_k_7_){
_start:
{
if (lean_obj_tag(v_t_6_) == 0)
{
return v_k_7_;
}
else
{
lean_object* v_fvarId_8_; lean_object* v___x_9_; 
v_fvarId_8_ = lean_ctor_get(v_t_6_, 0);
lean_inc(v_fvarId_8_);
lean_dec_ref_known(v_t_6_, 1);
v___x_9_ = lean_apply_1(v_k_7_, v_fvarId_8_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim(lean_object* v_motive_10_, lean_object* v_ctorIdx_11_, lean_object* v_t_12_, lean_object* v_h_13_, lean_object* v_k_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(v_t_12_, v_k_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_18_, v_h_19_, v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_target_elim___redArg(lean_object* v_t_22_, lean_object* v_target_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(v_t_22_, v_target_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_target_elim(lean_object* v_motive_25_, lean_object* v_t_26_, lean_object* v_h_27_, lean_object* v_target_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(v_t_26_, v_target_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_hyp_elim___redArg(lean_object* v_t_30_, lean_object* v_hyp_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(v_t_30_, v_hyp_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_hyp_elim(lean_object* v_motive_33_, lean_object* v_t_34_, lean_object* v_h_35_, lean_object* v_hyp_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_SplitPosition_ctorElim___redArg(v_t_34_, v_hyp_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(lean_object* v_e_38_, lean_object* v___y_39_){
_start:
{
uint8_t v___x_41_; 
v___x_41_ = l_Lean_Expr_hasMVar(v_e_38_);
if (v___x_41_ == 0)
{
lean_object* v___x_42_; 
v___x_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_42_, 0, v_e_38_);
return v___x_42_;
}
else
{
lean_object* v___x_43_; lean_object* v_mctx_44_; lean_object* v___x_45_; lean_object* v_fst_46_; lean_object* v_snd_47_; lean_object* v___x_48_; lean_object* v_cache_49_; lean_object* v_zetaDeltaFVarIds_50_; lean_object* v_postponed_51_; lean_object* v_diag_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_61_; 
v___x_43_ = lean_st_ref_get(v___y_39_);
v_mctx_44_ = lean_ctor_get(v___x_43_, 0);
lean_inc_ref(v_mctx_44_);
lean_dec(v___x_43_);
v___x_45_ = l_Lean_instantiateMVarsCore(v_mctx_44_, v_e_38_);
v_fst_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc(v_fst_46_);
v_snd_47_ = lean_ctor_get(v___x_45_, 1);
lean_inc(v_snd_47_);
lean_dec_ref(v___x_45_);
v___x_48_ = lean_st_ref_take(v___y_39_);
v_cache_49_ = lean_ctor_get(v___x_48_, 1);
v_zetaDeltaFVarIds_50_ = lean_ctor_get(v___x_48_, 2);
v_postponed_51_ = lean_ctor_get(v___x_48_, 3);
v_diag_52_ = lean_ctor_get(v___x_48_, 4);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_48_);
if (v_isSharedCheck_61_ == 0)
{
lean_object* v_unused_62_; 
v_unused_62_ = lean_ctor_get(v___x_48_, 0);
lean_dec(v_unused_62_);
v___x_54_ = v___x_48_;
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_diag_52_);
lean_inc(v_postponed_51_);
lean_inc(v_zetaDeltaFVarIds_50_);
lean_inc(v_cache_49_);
lean_dec(v___x_48_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_61_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 0, v_snd_47_);
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_snd_47_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v_cache_49_);
lean_ctor_set(v_reuseFailAlloc_60_, 2, v_zetaDeltaFVarIds_50_);
lean_ctor_set(v_reuseFailAlloc_60_, 3, v_postponed_51_);
lean_ctor_set(v_reuseFailAlloc_60_, 4, v_diag_52_);
v___x_57_ = v_reuseFailAlloc_60_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = lean_st_ref_set(v___y_39_, v___x_57_);
v___x_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_59_, 0, v_fst_46_);
return v___x_59_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg___boxed(lean_object* v_e_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(v_e_63_, v___y_64_);
lean_dec(v___y_64_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0(lean_object* v_e_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(v_e_67_, v___y_73_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___boxed(lean_object* v_e_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0(v_e_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_, v___y_84_, v___y_85_, v___y_86_);
lean_dec(v___y_86_);
lean_dec_ref(v___y_85_);
lean_dec(v___y_84_);
lean_dec_ref(v___y_83_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1(size_t v_sz_89_, size_t v_i_90_, lean_object* v_bs_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
uint8_t v___x_101_; 
v___x_101_ = lean_usize_dec_lt(v_i_90_, v_sz_89_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; 
v___x_102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_102_, 0, v_bs_91_);
return v___x_102_;
}
else
{
lean_object* v_v_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v_v_103_ = lean_array_uget(v_bs_91_, v_i_90_);
lean_inc(v_v_103_);
v___x_104_ = l_Lean_mkFVar(v_v_103_);
lean_inc(v___y_99_);
lean_inc_ref(v___y_98_);
lean_inc(v___y_97_);
lean_inc_ref(v___y_96_);
v___x_105_ = lean_infer_type(v___x_104_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
if (lean_obj_tag(v___x_105_) == 0)
{
lean_object* v_a_106_; lean_object* v___x_107_; 
v_a_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_a_106_);
lean_dec_ref_known(v___x_105_, 1);
v___x_107_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(v_a_106_, v___y_97_);
if (lean_obj_tag(v___x_107_) == 0)
{
lean_object* v_a_108_; lean_object* v___x_109_; lean_object* v_bs_x27_110_; lean_object* v___x_111_; lean_object* v___x_112_; size_t v___x_113_; size_t v___x_114_; lean_object* v___x_115_; 
v_a_108_ = lean_ctor_get(v___x_107_, 0);
lean_inc(v_a_108_);
lean_dec_ref_known(v___x_107_, 1);
v___x_109_ = lean_unsigned_to_nat(0u);
v_bs_x27_110_ = lean_array_uset(v_bs_91_, v_i_90_, v___x_109_);
v___x_111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_111_, 0, v_v_103_);
v___x_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_a_108_);
v___x_113_ = ((size_t)1ULL);
v___x_114_ = lean_usize_add(v_i_90_, v___x_113_);
v___x_115_ = lean_array_uset(v_bs_x27_110_, v_i_90_, v___x_112_);
v_i_90_ = v___x_114_;
v_bs_91_ = v___x_115_;
goto _start;
}
else
{
lean_object* v_a_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_124_; 
lean_dec(v_v_103_);
lean_dec_ref(v_bs_91_);
v_a_117_ = lean_ctor_get(v___x_107_, 0);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_107_);
if (v_isSharedCheck_124_ == 0)
{
v___x_119_ = v___x_107_;
v_isShared_120_ = v_isSharedCheck_124_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_a_117_);
lean_dec(v___x_107_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_124_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_122_; 
if (v_isShared_120_ == 0)
{
v___x_122_ = v___x_119_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v_a_117_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
}
}
else
{
lean_object* v_a_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
lean_dec(v_v_103_);
lean_dec_ref(v_bs_91_);
v_a_125_ = lean_ctor_get(v___x_105_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v___x_105_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_a_125_);
lean_dec(v___x_105_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_a_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1___boxed(lean_object* v_sz_133_, lean_object* v_i_134_, lean_object* v_bs_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_){
_start:
{
size_t v_sz_boxed_145_; size_t v_i_boxed_146_; lean_object* v_res_147_; 
v_sz_boxed_145_ = lean_unbox_usize(v_sz_133_);
lean_dec(v_sz_133_);
v_i_boxed_146_ = lean_unbox_usize(v_i_134_);
lean_dec(v_i_134_);
v_res_147_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1(v_sz_boxed_145_, v_i_boxed_146_, v_bs_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
lean_dec(v___y_141_);
lean_dec_ref(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2(size_t v_sz_148_, size_t v_i_149_, lean_object* v_bs_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
uint8_t v___x_160_; 
v___x_160_ = lean_usize_dec_lt(v_i_149_, v_sz_148_);
if (v___x_160_ == 0)
{
lean_object* v___x_161_; 
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v_bs_150_);
return v___x_161_;
}
else
{
lean_object* v_v_162_; lean_object* v___x_163_; 
v_v_162_ = lean_array_uget_borrowed(v_bs_150_, v_i_149_);
lean_inc(v_v_162_);
v___x_163_ = l_Lean_Elab_Tactic_getFVarId(v_v_162_, v___y_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, v___y_158_);
if (lean_obj_tag(v___x_163_) == 0)
{
lean_object* v_a_164_; lean_object* v___x_165_; lean_object* v_bs_x27_166_; size_t v___x_167_; size_t v___x_168_; lean_object* v___x_169_; 
v_a_164_ = lean_ctor_get(v___x_163_, 0);
lean_inc(v_a_164_);
lean_dec_ref_known(v___x_163_, 1);
v___x_165_ = lean_unsigned_to_nat(0u);
v_bs_x27_166_ = lean_array_uset(v_bs_150_, v_i_149_, v___x_165_);
v___x_167_ = ((size_t)1ULL);
v___x_168_ = lean_usize_add(v_i_149_, v___x_167_);
v___x_169_ = lean_array_uset(v_bs_x27_166_, v_i_149_, v_a_164_);
v_i_149_ = v___x_168_;
v_bs_150_ = v___x_169_;
goto _start;
}
else
{
lean_object* v_a_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_178_; 
lean_dec_ref(v_bs_150_);
v_a_171_ = lean_ctor_get(v___x_163_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_163_);
if (v_isSharedCheck_178_ == 0)
{
v___x_173_ = v___x_163_;
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_a_171_);
lean_dec(v___x_163_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_176_; 
if (v_isShared_174_ == 0)
{
v___x_176_ = v___x_173_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v_a_171_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2___boxed(lean_object* v_sz_179_, lean_object* v_i_180_, lean_object* v_bs_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_){
_start:
{
size_t v_sz_boxed_191_; size_t v_i_boxed_192_; lean_object* v_res_193_; 
v_sz_boxed_191_ = lean_unbox_usize(v_sz_179_);
lean_dec(v_sz_179_);
v_i_boxed_192_ = lean_unbox_usize(v_i_180_);
lean_dec(v_i_180_);
v_res_193_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2(v_sz_boxed_191_, v_i_boxed_192_, v_bs_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
lean_dec(v___y_187_);
lean_dec_ref(v___y_186_);
lean_dec(v___y_185_);
lean_dec_ref(v___y_184_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates(lean_object* v_loc_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_){
_start:
{
if (lean_obj_tag(v_loc_194_) == 0)
{
lean_object* v_lctx_204_; lean_object* v___x_205_; size_t v_sz_206_; size_t v___x_207_; lean_object* v___x_208_; 
v_lctx_204_ = lean_ctor_get(v_a_199_, 2);
v___x_205_ = l_Lean_LocalContext_getFVarIds(v_lctx_204_);
v_sz_206_ = lean_array_size(v___x_205_);
v___x_207_ = ((size_t)0ULL);
v___x_208_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1(v_sz_206_, v___x_207_, v___x_205_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_208_) == 0)
{
lean_object* v_a_209_; lean_object* v___x_210_; 
v_a_209_ = lean_ctor_get(v___x_208_, 0);
lean_inc(v_a_209_);
lean_dec_ref_known(v___x_208_, 1);
v___x_210_ = l_Lean_Elab_Tactic_getMainTarget(v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v_a_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_222_; 
v_a_211_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_222_ == 0)
{
v___x_213_ = v___x_210_;
v_isShared_214_ = v_isSharedCheck_222_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_a_211_);
lean_dec(v___x_210_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_222_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_220_; 
v___x_215_ = lean_box(0);
v___x_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
lean_ctor_set(v___x_216_, 1, v_a_211_);
v___x_217_ = lean_array_to_list(v_a_209_);
v___x_218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_216_);
lean_ctor_set(v___x_218_, 1, v___x_217_);
if (v_isShared_214_ == 0)
{
lean_ctor_set(v___x_213_, 0, v___x_218_);
v___x_220_ = v___x_213_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_218_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
return v___x_220_;
}
}
}
else
{
lean_object* v_a_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_230_; 
lean_dec(v_a_209_);
v_a_223_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_230_ == 0)
{
v___x_225_ = v___x_210_;
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_a_223_);
lean_dec(v___x_210_);
v___x_225_ = lean_box(0);
v_isShared_226_ = v_isSharedCheck_230_;
goto v_resetjp_224_;
}
v_resetjp_224_:
{
lean_object* v___x_228_; 
if (v_isShared_226_ == 0)
{
v___x_228_ = v___x_225_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_a_223_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
else
{
lean_object* v_a_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_238_; 
v_a_231_ = lean_ctor_get(v___x_208_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_208_);
if (v_isSharedCheck_238_ == 0)
{
v___x_233_ = v___x_208_;
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_a_231_);
lean_dec(v___x_208_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_236_; 
if (v_isShared_234_ == 0)
{
v___x_236_ = v___x_233_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_a_231_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
else
{
lean_object* v_hypotheses_239_; uint8_t v_type_240_; size_t v_sz_241_; size_t v___x_242_; lean_object* v___x_243_; 
v_hypotheses_239_ = lean_ctor_get(v_loc_194_, 0);
lean_inc_ref(v_hypotheses_239_);
v_type_240_ = lean_ctor_get_uint8(v_loc_194_, sizeof(void*)*1);
lean_dec_ref_known(v_loc_194_, 1);
v_sz_241_ = lean_array_size(v_hypotheses_239_);
v___x_242_ = ((size_t)0ULL);
v___x_243_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__2(v_sz_241_, v___x_242_, v_hypotheses_239_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_243_) == 0)
{
lean_object* v_a_244_; size_t v_sz_245_; lean_object* v___x_246_; 
v_a_244_ = lean_ctor_get(v___x_243_, 0);
lean_inc(v_a_244_);
lean_dec_ref_known(v___x_243_, 1);
v_sz_245_ = lean_array_size(v_a_244_);
v___x_246_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__1(v_sz_245_, v___x_242_, v_a_244_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_246_) == 0)
{
if (v_type_240_ == 0)
{
lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_255_; 
v_a_247_ = lean_ctor_get(v___x_246_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_246_);
if (v_isSharedCheck_255_ == 0)
{
v___x_249_ = v___x_246_;
v_isShared_250_ = v_isSharedCheck_255_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_246_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_255_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_251_; lean_object* v___x_253_; 
v___x_251_ = lean_array_to_list(v_a_247_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 0, v___x_251_);
v___x_253_ = v___x_249_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v___x_251_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_257_; 
v_a_256_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_a_256_);
lean_dec_ref_known(v___x_246_, 1);
v___x_257_ = l_Lean_Elab_Tactic_getMainTarget(v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_269_; 
v_a_258_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_269_ == 0)
{
v___x_260_ = v___x_257_;
v_isShared_261_ = v_isSharedCheck_269_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_a_258_);
lean_dec(v___x_257_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_269_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_267_; 
v___x_262_ = lean_box(0);
v___x_263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_258_);
v___x_264_ = lean_array_to_list(v_a_256_);
v___x_265_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_263_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
if (v_isShared_261_ == 0)
{
lean_ctor_set(v___x_260_, 0, v___x_265_);
v___x_267_ = v___x_260_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_265_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
}
else
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_277_; 
lean_dec(v_a_256_);
v_a_270_ = lean_ctor_get(v___x_257_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_277_ == 0)
{
v___x_272_ = v___x_257_;
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_257_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_277_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_275_; 
if (v_isShared_273_ == 0)
{
v___x_275_ = v___x_272_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_a_270_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
}
}
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
v_a_278_ = lean_ctor_get(v___x_246_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_246_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___x_246_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v___x_246_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_283_; 
if (v_isShared_281_ == 0)
{
v___x_283_ = v___x_280_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_a_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
}
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
v_a_286_ = lean_ctor_get(v___x_243_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_243_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_243_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_243_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates___boxed(lean_object* v_loc_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates(v_loc_294_, v_a_295_, v_a_296_, v_a_297_, v_a_298_, v_a_299_, v_a_300_, v_a_301_, v_a_302_);
lean_dec(v_a_302_);
lean_dec_ref(v_a_301_);
lean_dec(v_a_300_);
lean_dec_ref(v_a_299_);
lean_dec(v_a_298_);
lean_dec_ref(v_a_297_);
lean_dec(v_a_296_);
lean_dec_ref(v_a_295_);
return v_res_304_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0(lean_object* v_e_305_){
_start:
{
uint8_t v___y_307_; uint8_t v___x_312_; 
v___x_312_ = l_Lean_Expr_isIte(v_e_305_);
if (v___x_312_ == 0)
{
uint8_t v___x_313_; 
v___x_313_ = l_Lean_Expr_isDIte(v_e_305_);
v___y_307_ = v___x_313_;
goto v___jp_306_;
}
else
{
v___y_307_ = v___x_312_;
goto v___jp_306_;
}
v___jp_306_:
{
if (v___y_307_ == 0)
{
return v___y_307_;
}
else
{
lean_object* v___x_308_; lean_object* v___x_309_; uint8_t v___x_310_; 
v___x_308_ = lean_unsigned_to_nat(3u);
v___x_309_ = l_Lean_Expr_getRevArg_x21(v_e_305_, v___x_308_);
v___x_310_ = l_Lean_Expr_hasLooseBVars(v___x_309_);
lean_dec_ref(v___x_309_);
if (v___x_310_ == 0)
{
return v___y_307_;
}
else
{
uint8_t v___x_311_; 
v___x_311_ = 0;
return v___x_311_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0___boxed(lean_object* v_e_314_){
_start:
{
uint8_t v_res_315_; lean_object* v_r_316_; 
v_res_315_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___lam__0(v_e_314_);
lean_dec_ref(v_e_314_);
v_r_316_ = lean_box(v_res_315_);
return v_r_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f(lean_object* v_e_318_){
_start:
{
lean_object* v___f_319_; lean_object* v___x_320_; 
v___f_319_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___closed__0));
v___x_320_ = lean_find_expr(v___f_319_, v_e_318_);
if (lean_obj_tag(v___x_320_) == 0)
{
lean_object* v___x_321_; 
v___x_321_ = lean_box(0);
return v___x_321_;
}
else
{
lean_object* v_val_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_335_; 
v_val_322_ = lean_ctor_get(v___x_320_, 0);
v_isSharedCheck_335_ = !lean_is_exclusive(v___x_320_);
if (v_isSharedCheck_335_ == 0)
{
v___x_324_ = v___x_320_;
v_isShared_325_ = v_isSharedCheck_335_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_val_322_);
lean_dec(v___x_320_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_335_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___x_326_; lean_object* v_cond_327_; lean_object* v___x_328_; 
v___x_326_ = lean_unsigned_to_nat(3u);
v_cond_327_ = l_Lean_Expr_getRevArg_x21(v_val_322_, v___x_326_);
v___x_328_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f(v_cond_327_);
if (lean_obj_tag(v___x_328_) == 0)
{
lean_object* v___x_329_; lean_object* v_dec_330_; lean_object* v___x_331_; lean_object* v___x_333_; 
v___x_329_ = lean_unsigned_to_nat(2u);
v_dec_330_ = l_Lean_Expr_getRevArg_x21(v_val_322_, v___x_329_);
lean_dec(v_val_322_);
v___x_331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_331_, 0, v_cond_327_);
lean_ctor_set(v___x_331_, 1, v_dec_330_);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 0, v___x_331_);
v___x_333_ = v___x_324_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_331_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
else
{
lean_dec_ref(v_cond_327_);
lean_del_object(v___x_324_);
lean_dec(v_val_322_);
return v___x_328_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f___boxed(lean_object* v_e_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f(v_e_336_);
lean_dec_ref(v_e_336_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg(lean_object* v_as_x27_341_, lean_object* v_b_342_){
_start:
{
if (lean_obj_tag(v_as_x27_341_) == 0)
{
lean_object* v___x_344_; 
v___x_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_344_, 0, v_b_342_);
return v___x_344_;
}
else
{
lean_object* v_head_345_; lean_object* v_tail_346_; lean_object* v_fst_347_; lean_object* v_snd_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
lean_dec_ref(v_b_342_);
v_head_345_ = lean_ctor_get(v_as_x27_341_, 0);
v_tail_346_ = lean_ctor_get(v_as_x27_341_, 1);
v_fst_347_ = lean_ctor_get(v_head_345_, 0);
v_snd_348_ = lean_ctor_get(v_head_345_, 1);
v___x_349_ = lean_box(0);
v___x_350_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfToSplit_x3f(v_snd_348_);
if (lean_obj_tag(v___x_350_) == 1)
{
lean_object* v_val_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_370_; 
v_val_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_370_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_370_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_val_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_370_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v_fst_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_368_; 
v_fst_355_ = lean_ctor_get(v_val_351_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v_val_351_);
if (v_isSharedCheck_368_ == 0)
{
lean_object* v_unused_369_; 
v_unused_369_ = lean_ctor_get(v_val_351_, 1);
lean_dec(v_unused_369_);
v___x_357_ = v_val_351_;
v_isShared_358_ = v_isSharedCheck_368_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_fst_355_);
lean_dec(v_val_351_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_368_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_360_; 
lean_inc(v_fst_347_);
if (v_isShared_358_ == 0)
{
lean_ctor_set(v___x_357_, 1, v_fst_355_);
lean_ctor_set(v___x_357_, 0, v_fst_347_);
v___x_360_ = v___x_357_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_fst_347_);
lean_ctor_set(v_reuseFailAlloc_367_, 1, v_fst_355_);
v___x_360_ = v_reuseFailAlloc_367_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
lean_object* v___x_362_; 
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 0, v___x_360_);
v___x_362_ = v___x_353_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_360_);
v___x_362_ = v_reuseFailAlloc_366_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
v___x_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_349_);
v___x_365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
return v___x_365_;
}
}
}
}
}
else
{
lean_object* v___x_371_; 
lean_dec(v___x_350_);
v___x_371_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___closed__0));
v_as_x27_341_ = v_tail_346_;
v_b_342_ = v___x_371_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___boxed(lean_object* v_as_x27_373_, lean_object* v_b_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg(v_as_x27_373_, v_b_374_);
lean_dec(v_as_x27_373_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt(lean_object* v_loc_377_, lean_object* v_a_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_, lean_object* v_a_385_){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates(v_loc_377_, v_a_378_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_, v_a_385_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_404_; 
v_a_388_ = lean_ctor_get(v___x_387_, 0);
lean_inc(v_a_388_);
lean_dec_ref_known(v___x_387_, 1);
v___x_389_ = lean_box(0);
v___x_390_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg___closed__0));
v___x_391_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg(v_a_388_, v___x_390_);
lean_dec(v_a_388_);
v_a_392_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_404_ == 0)
{
v___x_394_ = v___x_391_;
v_isShared_395_ = v_isSharedCheck_404_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_391_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_404_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v_fst_396_; 
v_fst_396_ = lean_ctor_get(v_a_392_, 0);
lean_inc(v_fst_396_);
lean_dec(v_a_392_);
if (lean_obj_tag(v_fst_396_) == 0)
{
lean_object* v___x_398_; 
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 0, v___x_389_);
v___x_398_ = v___x_394_;
goto v_reusejp_397_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v___x_389_);
v___x_398_ = v_reuseFailAlloc_399_;
goto v_reusejp_397_;
}
v_reusejp_397_:
{
return v___x_398_;
}
}
else
{
lean_object* v_val_400_; lean_object* v___x_402_; 
v_val_400_ = lean_ctor_get(v_fst_396_, 0);
lean_inc(v_val_400_);
lean_dec_ref_known(v_fst_396_, 1);
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 0, v_val_400_);
v___x_402_ = v___x_394_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_val_400_);
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
v_a_405_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_387_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_387_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt___boxed(lean_object* v_loc_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_, lean_object* v_a_419_, lean_object* v_a_420_, lean_object* v_a_421_, lean_object* v_a_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt(v_loc_413_, v_a_414_, v_a_415_, v_a_416_, v_a_417_, v_a_418_, v_a_419_, v_a_420_, v_a_421_);
lean_dec(v_a_421_);
lean_dec_ref(v_a_420_);
lean_dec(v_a_419_);
lean_dec_ref(v_a_418_);
lean_dec(v_a_417_);
lean_dec_ref(v_a_416_);
lean_dec(v_a_415_);
lean_dec_ref(v_a_414_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0(lean_object* v_as_424_, lean_object* v_as_x27_425_, lean_object* v_b_426_, lean_object* v_a_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___redArg(v_as_x27_425_, v_b_426_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0___boxed(lean_object* v_as_438_, lean_object* v_as_x27_439_, lean_object* v_b_440_, lean_object* v_a_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt_spec__0(v_as_438_, v_as_x27_439_, v_b_440_, v_a_441_, v___y_442_, v___y_443_, v___y_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
lean_dec(v___y_445_);
lean_dec_ref(v___y_444_);
lean_dec(v___y_443_);
lean_dec_ref(v___y_442_);
lean_dec(v_as_x27_439_);
lean_dec(v_as_438_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg(lean_object* v_e_452_, lean_object* v___y_453_){
_start:
{
uint8_t v___x_455_; 
v___x_455_ = l_Lean_Expr_hasMVar(v_e_452_);
if (v___x_455_ == 0)
{
lean_object* v___x_456_; 
v___x_456_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_456_, 0, v_e_452_);
return v___x_456_;
}
else
{
lean_object* v___x_457_; lean_object* v_mctx_458_; lean_object* v___x_459_; lean_object* v_fst_460_; lean_object* v_snd_461_; lean_object* v___x_462_; lean_object* v_cache_463_; lean_object* v_zetaDeltaFVarIds_464_; lean_object* v_postponed_465_; lean_object* v_diag_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_475_; 
v___x_457_ = lean_st_ref_get(v___y_453_);
v_mctx_458_ = lean_ctor_get(v___x_457_, 0);
lean_inc_ref(v_mctx_458_);
lean_dec(v___x_457_);
v___x_459_ = l_Lean_instantiateMVarsCore(v_mctx_458_, v_e_452_);
v_fst_460_ = lean_ctor_get(v___x_459_, 0);
lean_inc(v_fst_460_);
v_snd_461_ = lean_ctor_get(v___x_459_, 1);
lean_inc(v_snd_461_);
lean_dec_ref(v___x_459_);
v___x_462_ = lean_st_ref_take(v___y_453_);
v_cache_463_ = lean_ctor_get(v___x_462_, 1);
v_zetaDeltaFVarIds_464_ = lean_ctor_get(v___x_462_, 2);
v_postponed_465_ = lean_ctor_get(v___x_462_, 3);
v_diag_466_ = lean_ctor_get(v___x_462_, 4);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_462_);
if (v_isSharedCheck_475_ == 0)
{
lean_object* v_unused_476_; 
v_unused_476_ = lean_ctor_get(v___x_462_, 0);
lean_dec(v_unused_476_);
v___x_468_ = v___x_462_;
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_diag_466_);
lean_inc(v_postponed_465_);
lean_inc(v_zetaDeltaFVarIds_464_);
lean_inc(v_cache_463_);
lean_dec(v___x_462_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___x_471_; 
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 0, v_snd_461_);
v___x_471_ = v___x_468_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_snd_461_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v_cache_463_);
lean_ctor_set(v_reuseFailAlloc_474_, 2, v_zetaDeltaFVarIds_464_);
lean_ctor_set(v_reuseFailAlloc_474_, 3, v_postponed_465_);
lean_ctor_set(v_reuseFailAlloc_474_, 4, v_diag_466_);
v___x_471_ = v_reuseFailAlloc_474_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_472_ = lean_st_ref_set(v___y_453_, v___x_471_);
v___x_473_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_473_, 0, v_fst_460_);
return v___x_473_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg___boxed(lean_object* v_e_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg(v_e_477_, v___y_478_);
lean_dec(v___y_478_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0(lean_object* v_e_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg(v_e_481_, v___y_486_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___boxed(lean_object* v_e_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0(v_e_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
lean_dec(v___y_492_);
return v_res_500_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4(void){
_start:
{
lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_508_ = lean_box(0);
v___x_509_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__3));
v___x_510_ = l_Lean_mkConst(v___x_509_, v___x_508_);
return v___x_510_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__4);
v___x_512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_512_, 0, v___x_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f(lean_object* v_e_513_, lean_object* v_a_514_, lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
lean_object* v___x_522_; lean_object* v_a_523_; uint8_t v___x_524_; lean_object* v___x_525_; 
v___x_522_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f_spec__0___redArg(v_e_513_, v_a_518_);
v_a_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_a_523_);
lean_dec_ref(v___x_522_);
v___x_524_ = 0;
v___x_525_ = l_Lean_Meta_SplitIf_mkDischarge_x3f___redArg(v___x_524_, v_a_517_);
if (lean_obj_tag(v___x_525_) == 0)
{
lean_object* v_a_526_; lean_object* v___x_527_; 
v_a_526_ = lean_ctor_get(v___x_525_, 0);
lean_inc(v_a_526_);
lean_dec_ref_known(v___x_525_, 1);
lean_inc(v_a_520_);
lean_inc_ref(v_a_519_);
lean_inc(v_a_518_);
lean_inc_ref(v_a_517_);
lean_inc(v_a_516_);
lean_inc_ref(v_a_515_);
lean_inc(v_a_514_);
lean_inc(v_a_523_);
v___x_527_ = lean_apply_9(v_a_526_, v_a_523_, v_a_514_, v_a_515_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_, lean_box(0));
if (lean_obj_tag(v___x_527_) == 0)
{
lean_object* v_a_528_; 
v_a_528_ = lean_ctor_get(v___x_527_, 0);
lean_inc(v_a_528_);
if (lean_obj_tag(v_a_528_) == 1)
{
lean_dec_ref_known(v_a_528_, 1);
lean_dec(v_a_523_);
return v___x_527_;
}
else
{
lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_542_; 
lean_dec(v_a_528_);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_527_);
if (v_isSharedCheck_542_ == 0)
{
lean_object* v_unused_543_; 
v_unused_543_ = lean_ctor_get(v___x_527_, 0);
lean_dec(v_unused_543_);
v___x_530_ = v___x_527_;
v_isShared_531_ = v_isSharedCheck_542_;
goto v_resetjp_529_;
}
else
{
lean_dec(v___x_527_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_542_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_532_; uint8_t v___x_533_; 
v___x_532_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__1));
v___x_533_ = l_Lean_Expr_isConstOf(v_a_523_, v___x_532_);
lean_dec(v_a_523_);
if (v___x_533_ == 0)
{
lean_object* v___x_534_; lean_object* v___x_536_; 
v___x_534_ = lean_box(0);
if (v_isShared_531_ == 0)
{
lean_ctor_set(v___x_530_, 0, v___x_534_);
v___x_536_ = v___x_530_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v___x_534_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
else
{
lean_object* v___x_538_; lean_object* v___x_540_; 
v___x_538_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___closed__5);
if (v_isShared_531_ == 0)
{
lean_ctor_set(v___x_530_, 0, v___x_538_);
v___x_540_ = v___x_530_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v___x_538_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
}
else
{
lean_dec(v_a_523_);
return v___x_527_;
}
}
else
{
lean_object* v_a_544_; lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_551_; 
lean_dec(v_a_523_);
v_a_544_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_551_ == 0)
{
v___x_546_ = v___x_525_;
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
else
{
lean_inc(v_a_544_);
lean_dec(v___x_525_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v___x_549_; 
if (v_isShared_547_ == 0)
{
v___x_549_ = v___x_546_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v_a_544_);
v___x_549_ = v_reuseFailAlloc_550_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
return v___x_549_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f___boxed(lean_object* v_e_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_, lean_object* v_a_556_, lean_object* v_a_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_discharge_x3f(v_e_552_, v_a_553_, v_a_554_, v_a_555_, v_a_556_, v_a_557_, v_a_558_, v_a_559_);
lean_dec(v_a_559_);
lean_dec_ref(v_a_558_);
lean_dec(v_a_557_);
lean_dec_ref(v_a_556_);
lean_dec(v_a_555_);
lean_dec_ref(v_a_554_);
lean_dec(v_a_553_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt(lean_object* v_loc_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_){
_start:
{
lean_object* v___x_580_; 
v___x_580_ = l_Lean_Meta_SplitIf_getSimpContext(v_a_575_, v_a_576_, v_a_577_, v_a_578_);
if (lean_obj_tag(v___x_580_) == 0)
{
lean_object* v_a_581_; uint8_t v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v_a_581_ = lean_ctor_get(v___x_580_, 0);
lean_inc(v_a_581_);
lean_dec_ref_known(v___x_580_, 1);
v___x_582_ = 0;
v___x_583_ = l_Lean_Meta_Simp_Context_setFailIfUnchanged(v_a_581_, v___x_582_);
v___x_584_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__0));
v___x_585_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__2));
v___x_586_ = l_Lean_Meta_Simp_SimprocsArray_add(v___x_584_, v___x_585_, v___x_582_, v_a_577_, v_a_578_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v_a_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v_a_587_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_a_587_);
lean_dec_ref_known(v___x_586_, 1);
v___x_588_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___closed__4));
v___x_589_ = l_Lean_Elab_Tactic_simpLocation(v___x_583_, v_a_587_, v___x_588_, v_loc_570_, v_a_571_, v_a_572_, v_a_573_, v_a_574_, v_a_575_, v_a_576_, v_a_577_, v_a_578_);
if (lean_obj_tag(v___x_589_) == 0)
{
lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_597_; 
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_589_);
if (v_isSharedCheck_597_ == 0)
{
lean_object* v_unused_598_; 
v_unused_598_ = lean_ctor_get(v___x_589_, 0);
lean_dec(v_unused_598_);
v___x_591_ = v___x_589_;
v_isShared_592_ = v_isSharedCheck_597_;
goto v_resetjp_590_;
}
else
{
lean_dec(v___x_589_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_597_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_593_; lean_object* v___x_595_; 
v___x_593_ = lean_box(0);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_593_);
v___x_595_ = v___x_591_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v___x_593_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
else
{
lean_object* v_a_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_606_; 
v_a_599_ = lean_ctor_get(v___x_589_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_589_);
if (v_isSharedCheck_606_ == 0)
{
v___x_601_ = v___x_589_;
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_a_599_);
lean_dec(v___x_589_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_604_; 
if (v_isShared_602_ == 0)
{
v___x_604_ = v___x_601_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_a_599_);
v___x_604_ = v_reuseFailAlloc_605_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
return v___x_604_;
}
}
}
}
else
{
lean_object* v_a_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_614_; 
lean_dec_ref(v___x_583_);
lean_dec(v_loc_570_);
v_a_607_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_614_ == 0)
{
v___x_609_ = v___x_586_;
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_a_607_);
lean_dec(v___x_586_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v___x_612_; 
if (v_isShared_610_ == 0)
{
v___x_612_ = v___x_609_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_a_607_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_loc_570_);
v_a_615_ = lean_ctor_get(v___x_580_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_580_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_580_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_580_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___boxed(lean_object* v_loc_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt(v_loc_623_, v_a_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_);
lean_dec(v_a_631_);
lean_dec_ref(v_a_630_);
lean_dec(v_a_629_);
lean_dec_ref(v_a_628_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1(lean_object* v_cond_642_, lean_object* v_hName_643_, lean_object* v_loc_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_, lean_object* v_a_651_, lean_object* v_a_652_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = l_Lean_Elab_Term_exprToSyntax(v_cond_642_, v_a_647_, v_a_648_, v_a_649_, v_a_650_, v_a_651_, v_a_652_);
if (lean_obj_tag(v___x_654_) == 0)
{
lean_object* v_a_655_; lean_object* v_ref_656_; uint8_t v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v_a_655_ = lean_ctor_get(v___x_654_, 0);
lean_inc(v_a_655_);
lean_dec_ref_known(v___x_654_, 1);
v_ref_656_ = lean_ctor_get(v_a_651_, 5);
v___x_657_ = 0;
v___x_658_ = l_Lean_SourceInfo_fromRef(v_ref_656_, v___x_657_);
v___x_659_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__1));
v___x_660_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__2));
lean_inc_n(v___x_658_, 3);
v___x_661_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_661_, 0, v___x_658_);
lean_ctor_set(v___x_661_, 1, v___x_660_);
v___x_662_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__4));
v___x_663_ = l_Lean_mkIdent(v_hName_643_);
v___x_664_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___closed__5));
v___x_665_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_665_, 0, v___x_658_);
lean_ctor_set(v___x_665_, 1, v___x_664_);
v___x_666_ = l_Lean_Syntax_node2(v___x_658_, v___x_662_, v___x_663_, v___x_665_);
v___x_667_ = l_Lean_Syntax_node3(v___x_658_, v___x_659_, v___x_661_, v___x_666_, v_a_655_);
v___x_668_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_668_, 0, v___x_667_);
v___x_669_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___boxed), 10, 1);
lean_closure_set(v___x_669_, 0, v_loc_644_);
v___x_670_ = lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals(v___x_668_, v___x_669_, v_a_645_, v_a_646_, v_a_647_, v_a_648_, v_a_649_, v_a_650_, v_a_651_, v_a_652_);
return v___x_670_;
}
else
{
lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
lean_dec(v_loc_644_);
lean_dec(v_hName_643_);
v_a_671_ = lean_ctor_get(v___x_654_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_654_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_654_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_dec(v___x_654_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___boxed(lean_object* v_cond_679_, lean_object* v_hName_680_, lean_object* v_loc_681_, lean_object* v_a_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1(v_cond_679_, v_hName_680_, v_loc_681_, v_a_682_, v_a_683_, v_a_684_, v_a_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_);
lean_dec(v_a_689_);
lean_dec_ref(v_a_688_);
lean_dec(v_a_687_);
lean_dec_ref(v_a_686_);
lean_dec(v_a_685_);
lean_dec_ref(v_a_684_);
lean_dec(v_a_683_);
lean_dec_ref(v_a_682_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg(lean_object* v_hNames_706_, lean_object* v_a_707_, lean_object* v_a_708_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lean_st_ref_get(v_hNames_706_);
if (lean_obj_tag(v___x_710_) == 0)
{
lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_711_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__1));
v___x_712_ = l_Lean_Core_mkFreshUserName(v___x_711_, v_a_707_, v_a_708_);
return v___x_712_;
}
else
{
lean_object* v_head_713_; lean_object* v_tail_714_; lean_object* v___x_715_; lean_object* v___x_716_; uint8_t v___x_717_; 
v_head_713_ = lean_ctor_get(v___x_710_, 0);
lean_inc_n(v_head_713_, 2);
v_tail_714_ = lean_ctor_get(v___x_710_, 1);
lean_inc(v_tail_714_);
lean_dec_ref_known(v___x_710_, 2);
v___x_715_ = lean_st_ref_set(v_hNames_706_, v_tail_714_);
v___x_716_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__4));
v___x_717_ = l_Lean_Syntax_isOfKind(v_head_713_, v___x_716_);
if (v___x_717_ == 0)
{
lean_object* v___x_718_; lean_object* v___x_719_; 
lean_dec(v_head_713_);
v___x_718_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__6));
v___x_719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
return v___x_719_;
}
else
{
lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; uint8_t v___x_723_; 
v___x_720_ = lean_unsigned_to_nat(0u);
v___x_721_ = l_Lean_Syntax_getArg(v_head_713_, v___x_720_);
lean_dec(v_head_713_);
v___x_722_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__8));
lean_inc(v___x_721_);
v___x_723_ = l_Lean_Syntax_isOfKind(v___x_721_, v___x_722_);
if (v___x_723_ == 0)
{
lean_object* v___x_724_; lean_object* v___x_725_; 
lean_dec(v___x_721_);
v___x_724_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___closed__6));
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
else
{
lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_726_ = l_Lean_TSyntax_getId(v___x_721_);
lean_dec(v___x_721_);
v___x_727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_727_, 0, v___x_726_);
return v___x_727_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg___boxed(lean_object* v_hNames_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_){
_start:
{
lean_object* v_res_732_; 
v_res_732_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg(v_hNames_728_, v_a_729_, v_a_730_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
lean_dec(v_hNames_728_);
return v_res_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName(lean_object* v_hNames_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_, lean_object* v_a_737_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg(v_hNames_733_, v_a_736_, v_a_737_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___boxed(lean_object* v_hNames_740_, lean_object* v_a_741_, lean_object* v_a_742_, lean_object* v_a_743_, lean_object* v_a_744_, lean_object* v_a_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName(v_hNames_740_, v_a_741_, v_a_742_, v_a_743_, v_a_744_);
lean_dec(v_a_744_);
lean_dec_ref(v_a_743_);
lean_dec(v_a_742_);
lean_dec_ref(v_a_741_);
lean_dec(v_hNames_740_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg(lean_object* v_cond_750_, lean_object* v_not__cond_751_, lean_object* v_as_752_, size_t v_sz_753_, size_t v_i_754_, lean_object* v_b_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
uint8_t v___x_761_; 
v___x_761_ = lean_usize_dec_lt(v_i_754_, v_sz_753_);
if (v___x_761_ == 0)
{
lean_object* v___x_762_; 
v___x_762_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_762_, 0, v_b_755_);
return v___x_762_;
}
else
{
lean_object* v_a_763_; lean_object* v___x_764_; 
lean_dec_ref(v_b_755_);
v_a_763_ = lean_array_uget_borrowed(v_as_752_, v_i_754_);
lean_inc(v___y_759_);
lean_inc_ref(v___y_758_);
lean_inc(v___y_757_);
lean_inc_ref(v___y_756_);
lean_inc(v_a_763_);
v___x_764_ = lean_infer_type(v_a_763_, v___y_756_, v___y_757_, v___y_758_, v___y_759_);
if (lean_obj_tag(v___x_764_) == 0)
{
lean_object* v_a_765_; lean_object* v___x_766_; 
v_a_765_ = lean_ctor_get(v___x_764_, 0);
lean_inc(v_a_765_);
lean_dec_ref_known(v___x_764_, 1);
v___x_766_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getSplitCandidates_spec__0___redArg(v_a_765_, v___y_757_);
if (lean_obj_tag(v___x_766_) == 0)
{
lean_object* v_a_767_; lean_object* v___x_769_; uint8_t v_isShared_770_; uint8_t v_isSharedCheck_790_; 
v_a_767_ = lean_ctor_get(v___x_766_, 0);
v_isSharedCheck_790_ = !lean_is_exclusive(v___x_766_);
if (v_isSharedCheck_790_ == 0)
{
v___x_769_ = v___x_766_;
v_isShared_770_ = v_isSharedCheck_790_;
goto v_resetjp_768_;
}
else
{
lean_inc(v_a_767_);
lean_dec(v___x_766_);
v___x_769_ = lean_box(0);
v_isShared_770_ = v_isSharedCheck_790_;
goto v_resetjp_768_;
}
v_resetjp_768_:
{
lean_object* v___x_771_; uint8_t v___x_772_; 
v___x_771_ = lean_box(0);
v___x_772_ = lean_expr_eqv(v_cond_750_, v_a_767_);
if (v___x_772_ == 0)
{
uint8_t v___x_773_; 
v___x_773_ = lean_expr_eqv(v_not__cond_751_, v_a_767_);
lean_dec(v_a_767_);
if (v___x_773_ == 0)
{
lean_object* v___x_774_; size_t v___x_775_; size_t v___x_776_; 
lean_del_object(v___x_769_);
v___x_774_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___closed__0));
v___x_775_ = ((size_t)1ULL);
v___x_776_ = lean_usize_add(v_i_754_, v___x_775_);
v_i_754_ = v___x_776_;
v_b_755_ = v___x_774_;
goto _start;
}
else
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_782_; 
v___x_778_ = lean_box(v___x_761_);
v___x_779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_779_, 0, v___x_778_);
v___x_780_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_780_, 0, v___x_779_);
lean_ctor_set(v___x_780_, 1, v___x_771_);
if (v_isShared_770_ == 0)
{
lean_ctor_set(v___x_769_, 0, v___x_780_);
v___x_782_ = v___x_769_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v___x_780_);
v___x_782_ = v_reuseFailAlloc_783_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
return v___x_782_;
}
}
}
else
{
lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_788_; 
lean_dec(v_a_767_);
v___x_784_ = lean_box(v___x_761_);
v___x_785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_785_, 0, v___x_784_);
v___x_786_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_786_, 0, v___x_785_);
lean_ctor_set(v___x_786_, 1, v___x_771_);
if (v_isShared_770_ == 0)
{
lean_ctor_set(v___x_769_, 0, v___x_786_);
v___x_788_ = v___x_769_;
goto v_reusejp_787_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v___x_786_);
v___x_788_ = v_reuseFailAlloc_789_;
goto v_reusejp_787_;
}
v_reusejp_787_:
{
return v___x_788_;
}
}
}
}
else
{
lean_object* v_a_791_; lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_798_; 
v_a_791_ = lean_ctor_get(v___x_766_, 0);
v_isSharedCheck_798_ = !lean_is_exclusive(v___x_766_);
if (v_isSharedCheck_798_ == 0)
{
v___x_793_ = v___x_766_;
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
else
{
lean_inc(v_a_791_);
lean_dec(v___x_766_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_796_; 
if (v_isShared_794_ == 0)
{
v___x_796_ = v___x_793_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v_a_791_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
return v___x_796_;
}
}
}
}
else
{
lean_object* v_a_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_806_; 
v_a_799_ = lean_ctor_get(v___x_764_, 0);
v_isSharedCheck_806_ = !lean_is_exclusive(v___x_764_);
if (v_isSharedCheck_806_ == 0)
{
v___x_801_ = v___x_764_;
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_a_799_);
lean_dec(v___x_764_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_804_; 
if (v_isShared_802_ == 0)
{
v___x_804_ = v___x_801_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v_a_799_);
v___x_804_ = v_reuseFailAlloc_805_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
return v___x_804_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___boxed(lean_object* v_cond_807_, lean_object* v_not__cond_808_, lean_object* v_as_809_, lean_object* v_sz_810_, lean_object* v_i_811_, lean_object* v_b_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
size_t v_sz_boxed_818_; size_t v_i_boxed_819_; lean_object* v_res_820_; 
v_sz_boxed_818_ = lean_unbox_usize(v_sz_810_);
lean_dec(v_sz_810_);
v_i_boxed_819_ = lean_unbox_usize(v_i_811_);
lean_dec(v_i_811_);
v_res_820_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg(v_cond_807_, v_not__cond_808_, v_as_809_, v_sz_boxed_818_, v_i_boxed_819_, v_b_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec(v___y_814_);
lean_dec_ref(v___y_813_);
lean_dec_ref(v_as_809_);
lean_dec_ref(v_not__cond_808_);
lean_dec_ref(v_cond_807_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(lean_object* v_as_821_, size_t v_sz_822_, size_t v_i_823_, lean_object* v_b_824_){
_start:
{
uint8_t v___x_826_; 
v___x_826_ = lean_usize_dec_lt(v_i_823_, v_sz_822_);
if (v___x_826_ == 0)
{
lean_object* v___x_827_; 
v___x_827_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_827_, 0, v_b_824_);
return v___x_827_;
}
else
{
lean_object* v_snd_828_; lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_846_; 
v_snd_828_ = lean_ctor_get(v_b_824_, 1);
v_isSharedCheck_846_ = !lean_is_exclusive(v_b_824_);
if (v_isSharedCheck_846_ == 0)
{
lean_object* v_unused_847_; 
v_unused_847_ = lean_ctor_get(v_b_824_, 0);
lean_dec(v_unused_847_);
v___x_830_ = v_b_824_;
v_isShared_831_ = v_isSharedCheck_846_;
goto v_resetjp_829_;
}
else
{
lean_inc(v_snd_828_);
lean_dec(v_b_824_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_846_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v___x_832_; lean_object* v_a_834_; lean_object* v_a_841_; 
v___x_832_ = lean_box(0);
v_a_841_ = lean_array_uget_borrowed(v_as_821_, v_i_823_);
if (lean_obj_tag(v_a_841_) == 0)
{
v_a_834_ = v_snd_828_;
goto v___jp_833_;
}
else
{
lean_object* v_val_842_; uint8_t v___x_843_; 
v_val_842_ = lean_ctor_get(v_a_841_, 0);
v___x_843_ = l_Lean_LocalDecl_isImplementationDetail(v_val_842_);
if (v___x_843_ == 0)
{
lean_object* v___x_844_; lean_object* v___x_845_; 
lean_inc(v_val_842_);
v___x_844_ = l_Lean_LocalDecl_toExpr(v_val_842_);
v___x_845_ = lean_array_push(v_snd_828_, v___x_844_);
v_a_834_ = v___x_845_;
goto v___jp_833_;
}
else
{
v_a_834_ = v_snd_828_;
goto v___jp_833_;
}
}
v___jp_833_:
{
lean_object* v___x_836_; 
if (v_isShared_831_ == 0)
{
lean_ctor_set(v___x_830_, 1, v_a_834_);
lean_ctor_set(v___x_830_, 0, v___x_832_);
v___x_836_ = v___x_830_;
goto v_reusejp_835_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_832_);
lean_ctor_set(v_reuseFailAlloc_840_, 1, v_a_834_);
v___x_836_ = v_reuseFailAlloc_840_;
goto v_reusejp_835_;
}
v_reusejp_835_:
{
size_t v___x_837_; size_t v___x_838_; 
v___x_837_ = ((size_t)1ULL);
v___x_838_ = lean_usize_add(v_i_823_, v___x_837_);
v_i_823_ = v___x_838_;
v_b_824_ = v___x_836_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg___boxed(lean_object* v_as_848_, lean_object* v_sz_849_, lean_object* v_i_850_, lean_object* v_b_851_, lean_object* v___y_852_){
_start:
{
size_t v_sz_boxed_853_; size_t v_i_boxed_854_; lean_object* v_res_855_; 
v_sz_boxed_853_ = lean_unbox_usize(v_sz_849_);
lean_dec(v_sz_849_);
v_i_boxed_854_ = lean_unbox_usize(v_i_850_);
lean_dec(v_i_850_);
v_res_855_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(v_as_848_, v_sz_boxed_853_, v_i_boxed_854_, v_b_851_);
lean_dec_ref(v_as_848_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4(lean_object* v_as_856_, size_t v_sz_857_, size_t v_i_858_, lean_object* v_b_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_){
_start:
{
uint8_t v___x_869_; 
v___x_869_ = lean_usize_dec_lt(v_i_858_, v_sz_857_);
if (v___x_869_ == 0)
{
lean_object* v___x_870_; 
v___x_870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_870_, 0, v_b_859_);
return v___x_870_;
}
else
{
lean_object* v_snd_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_889_; 
v_snd_871_ = lean_ctor_get(v_b_859_, 1);
v_isSharedCheck_889_ = !lean_is_exclusive(v_b_859_);
if (v_isSharedCheck_889_ == 0)
{
lean_object* v_unused_890_; 
v_unused_890_ = lean_ctor_get(v_b_859_, 0);
lean_dec(v_unused_890_);
v___x_873_ = v_b_859_;
v_isShared_874_ = v_isSharedCheck_889_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_snd_871_);
lean_dec(v_b_859_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_889_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; lean_object* v_a_877_; lean_object* v_a_884_; 
v___x_875_ = lean_box(0);
v_a_884_ = lean_array_uget_borrowed(v_as_856_, v_i_858_);
if (lean_obj_tag(v_a_884_) == 0)
{
v_a_877_ = v_snd_871_;
goto v___jp_876_;
}
else
{
lean_object* v_val_885_; uint8_t v___x_886_; 
v_val_885_ = lean_ctor_get(v_a_884_, 0);
v___x_886_ = l_Lean_LocalDecl_isImplementationDetail(v_val_885_);
if (v___x_886_ == 0)
{
lean_object* v___x_887_; lean_object* v___x_888_; 
lean_inc(v_val_885_);
v___x_887_ = l_Lean_LocalDecl_toExpr(v_val_885_);
v___x_888_ = lean_array_push(v_snd_871_, v___x_887_);
v_a_877_ = v___x_888_;
goto v___jp_876_;
}
else
{
v_a_877_ = v_snd_871_;
goto v___jp_876_;
}
}
v___jp_876_:
{
lean_object* v___x_879_; 
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v_a_877_);
lean_ctor_set(v___x_873_, 0, v___x_875_);
v___x_879_ = v___x_873_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v___x_875_);
lean_ctor_set(v_reuseFailAlloc_883_, 1, v_a_877_);
v___x_879_ = v_reuseFailAlloc_883_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
size_t v___x_880_; size_t v___x_881_; lean_object* v___x_882_; 
v___x_880_ = ((size_t)1ULL);
v___x_881_ = lean_usize_add(v_i_858_, v___x_880_);
v___x_882_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(v_as_856_, v_sz_857_, v___x_881_, v___x_879_);
return v___x_882_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_as_891_, lean_object* v_sz_892_, lean_object* v_i_893_, lean_object* v_b_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
size_t v_sz_boxed_904_; size_t v_i_boxed_905_; lean_object* v_res_906_; 
v_sz_boxed_904_ = lean_unbox_usize(v_sz_892_);
lean_dec(v_sz_892_);
v_i_boxed_905_ = lean_unbox_usize(v_i_893_);
lean_dec(v_i_893_);
v_res_906_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4(v_as_891_, v_sz_boxed_904_, v_i_boxed_905_, v_b_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, v___y_902_);
lean_dec(v___y_902_);
lean_dec_ref(v___y_901_);
lean_dec(v___y_900_);
lean_dec_ref(v___y_899_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
lean_dec_ref(v_as_891_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1(lean_object* v_init_907_, lean_object* v_n_908_, lean_object* v_b_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_){
_start:
{
if (lean_obj_tag(v_n_908_) == 0)
{
lean_object* v_cs_919_; lean_object* v___x_920_; lean_object* v___x_921_; size_t v_sz_922_; size_t v___x_923_; lean_object* v___x_924_; 
v_cs_919_ = lean_ctor_get(v_n_908_, 0);
v___x_920_ = lean_box(0);
v___x_921_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_920_);
lean_ctor_set(v___x_921_, 1, v_b_909_);
v_sz_922_ = lean_array_size(v_cs_919_);
v___x_923_ = ((size_t)0ULL);
v___x_924_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3(v_init_907_, v_cs_919_, v_sz_922_, v___x_923_, v___x_921_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_, v___y_917_);
if (lean_obj_tag(v___x_924_) == 0)
{
lean_object* v_a_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_939_; 
v_a_925_ = lean_ctor_get(v___x_924_, 0);
v_isSharedCheck_939_ = !lean_is_exclusive(v___x_924_);
if (v_isSharedCheck_939_ == 0)
{
v___x_927_ = v___x_924_;
v_isShared_928_ = v_isSharedCheck_939_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_a_925_);
lean_dec(v___x_924_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_939_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v_fst_929_; 
v_fst_929_ = lean_ctor_get(v_a_925_, 0);
if (lean_obj_tag(v_fst_929_) == 0)
{
lean_object* v_snd_930_; lean_object* v___x_931_; lean_object* v___x_933_; 
v_snd_930_ = lean_ctor_get(v_a_925_, 1);
lean_inc(v_snd_930_);
lean_dec(v_a_925_);
v___x_931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_931_, 0, v_snd_930_);
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 0, v___x_931_);
v___x_933_ = v___x_927_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v___x_931_);
v___x_933_ = v_reuseFailAlloc_934_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
return v___x_933_;
}
}
else
{
lean_object* v_val_935_; lean_object* v___x_937_; 
lean_inc_ref(v_fst_929_);
lean_dec(v_a_925_);
v_val_935_ = lean_ctor_get(v_fst_929_, 0);
lean_inc(v_val_935_);
lean_dec_ref_known(v_fst_929_, 1);
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 0, v_val_935_);
v___x_937_ = v___x_927_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v_val_935_);
v___x_937_ = v_reuseFailAlloc_938_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
return v___x_937_;
}
}
}
}
else
{
lean_object* v_a_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_947_; 
v_a_940_ = lean_ctor_get(v___x_924_, 0);
v_isSharedCheck_947_ = !lean_is_exclusive(v___x_924_);
if (v_isSharedCheck_947_ == 0)
{
v___x_942_ = v___x_924_;
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_a_940_);
lean_dec(v___x_924_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___x_945_; 
if (v_isShared_943_ == 0)
{
v___x_945_ = v___x_942_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_946_; 
v_reuseFailAlloc_946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_946_, 0, v_a_940_);
v___x_945_ = v_reuseFailAlloc_946_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
return v___x_945_;
}
}
}
}
else
{
lean_object* v_vs_948_; lean_object* v___x_949_; lean_object* v___x_950_; size_t v_sz_951_; size_t v___x_952_; lean_object* v___x_953_; 
v_vs_948_ = lean_ctor_get(v_n_908_, 0);
v___x_949_ = lean_box(0);
v___x_950_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_950_, 0, v___x_949_);
lean_ctor_set(v___x_950_, 1, v_b_909_);
v_sz_951_ = lean_array_size(v_vs_948_);
v___x_952_ = ((size_t)0ULL);
v___x_953_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4(v_vs_948_, v_sz_951_, v___x_952_, v___x_950_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_, v___y_917_);
if (lean_obj_tag(v___x_953_) == 0)
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_968_; 
v_a_954_ = lean_ctor_get(v___x_953_, 0);
v_isSharedCheck_968_ = !lean_is_exclusive(v___x_953_);
if (v_isSharedCheck_968_ == 0)
{
v___x_956_ = v___x_953_;
v_isShared_957_ = v_isSharedCheck_968_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_953_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_968_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v_fst_958_; 
v_fst_958_ = lean_ctor_get(v_a_954_, 0);
if (lean_obj_tag(v_fst_958_) == 0)
{
lean_object* v_snd_959_; lean_object* v___x_960_; lean_object* v___x_962_; 
v_snd_959_ = lean_ctor_get(v_a_954_, 1);
lean_inc(v_snd_959_);
lean_dec(v_a_954_);
v___x_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_960_, 0, v_snd_959_);
if (v_isShared_957_ == 0)
{
lean_ctor_set(v___x_956_, 0, v___x_960_);
v___x_962_ = v___x_956_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_963_; 
v_reuseFailAlloc_963_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_963_, 0, v___x_960_);
v___x_962_ = v_reuseFailAlloc_963_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
return v___x_962_;
}
}
else
{
lean_object* v_val_964_; lean_object* v___x_966_; 
lean_inc_ref(v_fst_958_);
lean_dec(v_a_954_);
v_val_964_ = lean_ctor_get(v_fst_958_, 0);
lean_inc(v_val_964_);
lean_dec_ref_known(v_fst_958_, 1);
if (v_isShared_957_ == 0)
{
lean_ctor_set(v___x_956_, 0, v_val_964_);
v___x_966_ = v___x_956_;
goto v_reusejp_965_;
}
else
{
lean_object* v_reuseFailAlloc_967_; 
v_reuseFailAlloc_967_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_967_, 0, v_val_964_);
v___x_966_ = v_reuseFailAlloc_967_;
goto v_reusejp_965_;
}
v_reusejp_965_:
{
return v___x_966_;
}
}
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
v_a_969_ = lean_ctor_get(v___x_953_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_953_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v___x_953_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_953_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3(lean_object* v_init_977_, lean_object* v_as_978_, size_t v_sz_979_, size_t v_i_980_, lean_object* v_b_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
uint8_t v___x_991_; 
v___x_991_ = lean_usize_dec_lt(v_i_980_, v_sz_979_);
if (v___x_991_ == 0)
{
lean_object* v___x_992_; 
v___x_992_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_992_, 0, v_b_981_);
return v___x_992_;
}
else
{
lean_object* v_snd_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1027_; 
v_snd_993_ = lean_ctor_get(v_b_981_, 1);
v_isSharedCheck_1027_ = !lean_is_exclusive(v_b_981_);
if (v_isSharedCheck_1027_ == 0)
{
lean_object* v_unused_1028_; 
v_unused_1028_ = lean_ctor_get(v_b_981_, 0);
lean_dec(v_unused_1028_);
v___x_995_ = v_b_981_;
v_isShared_996_ = v_isSharedCheck_1027_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_snd_993_);
lean_dec(v_b_981_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1027_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v_a_997_; lean_object* v___x_998_; 
v_a_997_ = lean_array_uget_borrowed(v_as_978_, v_i_980_);
lean_inc(v_snd_993_);
v___x_998_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1(v_init_977_, v_a_997_, v_snd_993_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_998_) == 0)
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1018_; 
v_a_999_ = lean_ctor_get(v___x_998_, 0);
v_isSharedCheck_1018_ = !lean_is_exclusive(v___x_998_);
if (v_isSharedCheck_1018_ == 0)
{
v___x_1001_ = v___x_998_;
v_isShared_1002_ = v_isSharedCheck_1018_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v___x_998_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1018_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
if (lean_obj_tag(v_a_999_) == 0)
{
lean_object* v___x_1003_; lean_object* v___x_1005_; 
v___x_1003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1003_, 0, v_a_999_);
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 0, v___x_1003_);
v___x_1005_ = v___x_995_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v___x_1003_);
lean_ctor_set(v_reuseFailAlloc_1009_, 1, v_snd_993_);
v___x_1005_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
lean_object* v___x_1007_; 
if (v_isShared_1002_ == 0)
{
lean_ctor_set(v___x_1001_, 0, v___x_1005_);
v___x_1007_ = v___x_1001_;
goto v_reusejp_1006_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v___x_1005_);
v___x_1007_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1006_;
}
v_reusejp_1006_:
{
return v___x_1007_;
}
}
}
else
{
lean_object* v_a_1010_; lean_object* v___x_1011_; lean_object* v___x_1013_; 
lean_del_object(v___x_1001_);
lean_dec(v_snd_993_);
v_a_1010_ = lean_ctor_get(v_a_999_, 0);
lean_inc(v_a_1010_);
lean_dec_ref_known(v_a_999_, 1);
v___x_1011_ = lean_box(0);
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 1, v_a_1010_);
lean_ctor_set(v___x_995_, 0, v___x_1011_);
v___x_1013_ = v___x_995_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v___x_1011_);
lean_ctor_set(v_reuseFailAlloc_1017_, 1, v_a_1010_);
v___x_1013_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
size_t v___x_1014_; size_t v___x_1015_; 
v___x_1014_ = ((size_t)1ULL);
v___x_1015_ = lean_usize_add(v_i_980_, v___x_1014_);
v_i_980_ = v___x_1015_;
v_b_981_ = v___x_1013_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1026_; 
lean_del_object(v___x_995_);
lean_dec(v_snd_993_);
v_a_1019_ = lean_ctor_get(v___x_998_, 0);
v_isSharedCheck_1026_ = !lean_is_exclusive(v___x_998_);
if (v_isSharedCheck_1026_ == 0)
{
v___x_1021_ = v___x_998_;
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_a_1019_);
lean_dec(v___x_998_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1026_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v___x_1024_; 
if (v_isShared_1022_ == 0)
{
v___x_1024_ = v___x_1021_;
goto v_reusejp_1023_;
}
else
{
lean_object* v_reuseFailAlloc_1025_; 
v_reuseFailAlloc_1025_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1025_, 0, v_a_1019_);
v___x_1024_ = v_reuseFailAlloc_1025_;
goto v_reusejp_1023_;
}
v_reusejp_1023_:
{
return v___x_1024_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_init_1029_, lean_object* v_as_1030_, lean_object* v_sz_1031_, lean_object* v_i_1032_, lean_object* v_b_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
size_t v_sz_boxed_1043_; size_t v_i_boxed_1044_; lean_object* v_res_1045_; 
v_sz_boxed_1043_ = lean_unbox_usize(v_sz_1031_);
lean_dec(v_sz_1031_);
v_i_boxed_1044_ = lean_unbox_usize(v_i_1032_);
lean_dec(v_i_1032_);
v_res_1045_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__3(v_init_1029_, v_as_1030_, v_sz_boxed_1043_, v_i_boxed_1044_, v_b_1033_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_);
lean_dec(v___y_1041_);
lean_dec_ref(v___y_1040_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
lean_dec_ref(v_as_1030_);
lean_dec_ref(v_init_1029_);
return v_res_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1___boxed(lean_object* v_init_1046_, lean_object* v_n_1047_, lean_object* v_b_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_){
_start:
{
lean_object* v_res_1058_; 
v_res_1058_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1(v_init_1046_, v_n_1047_, v_b_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
lean_dec(v___y_1054_);
lean_dec_ref(v___y_1053_);
lean_dec(v___y_1052_);
lean_dec_ref(v___y_1051_);
lean_dec(v___y_1050_);
lean_dec_ref(v___y_1049_);
lean_dec_ref(v_n_1047_);
lean_dec_ref(v_init_1046_);
return v_res_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg(lean_object* v_as_1059_, size_t v_sz_1060_, size_t v_i_1061_, lean_object* v_b_1062_){
_start:
{
uint8_t v___x_1064_; 
v___x_1064_ = lean_usize_dec_lt(v_i_1061_, v_sz_1060_);
if (v___x_1064_ == 0)
{
lean_object* v___x_1065_; 
v___x_1065_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1065_, 0, v_b_1062_);
return v___x_1065_;
}
else
{
lean_object* v_snd_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1084_; 
v_snd_1066_ = lean_ctor_get(v_b_1062_, 1);
v_isSharedCheck_1084_ = !lean_is_exclusive(v_b_1062_);
if (v_isSharedCheck_1084_ == 0)
{
lean_object* v_unused_1085_; 
v_unused_1085_ = lean_ctor_get(v_b_1062_, 0);
lean_dec(v_unused_1085_);
v___x_1068_ = v_b_1062_;
v_isShared_1069_ = v_isSharedCheck_1084_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_snd_1066_);
lean_dec(v_b_1062_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1084_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
lean_object* v___x_1070_; lean_object* v_a_1072_; lean_object* v_a_1079_; 
v___x_1070_ = lean_box(0);
v_a_1079_ = lean_array_uget_borrowed(v_as_1059_, v_i_1061_);
if (lean_obj_tag(v_a_1079_) == 0)
{
v_a_1072_ = v_snd_1066_;
goto v___jp_1071_;
}
else
{
lean_object* v_val_1080_; uint8_t v___x_1081_; 
v_val_1080_ = lean_ctor_get(v_a_1079_, 0);
v___x_1081_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1080_);
if (v___x_1081_ == 0)
{
lean_object* v___x_1082_; lean_object* v___x_1083_; 
lean_inc(v_val_1080_);
v___x_1082_ = l_Lean_LocalDecl_toExpr(v_val_1080_);
v___x_1083_ = lean_array_push(v_snd_1066_, v___x_1082_);
v_a_1072_ = v___x_1083_;
goto v___jp_1071_;
}
else
{
v_a_1072_ = v_snd_1066_;
goto v___jp_1071_;
}
}
v___jp_1071_:
{
lean_object* v___x_1074_; 
if (v_isShared_1069_ == 0)
{
lean_ctor_set(v___x_1068_, 1, v_a_1072_);
lean_ctor_set(v___x_1068_, 0, v___x_1070_);
v___x_1074_ = v___x_1068_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v___x_1070_);
lean_ctor_set(v_reuseFailAlloc_1078_, 1, v_a_1072_);
v___x_1074_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
size_t v___x_1075_; size_t v___x_1076_; 
v___x_1075_ = ((size_t)1ULL);
v___x_1076_ = lean_usize_add(v_i_1061_, v___x_1075_);
v_i_1061_ = v___x_1076_;
v_b_1062_ = v___x_1074_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object* v_as_1086_, lean_object* v_sz_1087_, lean_object* v_i_1088_, lean_object* v_b_1089_, lean_object* v___y_1090_){
_start:
{
size_t v_sz_boxed_1091_; size_t v_i_boxed_1092_; lean_object* v_res_1093_; 
v_sz_boxed_1091_ = lean_unbox_usize(v_sz_1087_);
lean_dec(v_sz_1087_);
v_i_boxed_1092_ = lean_unbox_usize(v_i_1088_);
lean_dec(v_i_1088_);
v_res_1093_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg(v_as_1086_, v_sz_boxed_1091_, v_i_boxed_1092_, v_b_1089_);
lean_dec_ref(v_as_1086_);
return v_res_1093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2(lean_object* v_as_1094_, size_t v_sz_1095_, size_t v_i_1096_, lean_object* v_b_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
uint8_t v___x_1107_; 
v___x_1107_ = lean_usize_dec_lt(v_i_1096_, v_sz_1095_);
if (v___x_1107_ == 0)
{
lean_object* v___x_1108_; 
v___x_1108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1108_, 0, v_b_1097_);
return v___x_1108_;
}
else
{
lean_object* v_snd_1109_; lean_object* v___x_1111_; uint8_t v_isShared_1112_; uint8_t v_isSharedCheck_1127_; 
v_snd_1109_ = lean_ctor_get(v_b_1097_, 1);
v_isSharedCheck_1127_ = !lean_is_exclusive(v_b_1097_);
if (v_isSharedCheck_1127_ == 0)
{
lean_object* v_unused_1128_; 
v_unused_1128_ = lean_ctor_get(v_b_1097_, 0);
lean_dec(v_unused_1128_);
v___x_1111_ = v_b_1097_;
v_isShared_1112_ = v_isSharedCheck_1127_;
goto v_resetjp_1110_;
}
else
{
lean_inc(v_snd_1109_);
lean_dec(v_b_1097_);
v___x_1111_ = lean_box(0);
v_isShared_1112_ = v_isSharedCheck_1127_;
goto v_resetjp_1110_;
}
v_resetjp_1110_:
{
lean_object* v___x_1113_; lean_object* v_a_1115_; lean_object* v_a_1122_; 
v___x_1113_ = lean_box(0);
v_a_1122_ = lean_array_uget_borrowed(v_as_1094_, v_i_1096_);
if (lean_obj_tag(v_a_1122_) == 0)
{
v_a_1115_ = v_snd_1109_;
goto v___jp_1114_;
}
else
{
lean_object* v_val_1123_; uint8_t v___x_1124_; 
v_val_1123_ = lean_ctor_get(v_a_1122_, 0);
v___x_1124_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1123_);
if (v___x_1124_ == 0)
{
lean_object* v___x_1125_; lean_object* v___x_1126_; 
lean_inc(v_val_1123_);
v___x_1125_ = l_Lean_LocalDecl_toExpr(v_val_1123_);
v___x_1126_ = lean_array_push(v_snd_1109_, v___x_1125_);
v_a_1115_ = v___x_1126_;
goto v___jp_1114_;
}
else
{
v_a_1115_ = v_snd_1109_;
goto v___jp_1114_;
}
}
v___jp_1114_:
{
lean_object* v___x_1117_; 
if (v_isShared_1112_ == 0)
{
lean_ctor_set(v___x_1111_, 1, v_a_1115_);
lean_ctor_set(v___x_1111_, 0, v___x_1113_);
v___x_1117_ = v___x_1111_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v___x_1113_);
lean_ctor_set(v_reuseFailAlloc_1121_, 1, v_a_1115_);
v___x_1117_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
size_t v___x_1118_; size_t v___x_1119_; lean_object* v___x_1120_; 
v___x_1118_ = ((size_t)1ULL);
v___x_1119_ = lean_usize_add(v_i_1096_, v___x_1118_);
v___x_1120_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg(v_as_1094_, v_sz_1095_, v___x_1119_, v___x_1117_);
return v___x_1120_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2___boxed(lean_object* v_as_1129_, lean_object* v_sz_1130_, lean_object* v_i_1131_, lean_object* v_b_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
size_t v_sz_boxed_1142_; size_t v_i_boxed_1143_; lean_object* v_res_1144_; 
v_sz_boxed_1142_ = lean_unbox_usize(v_sz_1130_);
lean_dec(v_sz_1130_);
v_i_boxed_1143_ = lean_unbox_usize(v_i_1131_);
lean_dec(v_i_1131_);
v_res_1144_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2(v_as_1129_, v_sz_boxed_1142_, v_i_boxed_1143_, v_b_1132_, v___y_1133_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec(v___y_1138_);
lean_dec_ref(v___y_1137_);
lean_dec(v___y_1136_);
lean_dec_ref(v___y_1135_);
lean_dec(v___y_1134_);
lean_dec_ref(v___y_1133_);
lean_dec_ref(v_as_1129_);
return v_res_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0(lean_object* v_t_1145_, lean_object* v_init_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_){
_start:
{
lean_object* v_root_1156_; lean_object* v_tail_1157_; lean_object* v___x_1158_; 
v_root_1156_ = lean_ctor_get(v_t_1145_, 0);
v_tail_1157_ = lean_ctor_get(v_t_1145_, 1);
lean_inc_ref(v_init_1146_);
v___x_1158_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1(v_init_1146_, v_root_1156_, v_init_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_);
lean_dec_ref(v_init_1146_);
if (lean_obj_tag(v___x_1158_) == 0)
{
lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1195_; 
v_a_1159_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1195_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1195_ == 0)
{
v___x_1161_ = v___x_1158_;
v_isShared_1162_ = v_isSharedCheck_1195_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v___x_1158_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1195_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
if (lean_obj_tag(v_a_1159_) == 0)
{
lean_object* v_a_1163_; lean_object* v___x_1165_; 
v_a_1163_ = lean_ctor_get(v_a_1159_, 0);
lean_inc(v_a_1163_);
lean_dec_ref_known(v_a_1159_, 1);
if (v_isShared_1162_ == 0)
{
lean_ctor_set(v___x_1161_, 0, v_a_1163_);
v___x_1165_ = v___x_1161_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_a_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
else
{
lean_object* v_a_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; size_t v_sz_1170_; size_t v___x_1171_; lean_object* v___x_1172_; 
lean_del_object(v___x_1161_);
v_a_1167_ = lean_ctor_get(v_a_1159_, 0);
lean_inc(v_a_1167_);
lean_dec_ref_known(v_a_1159_, 1);
v___x_1168_ = lean_box(0);
v___x_1169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1169_, 0, v___x_1168_);
lean_ctor_set(v___x_1169_, 1, v_a_1167_);
v_sz_1170_ = lean_array_size(v_tail_1157_);
v___x_1171_ = ((size_t)0ULL);
v___x_1172_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2(v_tail_1157_, v_sz_1170_, v___x_1171_, v___x_1169_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_);
if (lean_obj_tag(v___x_1172_) == 0)
{
lean_object* v_a_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1186_; 
v_a_1173_ = lean_ctor_get(v___x_1172_, 0);
v_isSharedCheck_1186_ = !lean_is_exclusive(v___x_1172_);
if (v_isSharedCheck_1186_ == 0)
{
v___x_1175_ = v___x_1172_;
v_isShared_1176_ = v_isSharedCheck_1186_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_a_1173_);
lean_dec(v___x_1172_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1186_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v_fst_1177_; 
v_fst_1177_ = lean_ctor_get(v_a_1173_, 0);
if (lean_obj_tag(v_fst_1177_) == 0)
{
lean_object* v_snd_1178_; lean_object* v___x_1180_; 
v_snd_1178_ = lean_ctor_get(v_a_1173_, 1);
lean_inc(v_snd_1178_);
lean_dec(v_a_1173_);
if (v_isShared_1176_ == 0)
{
lean_ctor_set(v___x_1175_, 0, v_snd_1178_);
v___x_1180_ = v___x_1175_;
goto v_reusejp_1179_;
}
else
{
lean_object* v_reuseFailAlloc_1181_; 
v_reuseFailAlloc_1181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1181_, 0, v_snd_1178_);
v___x_1180_ = v_reuseFailAlloc_1181_;
goto v_reusejp_1179_;
}
v_reusejp_1179_:
{
return v___x_1180_;
}
}
else
{
lean_object* v_val_1182_; lean_object* v___x_1184_; 
lean_inc_ref(v_fst_1177_);
lean_dec(v_a_1173_);
v_val_1182_ = lean_ctor_get(v_fst_1177_, 0);
lean_inc(v_val_1182_);
lean_dec_ref_known(v_fst_1177_, 1);
if (v_isShared_1176_ == 0)
{
lean_ctor_set(v___x_1175_, 0, v_val_1182_);
v___x_1184_ = v___x_1175_;
goto v_reusejp_1183_;
}
else
{
lean_object* v_reuseFailAlloc_1185_; 
v_reuseFailAlloc_1185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1185_, 0, v_val_1182_);
v___x_1184_ = v_reuseFailAlloc_1185_;
goto v_reusejp_1183_;
}
v_reusejp_1183_:
{
return v___x_1184_;
}
}
}
}
else
{
lean_object* v_a_1187_; lean_object* v___x_1189_; uint8_t v_isShared_1190_; uint8_t v_isSharedCheck_1194_; 
v_a_1187_ = lean_ctor_get(v___x_1172_, 0);
v_isSharedCheck_1194_ = !lean_is_exclusive(v___x_1172_);
if (v_isSharedCheck_1194_ == 0)
{
v___x_1189_ = v___x_1172_;
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
else
{
lean_inc(v_a_1187_);
lean_dec(v___x_1172_);
v___x_1189_ = lean_box(0);
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
v_resetjp_1188_:
{
lean_object* v___x_1192_; 
if (v_isShared_1190_ == 0)
{
v___x_1192_ = v___x_1189_;
goto v_reusejp_1191_;
}
else
{
lean_object* v_reuseFailAlloc_1193_; 
v_reuseFailAlloc_1193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1193_, 0, v_a_1187_);
v___x_1192_ = v_reuseFailAlloc_1193_;
goto v_reusejp_1191_;
}
v_reusejp_1191_:
{
return v___x_1192_;
}
}
}
}
}
}
else
{
lean_object* v_a_1196_; lean_object* v___x_1198_; uint8_t v_isShared_1199_; uint8_t v_isSharedCheck_1203_; 
v_a_1196_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1203_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1203_ == 0)
{
v___x_1198_ = v___x_1158_;
v_isShared_1199_ = v_isSharedCheck_1203_;
goto v_resetjp_1197_;
}
else
{
lean_inc(v_a_1196_);
lean_dec(v___x_1158_);
v___x_1198_ = lean_box(0);
v_isShared_1199_ = v_isSharedCheck_1203_;
goto v_resetjp_1197_;
}
v_resetjp_1197_:
{
lean_object* v___x_1201_; 
if (v_isShared_1199_ == 0)
{
v___x_1201_ = v___x_1198_;
goto v_reusejp_1200_;
}
else
{
lean_object* v_reuseFailAlloc_1202_; 
v_reuseFailAlloc_1202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1202_, 0, v_a_1196_);
v___x_1201_ = v_reuseFailAlloc_1202_;
goto v_reusejp_1200_;
}
v_reusejp_1200_:
{
return v___x_1201_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0___boxed(lean_object* v_t_1204_, lean_object* v_init_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v_res_1215_; 
v_res_1215_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0(v_t_1204_, v_init_1205_, v___y_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
lean_dec(v___y_1207_);
lean_dec_ref(v___y_1206_);
lean_dec_ref(v_t_1204_);
return v_res_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0(lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
lean_object* v_lctx_1227_; lean_object* v_decls_1228_; lean_object* v_hs_1229_; lean_object* v___x_1230_; 
v_lctx_1227_ = lean_ctor_get(v___y_1222_, 2);
v_decls_1228_ = lean_ctor_get(v_lctx_1227_, 1);
v_hs_1229_ = ((lean_object*)(lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___closed__0));
v___x_1230_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0(v_decls_1228_, v_hs_1229_, v___y_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_);
return v___x_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0___boxed(lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_){
_start:
{
lean_object* v_res_1240_; 
v_res_1240_ = lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0(v___y_1231_, v___y_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
lean_dec(v___y_1238_);
lean_dec_ref(v___y_1237_);
lean_dec(v___y_1236_);
lean_dec_ref(v___y_1235_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
lean_dec(v___y_1232_);
lean_dec_ref(v___y_1231_);
return v_res_1240_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2(void){
_start:
{
lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1244_ = lean_box(0);
v___x_1245_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__1));
v___x_1246_ = l_Lean_mkConst(v___x_1245_, v___x_1244_);
return v___x_1246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown(lean_object* v_cond_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_){
_start:
{
lean_object* v___x_1257_; 
v___x_1257_ = lp_mathlib_Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0(v_a_1248_, v_a_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_, v_a_1254_, v_a_1255_);
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_object* v_a_1258_; lean_object* v___x_1259_; lean_object* v_not__cond_1260_; lean_object* v___x_1261_; size_t v_sz_1262_; size_t v___x_1263_; lean_object* v___x_1264_; 
v_a_1258_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_a_1258_);
lean_dec_ref_known(v___x_1257_, 1);
v___x_1259_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__2);
lean_inc_ref(v_cond_1247_);
v_not__cond_1260_ = l_Lean_Expr_app___override(v___x_1259_, v_cond_1247_);
v___x_1261_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg___closed__0));
v_sz_1262_ = lean_array_size(v_a_1258_);
v___x_1263_ = ((size_t)0ULL);
v___x_1264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg(v_cond_1247_, v_not__cond_1260_, v_a_1258_, v_sz_1262_, v___x_1263_, v___x_1261_, v_a_1252_, v_a_1253_, v_a_1254_, v_a_1255_);
lean_dec(v_a_1258_);
lean_dec_ref(v_not__cond_1260_);
lean_dec_ref(v_cond_1247_);
if (lean_obj_tag(v___x_1264_) == 0)
{
lean_object* v_a_1265_; lean_object* v___x_1267_; uint8_t v_isShared_1268_; uint8_t v_isSharedCheck_1279_; 
v_a_1265_ = lean_ctor_get(v___x_1264_, 0);
v_isSharedCheck_1279_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1279_ == 0)
{
v___x_1267_ = v___x_1264_;
v_isShared_1268_ = v_isSharedCheck_1279_;
goto v_resetjp_1266_;
}
else
{
lean_inc(v_a_1265_);
lean_dec(v___x_1264_);
v___x_1267_ = lean_box(0);
v_isShared_1268_ = v_isSharedCheck_1279_;
goto v_resetjp_1266_;
}
v_resetjp_1266_:
{
lean_object* v_fst_1269_; 
v_fst_1269_ = lean_ctor_get(v_a_1265_, 0);
lean_inc(v_fst_1269_);
lean_dec(v_a_1265_);
if (lean_obj_tag(v_fst_1269_) == 0)
{
uint8_t v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1273_; 
v___x_1270_ = 0;
v___x_1271_ = lean_box(v___x_1270_);
if (v_isShared_1268_ == 0)
{
lean_ctor_set(v___x_1267_, 0, v___x_1271_);
v___x_1273_ = v___x_1267_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v___x_1271_);
v___x_1273_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
return v___x_1273_;
}
}
else
{
lean_object* v_val_1275_; lean_object* v___x_1277_; 
v_val_1275_ = lean_ctor_get(v_fst_1269_, 0);
lean_inc(v_val_1275_);
lean_dec_ref_known(v_fst_1269_, 1);
if (v_isShared_1268_ == 0)
{
lean_ctor_set(v___x_1267_, 0, v_val_1275_);
v___x_1277_ = v___x_1267_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1278_; 
v_reuseFailAlloc_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1278_, 0, v_val_1275_);
v___x_1277_ = v_reuseFailAlloc_1278_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
return v___x_1277_;
}
}
}
}
else
{
lean_object* v_a_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1287_; 
v_a_1280_ = lean_ctor_get(v___x_1264_, 0);
v_isSharedCheck_1287_ = !lean_is_exclusive(v___x_1264_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1282_ = v___x_1264_;
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_a_1280_);
lean_dec(v___x_1264_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
lean_object* v___x_1285_; 
if (v_isShared_1283_ == 0)
{
v___x_1285_ = v___x_1282_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1286_; 
v_reuseFailAlloc_1286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1286_, 0, v_a_1280_);
v___x_1285_ = v_reuseFailAlloc_1286_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
return v___x_1285_;
}
}
}
}
else
{
lean_object* v_a_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1295_; 
lean_dec_ref(v_cond_1247_);
v_a_1288_ = lean_ctor_get(v___x_1257_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1257_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1290_ = v___x_1257_;
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_a_1288_);
lean_dec(v___x_1257_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v___x_1293_; 
if (v_isShared_1291_ == 0)
{
v___x_1293_ = v___x_1290_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v_a_1288_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___boxed(lean_object* v_cond_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_, lean_object* v_a_1300_, lean_object* v_a_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_, lean_object* v_a_1305_){
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown(v_cond_1296_, v_a_1297_, v_a_1298_, v_a_1299_, v_a_1300_, v_a_1301_, v_a_1302_, v_a_1303_, v_a_1304_);
lean_dec(v_a_1304_);
lean_dec_ref(v_a_1303_);
lean_dec(v_a_1302_);
lean_dec_ref(v_a_1301_);
lean_dec(v_a_1300_);
lean_dec_ref(v_a_1299_);
lean_dec(v_a_1298_);
lean_dec_ref(v_a_1297_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1(lean_object* v_cond_1307_, lean_object* v_not__cond_1308_, lean_object* v_as_1309_, size_t v_sz_1310_, size_t v_i_1311_, lean_object* v_b_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___redArg(v_cond_1307_, v_not__cond_1308_, v_as_1309_, v_sz_1310_, v_i_1311_, v_b_1312_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1___boxed(lean_object* v_cond_1323_, lean_object* v_not__cond_1324_, lean_object* v_as_1325_, lean_object* v_sz_1326_, lean_object* v_i_1327_, lean_object* v_b_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_){
_start:
{
size_t v_sz_boxed_1338_; size_t v_i_boxed_1339_; lean_object* v_res_1340_; 
v_sz_boxed_1338_ = lean_unbox_usize(v_sz_1326_);
lean_dec(v_sz_1326_);
v_i_boxed_1339_ = lean_unbox_usize(v_i_1327_);
lean_dec(v_i_1327_);
v_res_1340_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__1(v_cond_1323_, v_not__cond_1324_, v_as_1325_, v_sz_boxed_1338_, v_i_boxed_1339_, v_b_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_);
lean_dec(v___y_1336_);
lean_dec_ref(v___y_1335_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
lean_dec(v___y_1330_);
lean_dec_ref(v___y_1329_);
lean_dec_ref(v_as_1325_);
lean_dec_ref(v_not__cond_1324_);
lean_dec_ref(v_cond_1323_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6(lean_object* v_as_1341_, size_t v_sz_1342_, size_t v_i_1343_, lean_object* v_b_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_){
_start:
{
lean_object* v___x_1354_; 
v___x_1354_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___redArg(v_as_1341_, v_sz_1342_, v_i_1343_, v_b_1344_);
return v___x_1354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6___boxed(lean_object* v_as_1355_, lean_object* v_sz_1356_, lean_object* v_i_1357_, lean_object* v_b_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_){
_start:
{
size_t v_sz_boxed_1368_; size_t v_i_boxed_1369_; lean_object* v_res_1370_; 
v_sz_boxed_1368_ = lean_unbox_usize(v_sz_1356_);
lean_dec(v_sz_1356_);
v_i_boxed_1369_ = lean_unbox_usize(v_i_1357_);
lean_dec(v_i_1357_);
v_res_1370_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__2_spec__6(v_as_1355_, v_sz_boxed_1368_, v_i_boxed_1369_, v_b_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_);
lean_dec(v___y_1366_);
lean_dec_ref(v___y_1365_);
lean_dec(v___y_1364_);
lean_dec_ref(v___y_1363_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
lean_dec_ref(v_as_1355_);
return v_res_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5(lean_object* v_as_1371_, size_t v_sz_1372_, size_t v_i_1373_, lean_object* v_b_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_){
_start:
{
lean_object* v___x_1384_; 
v___x_1384_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___redArg(v_as_1371_, v_sz_1372_, v_i_1373_, v_b_1374_);
return v___x_1384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5___boxed(lean_object* v_as_1385_, lean_object* v_sz_1386_, lean_object* v_i_1387_, lean_object* v_b_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_){
_start:
{
size_t v_sz_boxed_1398_; size_t v_i_boxed_1399_; lean_object* v_res_1400_; 
v_sz_boxed_1398_ = lean_unbox_usize(v_sz_1386_);
lean_dec(v_sz_1386_);
v_i_boxed_1399_ = lean_unbox_usize(v_i_1387_);
lean_dec(v_i_1387_);
v_res_1400_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_getLocalHyps___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown_spec__0_spec__0_spec__1_spec__4_spec__5(v_as_1385_, v_sz_boxed_1398_, v_i_boxed_1399_, v_b_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_, v___y_1396_);
lean_dec(v___y_1396_);
lean_dec_ref(v___y_1395_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec(v___y_1390_);
lean_dec_ref(v___y_1389_);
lean_dec_ref(v_as_1385_);
return v_res_1400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0(lean_object* v___x_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
lean_object* v___x_1411_; 
v___x_1411_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1403_, v___y_1405_, v___y_1407_, v___y_1409_);
if (lean_obj_tag(v___x_1411_) == 0)
{
lean_object* v_a_1412_; lean_object* v___x_1413_; 
v_a_1412_ = lean_ctor_get(v___x_1411_, 0);
lean_inc(v_a_1412_);
lean_dec_ref_known(v___x_1411_, 1);
v___x_1413_ = l_Lean_Elab_Tactic_withoutRecover___redArg(v___x_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_);
if (lean_obj_tag(v___x_1413_) == 0)
{
lean_dec(v_a_1412_);
return v___x_1413_;
}
else
{
lean_object* v_a_1414_; uint8_t v___y_1416_; uint8_t v___x_1427_; 
v_a_1414_ = lean_ctor_get(v___x_1413_, 0);
lean_inc(v_a_1414_);
v___x_1427_ = l_Lean_Exception_isInterrupt(v_a_1414_);
if (v___x_1427_ == 0)
{
uint8_t v___x_1428_; 
v___x_1428_ = l_Lean_Exception_isRuntime(v_a_1414_);
v___y_1416_ = v___x_1428_;
goto v___jp_1415_;
}
else
{
lean_dec(v_a_1414_);
v___y_1416_ = v___x_1427_;
goto v___jp_1415_;
}
v___jp_1415_:
{
if (v___y_1416_ == 0)
{
lean_object* v___x_1417_; 
lean_dec_ref_known(v___x_1413_, 1);
v___x_1417_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1412_, v___y_1416_, v___y_1403_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_, v___y_1409_);
if (lean_obj_tag(v___x_1417_) == 0)
{
lean_object* v___x_1419_; uint8_t v_isShared_1420_; uint8_t v_isSharedCheck_1425_; 
v_isSharedCheck_1425_ = !lean_is_exclusive(v___x_1417_);
if (v_isSharedCheck_1425_ == 0)
{
lean_object* v_unused_1426_; 
v_unused_1426_ = lean_ctor_get(v___x_1417_, 0);
lean_dec(v_unused_1426_);
v___x_1419_ = v___x_1417_;
v_isShared_1420_ = v_isSharedCheck_1425_;
goto v_resetjp_1418_;
}
else
{
lean_dec(v___x_1417_);
v___x_1419_ = lean_box(0);
v_isShared_1420_ = v_isSharedCheck_1425_;
goto v_resetjp_1418_;
}
v_resetjp_1418_:
{
lean_object* v___x_1421_; lean_object* v___x_1423_; 
v___x_1421_ = lean_box(0);
if (v_isShared_1420_ == 0)
{
lean_ctor_set(v___x_1419_, 0, v___x_1421_);
v___x_1423_ = v___x_1419_;
goto v_reusejp_1422_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v___x_1421_);
v___x_1423_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1422_;
}
v_reusejp_1422_:
{
return v___x_1423_;
}
}
}
else
{
return v___x_1417_;
}
}
else
{
lean_dec(v_a_1412_);
return v___x_1413_;
}
}
}
}
else
{
lean_object* v_a_1429_; lean_object* v___x_1431_; uint8_t v_isShared_1432_; uint8_t v_isSharedCheck_1436_; 
lean_dec_ref(v___x_1401_);
v_a_1429_ = lean_ctor_get(v___x_1411_, 0);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1411_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1431_ = v___x_1411_;
v_isShared_1432_ = v_isSharedCheck_1436_;
goto v_resetjp_1430_;
}
else
{
lean_inc(v_a_1429_);
lean_dec(v___x_1411_);
v___x_1431_ = lean_box(0);
v_isShared_1432_ = v_isSharedCheck_1436_;
goto v_resetjp_1430_;
}
v_resetjp_1430_:
{
lean_object* v___x_1434_; 
if (v_isShared_1432_ == 0)
{
v___x_1434_ = v___x_1431_;
goto v_reusejp_1433_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v_a_1429_);
v___x_1434_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1433_;
}
v_reusejp_1433_:
{
return v___x_1434_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0___boxed(lean_object* v___x_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v_res_1447_; 
v_res_1447_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0(v___x_1437_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_);
lean_dec(v___y_1445_);
lean_dec_ref(v___y_1444_);
lean_dec(v___y_1443_);
lean_dec_ref(v___y_1442_);
lean_dec(v___y_1441_);
lean_dec_ref(v___y_1440_);
lean_dec(v___y_1439_);
lean_dec_ref(v___y_1438_);
return v_res_1447_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0(lean_object* v_a_1448_, lean_object* v_x_1449_){
_start:
{
if (lean_obj_tag(v_x_1449_) == 0)
{
uint8_t v___x_1450_; 
v___x_1450_ = 0;
return v___x_1450_;
}
else
{
lean_object* v_head_1451_; lean_object* v_tail_1452_; uint8_t v___x_1453_; 
v_head_1451_ = lean_ctor_get(v_x_1449_, 0);
v_tail_1452_ = lean_ctor_get(v_x_1449_, 1);
v___x_1453_ = lean_expr_eqv(v_a_1448_, v_head_1451_);
if (v___x_1453_ == 0)
{
v_x_1449_ = v_tail_1452_;
goto _start;
}
else
{
return v___x_1453_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0___boxed(lean_object* v_a_1455_, lean_object* v_x_1456_){
_start:
{
uint8_t v_res_1457_; lean_object* v_r_1458_; 
v_res_1457_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0(v_a_1455_, v_x_1456_);
lean_dec(v_x_1456_);
lean_dec_ref(v_a_1455_);
v_r_1458_ = lean_box(v_res_1457_);
return v_r_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___boxed(lean_object* v_loc_1459_, lean_object* v_hNames_1460_, lean_object* v_done_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_, lean_object* v_a_1465_, lean_object* v_a_1466_, lean_object* v_a_1467_, lean_object* v_a_1468_, lean_object* v_a_1469_, lean_object* v_a_1470_){
_start:
{
lean_object* v_res_1471_; 
v_res_1471_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore(v_loc_1459_, v_hNames_1460_, v_done_1461_, v_a_1462_, v_a_1463_, v_a_1464_, v_a_1465_, v_a_1466_, v_a_1467_, v_a_1468_, v_a_1469_);
lean_dec(v_a_1469_);
lean_dec_ref(v_a_1468_);
lean_dec(v_a_1467_);
lean_dec_ref(v_a_1466_);
lean_dec(v_a_1465_);
lean_dec_ref(v_a_1464_);
lean_dec(v_a_1463_);
lean_dec_ref(v_a_1462_);
return v_res_1471_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0(void){
_start:
{
lean_object* v___x_1472_; lean_object* v_dummy_1473_; 
v___x_1472_ = lean_box(0);
v_dummy_1473_ = l_Lean_Expr_sort___override(v___x_1472_);
return v_dummy_1473_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5(void){
_start:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1480_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__4));
v___x_1481_ = l_Lean_MessageData_ofFormat(v___x_1480_);
return v___x_1481_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6(void){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1482_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__5);
v___x_1483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1483_, 0, v___x_1482_);
return v___x_1483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2(lean_object* v_loc_1484_, lean_object* v_done_1485_, lean_object* v_hNames_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_){
_start:
{
lean_object* v___x_1496_; 
lean_inc(v_loc_1484_);
v___x_1496_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_findIfCondAt(v_loc_1484_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1496_) == 0)
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1565_; 
v_a_1497_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1565_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1565_ == 0)
{
v___x_1499_ = v___x_1496_;
v_isShared_1500_ = v_isSharedCheck_1565_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1496_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1565_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v___y_1502_; 
if (lean_obj_tag(v_a_1497_) == 1)
{
lean_object* v_val_1539_; lean_object* v_snd_1540_; lean_object* v___x_1541_; uint8_t v___x_1542_; 
v_val_1539_ = lean_ctor_get(v_a_1497_, 0);
lean_inc(v_val_1539_);
lean_dec_ref_known(v_a_1497_, 1);
v_snd_1540_ = lean_ctor_get(v_val_1539_, 1);
lean_inc(v_snd_1540_);
lean_dec(v_val_1539_);
v___x_1541_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown___closed__1));
v___x_1542_ = l_Lean_Expr_isAppOf(v_snd_1540_, v___x_1541_);
if (v___x_1542_ == 0)
{
v___y_1502_ = v_snd_1540_;
goto v___jp_1501_;
}
else
{
lean_object* v___x_1543_; lean_object* v_dummy_1544_; lean_object* v_nargs_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; 
v___x_1543_ = l_Lean_instInhabitedExpr;
v_dummy_1544_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__0);
v_nargs_1545_ = l_Lean_Expr_getAppNumArgs(v_snd_1540_);
lean_inc(v_nargs_1545_);
v___x_1546_ = lean_mk_array(v_nargs_1545_, v_dummy_1544_);
v___x_1547_ = lean_unsigned_to_nat(1u);
v___x_1548_ = lean_nat_sub(v_nargs_1545_, v___x_1547_);
lean_dec(v_nargs_1545_);
v___x_1549_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_snd_1540_, v___x_1546_, v___x_1548_);
v___x_1550_ = lean_unsigned_to_nat(0u);
v___x_1551_ = lean_array_get(v___x_1543_, v___x_1549_, v___x_1550_);
lean_dec_ref(v___x_1549_);
v___y_1502_ = v___x_1551_;
goto v___jp_1501_;
}
}
else
{
lean_object* v___x_1552_; 
lean_del_object(v___x_1499_);
lean_dec(v_a_1497_);
lean_dec(v_hNames_1486_);
lean_dec(v_done_1485_);
lean_dec(v_loc_1484_);
v___x_1552_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1488_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1552_) == 0)
{
lean_object* v_a_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; 
v_a_1553_ = lean_ctor_get(v___x_1552_, 0);
lean_inc(v_a_1553_);
lean_dec_ref_known(v___x_1552_, 1);
v___x_1554_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__2));
v___x_1555_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6, &lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___closed__6);
v___x_1556_ = l_Lean_Meta_throwTacticEx___redArg(v___x_1554_, v_a_1553_, v___x_1555_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
return v___x_1556_;
}
else
{
lean_object* v_a_1557_; lean_object* v___x_1559_; uint8_t v_isShared_1560_; uint8_t v_isSharedCheck_1564_; 
v_a_1557_ = lean_ctor_get(v___x_1552_, 0);
v_isSharedCheck_1564_ = !lean_is_exclusive(v___x_1552_);
if (v_isSharedCheck_1564_ == 0)
{
v___x_1559_ = v___x_1552_;
v_isShared_1560_ = v_isSharedCheck_1564_;
goto v_resetjp_1558_;
}
else
{
lean_inc(v_a_1557_);
lean_dec(v___x_1552_);
v___x_1559_ = lean_box(0);
v_isShared_1560_ = v_isSharedCheck_1564_;
goto v_resetjp_1558_;
}
v_resetjp_1558_:
{
lean_object* v___x_1562_; 
if (v_isShared_1560_ == 0)
{
v___x_1562_ = v___x_1559_;
goto v_reusejp_1561_;
}
else
{
lean_object* v_reuseFailAlloc_1563_; 
v_reuseFailAlloc_1563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1563_, 0, v_a_1557_);
v___x_1562_ = v_reuseFailAlloc_1563_;
goto v_reusejp_1561_;
}
v_reusejp_1561_:
{
return v___x_1562_;
}
}
}
}
v___jp_1501_:
{
uint8_t v___x_1503_; 
v___x_1503_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore_spec__0(v___y_1502_, v_done_1485_);
if (v___x_1503_ == 0)
{
lean_object* v___x_1504_; 
lean_del_object(v___x_1499_);
lean_inc_ref(v___y_1502_);
v___x_1504_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_valueKnown(v___y_1502_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1504_) == 0)
{
lean_object* v_a_1505_; uint8_t v___x_1506_; 
v_a_1505_ = lean_ctor_get(v___x_1504_, 0);
lean_inc(v_a_1505_);
lean_dec_ref_known(v___x_1504_, 1);
v___x_1506_ = lean_unbox(v_a_1505_);
lean_dec(v_a_1505_);
if (v___x_1506_ == 0)
{
lean_object* v___x_1507_; 
v___x_1507_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_getNextName___redArg(v_hNames_1486_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1507_) == 0)
{
lean_object* v_a_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___f_1512_; lean_object* v___x_1513_; 
v_a_1508_ = lean_ctor_get(v___x_1507_, 0);
lean_inc(v_a_1508_);
lean_dec_ref_known(v___x_1507_, 1);
lean_inc(v_loc_1484_);
lean_inc_ref(v___y_1502_);
v___x_1509_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIf1___boxed), 12, 3);
lean_closure_set(v___x_1509_, 0, v___y_1502_);
lean_closure_set(v___x_1509_, 1, v_a_1508_);
lean_closure_set(v___x_1509_, 2, v_loc_1484_);
v___x_1510_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1510_, 0, v___y_1502_);
lean_ctor_set(v___x_1510_, 1, v_done_1485_);
v___x_1511_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___boxed), 12, 3);
lean_closure_set(v___x_1511_, 0, v_loc_1484_);
lean_closure_set(v___x_1511_, 1, v_hNames_1486_);
lean_closure_set(v___x_1511_, 2, v___x_1510_);
v___f_1512_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1512_, 0, v___x_1511_);
v___x_1513_ = lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals(v___x_1509_, v___f_1512_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
return v___x_1513_;
}
else
{
lean_object* v_a_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1521_; 
lean_dec_ref(v___y_1502_);
lean_dec(v_hNames_1486_);
lean_dec(v_done_1485_);
lean_dec(v_loc_1484_);
v_a_1514_ = lean_ctor_get(v___x_1507_, 0);
v_isSharedCheck_1521_ = !lean_is_exclusive(v___x_1507_);
if (v_isSharedCheck_1521_ == 0)
{
v___x_1516_ = v___x_1507_;
v_isShared_1517_ = v_isSharedCheck_1521_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_a_1514_);
lean_dec(v___x_1507_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1521_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
lean_object* v___x_1519_; 
if (v_isShared_1517_ == 0)
{
v___x_1519_ = v___x_1516_;
goto v_reusejp_1518_;
}
else
{
lean_object* v_reuseFailAlloc_1520_; 
v_reuseFailAlloc_1520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1520_, 0, v_a_1514_);
v___x_1519_ = v_reuseFailAlloc_1520_;
goto v_reusejp_1518_;
}
v_reusejp_1518_:
{
return v___x_1519_;
}
}
}
}
else
{
lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___f_1525_; lean_object* v___x_1526_; 
lean_inc(v_loc_1484_);
v___x_1522_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_reduceIfsAt___boxed), 10, 1);
lean_closure_set(v___x_1522_, 0, v_loc_1484_);
v___x_1523_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1523_, 0, v___y_1502_);
lean_ctor_set(v___x_1523_, 1, v_done_1485_);
v___x_1524_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___boxed), 12, 3);
lean_closure_set(v___x_1524_, 0, v_loc_1484_);
lean_closure_set(v___x_1524_, 1, v_hNames_1486_);
lean_closure_set(v___x_1524_, 2, v___x_1523_);
v___f_1525_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1525_, 0, v___x_1524_);
v___x_1526_ = lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals(v___x_1522_, v___f_1525_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
return v___x_1526_;
}
}
else
{
lean_object* v_a_1527_; lean_object* v___x_1529_; uint8_t v_isShared_1530_; uint8_t v_isSharedCheck_1534_; 
lean_dec_ref(v___y_1502_);
lean_dec(v_hNames_1486_);
lean_dec(v_done_1485_);
lean_dec(v_loc_1484_);
v_a_1527_ = lean_ctor_get(v___x_1504_, 0);
v_isSharedCheck_1534_ = !lean_is_exclusive(v___x_1504_);
if (v_isSharedCheck_1534_ == 0)
{
v___x_1529_ = v___x_1504_;
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
else
{
lean_inc(v_a_1527_);
lean_dec(v___x_1504_);
v___x_1529_ = lean_box(0);
v_isShared_1530_ = v_isSharedCheck_1534_;
goto v_resetjp_1528_;
}
v_resetjp_1528_:
{
lean_object* v___x_1532_; 
if (v_isShared_1530_ == 0)
{
v___x_1532_ = v___x_1529_;
goto v_reusejp_1531_;
}
else
{
lean_object* v_reuseFailAlloc_1533_; 
v_reuseFailAlloc_1533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1533_, 0, v_a_1527_);
v___x_1532_ = v_reuseFailAlloc_1533_;
goto v_reusejp_1531_;
}
v_reusejp_1531_:
{
return v___x_1532_;
}
}
}
}
else
{
lean_object* v___x_1535_; lean_object* v___x_1537_; 
lean_dec_ref(v___y_1502_);
lean_dec(v_hNames_1486_);
lean_dec(v_done_1485_);
lean_dec(v_loc_1484_);
v___x_1535_ = lean_box(0);
if (v_isShared_1500_ == 0)
{
lean_ctor_set(v___x_1499_, 0, v___x_1535_);
v___x_1537_ = v___x_1499_;
goto v_reusejp_1536_;
}
else
{
lean_object* v_reuseFailAlloc_1538_; 
v_reuseFailAlloc_1538_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1538_, 0, v___x_1535_);
v___x_1537_ = v_reuseFailAlloc_1538_;
goto v_reusejp_1536_;
}
v_reusejp_1536_:
{
return v___x_1537_;
}
}
}
}
}
else
{
lean_object* v_a_1566_; lean_object* v___x_1568_; uint8_t v_isShared_1569_; uint8_t v_isSharedCheck_1573_; 
lean_dec(v_hNames_1486_);
lean_dec(v_done_1485_);
lean_dec(v_loc_1484_);
v_a_1566_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1573_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1573_ == 0)
{
v___x_1568_ = v___x_1496_;
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
else
{
lean_inc(v_a_1566_);
lean_dec(v___x_1496_);
v___x_1568_ = lean_box(0);
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
v_resetjp_1567_:
{
lean_object* v___x_1571_; 
if (v_isShared_1569_ == 0)
{
v___x_1571_ = v___x_1568_;
goto v_reusejp_1570_;
}
else
{
lean_object* v_reuseFailAlloc_1572_; 
v_reuseFailAlloc_1572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1572_, 0, v_a_1566_);
v___x_1571_ = v_reuseFailAlloc_1572_;
goto v_reusejp_1570_;
}
v_reusejp_1570_:
{
return v___x_1571_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___boxed(lean_object* v_loc_1574_, lean_object* v_done_1575_, lean_object* v_hNames_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_){
_start:
{
lean_object* v_res_1586_; 
v_res_1586_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2(v_loc_1574_, v_done_1575_, v_hNames_1576_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec(v___y_1580_);
lean_dec_ref(v___y_1579_);
lean_dec(v___y_1578_);
lean_dec_ref(v___y_1577_);
return v_res_1586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore(lean_object* v_loc_1587_, lean_object* v_hNames_1588_, lean_object* v_done_1589_, lean_object* v_a_1590_, lean_object* v_a_1591_, lean_object* v_a_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_, lean_object* v_a_1595_, lean_object* v_a_1596_, lean_object* v_a_1597_){
_start:
{
lean_object* v___f_1599_; lean_object* v___x_1600_; 
v___f_1599_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore___lam__2___boxed), 12, 3);
lean_closure_set(v___f_1599_, 0, v_loc_1587_);
lean_closure_set(v___f_1599_, 1, v_done_1589_);
lean_closure_set(v___f_1599_, 2, v_hNames_1588_);
v___x_1600_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1599_, v_a_1590_, v_a_1591_, v_a_1592_, v_a_1593_, v_a_1594_, v_a_1595_, v_a_1596_, v_a_1597_);
return v___x_1600_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__9(void){
_start:
{
lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; 
v___x_1617_ = l_Lean_Parser_Tactic_location;
v___x_1618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__8));
v___x_1619_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1619_, 0, v___x_1618_);
lean_ctor_set(v___x_1619_, 1, v___x_1617_);
return v___x_1619_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__10(void){
_start:
{
lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1620_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__9, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__9);
v___x_1621_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__6));
v___x_1622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__5));
v___x_1623_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1623_, 0, v___x_1622_);
lean_ctor_set(v___x_1623_, 1, v___x_1621_);
lean_ctor_set(v___x_1623_, 2, v___x_1620_);
return v___x_1623_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__22(void){
_start:
{
lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; 
v___x_1644_ = l_Lean_binderIdent;
v___x_1645_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__21));
v___x_1646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__5));
v___x_1647_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1647_, 0, v___x_1646_);
lean_ctor_set(v___x_1647_, 1, v___x_1645_);
lean_ctor_set(v___x_1647_, 2, v___x_1644_);
return v___x_1647_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__23(void){
_start:
{
lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; 
v___x_1648_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__22, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__22);
v___x_1649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__14));
v___x_1650_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1650_, 0, v___x_1649_);
lean_ctor_set(v___x_1650_, 1, v___x_1648_);
return v___x_1650_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__24(void){
_start:
{
lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; 
v___x_1651_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__23, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__23);
v___x_1652_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__12));
v___x_1653_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__5));
v___x_1654_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1654_, 0, v___x_1653_);
lean_ctor_set(v___x_1654_, 1, v___x_1652_);
lean_ctor_set(v___x_1654_, 2, v___x_1651_);
return v___x_1654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__25(void){
_start:
{
lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; 
v___x_1655_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__24, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__24);
v___x_1656_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__8));
v___x_1657_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1657_, 0, v___x_1656_);
lean_ctor_set(v___x_1657_, 1, v___x_1655_);
return v___x_1657_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__26(void){
_start:
{
lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; 
v___x_1658_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__25, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__25);
v___x_1659_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__10, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__10);
v___x_1660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__5));
v___x_1661_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1661_, 0, v___x_1660_);
lean_ctor_set(v___x_1661_, 1, v___x_1659_);
lean_ctor_set(v___x_1661_, 2, v___x_1658_);
return v___x_1661_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__27(void){
_start:
{
lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1662_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__26, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__26);
v___x_1663_ = lean_unsigned_to_nat(1022u);
v___x_1664_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__3));
v___x_1665_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1665_, 0, v___x_1664_);
lean_ctor_set(v___x_1665_, 1, v___x_1663_);
lean_ctor_set(v___x_1665_, 2, v___x_1662_);
return v___x_1665_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_splitIfs(void){
_start:
{
lean_object* v___x_1666_; 
v___x_1666_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_splitIfs___closed__27, &lp_mathlib_Mathlib_Tactic_splitIfs___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_splitIfs___closed__27);
return v___x_1666_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; 
v___x_1667_ = lean_box(0);
v___x_1668_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1669_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1669_, 0, v___x_1668_);
lean_ctor_set(v___x_1669_, 1, v___x_1667_);
return v___x_1669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1671_; lean_object* v___x_1672_; 
v___x_1671_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___closed__0);
v___x_1672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1672_, 0, v___x_1671_);
return v___x_1672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg___boxed(lean_object* v___y_1673_){
_start:
{
lean_object* v_res_1674_; 
v_res_1674_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v_res_1674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0(lean_object* v_00_u03b1_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_){
_start:
{
lean_object* v___x_1685_; 
v___x_1685_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v___x_1685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___boxed(lean_object* v_00_u03b1_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_){
_start:
{
lean_object* v_res_1696_; 
v_res_1696_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0(v_00_u03b1_1686_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
lean_dec(v___y_1690_);
lean_dec_ref(v___y_1689_);
lean_dec(v___y_1688_);
lean_dec_ref(v___y_1687_);
return v_res_1696_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0(uint8_t v___y_1704_, uint8_t v_suppressElabErrors_1705_, lean_object* v_x_1706_){
_start:
{
if (lean_obj_tag(v_x_1706_) == 1)
{
lean_object* v_pre_1707_; 
v_pre_1707_ = lean_ctor_get(v_x_1706_, 0);
switch(lean_obj_tag(v_pre_1707_))
{
case 1:
{
lean_object* v_pre_1708_; 
v_pre_1708_ = lean_ctor_get(v_pre_1707_, 0);
switch(lean_obj_tag(v_pre_1708_))
{
case 0:
{
lean_object* v_str_1709_; lean_object* v_str_1710_; lean_object* v___x_1711_; uint8_t v___x_1712_; 
v_str_1709_ = lean_ctor_get(v_x_1706_, 1);
v_str_1710_ = lean_ctor_get(v_pre_1707_, 1);
v___x_1711_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__0));
v___x_1712_ = lean_string_dec_eq(v_str_1710_, v___x_1711_);
if (v___x_1712_ == 0)
{
lean_object* v___x_1713_; uint8_t v___x_1714_; 
v___x_1713_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__1));
v___x_1714_ = lean_string_dec_eq(v_str_1710_, v___x_1713_);
if (v___x_1714_ == 0)
{
return v___y_1704_;
}
else
{
lean_object* v___x_1715_; uint8_t v___x_1716_; 
v___x_1715_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__1));
v___x_1716_ = lean_string_dec_eq(v_str_1709_, v___x_1715_);
if (v___x_1716_ == 0)
{
return v___y_1704_;
}
else
{
return v_suppressElabErrors_1705_;
}
}
}
else
{
lean_object* v___x_1717_; uint8_t v___x_1718_; 
v___x_1717_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__2));
v___x_1718_ = lean_string_dec_eq(v_str_1709_, v___x_1717_);
if (v___x_1718_ == 0)
{
return v___y_1704_;
}
else
{
return v_suppressElabErrors_1705_;
}
}
}
case 1:
{
lean_object* v_pre_1719_; 
v_pre_1719_ = lean_ctor_get(v_pre_1708_, 0);
if (lean_obj_tag(v_pre_1719_) == 0)
{
lean_object* v_str_1720_; lean_object* v_str_1721_; lean_object* v_str_1722_; lean_object* v___x_1723_; uint8_t v___x_1724_; 
v_str_1720_ = lean_ctor_get(v_x_1706_, 1);
v_str_1721_ = lean_ctor_get(v_pre_1707_, 1);
v_str_1722_ = lean_ctor_get(v_pre_1708_, 1);
v___x_1723_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__3));
v___x_1724_ = lean_string_dec_eq(v_str_1722_, v___x_1723_);
if (v___x_1724_ == 0)
{
return v___y_1704_;
}
else
{
lean_object* v___x_1725_; uint8_t v___x_1726_; 
v___x_1725_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__4));
v___x_1726_ = lean_string_dec_eq(v_str_1721_, v___x_1725_);
if (v___x_1726_ == 0)
{
return v___y_1704_;
}
else
{
lean_object* v___x_1727_; uint8_t v___x_1728_; 
v___x_1727_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__5));
v___x_1728_ = lean_string_dec_eq(v_str_1720_, v___x_1727_);
if (v___x_1728_ == 0)
{
return v___y_1704_;
}
else
{
return v_suppressElabErrors_1705_;
}
}
}
}
else
{
return v___y_1704_;
}
}
default: 
{
return v___y_1704_;
}
}
}
case 0:
{
lean_object* v_str_1729_; lean_object* v___x_1730_; uint8_t v___x_1731_; 
v_str_1729_ = lean_ctor_get(v_x_1706_, 1);
v___x_1730_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___closed__6));
v___x_1731_ = lean_string_dec_eq(v_str_1729_, v___x_1730_);
if (v___x_1731_ == 0)
{
return v___y_1704_;
}
else
{
return v_suppressElabErrors_1705_;
}
}
default: 
{
return v___y_1704_;
}
}
}
else
{
return v___y_1704_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v___y_1732_, lean_object* v_suppressElabErrors_1733_, lean_object* v_x_1734_){
_start:
{
uint8_t v___y_7551__boxed_1735_; uint8_t v_suppressElabErrors_boxed_1736_; uint8_t v_res_1737_; lean_object* v_r_1738_; 
v___y_7551__boxed_1735_ = lean_unbox(v___y_1732_);
v_suppressElabErrors_boxed_1736_ = lean_unbox(v_suppressElabErrors_1733_);
v_res_1737_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0(v___y_7551__boxed_1735_, v_suppressElabErrors_boxed_1736_, v_x_1734_);
lean_dec(v_x_1734_);
v_r_1738_ = lean_box(v_res_1737_);
return v_r_1738_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3(lean_object* v_opts_1739_, lean_object* v_opt_1740_){
_start:
{
lean_object* v_name_1741_; lean_object* v_defValue_1742_; lean_object* v_map_1743_; lean_object* v___x_1744_; 
v_name_1741_ = lean_ctor_get(v_opt_1740_, 0);
v_defValue_1742_ = lean_ctor_get(v_opt_1740_, 1);
v_map_1743_ = lean_ctor_get(v_opts_1739_, 0);
v___x_1744_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1743_, v_name_1741_);
if (lean_obj_tag(v___x_1744_) == 0)
{
uint8_t v___x_1745_; 
v___x_1745_ = lean_unbox(v_defValue_1742_);
return v___x_1745_;
}
else
{
lean_object* v_val_1746_; 
v_val_1746_ = lean_ctor_get(v___x_1744_, 0);
lean_inc(v_val_1746_);
lean_dec_ref_known(v___x_1744_, 1);
if (lean_obj_tag(v_val_1746_) == 1)
{
uint8_t v_v_1747_; 
v_v_1747_ = lean_ctor_get_uint8(v_val_1746_, 0);
lean_dec_ref_known(v_val_1746_, 0);
return v_v_1747_;
}
else
{
uint8_t v___x_1748_; 
lean_dec(v_val_1746_);
v___x_1748_ = lean_unbox(v_defValue_1742_);
return v___x_1748_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3___boxed(lean_object* v_opts_1749_, lean_object* v_opt_1750_){
_start:
{
uint8_t v_res_1751_; lean_object* v_r_1752_; 
v_res_1751_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3(v_opts_1749_, v_opt_1750_);
lean_dec_ref(v_opt_1750_);
lean_dec_ref(v_opts_1749_);
v_r_1752_ = lean_box(v_res_1751_);
return v_r_1752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2(lean_object* v_msgData_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_){
_start:
{
lean_object* v___x_1759_; lean_object* v_env_1760_; lean_object* v___x_1761_; lean_object* v_mctx_1762_; lean_object* v_lctx_1763_; lean_object* v_options_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; 
v___x_1759_ = lean_st_ref_get(v___y_1757_);
v_env_1760_ = lean_ctor_get(v___x_1759_, 0);
lean_inc_ref(v_env_1760_);
lean_dec(v___x_1759_);
v___x_1761_ = lean_st_ref_get(v___y_1755_);
v_mctx_1762_ = lean_ctor_get(v___x_1761_, 0);
lean_inc_ref(v_mctx_1762_);
lean_dec(v___x_1761_);
v_lctx_1763_ = lean_ctor_get(v___y_1754_, 2);
v_options_1764_ = lean_ctor_get(v___y_1756_, 2);
lean_inc_ref(v_options_1764_);
lean_inc_ref(v_lctx_1763_);
v___x_1765_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1765_, 0, v_env_1760_);
lean_ctor_set(v___x_1765_, 1, v_mctx_1762_);
lean_ctor_set(v___x_1765_, 2, v_lctx_1763_);
lean_ctor_set(v___x_1765_, 3, v_options_1764_);
v___x_1766_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1766_, 0, v___x_1765_);
lean_ctor_set(v___x_1766_, 1, v_msgData_1753_);
v___x_1767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1767_, 0, v___x_1766_);
return v___x_1767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_){
_start:
{
lean_object* v_res_1774_; 
v_res_1774_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2(v_msgData_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_);
lean_dec(v___y_1772_);
lean_dec_ref(v___y_1771_);
lean_dec(v___y_1770_);
lean_dec_ref(v___y_1769_);
return v_res_1774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg(lean_object* v_ref_1776_, lean_object* v_msgData_1777_, uint8_t v_severity_1778_, uint8_t v_isSilent_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_){
_start:
{
lean_object* v___y_1786_; lean_object* v___y_1787_; lean_object* v___y_1788_; uint8_t v___y_1789_; uint8_t v___y_1790_; lean_object* v___y_1791_; lean_object* v___y_1792_; lean_object* v___y_1793_; lean_object* v___y_1794_; lean_object* v___y_1822_; lean_object* v___y_1823_; lean_object* v___y_1824_; lean_object* v___y_1825_; uint8_t v___y_1826_; uint8_t v___y_1827_; uint8_t v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; uint8_t v___y_1850_; uint8_t v___y_1851_; lean_object* v___y_1852_; uint8_t v___y_1853_; lean_object* v___y_1854_; lean_object* v___y_1858_; lean_object* v___y_1859_; lean_object* v___y_1860_; lean_object* v___y_1861_; uint8_t v___y_1862_; uint8_t v___y_1863_; uint8_t v___y_1864_; uint8_t v___x_1869_; lean_object* v___y_1871_; lean_object* v___y_1872_; lean_object* v___y_1873_; lean_object* v___y_1874_; uint8_t v___y_1875_; uint8_t v___y_1876_; uint8_t v___y_1877_; uint8_t v___y_1879_; uint8_t v___x_1894_; 
v___x_1869_ = 2;
v___x_1894_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1778_, v___x_1869_);
if (v___x_1894_ == 0)
{
v___y_1879_ = v___x_1894_;
goto v___jp_1878_;
}
else
{
uint8_t v___x_1895_; 
lean_inc_ref(v_msgData_1777_);
v___x_1895_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1777_);
v___y_1879_ = v___x_1895_;
goto v___jp_1878_;
}
v___jp_1785_:
{
lean_object* v___x_1795_; lean_object* v_currNamespace_1796_; lean_object* v_openDecls_1797_; lean_object* v_env_1798_; lean_object* v_nextMacroScope_1799_; lean_object* v_ngen_1800_; lean_object* v_auxDeclNGen_1801_; lean_object* v_traceState_1802_; lean_object* v_cache_1803_; lean_object* v_messages_1804_; lean_object* v_infoState_1805_; lean_object* v_snapshotTasks_1806_; lean_object* v___x_1808_; uint8_t v_isShared_1809_; uint8_t v_isSharedCheck_1820_; 
v___x_1795_ = lean_st_ref_take(v___y_1794_);
v_currNamespace_1796_ = lean_ctor_get(v___y_1793_, 6);
v_openDecls_1797_ = lean_ctor_get(v___y_1793_, 7);
v_env_1798_ = lean_ctor_get(v___x_1795_, 0);
v_nextMacroScope_1799_ = lean_ctor_get(v___x_1795_, 1);
v_ngen_1800_ = lean_ctor_get(v___x_1795_, 2);
v_auxDeclNGen_1801_ = lean_ctor_get(v___x_1795_, 3);
v_traceState_1802_ = lean_ctor_get(v___x_1795_, 4);
v_cache_1803_ = lean_ctor_get(v___x_1795_, 5);
v_messages_1804_ = lean_ctor_get(v___x_1795_, 6);
v_infoState_1805_ = lean_ctor_get(v___x_1795_, 7);
v_snapshotTasks_1806_ = lean_ctor_get(v___x_1795_, 8);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1795_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1808_ = v___x_1795_;
v_isShared_1809_ = v_isSharedCheck_1820_;
goto v_resetjp_1807_;
}
else
{
lean_inc(v_snapshotTasks_1806_);
lean_inc(v_infoState_1805_);
lean_inc(v_messages_1804_);
lean_inc(v_cache_1803_);
lean_inc(v_traceState_1802_);
lean_inc(v_auxDeclNGen_1801_);
lean_inc(v_ngen_1800_);
lean_inc(v_nextMacroScope_1799_);
lean_inc(v_env_1798_);
lean_dec(v___x_1795_);
v___x_1808_ = lean_box(0);
v_isShared_1809_ = v_isSharedCheck_1820_;
goto v_resetjp_1807_;
}
v_resetjp_1807_:
{
lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1815_; 
lean_inc(v_openDecls_1797_);
lean_inc(v_currNamespace_1796_);
v___x_1810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1810_, 0, v_currNamespace_1796_);
lean_ctor_set(v___x_1810_, 1, v_openDecls_1797_);
v___x_1811_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1811_, 0, v___x_1810_);
lean_ctor_set(v___x_1811_, 1, v___y_1786_);
lean_inc_ref(v___y_1787_);
lean_inc_ref(v___y_1788_);
v___x_1812_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1812_, 0, v___y_1788_);
lean_ctor_set(v___x_1812_, 1, v___y_1791_);
lean_ctor_set(v___x_1812_, 2, v___y_1792_);
lean_ctor_set(v___x_1812_, 3, v___y_1787_);
lean_ctor_set(v___x_1812_, 4, v___x_1811_);
lean_ctor_set_uint8(v___x_1812_, sizeof(void*)*5, v___y_1790_);
lean_ctor_set_uint8(v___x_1812_, sizeof(void*)*5 + 1, v___y_1789_);
lean_ctor_set_uint8(v___x_1812_, sizeof(void*)*5 + 2, v_isSilent_1779_);
v___x_1813_ = l_Lean_MessageLog_add(v___x_1812_, v_messages_1804_);
if (v_isShared_1809_ == 0)
{
lean_ctor_set(v___x_1808_, 6, v___x_1813_);
v___x_1815_ = v___x_1808_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_env_1798_);
lean_ctor_set(v_reuseFailAlloc_1819_, 1, v_nextMacroScope_1799_);
lean_ctor_set(v_reuseFailAlloc_1819_, 2, v_ngen_1800_);
lean_ctor_set(v_reuseFailAlloc_1819_, 3, v_auxDeclNGen_1801_);
lean_ctor_set(v_reuseFailAlloc_1819_, 4, v_traceState_1802_);
lean_ctor_set(v_reuseFailAlloc_1819_, 5, v_cache_1803_);
lean_ctor_set(v_reuseFailAlloc_1819_, 6, v___x_1813_);
lean_ctor_set(v_reuseFailAlloc_1819_, 7, v_infoState_1805_);
lean_ctor_set(v_reuseFailAlloc_1819_, 8, v_snapshotTasks_1806_);
v___x_1815_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; 
v___x_1816_ = lean_st_ref_set(v___y_1794_, v___x_1815_);
v___x_1817_ = lean_box(0);
v___x_1818_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1818_, 0, v___x_1817_);
return v___x_1818_;
}
}
}
v___jp_1821_:
{
lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v_a_1832_; lean_object* v___x_1834_; uint8_t v_isShared_1835_; uint8_t v_isSharedCheck_1845_; 
v___x_1830_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1777_);
v___x_1831_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__2(v___x_1830_, v___y_1780_, v___y_1781_, v___y_1782_, v___y_1783_);
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1834_ = v___x_1831_;
v_isShared_1835_ = v_isSharedCheck_1845_;
goto v_resetjp_1833_;
}
else
{
lean_inc(v_a_1832_);
lean_dec(v___x_1831_);
v___x_1834_ = lean_box(0);
v_isShared_1835_ = v_isSharedCheck_1845_;
goto v_resetjp_1833_;
}
v_resetjp_1833_:
{
lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; 
lean_inc_ref_n(v___y_1824_, 2);
v___x_1836_ = l_Lean_FileMap_toPosition(v___y_1824_, v___y_1823_);
lean_dec(v___y_1823_);
v___x_1837_ = l_Lean_FileMap_toPosition(v___y_1824_, v___y_1829_);
lean_dec(v___y_1829_);
v___x_1838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1838_, 0, v___x_1837_);
v___x_1839_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___closed__0));
if (v___y_1828_ == 0)
{
lean_del_object(v___x_1834_);
lean_dec_ref(v___y_1822_);
v___y_1786_ = v_a_1832_;
v___y_1787_ = v___x_1839_;
v___y_1788_ = v___y_1825_;
v___y_1789_ = v___y_1827_;
v___y_1790_ = v___y_1826_;
v___y_1791_ = v___x_1836_;
v___y_1792_ = v___x_1838_;
v___y_1793_ = v___y_1782_;
v___y_1794_ = v___y_1783_;
goto v___jp_1785_;
}
else
{
uint8_t v___x_1840_; 
lean_inc(v_a_1832_);
v___x_1840_ = l_Lean_MessageData_hasTag(v___y_1822_, v_a_1832_);
if (v___x_1840_ == 0)
{
lean_object* v___x_1841_; lean_object* v___x_1843_; 
lean_dec_ref_known(v___x_1838_, 1);
lean_dec_ref(v___x_1836_);
lean_dec(v_a_1832_);
v___x_1841_ = lean_box(0);
if (v_isShared_1835_ == 0)
{
lean_ctor_set(v___x_1834_, 0, v___x_1841_);
v___x_1843_ = v___x_1834_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v___x_1841_);
v___x_1843_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
return v___x_1843_;
}
}
else
{
lean_del_object(v___x_1834_);
v___y_1786_ = v_a_1832_;
v___y_1787_ = v___x_1839_;
v___y_1788_ = v___y_1825_;
v___y_1789_ = v___y_1827_;
v___y_1790_ = v___y_1826_;
v___y_1791_ = v___x_1836_;
v___y_1792_ = v___x_1838_;
v___y_1793_ = v___y_1782_;
v___y_1794_ = v___y_1783_;
goto v___jp_1785_;
}
}
}
}
v___jp_1846_:
{
lean_object* v___x_1855_; 
v___x_1855_ = l_Lean_Syntax_getTailPos_x3f(v___y_1852_, v___y_1851_);
lean_dec(v___y_1852_);
if (lean_obj_tag(v___x_1855_) == 0)
{
lean_inc(v___y_1854_);
v___y_1822_ = v___y_1847_;
v___y_1823_ = v___y_1854_;
v___y_1824_ = v___y_1848_;
v___y_1825_ = v___y_1849_;
v___y_1826_ = v___y_1851_;
v___y_1827_ = v___y_1850_;
v___y_1828_ = v___y_1853_;
v___y_1829_ = v___y_1854_;
goto v___jp_1821_;
}
else
{
lean_object* v_val_1856_; 
v_val_1856_ = lean_ctor_get(v___x_1855_, 0);
lean_inc(v_val_1856_);
lean_dec_ref_known(v___x_1855_, 1);
v___y_1822_ = v___y_1847_;
v___y_1823_ = v___y_1854_;
v___y_1824_ = v___y_1848_;
v___y_1825_ = v___y_1849_;
v___y_1826_ = v___y_1851_;
v___y_1827_ = v___y_1850_;
v___y_1828_ = v___y_1853_;
v___y_1829_ = v_val_1856_;
goto v___jp_1821_;
}
}
v___jp_1857_:
{
lean_object* v_ref_1865_; lean_object* v___x_1866_; 
v_ref_1865_ = l_Lean_replaceRef(v_ref_1776_, v___y_1860_);
v___x_1866_ = l_Lean_Syntax_getPos_x3f(v_ref_1865_, v___y_1862_);
if (lean_obj_tag(v___x_1866_) == 0)
{
lean_object* v___x_1867_; 
v___x_1867_ = lean_unsigned_to_nat(0u);
v___y_1847_ = v___y_1858_;
v___y_1848_ = v___y_1859_;
v___y_1849_ = v___y_1861_;
v___y_1850_ = v___y_1864_;
v___y_1851_ = v___y_1862_;
v___y_1852_ = v_ref_1865_;
v___y_1853_ = v___y_1863_;
v___y_1854_ = v___x_1867_;
goto v___jp_1846_;
}
else
{
lean_object* v_val_1868_; 
v_val_1868_ = lean_ctor_get(v___x_1866_, 0);
lean_inc(v_val_1868_);
lean_dec_ref_known(v___x_1866_, 1);
v___y_1847_ = v___y_1858_;
v___y_1848_ = v___y_1859_;
v___y_1849_ = v___y_1861_;
v___y_1850_ = v___y_1864_;
v___y_1851_ = v___y_1862_;
v___y_1852_ = v_ref_1865_;
v___y_1853_ = v___y_1863_;
v___y_1854_ = v_val_1868_;
goto v___jp_1846_;
}
}
v___jp_1870_:
{
if (v___y_1877_ == 0)
{
v___y_1858_ = v___y_1872_;
v___y_1859_ = v___y_1871_;
v___y_1860_ = v___y_1873_;
v___y_1861_ = v___y_1874_;
v___y_1862_ = v___y_1876_;
v___y_1863_ = v___y_1875_;
v___y_1864_ = v_severity_1778_;
goto v___jp_1857_;
}
else
{
v___y_1858_ = v___y_1872_;
v___y_1859_ = v___y_1871_;
v___y_1860_ = v___y_1873_;
v___y_1861_ = v___y_1874_;
v___y_1862_ = v___y_1876_;
v___y_1863_ = v___y_1875_;
v___y_1864_ = v___x_1869_;
goto v___jp_1857_;
}
}
v___jp_1878_:
{
if (v___y_1879_ == 0)
{
lean_object* v_fileName_1880_; lean_object* v_fileMap_1881_; lean_object* v_options_1882_; lean_object* v_ref_1883_; uint8_t v_suppressElabErrors_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___f_1887_; uint8_t v___x_1888_; uint8_t v___x_1889_; 
v_fileName_1880_ = lean_ctor_get(v___y_1782_, 0);
v_fileMap_1881_ = lean_ctor_get(v___y_1782_, 1);
v_options_1882_ = lean_ctor_get(v___y_1782_, 2);
v_ref_1883_ = lean_ctor_get(v___y_1782_, 5);
v_suppressElabErrors_1884_ = lean_ctor_get_uint8(v___y_1782_, sizeof(void*)*14 + 1);
v___x_1885_ = lean_box(v___y_1879_);
v___x_1886_ = lean_box(v_suppressElabErrors_1884_);
v___f_1887_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1887_, 0, v___x_1885_);
lean_closure_set(v___f_1887_, 1, v___x_1886_);
v___x_1888_ = 1;
v___x_1889_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1778_, v___x_1888_);
if (v___x_1889_ == 0)
{
v___y_1871_ = v_fileMap_1881_;
v___y_1872_ = v___f_1887_;
v___y_1873_ = v_ref_1883_;
v___y_1874_ = v_fileName_1880_;
v___y_1875_ = v_suppressElabErrors_1884_;
v___y_1876_ = v___y_1879_;
v___y_1877_ = v___x_1889_;
goto v___jp_1870_;
}
else
{
lean_object* v___x_1890_; uint8_t v___x_1891_; 
v___x_1890_ = l_Lean_warningAsError;
v___x_1891_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1_spec__3(v_options_1882_, v___x_1890_);
v___y_1871_ = v_fileMap_1881_;
v___y_1872_ = v___f_1887_;
v___y_1873_ = v_ref_1883_;
v___y_1874_ = v_fileName_1880_;
v___y_1875_ = v_suppressElabErrors_1884_;
v___y_1876_ = v___y_1879_;
v___y_1877_ = v___x_1891_;
goto v___jp_1870_;
}
}
else
{
lean_object* v___x_1892_; lean_object* v___x_1893_; 
lean_dec_ref(v_msgData_1777_);
v___x_1892_ = lean_box(0);
v___x_1893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1893_, 0, v___x_1892_);
return v___x_1893_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg___boxed(lean_object* v_ref_1896_, lean_object* v_msgData_1897_, lean_object* v_severity_1898_, lean_object* v_isSilent_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
uint8_t v_severity_boxed_1905_; uint8_t v_isSilent_boxed_1906_; lean_object* v_res_1907_; 
v_severity_boxed_1905_ = lean_unbox(v_severity_1898_);
v_isSilent_boxed_1906_ = lean_unbox(v_isSilent_1899_);
v_res_1907_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg(v_ref_1896_, v_msgData_1897_, v_severity_boxed_1905_, v_isSilent_boxed_1906_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1901_);
lean_dec_ref(v___y_1900_);
lean_dec(v_ref_1896_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1(lean_object* v_ref_1908_, lean_object* v_msgData_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
uint8_t v___x_1919_; uint8_t v___x_1920_; lean_object* v___x_1921_; 
v___x_1919_ = 1;
v___x_1920_ = 0;
v___x_1921_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg(v_ref_1908_, v_msgData_1909_, v___x_1919_, v___x_1920_, v___y_1914_, v___y_1915_, v___y_1916_, v___y_1917_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1___boxed(lean_object* v_ref_1922_, lean_object* v_msgData_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_){
_start:
{
lean_object* v_res_1933_; 
v_res_1933_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1(v_ref_1922_, v_msgData_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_, v___y_1931_);
lean_dec(v___y_1931_);
lean_dec_ref(v___y_1930_);
lean_dec(v___y_1929_);
lean_dec_ref(v___y_1928_);
lean_dec(v___y_1927_);
lean_dec_ref(v___y_1926_);
lean_dec(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec(v_ref_1922_);
return v_res_1933_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_1935_; lean_object* v___x_1936_; 
v___x_1935_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__0));
v___x_1936_ = l_Lean_stringToMessageData(v___x_1935_);
return v___x_1936_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg(lean_object* v_as_x27_1937_, lean_object* v_b_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_){
_start:
{
if (lean_obj_tag(v_as_x27_1937_) == 0)
{
lean_object* v___x_1948_; 
v___x_1948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1948_, 0, v_b_1938_);
return v___x_1948_;
}
else
{
lean_object* v_head_1949_; lean_object* v_tail_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; 
v_head_1949_ = lean_ctor_get(v_as_x27_1937_, 0);
v_tail_1950_ = lean_ctor_get(v_as_x27_1937_, 1);
v___x_1951_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___closed__1);
lean_inc(v_head_1949_);
v___x_1952_ = l_Lean_MessageData_ofSyntax(v_head_1949_);
v___x_1953_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1953_, 0, v___x_1951_);
lean_ctor_set(v___x_1953_, 1, v___x_1952_);
v___x_1954_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1(v_head_1949_, v___x_1953_, v___y_1939_, v___y_1940_, v___y_1941_, v___y_1942_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_);
if (lean_obj_tag(v___x_1954_) == 0)
{
lean_object* v___x_1955_; 
lean_dec_ref_known(v___x_1954_, 1);
v___x_1955_ = lean_box(0);
v_as_x27_1937_ = v_tail_1950_;
v_b_1938_ = v___x_1955_;
goto _start;
}
else
{
return v___x_1954_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg___boxed(lean_object* v_as_x27_1957_, lean_object* v_b_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_){
_start:
{
lean_object* v_res_1968_; 
v_res_1968_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg(v_as_x27_1957_, v_b_1958_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_);
lean_dec(v___y_1966_);
lean_dec_ref(v___y_1965_);
lean_dec(v___y_1964_);
lean_dec_ref(v___y_1963_);
lean_dec(v___y_1962_);
lean_dec_ref(v___y_1961_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v_as_x27_1957_);
return v_res_1968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0(lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_){
_start:
{
lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; 
v___x_1980_ = lean_st_mk_ref(v___y_1969_);
v___x_1981_ = lean_box(0);
lean_inc(v___x_1980_);
v___x_1982_ = lp_mathlib___private_Mathlib_Tactic_SplitIfs_0__Mathlib_Tactic_splitIfsCore(v___y_1970_, v___x_1980_, v___x_1981_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_);
if (lean_obj_tag(v___x_1982_) == 0)
{
lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
lean_dec_ref_known(v___x_1982_, 1);
v___x_1983_ = lean_st_ref_get(v___x_1980_);
lean_dec(v___x_1980_);
v___x_1984_ = lean_box(0);
v___x_1985_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg(v___x_1983_, v___x_1984_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_);
lean_dec(v___x_1983_);
if (lean_obj_tag(v___x_1985_) == 0)
{
lean_object* v___x_1987_; uint8_t v_isShared_1988_; uint8_t v_isSharedCheck_1992_; 
v_isSharedCheck_1992_ = !lean_is_exclusive(v___x_1985_);
if (v_isSharedCheck_1992_ == 0)
{
lean_object* v_unused_1993_; 
v_unused_1993_ = lean_ctor_get(v___x_1985_, 0);
lean_dec(v_unused_1993_);
v___x_1987_ = v___x_1985_;
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
else
{
lean_dec(v___x_1985_);
v___x_1987_ = lean_box(0);
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
v_resetjp_1986_:
{
lean_object* v___x_1990_; 
if (v_isShared_1988_ == 0)
{
lean_ctor_set(v___x_1987_, 0, v___x_1984_);
v___x_1990_ = v___x_1987_;
goto v_reusejp_1989_;
}
else
{
lean_object* v_reuseFailAlloc_1991_; 
v_reuseFailAlloc_1991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1991_, 0, v___x_1984_);
v___x_1990_ = v_reuseFailAlloc_1991_;
goto v_reusejp_1989_;
}
v_reusejp_1989_:
{
return v___x_1990_;
}
}
}
else
{
return v___x_1985_;
}
}
else
{
lean_dec(v___x_1980_);
return v___x_1982_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0___boxed(lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_){
_start:
{
lean_object* v_res_2005_; 
v_res_2005_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0(v___y_1994_, v___y_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
lean_dec(v___y_2003_);
lean_dec_ref(v___y_2002_);
lean_dec(v___y_2001_);
lean_dec_ref(v___y_2000_);
lean_dec(v___y_1999_);
lean_dec_ref(v___y_1998_);
lean_dec(v___y_1997_);
lean_dec_ref(v___y_1996_);
return v_res_2005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1(lean_object* v_x_2015_, lean_object* v_a_2016_, lean_object* v_a_2017_, lean_object* v_a_2018_, lean_object* v_a_2019_, lean_object* v_a_2020_, lean_object* v_a_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_){
_start:
{
lean_object* v___y_2026_; lean_object* v___y_2027_; lean_object* v___y_2028_; lean_object* v___y_2029_; lean_object* v___y_2030_; lean_object* v___y_2031_; lean_object* v___y_2032_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; lean_object* v___y_2039_; lean_object* v___y_2040_; lean_object* v___y_2041_; lean_object* v___y_2042_; lean_object* v___y_2043_; lean_object* v___y_2044_; lean_object* v___y_2045_; lean_object* v___y_2046_; lean_object* v___y_2047_; lean_object* v___y_2048_; lean_object* v___x_2052_; uint8_t v___x_2053_; lean_object* v___y_2055_; lean_object* v_withArg_2056_; lean_object* v___y_2057_; lean_object* v___y_2058_; lean_object* v___y_2059_; lean_object* v___y_2060_; lean_object* v___y_2061_; lean_object* v___y_2062_; lean_object* v___y_2063_; lean_object* v___y_2064_; 
v___x_2052_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_splitIfs___closed__3));
lean_inc(v_x_2015_);
v___x_2053_ = l_Lean_Syntax_isOfKind(v_x_2015_, v___x_2052_);
if (v___x_2053_ == 0)
{
lean_object* v___x_2069_; 
lean_dec(v_x_2015_);
v___x_2069_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v___x_2069_;
}
else
{
lean_object* v___x_2070_; lean_object* v_loc_2072_; lean_object* v___y_2073_; lean_object* v___y_2074_; lean_object* v___y_2075_; lean_object* v___y_2076_; lean_object* v___y_2077_; lean_object* v___y_2078_; lean_object* v___y_2079_; lean_object* v___y_2080_; lean_object* v___x_2090_; uint8_t v___x_2091_; 
v___x_2070_ = lean_unsigned_to_nat(1u);
v___x_2090_ = l_Lean_Syntax_getArg(v_x_2015_, v___x_2070_);
v___x_2091_ = l_Lean_Syntax_isNone(v___x_2090_);
if (v___x_2091_ == 0)
{
uint8_t v___x_2092_; 
lean_inc(v___x_2090_);
v___x_2092_ = l_Lean_Syntax_matchesNull(v___x_2090_, v___x_2070_);
if (v___x_2092_ == 0)
{
lean_object* v___x_2093_; 
lean_dec(v___x_2090_);
lean_dec(v_x_2015_);
v___x_2093_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v___x_2093_;
}
else
{
lean_object* v___x_2094_; lean_object* v_loc_2095_; lean_object* v___x_2096_; uint8_t v___x_2097_; 
v___x_2094_ = lean_unsigned_to_nat(0u);
v_loc_2095_ = l_Lean_Syntax_getArg(v___x_2090_, v___x_2094_);
lean_dec(v___x_2090_);
v___x_2096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__3));
lean_inc(v_loc_2095_);
v___x_2097_ = l_Lean_Syntax_isOfKind(v_loc_2095_, v___x_2096_);
if (v___x_2097_ == 0)
{
lean_object* v___x_2098_; 
lean_dec(v_loc_2095_);
lean_dec(v_x_2015_);
v___x_2098_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v___x_2098_;
}
else
{
lean_object* v___x_2099_; 
v___x_2099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2099_, 0, v_loc_2095_);
v_loc_2072_ = v___x_2099_;
v___y_2073_ = v_a_2016_;
v___y_2074_ = v_a_2017_;
v___y_2075_ = v_a_2018_;
v___y_2076_ = v_a_2019_;
v___y_2077_ = v_a_2020_;
v___y_2078_ = v_a_2021_;
v___y_2079_ = v_a_2022_;
v___y_2080_ = v_a_2023_;
goto v___jp_2071_;
}
}
}
else
{
lean_object* v___x_2100_; 
lean_dec(v___x_2090_);
v___x_2100_ = lean_box(0);
v_loc_2072_ = v___x_2100_;
v___y_2073_ = v_a_2016_;
v___y_2074_ = v_a_2017_;
v___y_2075_ = v_a_2018_;
v___y_2076_ = v_a_2019_;
v___y_2077_ = v_a_2020_;
v___y_2078_ = v_a_2021_;
v___y_2079_ = v_a_2022_;
v___y_2080_ = v_a_2023_;
goto v___jp_2071_;
}
v___jp_2071_:
{
lean_object* v___x_2081_; lean_object* v___x_2082_; uint8_t v___x_2083_; 
v___x_2081_ = lean_unsigned_to_nat(2u);
v___x_2082_ = l_Lean_Syntax_getArg(v_x_2015_, v___x_2081_);
lean_dec(v_x_2015_);
v___x_2083_ = l_Lean_Syntax_isNone(v___x_2082_);
if (v___x_2083_ == 0)
{
uint8_t v___x_2084_; 
lean_inc(v___x_2082_);
v___x_2084_ = l_Lean_Syntax_matchesNull(v___x_2082_, v___x_2081_);
if (v___x_2084_ == 0)
{
lean_object* v___x_2085_; 
lean_dec(v___x_2082_);
lean_dec(v_loc_2072_);
v___x_2085_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__0___redArg();
return v___x_2085_;
}
else
{
lean_object* v___x_2086_; lean_object* v_withArg_2087_; lean_object* v___x_2088_; 
v___x_2086_ = l_Lean_Syntax_getArg(v___x_2082_, v___x_2070_);
lean_dec(v___x_2082_);
v_withArg_2087_ = l_Lean_Syntax_getArgs(v___x_2086_);
lean_dec(v___x_2086_);
v___x_2088_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2088_, 0, v_withArg_2087_);
v___y_2055_ = v_loc_2072_;
v_withArg_2056_ = v___x_2088_;
v___y_2057_ = v___y_2073_;
v___y_2058_ = v___y_2074_;
v___y_2059_ = v___y_2075_;
v___y_2060_ = v___y_2076_;
v___y_2061_ = v___y_2077_;
v___y_2062_ = v___y_2078_;
v___y_2063_ = v___y_2079_;
v___y_2064_ = v___y_2080_;
goto v___jp_2054_;
}
}
else
{
lean_object* v___x_2089_; 
lean_dec(v___x_2082_);
v___x_2089_ = lean_box(0);
v___y_2055_ = v_loc_2072_;
v_withArg_2056_ = v___x_2089_;
v___y_2057_ = v___y_2073_;
v___y_2058_ = v___y_2074_;
v___y_2059_ = v___y_2075_;
v___y_2060_ = v___y_2076_;
v___y_2061_ = v___y_2077_;
v___y_2062_ = v___y_2078_;
v___y_2063_ = v___y_2079_;
v___y_2064_ = v___y_2080_;
goto v___jp_2054_;
}
}
}
v___jp_2025_:
{
lean_object* v___f_2036_; lean_object* v___x_2037_; 
v___f_2036_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___lam__0___boxed), 11, 2);
lean_closure_set(v___f_2036_, 0, v___y_2035_);
lean_closure_set(v___f_2036_, 1, v___y_2027_);
v___x_2037_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2036_, v___y_2032_, v___y_2034_, v___y_2026_, v___y_2031_, v___y_2030_, v___y_2029_, v___y_2028_, v___y_2033_);
return v___x_2037_;
}
v___jp_2038_:
{
if (lean_obj_tag(v___y_2043_) == 0)
{
lean_object* v___x_2049_; 
v___x_2049_ = lean_box(0);
v___y_2026_ = v___y_2045_;
v___y_2027_ = v___y_2048_;
v___y_2028_ = v___y_2046_;
v___y_2029_ = v___y_2039_;
v___y_2030_ = v___y_2040_;
v___y_2031_ = v___y_2041_;
v___y_2032_ = v___y_2042_;
v___y_2033_ = v___y_2047_;
v___y_2034_ = v___y_2044_;
v___y_2035_ = v___x_2049_;
goto v___jp_2025_;
}
else
{
lean_object* v_val_2050_; lean_object* v___x_2051_; 
v_val_2050_ = lean_ctor_get(v___y_2043_, 0);
lean_inc(v_val_2050_);
lean_dec_ref_known(v___y_2043_, 1);
v___x_2051_ = lean_array_to_list(v_val_2050_);
v___y_2026_ = v___y_2045_;
v___y_2027_ = v___y_2048_;
v___y_2028_ = v___y_2046_;
v___y_2029_ = v___y_2039_;
v___y_2030_ = v___y_2040_;
v___y_2031_ = v___y_2041_;
v___y_2032_ = v___y_2042_;
v___y_2033_ = v___y_2047_;
v___y_2034_ = v___y_2044_;
v___y_2035_ = v___x_2051_;
goto v___jp_2025_;
}
}
v___jp_2054_:
{
if (lean_obj_tag(v___y_2055_) == 0)
{
lean_object* v___x_2065_; lean_object* v___x_2066_; 
v___x_2065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___closed__0));
v___x_2066_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_2066_, 0, v___x_2065_);
lean_ctor_set_uint8(v___x_2066_, sizeof(void*)*1, v___x_2053_);
v___y_2039_ = v___y_2062_;
v___y_2040_ = v___y_2061_;
v___y_2041_ = v___y_2060_;
v___y_2042_ = v___y_2057_;
v___y_2043_ = v_withArg_2056_;
v___y_2044_ = v___y_2058_;
v___y_2045_ = v___y_2059_;
v___y_2046_ = v___y_2063_;
v___y_2047_ = v___y_2064_;
v___y_2048_ = v___x_2066_;
goto v___jp_2038_;
}
else
{
lean_object* v_val_2067_; lean_object* v___x_2068_; 
v_val_2067_ = lean_ctor_get(v___y_2055_, 0);
lean_inc(v_val_2067_);
lean_dec_ref_known(v___y_2055_, 1);
v___x_2068_ = l_Lean_Elab_Tactic_expandLocation(v_val_2067_);
lean_dec(v_val_2067_);
v___y_2039_ = v___y_2062_;
v___y_2040_ = v___y_2061_;
v___y_2041_ = v___y_2060_;
v___y_2042_ = v___y_2057_;
v___y_2043_ = v_withArg_2056_;
v___y_2044_ = v___y_2058_;
v___y_2045_ = v___y_2059_;
v___y_2046_ = v___y_2063_;
v___y_2047_ = v___y_2064_;
v___y_2048_ = v___x_2068_;
goto v___jp_2038_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1___boxed(lean_object* v_x_2101_, lean_object* v_a_2102_, lean_object* v_a_2103_, lean_object* v_a_2104_, lean_object* v_a_2105_, lean_object* v_a_2106_, lean_object* v_a_2107_, lean_object* v_a_2108_, lean_object* v_a_2109_, lean_object* v_a_2110_){
_start:
{
lean_object* v_res_2111_; 
v_res_2111_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1(v_x_2101_, v_a_2102_, v_a_2103_, v_a_2104_, v_a_2105_, v_a_2106_, v_a_2107_, v_a_2108_, v_a_2109_);
lean_dec(v_a_2109_);
lean_dec_ref(v_a_2108_);
lean_dec(v_a_2107_);
lean_dec_ref(v_a_2106_);
lean_dec(v_a_2105_);
lean_dec_ref(v_a_2104_);
lean_dec(v_a_2103_);
lean_dec_ref(v_a_2102_);
return v_res_2111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2(lean_object* v_as_2112_, lean_object* v_as_x27_2113_, lean_object* v_b_2114_, lean_object* v_a_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_){
_start:
{
lean_object* v___x_2125_; 
v___x_2125_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___redArg(v_as_x27_2113_, v_b_2114_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
return v___x_2125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2___boxed(lean_object* v_as_2126_, lean_object* v_as_x27_2127_, lean_object* v_b_2128_, lean_object* v_a_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_){
_start:
{
lean_object* v_res_2139_; 
v_res_2139_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__2(v_as_2126_, v_as_x27_2127_, v_b_2128_, v_a_2129_, v___y_2130_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_);
lean_dec(v___y_2137_);
lean_dec_ref(v___y_2136_);
lean_dec(v___y_2135_);
lean_dec_ref(v___y_2134_);
lean_dec(v___y_2133_);
lean_dec_ref(v___y_2132_);
lean_dec(v___y_2131_);
lean_dec_ref(v___y_2130_);
lean_dec(v_as_x27_2127_);
lean_dec(v_as_2126_);
return v_res_2139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1(lean_object* v_ref_2140_, lean_object* v_msgData_2141_, uint8_t v_severity_2142_, uint8_t v_isSilent_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_){
_start:
{
lean_object* v___x_2153_; 
v___x_2153_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___redArg(v_ref_2140_, v_msgData_2141_, v_severity_2142_, v_isSilent_2143_, v___y_2148_, v___y_2149_, v___y_2150_, v___y_2151_);
return v___x_2153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1___boxed(lean_object* v_ref_2154_, lean_object* v_msgData_2155_, lean_object* v_severity_2156_, lean_object* v_isSilent_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_){
_start:
{
uint8_t v_severity_boxed_2167_; uint8_t v_isSilent_boxed_2168_; lean_object* v_res_2169_; 
v_severity_boxed_2167_ = lean_unbox(v_severity_2156_);
v_isSilent_boxed_2168_ = lean_unbox(v_isSilent_2157_);
v_res_2169_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SplitIfs______elabRules__Mathlib__Tactic__splitIfs__1_spec__1_spec__1(v_ref_2154_, v_msgData_2155_, v_severity_boxed_2167_, v_isSilent_boxed_2168_, v___y_2158_, v___y_2159_, v___y_2160_, v___y_2161_, v___y_2162_, v___y_2163_, v___y_2164_, v___y_2165_);
lean_dec(v___y_2165_);
lean_dec_ref(v___y_2164_);
lean_dec(v___y_2163_);
lean_dec_ref(v___y_2162_);
lean_dec(v___y_2161_);
lean_dec_ref(v___y_2160_);
lean_dec(v___y_2159_);
lean_dec_ref(v___y_2158_);
lean_dec(v_ref_2154_);
return v_res_2169_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_SplitIf(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_SplitIf(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_splitIfs = _init_lp_mathlib_Mathlib_Tactic_splitIfs();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_splitIfs);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Location(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_SplitIf(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Location(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_SplitIf(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
}
#ifdef __cplusplus
}
#endif
