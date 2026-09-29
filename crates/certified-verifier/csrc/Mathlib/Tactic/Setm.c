// Lean compiler output
// Module: Mathlib.Tactic.Setm
// Imports: public import Init public meta import Init public meta import Mathlib.Lean.Elab.Tactic.Basic public import Mathlib.Init
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
lean_object* l_Lean_FVarId_getValue_x3f___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_kabstract(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_changeLocalDecl(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replaceTargetDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_expandLocation(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Elab_Term_registerMVarErrorCustomInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_define(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_addPPExplicitToExposeDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofLazyM(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Name_quickLt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_logUnassignedAndAbort(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getMVarsNoDelayed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_filterOldMVars___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Meta_getLocalDeclFromUserName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "` could not be assigned"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Pattern"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "\nis not definitionally equal to the target"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "setm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 164, 206, 67, 225, 44, 141, 128)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SetM"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "setM"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__2_value),LEAN_SCALAR_PTR_LITERAL(229, 232, 102, 62, 88, 43, 150, 84)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__3_value),LEAN_SCALAR_PTR_LITERAL(7, 7, 199, 172, 130, 192, 145, 227)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "setm "};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__13_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM_setM___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM_setM;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Rewriting failed"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 91, .m_capacity = 91, .m_length = 90, .m_data = "No holes (`\?n`, `\?_`) were present in the `setm` pattern. This means `setm` has no effect."};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__0));
v___x_3_ = l_Lean_stringToMessageData(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__2));
v___x_6_ = l_Lean_stringToMessageData(v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(lean_object* v_stx_7_, lean_object* v_name_8_, lean_object* v_a_9_, lean_object* v_a_10_, lean_object* v_a_11_, lean_object* v_a_12_, lean_object* v_a_13_, lean_object* v_a_14_){
_start:
{
lean_object* v___x_16_; uint8_t v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_16_ = lean_box(0);
v___x_17_ = 0;
v___x_18_ = lean_box(0);
v___x_19_ = l_Lean_Meta_mkFreshExprMVar(v___x_16_, v___x_17_, v___x_18_, v_a_11_, v_a_12_, v_a_13_, v_a_14_);
if (lean_obj_tag(v___x_19_) == 0)
{
lean_object* v_a_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v_a_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc(v_a_20_);
lean_dec_ref_known(v___x_19_, 1);
v___x_21_ = l_Lean_Expr_mvarId_x21(v_a_20_);
v___x_22_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__1);
lean_inc(v_stx_7_);
v___x_23_ = l_Lean_MessageData_ofSyntax(v_stx_7_);
v___x_24_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_24_, 0, v___x_22_);
lean_ctor_set(v___x_24_, 1, v___x_23_);
v___x_25_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___closed__3);
v___x_26_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_26_, 0, v___x_24_);
lean_ctor_set(v___x_26_, 1, v___x_25_);
lean_inc(v___x_21_);
v___x_27_ = l_Lean_Elab_Term_registerMVarErrorCustomInfo___redArg(v___x_21_, v_stx_7_, v___x_26_, v_a_10_);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v___x_28_; 
lean_dec_ref_known(v___x_27_, 1);
lean_inc(v___x_21_);
v___x_28_ = l_Lean_MVarId_getType(v___x_21_, v_a_11_, v_a_12_, v_a_13_, v_a_14_);
if (lean_obj_tag(v___x_28_) == 0)
{
lean_object* v_a_29_; lean_object* v_goal_30_; lean_object* v_holes_31_; lean_object* v_newMVars_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_78_; 
v_a_29_ = lean_ctor_get(v___x_28_, 0);
lean_inc(v_a_29_);
lean_dec_ref_known(v___x_28_, 1);
v_goal_30_ = lean_ctor_get(v_a_9_, 0);
v_holes_31_ = lean_ctor_get(v_a_9_, 1);
v_newMVars_32_ = lean_ctor_get(v_a_9_, 2);
v_isSharedCheck_78_ = !lean_is_exclusive(v_a_9_);
if (v_isSharedCheck_78_ == 0)
{
v___x_34_ = v_a_9_;
v_isShared_35_ = v_isSharedCheck_78_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_newMVars_32_);
lean_inc(v_holes_31_);
lean_inc(v_goal_30_);
lean_dec(v_a_9_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_78_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___x_36_; 
lean_inc(v_name_8_);
v___x_36_ = l_Lean_MVarId_define(v_goal_30_, v_name_8_, v_a_29_, v_a_20_, v_a_11_, v_a_12_, v_a_13_, v_a_14_);
if (lean_obj_tag(v___x_36_) == 0)
{
lean_object* v_a_37_; uint8_t v___x_38_; lean_object* v___x_39_; 
v_a_37_ = lean_ctor_get(v___x_36_, 0);
lean_inc(v_a_37_);
lean_dec_ref_known(v___x_36_, 1);
v___x_38_ = 1;
v___x_39_ = l_Lean_Meta_intro1Core(v_a_37_, v___x_38_, v_a_11_, v_a_12_, v_a_13_, v_a_14_);
if (lean_obj_tag(v___x_39_) == 0)
{
lean_object* v_a_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_61_; 
v_a_40_ = lean_ctor_get(v___x_39_, 0);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_61_ == 0)
{
v___x_42_ = v___x_39_;
v_isShared_43_ = v_isSharedCheck_61_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_a_40_);
lean_dec(v___x_39_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_61_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v_fst_44_; lean_object* v_snd_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_60_; 
v_fst_44_ = lean_ctor_get(v_a_40_, 0);
v_snd_45_ = lean_ctor_get(v_a_40_, 1);
v_isSharedCheck_60_ = !lean_is_exclusive(v_a_40_);
if (v_isSharedCheck_60_ == 0)
{
v___x_47_ = v_a_40_;
v_isShared_48_ = v_isSharedCheck_60_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_snd_45_);
lean_inc(v_fst_44_);
lean_dec(v_a_40_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_60_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_52_; 
lean_inc(v_fst_44_);
v___x_49_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_name_8_, v_fst_44_, v_holes_31_);
v___x_50_ = lean_array_push(v_newMVars_32_, v___x_21_);
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 2, v___x_50_);
lean_ctor_set(v___x_34_, 1, v___x_49_);
lean_ctor_set(v___x_34_, 0, v_snd_45_);
v___x_52_ = v___x_34_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v_snd_45_);
lean_ctor_set(v_reuseFailAlloc_59_, 1, v___x_49_);
lean_ctor_set(v_reuseFailAlloc_59_, 2, v___x_50_);
v___x_52_ = v_reuseFailAlloc_59_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
lean_object* v___x_54_; 
if (v_isShared_48_ == 0)
{
lean_ctor_set(v___x_47_, 1, v___x_52_);
v___x_54_ = v___x_47_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_fst_44_);
lean_ctor_set(v_reuseFailAlloc_58_, 1, v___x_52_);
v___x_54_ = v_reuseFailAlloc_58_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
lean_object* v___x_56_; 
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 0, v___x_54_);
v___x_56_ = v___x_42_;
goto v_reusejp_55_;
}
else
{
lean_object* v_reuseFailAlloc_57_; 
v_reuseFailAlloc_57_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_57_, 0, v___x_54_);
v___x_56_ = v_reuseFailAlloc_57_;
goto v_reusejp_55_;
}
v_reusejp_55_:
{
return v___x_56_;
}
}
}
}
}
}
else
{
lean_object* v_a_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_69_; 
lean_del_object(v___x_34_);
lean_dec_ref(v_newMVars_32_);
lean_dec(v_holes_31_);
lean_dec(v___x_21_);
lean_dec(v_name_8_);
v_a_62_ = lean_ctor_get(v___x_39_, 0);
v_isSharedCheck_69_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_69_ == 0)
{
v___x_64_ = v___x_39_;
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_a_62_);
lean_dec(v___x_39_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_67_; 
if (v_isShared_65_ == 0)
{
v___x_67_ = v___x_64_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v_a_62_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
else
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_77_; 
lean_del_object(v___x_34_);
lean_dec_ref(v_newMVars_32_);
lean_dec(v_holes_31_);
lean_dec(v___x_21_);
lean_dec(v_name_8_);
v_a_70_ = lean_ctor_get(v___x_36_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_36_);
if (v_isSharedCheck_77_ == 0)
{
v___x_72_ = v___x_36_;
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v___x_36_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_75_; 
if (v_isShared_73_ == 0)
{
v___x_75_ = v___x_72_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v_a_70_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
}
else
{
lean_object* v_a_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_86_; 
lean_dec(v___x_21_);
lean_dec(v_a_20_);
lean_dec_ref(v_a_9_);
lean_dec(v_name_8_);
v_a_79_ = lean_ctor_get(v___x_28_, 0);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_28_);
if (v_isSharedCheck_86_ == 0)
{
v___x_81_ = v___x_28_;
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_a_79_);
lean_dec(v___x_28_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v___x_84_; 
if (v_isShared_82_ == 0)
{
v___x_84_ = v___x_81_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_a_79_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
else
{
lean_object* v_a_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_94_; 
lean_dec(v___x_21_);
lean_dec(v_a_20_);
lean_dec_ref(v_a_9_);
lean_dec(v_name_8_);
v_a_87_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_94_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_94_ == 0)
{
v___x_89_ = v___x_27_;
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_a_87_);
lean_dec(v___x_27_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_92_; 
if (v_isShared_90_ == 0)
{
v___x_92_ = v___x_89_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v_a_87_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
else
{
lean_object* v_a_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_102_; 
lean_dec_ref(v_a_9_);
lean_dec(v_name_8_);
lean_dec(v_stx_7_);
v_a_95_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_102_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_102_ == 0)
{
v___x_97_ = v___x_19_;
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_a_95_);
lean_dec(v___x_19_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_102_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_100_; 
if (v_isShared_98_ == 0)
{
v___x_100_ = v___x_97_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_a_95_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg___boxed(lean_object* v_stx_103_, lean_object* v_name_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(v_stx_103_, v_name_104_, v_a_105_, v_a_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_);
lean_dec(v_a_110_);
lean_dec_ref(v_a_109_);
lean_dec(v_a_108_);
lean_dec_ref(v_a_107_);
lean_dec(v_a_106_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl(lean_object* v_stx_113_, lean_object* v_name_114_, lean_object* v_a_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(v_stx_113_, v_name_114_, v_a_115_, v_a_117_, v_a_118_, v_a_119_, v_a_120_, v_a_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___boxed(lean_object* v_stx_124_, lean_object* v_name_125_, lean_object* v_a_126_, lean_object* v_a_127_, lean_object* v_a_128_, lean_object* v_a_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl(v_stx_124_, v_name_125_, v_a_126_, v_a_127_, v_a_128_, v_a_129_, v_a_130_, v_a_131_, v_a_132_);
lean_dec(v_a_132_);
lean_dec_ref(v_a_131_);
lean_dec(v_a_130_);
lean_dec_ref(v_a_129_);
lean_dec(v_a_128_);
lean_dec_ref(v_a_127_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0(lean_object* v_x_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_){
_start:
{
lean_object* v___x_144_; 
lean_inc(v___y_138_);
lean_inc_ref(v___y_137_);
v___x_144_ = lean_apply_8(v_x_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, lean_box(0));
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0___boxed(lean_object* v_x_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0(v_x_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg(lean_object* v_mvarId_155_, lean_object* v_x_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v___f_165_; lean_object* v___x_166_; 
lean_inc(v___y_159_);
lean_inc_ref(v___y_158_);
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_165_, 0, v_x_156_);
lean_closure_set(v___f_165_, 1, v___y_157_);
lean_closure_set(v___f_165_, 2, v___y_158_);
lean_closure_set(v___f_165_, 3, v___y_159_);
v___x_166_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_155_, v___f_165_, v___y_160_, v___y_161_, v___y_162_, v___y_163_);
if (lean_obj_tag(v___x_166_) == 0)
{
lean_object* v_a_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_174_; 
v_a_167_ = lean_ctor_get(v___x_166_, 0);
v_isSharedCheck_174_ = !lean_is_exclusive(v___x_166_);
if (v_isSharedCheck_174_ == 0)
{
v___x_169_ = v___x_166_;
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_a_167_);
lean_dec(v___x_166_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_174_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
lean_object* v___x_172_; 
if (v_isShared_170_ == 0)
{
v___x_172_ = v___x_169_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_173_; 
v_reuseFailAlloc_173_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_173_, 0, v_a_167_);
v___x_172_ = v_reuseFailAlloc_173_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
return v___x_172_;
}
}
}
else
{
lean_object* v_a_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_182_; 
v_a_175_ = lean_ctor_get(v___x_166_, 0);
v_isSharedCheck_182_ = !lean_is_exclusive(v___x_166_);
if (v_isSharedCheck_182_ == 0)
{
v___x_177_ = v___x_166_;
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_a_175_);
lean_dec(v___x_166_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_182_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_180_; 
if (v_isShared_178_ == 0)
{
v___x_180_ = v___x_177_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_181_; 
v_reuseFailAlloc_181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_181_, 0, v_a_175_);
v___x_180_ = v_reuseFailAlloc_181_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
return v___x_180_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg___boxed(lean_object* v_mvarId_183_, lean_object* v_x_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg(v_mvarId_183_, v_x_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec(v___y_189_);
lean_dec_ref(v___y_188_);
lean_dec(v___y_187_);
lean_dec_ref(v___y_186_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0(lean_object* v_00_u03b1_194_, lean_object* v_mvarId_195_, lean_object* v_x_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg(v_mvarId_195_, v_x_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___boxed(lean_object* v_00_u03b1_206_, lean_object* v_mvarId_207_, lean_object* v_x_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0(v_00_u03b1_206_, v_mvarId_207_, v_x_208_, v___y_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg(lean_object* v_t_218_, lean_object* v_k_219_){
_start:
{
if (lean_obj_tag(v_t_218_) == 0)
{
lean_object* v_k_220_; lean_object* v_v_221_; lean_object* v_l_222_; lean_object* v_r_223_; uint8_t v___x_224_; 
v_k_220_ = lean_ctor_get(v_t_218_, 1);
v_v_221_ = lean_ctor_get(v_t_218_, 2);
v_l_222_ = lean_ctor_get(v_t_218_, 3);
v_r_223_ = lean_ctor_get(v_t_218_, 4);
v___x_224_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_219_, v_k_220_);
switch(v___x_224_)
{
case 0:
{
v_t_218_ = v_l_222_;
goto _start;
}
case 1:
{
lean_object* v___x_226_; 
lean_inc(v_v_221_);
v___x_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_226_, 0, v_v_221_);
return v___x_226_;
}
default: 
{
v_t_218_ = v_r_223_;
goto _start;
}
}
}
else
{
lean_object* v___x_228_; 
v___x_228_ = lean_box(0);
return v___x_228_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg___boxed(lean_object* v_t_229_, lean_object* v_k_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg(v_t_229_, v_k_230_);
lean_dec(v_k_230_);
lean_dec(v_t_229_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0(lean_object* v___x_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = l_Lean_Elab_Term_exprToSyntax(v___x_232_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
if (lean_obj_tag(v___x_241_) == 0)
{
lean_object* v_a_242_; lean_object* v___x_244_; uint8_t v_isShared_245_; uint8_t v_isSharedCheck_250_; 
v_a_242_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_250_ == 0)
{
v___x_244_ = v___x_241_;
v_isShared_245_ = v_isSharedCheck_250_;
goto v_resetjp_243_;
}
else
{
lean_inc(v_a_242_);
lean_dec(v___x_241_);
v___x_244_ = lean_box(0);
v_isShared_245_ = v_isSharedCheck_250_;
goto v_resetjp_243_;
}
v_resetjp_243_:
{
lean_object* v___x_246_; lean_object* v___x_248_; 
v___x_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_246_, 0, v_a_242_);
lean_ctor_set(v___x_246_, 1, v___y_233_);
if (v_isShared_245_ == 0)
{
lean_ctor_set(v___x_244_, 0, v___x_246_);
v___x_248_ = v___x_244_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v___x_246_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
else
{
lean_object* v_a_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
lean_dec_ref(v___y_233_);
v_a_251_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v___x_241_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_a_251_);
lean_dec(v___x_241_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0___boxed(lean_object* v___x_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0(v___x_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1(lean_object* v_stx_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
lean_object* v_fvar_294_; lean_object* v___y_295_; lean_object* v_goal_296_; lean_object* v___y_297_; lean_object* v___y_298_; lean_object* v___y_299_; lean_object* v___y_300_; lean_object* v___y_301_; lean_object* v___y_302_; lean_object* v_fvar_351_; lean_object* v___y_352_; lean_object* v___y_353_; lean_object* v___y_354_; lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___y_357_; lean_object* v___y_358_; lean_object* v___x_360_; uint8_t v___x_361_; 
v___x_360_ = ((lean_object*)(lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__4));
lean_inc(v_stx_284_);
v___x_361_ = l_Lean_Syntax_isOfKind(v_stx_284_, v___x_360_);
if (v___x_361_ == 0)
{
lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
lean_dec(v_stx_284_);
v___x_362_ = lean_box(0);
v___x_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v___y_285_);
v___x_364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
return v___x_364_;
}
else
{
lean_object* v___x_365_; lean_object* v_n_366_; lean_object* v___x_367_; uint8_t v___x_368_; 
v___x_365_ = lean_unsigned_to_nat(1u);
v_n_366_ = l_Lean_Syntax_getArg(v_stx_284_, v___x_365_);
v___x_367_ = ((lean_object*)(lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__6));
lean_inc(v_n_366_);
v___x_368_ = l_Lean_Syntax_isOfKind(v_n_366_, v___x_367_);
if (v___x_368_ == 0)
{
lean_object* v___x_369_; lean_object* v___x_370_; 
lean_dec(v_n_366_);
v___x_369_ = ((lean_object*)(lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___closed__8));
v___x_370_ = l_Lean_Core_mkFreshUserName(v___x_369_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v_a_371_; lean_object* v___x_372_; 
v_a_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc(v_a_371_);
lean_dec_ref_known(v___x_370_, 1);
lean_inc(v_stx_284_);
v___x_372_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(v_stx_284_, v_a_371_, v___y_285_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_372_) == 0)
{
lean_object* v_a_373_; lean_object* v_fst_374_; lean_object* v_snd_375_; 
v_a_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc(v_a_373_);
lean_dec_ref_known(v___x_372_, 1);
v_fst_374_ = lean_ctor_get(v_a_373_, 0);
lean_inc(v_fst_374_);
v_snd_375_ = lean_ctor_get(v_a_373_, 1);
lean_inc(v_snd_375_);
lean_dec(v_a_373_);
v_fvar_351_ = v_fst_374_;
v___y_352_ = v_snd_375_;
v___y_353_ = v___y_286_;
v___y_354_ = v___y_287_;
v___y_355_ = v___y_288_;
v___y_356_ = v___y_289_;
v___y_357_ = v___y_290_;
v___y_358_ = v___y_291_;
goto v___jp_350_;
}
else
{
lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
lean_dec(v_stx_284_);
v_a_376_ = lean_ctor_get(v___x_372_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_372_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v___x_372_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_372_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_a_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
else
{
lean_object* v_a_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_391_; 
lean_dec_ref(v___y_285_);
lean_dec(v_stx_284_);
v_a_384_ = lean_ctor_get(v___x_370_, 0);
v_isSharedCheck_391_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_391_ == 0)
{
v___x_386_ = v___x_370_;
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_a_384_);
lean_dec(v___x_370_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_391_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_389_; 
if (v_isShared_387_ == 0)
{
v___x_389_ = v___x_386_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v_a_384_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
else
{
lean_object* v_goal_392_; lean_object* v_holes_393_; lean_object* v_name_394_; lean_object* v___x_395_; 
v_goal_392_ = lean_ctor_get(v___y_285_, 0);
v_holes_393_ = lean_ctor_get(v___y_285_, 1);
v_name_394_ = l_Lean_TSyntax_getId(v_n_366_);
lean_dec(v_n_366_);
v___x_395_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg(v_holes_393_, v_name_394_);
if (lean_obj_tag(v___x_395_) == 0)
{
lean_object* v___x_396_; 
lean_inc(v_stx_284_);
v___x_396_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_createLDecl___redArg(v_stx_284_, v_name_394_, v___y_285_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v_fst_398_; lean_object* v_snd_399_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc(v_a_397_);
lean_dec_ref_known(v___x_396_, 1);
v_fst_398_ = lean_ctor_get(v_a_397_, 0);
lean_inc(v_fst_398_);
v_snd_399_ = lean_ctor_get(v_a_397_, 1);
lean_inc(v_snd_399_);
lean_dec(v_a_397_);
v_fvar_351_ = v_fst_398_;
v___y_352_ = v_snd_399_;
v___y_353_ = v___y_286_;
v___y_354_ = v___y_287_;
v___y_355_ = v___y_288_;
v___y_356_ = v___y_289_;
v___y_357_ = v___y_290_;
v___y_358_ = v___y_291_;
goto v___jp_350_;
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec(v_stx_284_);
v_a_400_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_396_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_396_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_405_; 
if (v_isShared_403_ == 0)
{
v___x_405_ = v___x_402_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v_a_400_);
v___x_405_ = v_reuseFailAlloc_406_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
return v___x_405_;
}
}
}
}
else
{
lean_object* v_val_408_; 
lean_inc(v_goal_392_);
lean_dec(v_name_394_);
v_val_408_ = lean_ctor_get(v___x_395_, 0);
lean_inc(v_val_408_);
lean_dec_ref_known(v___x_395_, 1);
v_fvar_294_ = v_val_408_;
v___y_295_ = v___y_285_;
v_goal_296_ = v_goal_392_;
v___y_297_ = v___y_286_;
v___y_298_ = v___y_287_;
v___y_299_ = v___y_288_;
v___y_300_ = v___y_289_;
v___y_301_ = v___y_290_;
v___y_302_ = v___y_291_;
goto v___jp_293_;
}
}
}
v___jp_293_:
{
lean_object* v_fileName_303_; lean_object* v_fileMap_304_; lean_object* v_options_305_; lean_object* v_currRecDepth_306_; lean_object* v_maxRecDepth_307_; lean_object* v_ref_308_; lean_object* v_currNamespace_309_; lean_object* v_openDecls_310_; lean_object* v_initHeartbeats_311_; lean_object* v_maxHeartbeats_312_; lean_object* v_quotContext_313_; lean_object* v_currMacroScope_314_; uint8_t v_diag_315_; lean_object* v_cancelTk_x3f_316_; uint8_t v_suppressElabErrors_317_; lean_object* v_inheritedTraceOptions_318_; lean_object* v___x_319_; lean_object* v___f_320_; lean_object* v_ref_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v_fileName_303_ = lean_ctor_get(v___y_301_, 0);
v_fileMap_304_ = lean_ctor_get(v___y_301_, 1);
v_options_305_ = lean_ctor_get(v___y_301_, 2);
v_currRecDepth_306_ = lean_ctor_get(v___y_301_, 3);
v_maxRecDepth_307_ = lean_ctor_get(v___y_301_, 4);
v_ref_308_ = lean_ctor_get(v___y_301_, 5);
v_currNamespace_309_ = lean_ctor_get(v___y_301_, 6);
v_openDecls_310_ = lean_ctor_get(v___y_301_, 7);
v_initHeartbeats_311_ = lean_ctor_get(v___y_301_, 8);
v_maxHeartbeats_312_ = lean_ctor_get(v___y_301_, 9);
v_quotContext_313_ = lean_ctor_get(v___y_301_, 10);
v_currMacroScope_314_ = lean_ctor_get(v___y_301_, 11);
v_diag_315_ = lean_ctor_get_uint8(v___y_301_, sizeof(void*)*14);
v_cancelTk_x3f_316_ = lean_ctor_get(v___y_301_, 12);
v_suppressElabErrors_317_ = lean_ctor_get_uint8(v___y_301_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_318_ = lean_ctor_get(v___y_301_, 13);
v___x_319_ = l_Lean_Expr_fvar___override(v_fvar_294_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__0___boxed), 9, 1);
lean_closure_set(v___f_320_, 0, v___x_319_);
v_ref_321_ = l_Lean_replaceRef(v_stx_284_, v_ref_308_);
lean_dec(v_stx_284_);
lean_inc_ref(v_inheritedTraceOptions_318_);
lean_inc(v_cancelTk_x3f_316_);
lean_inc(v_currMacroScope_314_);
lean_inc(v_quotContext_313_);
lean_inc(v_maxHeartbeats_312_);
lean_inc(v_initHeartbeats_311_);
lean_inc(v_openDecls_310_);
lean_inc(v_currNamespace_309_);
lean_inc(v_maxRecDepth_307_);
lean_inc(v_currRecDepth_306_);
lean_inc_ref(v_options_305_);
lean_inc_ref(v_fileMap_304_);
lean_inc_ref(v_fileName_303_);
v___x_322_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_322_, 0, v_fileName_303_);
lean_ctor_set(v___x_322_, 1, v_fileMap_304_);
lean_ctor_set(v___x_322_, 2, v_options_305_);
lean_ctor_set(v___x_322_, 3, v_currRecDepth_306_);
lean_ctor_set(v___x_322_, 4, v_maxRecDepth_307_);
lean_ctor_set(v___x_322_, 5, v_ref_321_);
lean_ctor_set(v___x_322_, 6, v_currNamespace_309_);
lean_ctor_set(v___x_322_, 7, v_openDecls_310_);
lean_ctor_set(v___x_322_, 8, v_initHeartbeats_311_);
lean_ctor_set(v___x_322_, 9, v_maxHeartbeats_312_);
lean_ctor_set(v___x_322_, 10, v_quotContext_313_);
lean_ctor_set(v___x_322_, 11, v_currMacroScope_314_);
lean_ctor_set(v___x_322_, 12, v_cancelTk_x3f_316_);
lean_ctor_set(v___x_322_, 13, v_inheritedTraceOptions_318_);
lean_ctor_set_uint8(v___x_322_, sizeof(void*)*14, v_diag_315_);
lean_ctor_set_uint8(v___x_322_, sizeof(void*)*14 + 1, v_suppressElabErrors_317_);
v___x_323_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__0___redArg(v_goal_296_, v___f_320_, v___y_295_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___x_322_, v___y_302_);
lean_dec_ref_known(v___x_322_, 14);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_341_; 
v_a_324_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_341_ == 0)
{
v___x_326_ = v___x_323_;
v_isShared_327_ = v_isSharedCheck_341_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_341_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v_fst_328_; lean_object* v_snd_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_340_; 
v_fst_328_ = lean_ctor_get(v_a_324_, 0);
v_snd_329_ = lean_ctor_get(v_a_324_, 1);
v_isSharedCheck_340_ = !lean_is_exclusive(v_a_324_);
if (v_isSharedCheck_340_ == 0)
{
v___x_331_ = v_a_324_;
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_snd_329_);
lean_inc(v_fst_328_);
lean_dec(v_a_324_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_333_, 0, v_fst_328_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 0, v___x_333_);
v___x_335_ = v___x_331_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_333_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v_snd_329_);
v___x_335_ = v_reuseFailAlloc_339_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_337_; 
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_335_);
v___x_337_ = v___x_326_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_335_);
v___x_337_ = v_reuseFailAlloc_338_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
return v___x_337_;
}
}
}
}
}
else
{
lean_object* v_a_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_349_; 
v_a_342_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_349_ == 0)
{
v___x_344_ = v___x_323_;
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_a_342_);
lean_dec(v___x_323_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_349_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v___x_347_; 
if (v_isShared_345_ == 0)
{
v___x_347_ = v___x_344_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v_a_342_);
v___x_347_ = v_reuseFailAlloc_348_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
return v___x_347_;
}
}
}
}
v___jp_350_:
{
lean_object* v_goal_359_; 
v_goal_359_ = lean_ctor_get(v___y_352_, 0);
lean_inc(v_goal_359_);
v_fvar_294_ = v_fvar_351_;
v___y_295_ = v___y_352_;
v_goal_296_ = v_goal_359_;
v___y_297_ = v___y_353_;
v___y_298_ = v___y_354_;
v___y_299_ = v___y_355_;
v___y_300_ = v___y_356_;
v___y_301_ = v___y_357_;
v___y_302_ = v___y_358_;
goto v___jp_293_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1___boxed(lean_object* v_stx_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1(v_stx_409_, v___y_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_, v___y_416_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(lean_object* v_x_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
if (lean_obj_tag(v_x_419_) == 1)
{
lean_object* v_info_428_; lean_object* v_kind_429_; lean_object* v_args_430_; lean_object* v___x_431_; 
v_info_428_ = lean_ctor_get(v_x_419_, 0);
lean_inc(v_info_428_);
v_kind_429_ = lean_ctor_get(v_x_419_, 1);
lean_inc(v_kind_429_);
v_args_430_ = lean_ctor_get(v_x_419_, 2);
lean_inc_ref(v_args_430_);
v___x_431_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1(v_x_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
if (lean_obj_tag(v___x_431_) == 0)
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_480_; 
v_a_432_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_480_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_480_ == 0)
{
v___x_434_ = v___x_431_;
v_isShared_435_ = v_isSharedCheck_480_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_431_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_480_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v_fst_436_; 
v_fst_436_ = lean_ctor_get(v_a_432_, 0);
if (lean_obj_tag(v_fst_436_) == 0)
{
lean_object* v_snd_437_; size_t v_sz_438_; size_t v___x_439_; lean_object* v___x_440_; 
lean_del_object(v___x_434_);
v_snd_437_ = lean_ctor_get(v_a_432_, 1);
lean_inc(v_snd_437_);
lean_dec(v_a_432_);
v_sz_438_ = lean_array_size(v_args_430_);
v___x_439_ = ((size_t)0ULL);
v___x_440_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2(v_sz_438_, v___x_439_, v_args_430_, v_snd_437_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
if (lean_obj_tag(v___x_440_) == 0)
{
lean_object* v_a_441_; lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_458_; 
v_a_441_ = lean_ctor_get(v___x_440_, 0);
v_isSharedCheck_458_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_458_ == 0)
{
v___x_443_ = v___x_440_;
v_isShared_444_ = v_isSharedCheck_458_;
goto v_resetjp_442_;
}
else
{
lean_inc(v_a_441_);
lean_dec(v___x_440_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_458_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v_fst_445_; lean_object* v_snd_446_; lean_object* v___x_448_; uint8_t v_isShared_449_; uint8_t v_isSharedCheck_457_; 
v_fst_445_ = lean_ctor_get(v_a_441_, 0);
v_snd_446_ = lean_ctor_get(v_a_441_, 1);
v_isSharedCheck_457_ = !lean_is_exclusive(v_a_441_);
if (v_isSharedCheck_457_ == 0)
{
v___x_448_ = v_a_441_;
v_isShared_449_ = v_isSharedCheck_457_;
goto v_resetjp_447_;
}
else
{
lean_inc(v_snd_446_);
lean_inc(v_fst_445_);
lean_dec(v_a_441_);
v___x_448_ = lean_box(0);
v_isShared_449_ = v_isSharedCheck_457_;
goto v_resetjp_447_;
}
v_resetjp_447_:
{
lean_object* v___x_450_; lean_object* v___x_452_; 
v___x_450_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_450_, 0, v_info_428_);
lean_ctor_set(v___x_450_, 1, v_kind_429_);
lean_ctor_set(v___x_450_, 2, v_fst_445_);
if (v_isShared_449_ == 0)
{
lean_ctor_set(v___x_448_, 0, v___x_450_);
v___x_452_ = v___x_448_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_456_; 
v_reuseFailAlloc_456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_456_, 0, v___x_450_);
lean_ctor_set(v_reuseFailAlloc_456_, 1, v_snd_446_);
v___x_452_ = v_reuseFailAlloc_456_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
lean_object* v___x_454_; 
if (v_isShared_444_ == 0)
{
lean_ctor_set(v___x_443_, 0, v___x_452_);
v___x_454_ = v___x_443_;
goto v_reusejp_453_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v___x_452_);
v___x_454_ = v_reuseFailAlloc_455_;
goto v_reusejp_453_;
}
v_reusejp_453_:
{
return v___x_454_;
}
}
}
}
}
else
{
lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_466_; 
lean_dec(v_kind_429_);
lean_dec(v_info_428_);
v_a_459_ = lean_ctor_get(v___x_440_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_440_);
if (v_isSharedCheck_466_ == 0)
{
v___x_461_ = v___x_440_;
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_440_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_464_; 
if (v_isShared_462_ == 0)
{
v___x_464_ = v___x_461_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_459_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
}
else
{
lean_object* v_snd_467_; lean_object* v___x_469_; uint8_t v_isShared_470_; uint8_t v_isSharedCheck_478_; 
lean_inc_ref(v_fst_436_);
lean_dec_ref(v_args_430_);
lean_dec(v_kind_429_);
lean_dec(v_info_428_);
v_snd_467_ = lean_ctor_get(v_a_432_, 1);
v_isSharedCheck_478_ = !lean_is_exclusive(v_a_432_);
if (v_isSharedCheck_478_ == 0)
{
lean_object* v_unused_479_; 
v_unused_479_ = lean_ctor_get(v_a_432_, 0);
lean_dec(v_unused_479_);
v___x_469_ = v_a_432_;
v_isShared_470_ = v_isSharedCheck_478_;
goto v_resetjp_468_;
}
else
{
lean_inc(v_snd_467_);
lean_dec(v_a_432_);
v___x_469_ = lean_box(0);
v_isShared_470_ = v_isSharedCheck_478_;
goto v_resetjp_468_;
}
v_resetjp_468_:
{
lean_object* v_val_471_; lean_object* v___x_473_; 
v_val_471_ = lean_ctor_get(v_fst_436_, 0);
lean_inc(v_val_471_);
lean_dec_ref_known(v_fst_436_, 1);
if (v_isShared_470_ == 0)
{
lean_ctor_set(v___x_469_, 0, v_val_471_);
v___x_473_ = v___x_469_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_val_471_);
lean_ctor_set(v_reuseFailAlloc_477_, 1, v_snd_467_);
v___x_473_ = v_reuseFailAlloc_477_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
lean_object* v___x_475_; 
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_473_);
v___x_475_ = v___x_434_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_476_; 
v_reuseFailAlloc_476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_476_, 0, v___x_473_);
v___x_475_ = v_reuseFailAlloc_476_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
return v___x_475_;
}
}
}
}
}
}
else
{
lean_object* v_a_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_488_; 
lean_dec_ref(v_args_430_);
lean_dec(v_kind_429_);
lean_dec(v_info_428_);
v_a_481_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_488_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_488_ == 0)
{
v___x_483_ = v___x_431_;
v_isShared_484_ = v_isSharedCheck_488_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_a_481_);
lean_dec(v___x_431_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_488_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v___x_486_; 
if (v_isShared_484_ == 0)
{
v___x_486_ = v___x_483_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v_a_481_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
}
}
}
}
else
{
lean_object* v___x_489_; 
lean_inc(v_x_419_);
v___x_489_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___lam__1(v_x_419_, v___y_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
if (lean_obj_tag(v___x_489_) == 0)
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_520_; 
v_a_490_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_520_ == 0)
{
v___x_492_ = v___x_489_;
v_isShared_493_ = v_isSharedCheck_520_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v___x_489_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_520_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v_fst_494_; 
v_fst_494_ = lean_ctor_get(v_a_490_, 0);
if (lean_obj_tag(v_fst_494_) == 0)
{
lean_object* v_snd_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_505_; 
v_snd_495_ = lean_ctor_get(v_a_490_, 1);
v_isSharedCheck_505_ = !lean_is_exclusive(v_a_490_);
if (v_isSharedCheck_505_ == 0)
{
lean_object* v_unused_506_; 
v_unused_506_ = lean_ctor_get(v_a_490_, 0);
lean_dec(v_unused_506_);
v___x_497_ = v_a_490_;
v_isShared_498_ = v_isSharedCheck_505_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_snd_495_);
lean_dec(v_a_490_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_505_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_500_; 
if (v_isShared_498_ == 0)
{
lean_ctor_set(v___x_497_, 0, v_x_419_);
v___x_500_ = v___x_497_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v_x_419_);
lean_ctor_set(v_reuseFailAlloc_504_, 1, v_snd_495_);
v___x_500_ = v_reuseFailAlloc_504_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
lean_object* v___x_502_; 
if (v_isShared_493_ == 0)
{
lean_ctor_set(v___x_492_, 0, v___x_500_);
v___x_502_ = v___x_492_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v___x_500_);
v___x_502_ = v_reuseFailAlloc_503_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
return v___x_502_;
}
}
}
}
else
{
lean_object* v_snd_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_518_; 
lean_inc_ref(v_fst_494_);
lean_dec(v_x_419_);
v_snd_507_ = lean_ctor_get(v_a_490_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v_a_490_);
if (v_isSharedCheck_518_ == 0)
{
lean_object* v_unused_519_; 
v_unused_519_ = lean_ctor_get(v_a_490_, 0);
lean_dec(v_unused_519_);
v___x_509_ = v_a_490_;
v_isShared_510_ = v_isSharedCheck_518_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_snd_507_);
lean_dec(v_a_490_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_518_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v_val_511_; lean_object* v___x_513_; 
v_val_511_ = lean_ctor_get(v_fst_494_, 0);
lean_inc(v_val_511_);
lean_dec_ref_known(v_fst_494_, 1);
if (v_isShared_510_ == 0)
{
lean_ctor_set(v___x_509_, 0, v_val_511_);
v___x_513_ = v___x_509_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_val_511_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_snd_507_);
v___x_513_ = v_reuseFailAlloc_517_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_515_; 
if (v_isShared_493_ == 0)
{
lean_ctor_set(v___x_492_, 0, v___x_513_);
v___x_515_ = v___x_492_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v___x_513_);
v___x_515_ = v_reuseFailAlloc_516_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
return v___x_515_;
}
}
}
}
}
}
else
{
lean_object* v_a_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_528_; 
lean_dec(v_x_419_);
v_a_521_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_528_ == 0)
{
v___x_523_ = v___x_489_;
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_a_521_);
lean_dec(v___x_489_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_524_ == 0)
{
v___x_526_ = v___x_523_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_a_521_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2(size_t v_sz_529_, size_t v_i_530_, lean_object* v_bs_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_){
_start:
{
uint8_t v___x_540_; 
v___x_540_ = lean_usize_dec_lt(v_i_530_, v_sz_529_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_541_, 0, v_bs_531_);
lean_ctor_set(v___x_541_, 1, v___y_532_);
v___x_542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
return v___x_542_;
}
else
{
lean_object* v_v_543_; lean_object* v___x_544_; 
v_v_543_ = lean_array_uget_borrowed(v_bs_531_, v_i_530_);
lean_inc(v_v_543_);
v___x_544_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(v_v_543_, v___y_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_);
if (lean_obj_tag(v___x_544_) == 0)
{
lean_object* v_a_545_; lean_object* v_fst_546_; lean_object* v_snd_547_; lean_object* v___x_548_; lean_object* v_bs_x27_549_; size_t v___x_550_; size_t v___x_551_; lean_object* v___x_552_; 
v_a_545_ = lean_ctor_get(v___x_544_, 0);
lean_inc(v_a_545_);
lean_dec_ref_known(v___x_544_, 1);
v_fst_546_ = lean_ctor_get(v_a_545_, 0);
lean_inc(v_fst_546_);
v_snd_547_ = lean_ctor_get(v_a_545_, 1);
lean_inc(v_snd_547_);
lean_dec(v_a_545_);
v___x_548_ = lean_unsigned_to_nat(0u);
v_bs_x27_549_ = lean_array_uset(v_bs_531_, v_i_530_, v___x_548_);
v___x_550_ = ((size_t)1ULL);
v___x_551_ = lean_usize_add(v_i_530_, v___x_550_);
v___x_552_ = lean_array_uset(v_bs_x27_549_, v_i_530_, v_fst_546_);
v_i_530_ = v___x_551_;
v_bs_531_ = v___x_552_;
v___y_532_ = v_snd_547_;
goto _start;
}
else
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_561_; 
lean_dec_ref(v_bs_531_);
v_a_554_ = lean_ctor_get(v___x_544_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_561_ == 0)
{
v___x_556_ = v___x_544_;
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_544_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_559_; 
if (v_isShared_557_ == 0)
{
v___x_559_ = v___x_556_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v_a_554_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2___boxed(lean_object* v_sz_562_, lean_object* v_i_563_, lean_object* v_bs_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_){
_start:
{
size_t v_sz_boxed_573_; size_t v_i_boxed_574_; lean_object* v_res_575_; 
v_sz_boxed_573_ = lean_unbox_usize(v_sz_562_);
lean_dec(v_sz_562_);
v_i_boxed_574_ = lean_unbox_usize(v_i_563_);
lean_dec(v_i_563_);
v_res_575_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2_spec__2(v_sz_boxed_573_, v_i_boxed_574_, v_bs_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_, v___y_570_, v___y_571_);
lean_dec(v___y_571_);
lean_dec_ref(v___y_570_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
lean_dec(v___y_567_);
lean_dec_ref(v___y_566_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2___boxed(lean_object* v_x_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(v_x_576_, v___y_577_, v___y_578_, v___y_579_, v___y_580_, v___y_581_, v___y_582_, v___y_583_);
lean_dec(v___y_583_);
lean_dec_ref(v___y_582_);
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls(lean_object* v_stx_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(v_stx_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_, v_a_593_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls___boxed(lean_object* v_stx_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls(v_stx_596_, v_a_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_);
lean_dec(v_a_603_);
lean_dec_ref(v_a_602_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec_ref(v_a_598_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1(lean_object* v_00_u03b4_606_, lean_object* v_t_607_, lean_object* v_k_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___redArg(v_t_607_, v_k_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1___boxed(lean_object* v_00_u03b4_610_, lean_object* v_t_611_, lean_object* v_k_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__1(v_00_u03b4_610_, v_t_611_, v_k_612_);
lean_dec(v_k_612_);
lean_dec(v_t_611_);
return v_res_613_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1(void){
_start:
{
lean_object* v___x_615_; lean_object* v___x_616_; 
v___x_615_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__0));
v___x_616_ = l_Lean_stringToMessageData(v___x_615_);
return v___x_616_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3(void){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; 
v___x_618_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__2));
v___x_619_ = l_Lean_stringToMessageData(v___x_618_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0(lean_object* v_p_620_, lean_object* v_e_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = l_Lean_Meta_addPPExplicitToExposeDiff(v_p_620_, v_e_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_650_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_650_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_650_ == 0)
{
v___x_630_ = v___x_627_;
v_isShared_631_ = v_isSharedCheck_650_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_a_628_);
lean_dec(v___x_627_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_650_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v_fst_632_; lean_object* v_snd_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_649_; 
v_fst_632_ = lean_ctor_get(v_a_628_, 0);
v_snd_633_ = lean_ctor_get(v_a_628_, 1);
v_isSharedCheck_649_ = !lean_is_exclusive(v_a_628_);
if (v_isSharedCheck_649_ == 0)
{
v___x_635_ = v_a_628_;
v_isShared_636_ = v_isSharedCheck_649_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_snd_633_);
lean_inc(v_fst_632_);
lean_dec(v_a_628_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_649_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_640_; 
v___x_637_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__1);
v___x_638_ = l_Lean_indentExpr(v_fst_632_);
if (v_isShared_636_ == 0)
{
lean_ctor_set_tag(v___x_635_, 7);
lean_ctor_set(v___x_635_, 1, v___x_638_);
lean_ctor_set(v___x_635_, 0, v___x_637_);
v___x_640_ = v___x_635_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v___x_637_);
lean_ctor_set(v_reuseFailAlloc_648_, 1, v___x_638_);
v___x_640_ = v_reuseFailAlloc_648_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_646_; 
v___x_641_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3, &lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___closed__3);
v___x_642_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_642_, 0, v___x_640_);
lean_ctor_set(v___x_642_, 1, v___x_641_);
v___x_643_ = l_Lean_indentExpr(v_snd_633_);
v___x_644_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_644_, 0, v___x_642_);
lean_ctor_set(v___x_644_, 1, v___x_643_);
if (v_isShared_631_ == 0)
{
lean_ctor_set(v___x_630_, 0, v___x_644_);
v___x_646_ = v___x_630_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_647_; 
v_reuseFailAlloc_647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_647_, 0, v___x_644_);
v___x_646_ = v_reuseFailAlloc_647_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
return v___x_646_;
}
}
}
}
}
else
{
lean_object* v_a_651_; lean_object* v___x_653_; uint8_t v_isShared_654_; uint8_t v_isSharedCheck_658_; 
v_a_651_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_658_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_658_ == 0)
{
v___x_653_ = v___x_627_;
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
else
{
lean_inc(v_a_651_);
lean_dec(v___x_627_);
v___x_653_ = lean_box(0);
v_isShared_654_ = v_isSharedCheck_658_;
goto v_resetjp_652_;
}
v_resetjp_652_:
{
lean_object* v___x_656_; 
if (v_isShared_654_ == 0)
{
v___x_656_ = v___x_653_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_657_; 
v_reuseFailAlloc_657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_657_, 0, v_a_651_);
v___x_656_ = v_reuseFailAlloc_657_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
return v___x_656_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___boxed(lean_object* v_p_659_, lean_object* v_e_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0(v_p_659_, v_e_660_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
lean_dec(v___y_664_);
lean_dec_ref(v___y_663_);
lean_dec(v___y_662_);
lean_dec_ref(v___y_661_);
return v_res_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError(lean_object* v_goal_670_, lean_object* v_p_671_, lean_object* v_e_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_){
_start:
{
lean_object* v_keyedConfig_678_; uint8_t v_trackZetaDelta_679_; lean_object* v_zetaDeltaSet_680_; lean_object* v_lctx_681_; lean_object* v_localInstances_682_; lean_object* v_defEqCtx_x3f_683_; lean_object* v_synthPendingDepth_684_; lean_object* v_customCanUnfoldPredicate_x3f_685_; uint8_t v_univApprox_686_; uint8_t v_inTypeClassResolution_687_; uint8_t v_cacheInferType_688_; uint8_t v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; uint8_t v_foApprox_693_; uint8_t v_ctxApprox_694_; uint8_t v_quasiPatternApprox_695_; uint8_t v_constApprox_696_; uint8_t v_isDefEqStuckEx_697_; uint8_t v_unificationHints_698_; uint8_t v_assignSyntheticOpaque_699_; uint8_t v_offsetCnstrs_700_; uint8_t v_transparency_701_; uint8_t v_etaStruct_702_; uint8_t v_univApprox_703_; uint8_t v_iota_704_; uint8_t v_beta_705_; uint8_t v_proj_706_; uint8_t v_zeta_707_; uint8_t v_zetaDelta_708_; uint8_t v_zetaUnused_709_; uint8_t v_zetaHave_710_; uint8_t v_canUnfoldPredicateConfig_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_781_; 
v_keyedConfig_678_ = lean_ctor_get(v_a_673_, 0);
v_trackZetaDelta_679_ = lean_ctor_get_uint8(v_a_673_, sizeof(void*)*7);
v_zetaDeltaSet_680_ = lean_ctor_get(v_a_673_, 1);
v_lctx_681_ = lean_ctor_get(v_a_673_, 2);
v_localInstances_682_ = lean_ctor_get(v_a_673_, 3);
v_defEqCtx_x3f_683_ = lean_ctor_get(v_a_673_, 4);
v_synthPendingDepth_684_ = lean_ctor_get(v_a_673_, 5);
v_customCanUnfoldPredicate_x3f_685_ = lean_ctor_get(v_a_673_, 6);
v_univApprox_686_ = lean_ctor_get_uint8(v_a_673_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_687_ = lean_ctor_get_uint8(v_a_673_, sizeof(void*)*7 + 2);
v_cacheInferType_688_ = lean_ctor_get_uint8(v_a_673_, sizeof(void*)*7 + 3);
v___x_689_ = 2;
lean_inc_ref(v_keyedConfig_678_);
v___x_690_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_689_, v_keyedConfig_678_);
lean_inc(v_customCanUnfoldPredicate_x3f_685_);
lean_inc(v_synthPendingDepth_684_);
lean_inc(v_defEqCtx_x3f_683_);
lean_inc_ref(v_localInstances_682_);
lean_inc_ref(v_lctx_681_);
lean_inc(v_zetaDeltaSet_680_);
v___x_691_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_691_, 0, v___x_690_);
lean_ctor_set(v___x_691_, 1, v_zetaDeltaSet_680_);
lean_ctor_set(v___x_691_, 2, v_lctx_681_);
lean_ctor_set(v___x_691_, 3, v_localInstances_682_);
lean_ctor_set(v___x_691_, 4, v_defEqCtx_x3f_683_);
lean_ctor_set(v___x_691_, 5, v_synthPendingDepth_684_);
lean_ctor_set(v___x_691_, 6, v_customCanUnfoldPredicate_x3f_685_);
lean_ctor_set_uint8(v___x_691_, sizeof(void*)*7, v_trackZetaDelta_679_);
lean_ctor_set_uint8(v___x_691_, sizeof(void*)*7 + 1, v_univApprox_686_);
lean_ctor_set_uint8(v___x_691_, sizeof(void*)*7 + 2, v_inTypeClassResolution_687_);
lean_ctor_set_uint8(v___x_691_, sizeof(void*)*7 + 3, v_cacheInferType_688_);
v___x_692_ = l_Lean_Meta_Context_config(v___x_691_);
lean_dec_ref_known(v___x_691_, 7);
v_foApprox_693_ = lean_ctor_get_uint8(v___x_692_, 0);
v_ctxApprox_694_ = lean_ctor_get_uint8(v___x_692_, 1);
v_quasiPatternApprox_695_ = lean_ctor_get_uint8(v___x_692_, 2);
v_constApprox_696_ = lean_ctor_get_uint8(v___x_692_, 3);
v_isDefEqStuckEx_697_ = lean_ctor_get_uint8(v___x_692_, 4);
v_unificationHints_698_ = lean_ctor_get_uint8(v___x_692_, 5);
v_assignSyntheticOpaque_699_ = lean_ctor_get_uint8(v___x_692_, 7);
v_offsetCnstrs_700_ = lean_ctor_get_uint8(v___x_692_, 8);
v_transparency_701_ = lean_ctor_get_uint8(v___x_692_, 9);
v_etaStruct_702_ = lean_ctor_get_uint8(v___x_692_, 10);
v_univApprox_703_ = lean_ctor_get_uint8(v___x_692_, 11);
v_iota_704_ = lean_ctor_get_uint8(v___x_692_, 12);
v_beta_705_ = lean_ctor_get_uint8(v___x_692_, 13);
v_proj_706_ = lean_ctor_get_uint8(v___x_692_, 14);
v_zeta_707_ = lean_ctor_get_uint8(v___x_692_, 15);
v_zetaDelta_708_ = lean_ctor_get_uint8(v___x_692_, 16);
v_zetaUnused_709_ = lean_ctor_get_uint8(v___x_692_, 17);
v_zetaHave_710_ = lean_ctor_get_uint8(v___x_692_, 18);
v_canUnfoldPredicateConfig_711_ = lean_ctor_get_uint8(v___x_692_, 19);
v_isSharedCheck_781_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_781_ == 0)
{
v___x_713_ = v___x_692_;
v_isShared_714_ = v_isSharedCheck_781_;
goto v_resetjp_712_;
}
else
{
lean_dec(v___x_692_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_781_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
uint8_t v___x_715_; lean_object* v___x_717_; 
v___x_715_ = 0;
if (v_isShared_714_ == 0)
{
v___x_717_ = v___x_713_;
goto v_reusejp_716_;
}
else
{
lean_object* v_reuseFailAlloc_780_; 
v_reuseFailAlloc_780_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 0, v_foApprox_693_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 1, v_ctxApprox_694_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 2, v_quasiPatternApprox_695_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 3, v_constApprox_696_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 4, v_isDefEqStuckEx_697_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 5, v_unificationHints_698_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 7, v_assignSyntheticOpaque_699_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 8, v_offsetCnstrs_700_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 9, v_transparency_701_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 10, v_etaStruct_702_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 11, v_univApprox_703_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 12, v_iota_704_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 13, v_beta_705_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 14, v_proj_706_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 15, v_zeta_707_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 16, v_zetaDelta_708_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 17, v_zetaUnused_709_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 18, v_zetaHave_710_);
lean_ctor_set_uint8(v_reuseFailAlloc_780_, 19, v_canUnfoldPredicateConfig_711_);
v___x_717_ = v_reuseFailAlloc_780_;
goto v_reusejp_716_;
}
v_reusejp_716_:
{
uint64_t v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; uint8_t v_foApprox_722_; uint8_t v_ctxApprox_723_; uint8_t v_quasiPatternApprox_724_; uint8_t v_constApprox_725_; uint8_t v_isDefEqStuckEx_726_; uint8_t v_unificationHints_727_; uint8_t v_proofIrrelevance_728_; uint8_t v_offsetCnstrs_729_; uint8_t v_transparency_730_; uint8_t v_etaStruct_731_; uint8_t v_univApprox_732_; uint8_t v_iota_733_; uint8_t v_beta_734_; uint8_t v_proj_735_; uint8_t v_zeta_736_; uint8_t v_zetaDelta_737_; uint8_t v_zetaUnused_738_; uint8_t v_zetaHave_739_; uint8_t v_canUnfoldPredicateConfig_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_779_; 
lean_ctor_set_uint8(v___x_717_, 6, v___x_715_);
v___x_718_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_717_);
v___x_719_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_719_, 0, v___x_717_);
lean_ctor_set_uint64(v___x_719_, sizeof(void*)*1, v___x_718_);
lean_inc(v_customCanUnfoldPredicate_x3f_685_);
lean_inc(v_synthPendingDepth_684_);
lean_inc(v_defEqCtx_x3f_683_);
lean_inc_ref(v_localInstances_682_);
lean_inc_ref(v_lctx_681_);
lean_inc(v_zetaDeltaSet_680_);
v___x_720_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_720_, 0, v___x_719_);
lean_ctor_set(v___x_720_, 1, v_zetaDeltaSet_680_);
lean_ctor_set(v___x_720_, 2, v_lctx_681_);
lean_ctor_set(v___x_720_, 3, v_localInstances_682_);
lean_ctor_set(v___x_720_, 4, v_defEqCtx_x3f_683_);
lean_ctor_set(v___x_720_, 5, v_synthPendingDepth_684_);
lean_ctor_set(v___x_720_, 6, v_customCanUnfoldPredicate_x3f_685_);
lean_ctor_set_uint8(v___x_720_, sizeof(void*)*7, v_trackZetaDelta_679_);
lean_ctor_set_uint8(v___x_720_, sizeof(void*)*7 + 1, v_univApprox_686_);
lean_ctor_set_uint8(v___x_720_, sizeof(void*)*7 + 2, v_inTypeClassResolution_687_);
lean_ctor_set_uint8(v___x_720_, sizeof(void*)*7 + 3, v_cacheInferType_688_);
v___x_721_ = l_Lean_Meta_Context_config(v___x_720_);
lean_dec_ref_known(v___x_720_, 7);
v_foApprox_722_ = lean_ctor_get_uint8(v___x_721_, 0);
v_ctxApprox_723_ = lean_ctor_get_uint8(v___x_721_, 1);
v_quasiPatternApprox_724_ = lean_ctor_get_uint8(v___x_721_, 2);
v_constApprox_725_ = lean_ctor_get_uint8(v___x_721_, 3);
v_isDefEqStuckEx_726_ = lean_ctor_get_uint8(v___x_721_, 4);
v_unificationHints_727_ = lean_ctor_get_uint8(v___x_721_, 5);
v_proofIrrelevance_728_ = lean_ctor_get_uint8(v___x_721_, 6);
v_offsetCnstrs_729_ = lean_ctor_get_uint8(v___x_721_, 8);
v_transparency_730_ = lean_ctor_get_uint8(v___x_721_, 9);
v_etaStruct_731_ = lean_ctor_get_uint8(v___x_721_, 10);
v_univApprox_732_ = lean_ctor_get_uint8(v___x_721_, 11);
v_iota_733_ = lean_ctor_get_uint8(v___x_721_, 12);
v_beta_734_ = lean_ctor_get_uint8(v___x_721_, 13);
v_proj_735_ = lean_ctor_get_uint8(v___x_721_, 14);
v_zeta_736_ = lean_ctor_get_uint8(v___x_721_, 15);
v_zetaDelta_737_ = lean_ctor_get_uint8(v___x_721_, 16);
v_zetaUnused_738_ = lean_ctor_get_uint8(v___x_721_, 17);
v_zetaHave_739_ = lean_ctor_get_uint8(v___x_721_, 18);
v_canUnfoldPredicateConfig_740_ = lean_ctor_get_uint8(v___x_721_, 19);
v_isSharedCheck_779_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_779_ == 0)
{
v___x_742_ = v___x_721_;
v_isShared_743_ = v_isSharedCheck_779_;
goto v_resetjp_741_;
}
else
{
lean_dec(v___x_721_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_779_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
uint8_t v___x_744_; lean_object* v___x_746_; 
v___x_744_ = 1;
if (v_isShared_743_ == 0)
{
v___x_746_ = v___x_742_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 0, v_foApprox_722_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 1, v_ctxApprox_723_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 2, v_quasiPatternApprox_724_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 3, v_constApprox_725_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 4, v_isDefEqStuckEx_726_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 5, v_unificationHints_727_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 6, v_proofIrrelevance_728_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 8, v_offsetCnstrs_729_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 9, v_transparency_730_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 10, v_etaStruct_731_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 11, v_univApprox_732_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 12, v_iota_733_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 13, v_beta_734_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 14, v_proj_735_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 15, v_zeta_736_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 16, v_zetaDelta_737_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 17, v_zetaUnused_738_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 18, v_zetaHave_739_);
lean_ctor_set_uint8(v_reuseFailAlloc_778_, 19, v_canUnfoldPredicateConfig_740_);
v___x_746_ = v_reuseFailAlloc_778_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
uint64_t v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
lean_ctor_set_uint8(v___x_746_, 7, v___x_744_);
v___x_747_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_746_);
v___x_748_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_748_, 0, v___x_746_);
lean_ctor_set_uint64(v___x_748_, sizeof(void*)*1, v___x_747_);
lean_inc(v_customCanUnfoldPredicate_x3f_685_);
lean_inc(v_synthPendingDepth_684_);
lean_inc(v_defEqCtx_x3f_683_);
lean_inc_ref(v_localInstances_682_);
lean_inc_ref(v_lctx_681_);
lean_inc(v_zetaDeltaSet_680_);
v___x_749_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_749_, 0, v___x_748_);
lean_ctor_set(v___x_749_, 1, v_zetaDeltaSet_680_);
lean_ctor_set(v___x_749_, 2, v_lctx_681_);
lean_ctor_set(v___x_749_, 3, v_localInstances_682_);
lean_ctor_set(v___x_749_, 4, v_defEqCtx_x3f_683_);
lean_ctor_set(v___x_749_, 5, v_synthPendingDepth_684_);
lean_ctor_set(v___x_749_, 6, v_customCanUnfoldPredicate_x3f_685_);
lean_ctor_set_uint8(v___x_749_, sizeof(void*)*7, v_trackZetaDelta_679_);
lean_ctor_set_uint8(v___x_749_, sizeof(void*)*7 + 1, v_univApprox_686_);
lean_ctor_set_uint8(v___x_749_, sizeof(void*)*7 + 2, v_inTypeClassResolution_687_);
lean_ctor_set_uint8(v___x_749_, sizeof(void*)*7 + 3, v_cacheInferType_688_);
lean_inc_ref(v_e_672_);
lean_inc_ref(v_p_671_);
v___x_750_ = l_Lean_Meta_isExprDefEq(v_p_671_, v_e_672_, v___x_749_, v_a_674_, v_a_675_, v_a_676_);
lean_dec_ref_known(v___x_749_, 7);
if (lean_obj_tag(v___x_750_) == 0)
{
lean_object* v_a_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_769_; 
v_a_751_ = lean_ctor_get(v___x_750_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_750_);
if (v_isSharedCheck_769_ == 0)
{
v___x_753_ = v___x_750_;
v_isShared_754_ = v_isSharedCheck_769_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_a_751_);
lean_dec(v___x_750_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_769_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
uint8_t v___x_755_; 
v___x_755_ = lean_unbox(v_a_751_);
lean_dec(v_a_751_);
if (v___x_755_ == 0)
{
lean_object* v___f_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
lean_del_object(v___x_753_);
lean_inc_ref(v_e_672_);
lean_inc_ref(v_p_671_);
v___f_756_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___lam__0___boxed), 7, 2);
lean_closure_set(v___f_756_, 0, v_p_671_);
lean_closure_set(v___f_756_, 1, v_e_672_);
v___x_757_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__1));
v___x_758_ = lean_unsigned_to_nat(2u);
v___x_759_ = lean_mk_empty_array_with_capacity(v___x_758_);
v___x_760_ = lean_array_push(v___x_759_, v_p_671_);
v___x_761_ = lean_array_push(v___x_760_, v_e_672_);
v___x_762_ = l_Lean_MessageData_ofLazyM(v___f_756_, v___x_761_);
v___x_763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_763_, 0, v___x_762_);
v___x_764_ = l_Lean_Meta_throwTacticEx___redArg(v___x_757_, v_goal_670_, v___x_763_, v_a_673_, v_a_674_, v_a_675_, v_a_676_);
return v___x_764_;
}
else
{
lean_object* v___x_765_; lean_object* v___x_767_; 
lean_dec_ref(v_e_672_);
lean_dec_ref(v_p_671_);
lean_dec(v_goal_670_);
v___x_765_ = lean_box(0);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 0, v___x_765_);
v___x_767_ = v___x_753_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_765_);
v___x_767_ = v_reuseFailAlloc_768_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
return v___x_767_;
}
}
}
}
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
lean_dec_ref(v_e_672_);
lean_dec_ref(v_p_671_);
lean_dec(v_goal_670_);
v_a_770_ = lean_ctor_get(v___x_750_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_750_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_750_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_750_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___boxed(lean_object* v_goal_782_, lean_object* v_p_783_, lean_object* v_e_784_, lean_object* v_a_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError(v_goal_782_, v_p_783_, v_e_784_, v_a_785_, v_a_786_, v_a_787_, v_a_788_);
lean_dec(v_a_788_);
lean_dec_ref(v_a_787_);
lean_dec(v_a_786_);
lean_dec_ref(v_a_785_);
return v_res_790_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21(void){
_start:
{
lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_836_ = l_Lean_Parser_Tactic_location;
v___x_837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__14));
v___x_838_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
lean_ctor_set(v___x_838_, 1, v___x_836_);
return v___x_838_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22(void){
_start:
{
lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; 
v___x_839_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21, &lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__21);
v___x_840_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__20));
v___x_841_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__6));
v___x_842_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_842_, 0, v___x_841_);
lean_ctor_set(v___x_842_, 1, v___x_840_);
lean_ctor_set(v___x_842_, 2, v___x_839_);
return v___x_842_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23(void){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_843_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22, &lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__22);
v___x_844_ = lean_unsigned_to_nat(1022u);
v___x_845_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4));
v___x_846_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_846_, 0, v___x_845_);
lean_ctor_set(v___x_846_, 1, v___x_844_);
lean_ctor_set(v___x_846_, 2, v___x_843_);
return v___x_846_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_SetM_setM(void){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23, &lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_SetM_setM___closed__23);
return v___x_847_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; 
v___x_848_ = lean_box(0);
v___x_849_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_850_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_850_, 0, v___x_849_);
lean_ctor_set(v___x_850_, 1, v___x_848_);
return v___x_850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg(){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; 
v___x_852_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___closed__0);
v___x_853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_853_, 0, v___x_852_);
return v___x_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg___boxed(lean_object* v___y_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0(lean_object* v_00_u03b1_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_){
_start:
{
lean_object* v___x_866_; 
v___x_866_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v___x_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___boxed(lean_object* v_00_u03b1_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0(v_00_u03b1_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, v___y_875_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v___y_873_);
lean_dec_ref(v___y_872_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(lean_object* v_e_878_, lean_object* v___y_879_){
_start:
{
uint8_t v___x_881_; 
v___x_881_ = l_Lean_Expr_hasMVar(v_e_878_);
if (v___x_881_ == 0)
{
lean_object* v___x_882_; 
v___x_882_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_882_, 0, v_e_878_);
return v___x_882_;
}
else
{
lean_object* v___x_883_; lean_object* v_mctx_884_; lean_object* v___x_885_; lean_object* v_fst_886_; lean_object* v_snd_887_; lean_object* v___x_888_; lean_object* v_cache_889_; lean_object* v_zetaDeltaFVarIds_890_; lean_object* v_postponed_891_; lean_object* v_diag_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_901_; 
v___x_883_ = lean_st_ref_get(v___y_879_);
v_mctx_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc_ref(v_mctx_884_);
lean_dec(v___x_883_);
v___x_885_ = l_Lean_instantiateMVarsCore(v_mctx_884_, v_e_878_);
v_fst_886_ = lean_ctor_get(v___x_885_, 0);
lean_inc(v_fst_886_);
v_snd_887_ = lean_ctor_get(v___x_885_, 1);
lean_inc(v_snd_887_);
lean_dec_ref(v___x_885_);
v___x_888_ = lean_st_ref_take(v___y_879_);
v_cache_889_ = lean_ctor_get(v___x_888_, 1);
v_zetaDeltaFVarIds_890_ = lean_ctor_get(v___x_888_, 2);
v_postponed_891_ = lean_ctor_get(v___x_888_, 3);
v_diag_892_ = lean_ctor_get(v___x_888_, 4);
v_isSharedCheck_901_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_901_ == 0)
{
lean_object* v_unused_902_; 
v_unused_902_ = lean_ctor_get(v___x_888_, 0);
lean_dec(v_unused_902_);
v___x_894_ = v___x_888_;
v_isShared_895_ = v_isSharedCheck_901_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_diag_892_);
lean_inc(v_postponed_891_);
lean_inc(v_zetaDeltaFVarIds_890_);
lean_inc(v_cache_889_);
lean_dec(v___x_888_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_901_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
lean_ctor_set(v___x_894_, 0, v_snd_887_);
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_900_; 
v_reuseFailAlloc_900_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_900_, 0, v_snd_887_);
lean_ctor_set(v_reuseFailAlloc_900_, 1, v_cache_889_);
lean_ctor_set(v_reuseFailAlloc_900_, 2, v_zetaDeltaFVarIds_890_);
lean_ctor_set(v_reuseFailAlloc_900_, 3, v_postponed_891_);
lean_ctor_set(v_reuseFailAlloc_900_, 4, v_diag_892_);
v___x_897_ = v_reuseFailAlloc_900_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
lean_object* v___x_898_; lean_object* v___x_899_; 
v___x_898_ = lean_st_ref_set(v___y_879_, v___x_897_);
v___x_899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_899_, 0, v_fst_886_);
return v___x_899_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg___boxed(lean_object* v_e_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
lean_object* v_res_906_; 
v_res_906_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(v_e_903_, v___y_904_);
lean_dec(v___y_904_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6(lean_object* v_e_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_){
_start:
{
lean_object* v___x_913_; 
v___x_913_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(v_e_907_, v___y_909_);
return v___x_913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___boxed(lean_object* v_e_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6(v_e_914_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
lean_dec(v___y_916_);
lean_dec_ref(v___y_915_);
return v_res_920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0(lean_object* v_x_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_){
_start:
{
lean_object* v___x_931_; 
lean_inc(v___y_925_);
lean_inc_ref(v___y_924_);
lean_inc(v___y_923_);
lean_inc_ref(v___y_922_);
v___x_931_ = lean_apply_9(v_x_921_, v___y_922_, v___y_923_, v___y_924_, v___y_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_, lean_box(0));
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0___boxed(lean_object* v_x_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0(v_x_932_, v___y_933_, v___y_934_, v___y_935_, v___y_936_, v___y_937_, v___y_938_, v___y_939_, v___y_940_);
lean_dec(v___y_936_);
lean_dec_ref(v___y_935_);
lean_dec(v___y_934_);
lean_dec_ref(v___y_933_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg(lean_object* v_mvarId_943_, lean_object* v_x_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
lean_object* v___f_954_; lean_object* v___x_955_; 
lean_inc(v___y_948_);
lean_inc_ref(v___y_947_);
lean_inc(v___y_946_);
lean_inc_ref(v___y_945_);
v___f_954_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_954_, 0, v_x_944_);
lean_closure_set(v___f_954_, 1, v___y_945_);
lean_closure_set(v___f_954_, 2, v___y_946_);
lean_closure_set(v___f_954_, 3, v___y_947_);
lean_closure_set(v___f_954_, 4, v___y_948_);
v___x_955_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_943_, v___f_954_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
if (lean_obj_tag(v___x_955_) == 0)
{
return v___x_955_;
}
else
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
v_a_956_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_955_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_955_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg___boxed(lean_object* v_mvarId_964_, lean_object* v_x_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
lean_object* v_res_975_; 
v_res_975_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg(v_mvarId_964_, v_x_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8(lean_object* v_00_u03b1_976_, lean_object* v_mvarId_977_, lean_object* v_x_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg(v_mvarId_977_, v_x_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___boxed(lean_object* v_00_u03b1_989_, lean_object* v_mvarId_990_, lean_object* v_x_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_){
_start:
{
lean_object* v_res_1001_; 
v_res_1001_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8(v_00_u03b1_989_, v_mvarId_990_, v_x_991_, v___y_992_, v___y_993_, v___y_994_, v___y_995_, v___y_996_, v___y_997_, v___y_998_, v___y_999_);
lean_dec(v___y_999_);
lean_dec_ref(v___y_998_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
lean_dec(v___y_993_);
lean_dec_ref(v___y_992_);
return v_res_1001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg(lean_object* v___x_1002_, lean_object* v_hi_1003_, lean_object* v_pivot_1004_, lean_object* v_as_1005_, lean_object* v_i_1006_, lean_object* v_k_1007_){
_start:
{
uint8_t v___y_1009_; uint8_t v___x_1018_; 
v___x_1018_ = lean_nat_dec_lt(v_k_1007_, v_hi_1003_);
if (v___x_1018_ == 0)
{
lean_object* v___x_1019_; lean_object* v___x_1020_; 
lean_dec(v_k_1007_);
lean_dec(v_pivot_1004_);
v___x_1019_ = lean_array_fswap(v_as_1005_, v_i_1006_, v_hi_1003_);
v___x_1020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1020_, 0, v_i_1006_);
lean_ctor_set(v___x_1020_, 1, v___x_1019_);
return v___x_1020_;
}
else
{
lean_object* v___x_1021_; lean_object* v_decl_u2081_1022_; lean_object* v_index_1023_; lean_object* v_decl_u2082_1024_; lean_object* v_index_1025_; uint8_t v___x_1026_; 
v___x_1021_ = lean_array_fget_borrowed(v_as_1005_, v_k_1007_);
lean_inc(v___x_1021_);
v_decl_u2081_1022_ = l_Lean_MetavarContext_getDecl(v___x_1002_, v___x_1021_);
v_index_1023_ = lean_ctor_get(v_decl_u2081_1022_, 6);
lean_inc(v_index_1023_);
lean_dec_ref(v_decl_u2081_1022_);
lean_inc(v_pivot_1004_);
v_decl_u2082_1024_ = l_Lean_MetavarContext_getDecl(v___x_1002_, v_pivot_1004_);
v_index_1025_ = lean_ctor_get(v_decl_u2082_1024_, 6);
lean_inc(v_index_1025_);
lean_dec_ref(v_decl_u2082_1024_);
v___x_1026_ = lean_nat_dec_eq(v_index_1023_, v_index_1025_);
if (v___x_1026_ == 0)
{
uint8_t v___x_1027_; 
v___x_1027_ = lean_nat_dec_lt(v_index_1023_, v_index_1025_);
lean_dec(v_index_1025_);
lean_dec(v_index_1023_);
v___y_1009_ = v___x_1027_;
goto v___jp_1008_;
}
else
{
uint8_t v___x_1028_; 
lean_dec(v_index_1025_);
lean_dec(v_index_1023_);
v___x_1028_ = l_Lean_Name_quickLt(v___x_1021_, v_pivot_1004_);
v___y_1009_ = v___x_1028_;
goto v___jp_1008_;
}
}
v___jp_1008_:
{
if (v___y_1009_ == 0)
{
lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1010_ = lean_unsigned_to_nat(1u);
v___x_1011_ = lean_nat_add(v_k_1007_, v___x_1010_);
lean_dec(v_k_1007_);
v_k_1007_ = v___x_1011_;
goto _start;
}
else
{
lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; 
v___x_1013_ = lean_array_fswap(v_as_1005_, v_i_1006_, v_k_1007_);
v___x_1014_ = lean_unsigned_to_nat(1u);
v___x_1015_ = lean_nat_add(v_i_1006_, v___x_1014_);
lean_dec(v_i_1006_);
v___x_1016_ = lean_nat_add(v_k_1007_, v___x_1014_);
lean_dec(v_k_1007_);
v_as_1005_ = v___x_1013_;
v_i_1006_ = v___x_1015_;
v_k_1007_ = v___x_1016_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg___boxed(lean_object* v___x_1029_, lean_object* v_hi_1030_, lean_object* v_pivot_1031_, lean_object* v_as_1032_, lean_object* v_i_1033_, lean_object* v_k_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg(v___x_1029_, v_hi_1030_, v_pivot_1031_, v_as_1032_, v_i_1033_, v_k_1034_);
lean_dec(v_hi_1030_);
lean_dec_ref(v___x_1029_);
return v_res_1035_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(lean_object* v___x_1036_, lean_object* v_mvarId_u2081_1037_, lean_object* v_mvarId_u2082_1038_){
_start:
{
lean_object* v_decl_u2081_1039_; lean_object* v_index_1040_; lean_object* v_decl_u2082_1041_; lean_object* v_index_1042_; uint8_t v___x_1043_; 
lean_inc(v_mvarId_u2081_1037_);
v_decl_u2081_1039_ = l_Lean_MetavarContext_getDecl(v___x_1036_, v_mvarId_u2081_1037_);
v_index_1040_ = lean_ctor_get(v_decl_u2081_1039_, 6);
lean_inc(v_index_1040_);
lean_dec_ref(v_decl_u2081_1039_);
lean_inc(v_mvarId_u2082_1038_);
v_decl_u2082_1041_ = l_Lean_MetavarContext_getDecl(v___x_1036_, v_mvarId_u2082_1038_);
v_index_1042_ = lean_ctor_get(v_decl_u2082_1041_, 6);
lean_inc(v_index_1042_);
lean_dec_ref(v_decl_u2082_1041_);
v___x_1043_ = lean_nat_dec_eq(v_index_1040_, v_index_1042_);
if (v___x_1043_ == 0)
{
uint8_t v___x_1044_; 
lean_dec(v_mvarId_u2082_1038_);
lean_dec(v_mvarId_u2081_1037_);
v___x_1044_ = lean_nat_dec_lt(v_index_1040_, v_index_1042_);
lean_dec(v_index_1042_);
lean_dec(v_index_1040_);
return v___x_1044_;
}
else
{
uint8_t v___x_1045_; 
lean_dec(v_index_1042_);
lean_dec(v_index_1040_);
v___x_1045_ = l_Lean_Name_quickLt(v_mvarId_u2081_1037_, v_mvarId_u2082_1038_);
lean_dec(v_mvarId_u2082_1038_);
lean_dec(v_mvarId_u2081_1037_);
return v___x_1045_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0___boxed(lean_object* v___x_1046_, lean_object* v_mvarId_u2081_1047_, lean_object* v_mvarId_u2082_1048_){
_start:
{
uint8_t v_res_1049_; lean_object* v_r_1050_; 
v_res_1049_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(v___x_1046_, v_mvarId_u2081_1047_, v_mvarId_u2082_1048_);
lean_dec_ref(v___x_1046_);
v_r_1050_ = lean_box(v_res_1049_);
return v_r_1050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(lean_object* v___x_1051_, lean_object* v_n_1052_, lean_object* v_as_1053_, lean_object* v_lo_1054_, lean_object* v_hi_1055_){
_start:
{
lean_object* v___y_1057_; uint8_t v___x_1067_; 
v___x_1067_ = lean_nat_dec_lt(v_lo_1054_, v_hi_1055_);
if (v___x_1067_ == 0)
{
lean_dec(v_lo_1054_);
return v_as_1053_;
}
else
{
lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v_mid_1070_; lean_object* v___y_1072_; lean_object* v___y_1078_; lean_object* v___x_1083_; lean_object* v___x_1084_; uint8_t v___x_1085_; 
v___x_1068_ = lean_nat_add(v_lo_1054_, v_hi_1055_);
v___x_1069_ = lean_unsigned_to_nat(1u);
v_mid_1070_ = lean_nat_shiftr(v___x_1068_, v___x_1069_);
lean_dec(v___x_1068_);
v___x_1083_ = lean_array_fget_borrowed(v_as_1053_, v_mid_1070_);
v___x_1084_ = lean_array_fget_borrowed(v_as_1053_, v_lo_1054_);
lean_inc(v___x_1084_);
lean_inc(v___x_1083_);
v___x_1085_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(v___x_1051_, v___x_1083_, v___x_1084_);
if (v___x_1085_ == 0)
{
v___y_1078_ = v_as_1053_;
goto v___jp_1077_;
}
else
{
lean_object* v___x_1086_; 
v___x_1086_ = lean_array_fswap(v_as_1053_, v_lo_1054_, v_mid_1070_);
v___y_1078_ = v___x_1086_;
goto v___jp_1077_;
}
v___jp_1071_:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; uint8_t v___x_1075_; 
v___x_1073_ = lean_array_fget_borrowed(v___y_1072_, v_mid_1070_);
v___x_1074_ = lean_array_fget_borrowed(v___y_1072_, v_hi_1055_);
lean_inc(v___x_1074_);
lean_inc(v___x_1073_);
v___x_1075_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(v___x_1051_, v___x_1073_, v___x_1074_);
if (v___x_1075_ == 0)
{
lean_dec(v_mid_1070_);
v___y_1057_ = v___y_1072_;
goto v___jp_1056_;
}
else
{
lean_object* v___x_1076_; 
v___x_1076_ = lean_array_fswap(v___y_1072_, v_mid_1070_, v_hi_1055_);
lean_dec(v_mid_1070_);
v___y_1057_ = v___x_1076_;
goto v___jp_1056_;
}
}
v___jp_1077_:
{
lean_object* v___x_1079_; lean_object* v___x_1080_; uint8_t v___x_1081_; 
v___x_1079_ = lean_array_fget_borrowed(v___y_1078_, v_hi_1055_);
v___x_1080_ = lean_array_fget_borrowed(v___y_1078_, v_lo_1054_);
lean_inc(v___x_1080_);
lean_inc(v___x_1079_);
v___x_1081_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___lam__0(v___x_1051_, v___x_1079_, v___x_1080_);
if (v___x_1081_ == 0)
{
v___y_1072_ = v___y_1078_;
goto v___jp_1071_;
}
else
{
lean_object* v___x_1082_; 
v___x_1082_ = lean_array_fswap(v___y_1078_, v_lo_1054_, v_hi_1055_);
v___y_1072_ = v___x_1082_;
goto v___jp_1071_;
}
}
}
v___jp_1056_:
{
lean_object* v_pivot_1058_; lean_object* v___x_1059_; lean_object* v_fst_1060_; lean_object* v_snd_1061_; uint8_t v___x_1062_; 
v_pivot_1058_ = lean_array_fget(v___y_1057_, v_hi_1055_);
lean_inc_n(v_lo_1054_, 2);
v___x_1059_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg(v___x_1051_, v_hi_1055_, v_pivot_1058_, v___y_1057_, v_lo_1054_, v_lo_1054_);
v_fst_1060_ = lean_ctor_get(v___x_1059_, 0);
lean_inc(v_fst_1060_);
v_snd_1061_ = lean_ctor_get(v___x_1059_, 1);
lean_inc(v_snd_1061_);
lean_dec_ref(v___x_1059_);
v___x_1062_ = lean_nat_dec_le(v_hi_1055_, v_fst_1060_);
if (v___x_1062_ == 0)
{
lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1063_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(v___x_1051_, v_n_1052_, v_snd_1061_, v_lo_1054_, v_fst_1060_);
v___x_1064_ = lean_unsigned_to_nat(1u);
v___x_1065_ = lean_nat_add(v_fst_1060_, v___x_1064_);
lean_dec(v_fst_1060_);
v_as_1053_ = v___x_1063_;
v_lo_1054_ = v___x_1065_;
goto _start;
}
else
{
lean_dec(v_fst_1060_);
lean_dec(v_lo_1054_);
return v_snd_1061_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg___boxed(lean_object* v___x_1087_, lean_object* v_n_1088_, lean_object* v_as_1089_, lean_object* v_lo_1090_, lean_object* v_hi_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(v___x_1087_, v_n_1088_, v_as_1089_, v_lo_1090_, v_hi_1091_);
lean_dec(v_hi_1091_);
lean_dec(v_n_1088_);
lean_dec_ref(v___x_1087_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg(lean_object* v_mvarIds_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v___x_1096_; lean_object* v_mctx_1097_; lean_object* v___x_1098_; lean_object* v___y_1100_; lean_object* v___y_1101_; lean_object* v___x_1104_; uint8_t v___x_1105_; 
v___x_1096_ = lean_st_ref_get(v___y_1094_);
v_mctx_1097_ = lean_ctor_get(v___x_1096_, 0);
lean_inc_ref(v_mctx_1097_);
lean_dec(v___x_1096_);
v___x_1098_ = lean_array_get_size(v_mvarIds_1093_);
v___x_1104_ = lean_unsigned_to_nat(0u);
v___x_1105_ = lean_nat_dec_eq(v___x_1098_, v___x_1104_);
if (v___x_1105_ == 0)
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___y_1109_; uint8_t v___x_1111_; 
v___x_1106_ = lean_unsigned_to_nat(1u);
v___x_1107_ = lean_nat_sub(v___x_1098_, v___x_1106_);
v___x_1111_ = lean_nat_dec_le(v___x_1104_, v___x_1107_);
if (v___x_1111_ == 0)
{
lean_inc(v___x_1107_);
v___y_1109_ = v___x_1107_;
goto v___jp_1108_;
}
else
{
v___y_1109_ = v___x_1104_;
goto v___jp_1108_;
}
v___jp_1108_:
{
uint8_t v___x_1110_; 
v___x_1110_ = lean_nat_dec_le(v___y_1109_, v___x_1107_);
if (v___x_1110_ == 0)
{
lean_dec(v___x_1107_);
lean_inc(v___y_1109_);
v___y_1100_ = v___y_1109_;
v___y_1101_ = v___y_1109_;
goto v___jp_1099_;
}
else
{
v___y_1100_ = v___y_1109_;
v___y_1101_ = v___x_1107_;
goto v___jp_1099_;
}
}
}
else
{
lean_object* v___x_1112_; 
lean_dec_ref(v_mctx_1097_);
v___x_1112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1112_, 0, v_mvarIds_1093_);
return v___x_1112_;
}
v___jp_1099_:
{
lean_object* v___x_1102_; lean_object* v___x_1103_; 
v___x_1102_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(v_mctx_1097_, v___x_1098_, v_mvarIds_1093_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
lean_dec_ref(v_mctx_1097_);
v___x_1103_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1103_, 0, v___x_1102_);
return v___x_1103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg___boxed(lean_object* v_mvarIds_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_){
_start:
{
lean_object* v_res_1116_; 
v_res_1116_ = lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg(v_mvarIds_1113_, v___y_1114_);
lean_dec(v___y_1114_);
return v_res_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2(lean_object* v_k_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_){
_start:
{
lean_object* v___x_1127_; lean_object* v_mctx_1128_; lean_object* v_mvarCounter_1129_; lean_object* v___x_1130_; 
v___x_1127_ = lean_st_ref_get(v___y_1123_);
v_mctx_1128_ = lean_ctor_get(v___x_1127_, 0);
lean_inc_ref(v_mctx_1128_);
lean_dec(v___x_1127_);
v_mvarCounter_1129_ = lean_ctor_get(v_mctx_1128_, 3);
lean_inc(v_mvarCounter_1129_);
lean_dec_ref(v_mctx_1128_);
lean_inc(v___y_1125_);
lean_inc_ref(v___y_1124_);
lean_inc(v___y_1123_);
lean_inc_ref(v___y_1122_);
lean_inc(v___y_1121_);
lean_inc_ref(v___y_1120_);
lean_inc(v___y_1119_);
lean_inc_ref(v___y_1118_);
v___x_1130_ = lean_apply_9(v_k_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_, lean_box(0));
if (lean_obj_tag(v___x_1130_) == 0)
{
lean_object* v_a_1131_; lean_object* v___x_1132_; 
v_a_1131_ = lean_ctor_get(v___x_1130_, 0);
lean_inc_n(v_a_1131_, 2);
lean_dec_ref_known(v___x_1130_, 1);
v___x_1132_ = l_Lean_Meta_getMVarsNoDelayed(v_a_1131_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_);
if (lean_obj_tag(v___x_1132_) == 0)
{
lean_object* v_a_1133_; lean_object* v___x_1134_; 
v_a_1133_ = lean_ctor_get(v___x_1132_, 0);
lean_inc(v_a_1133_);
lean_dec_ref_known(v___x_1132_, 1);
v___x_1134_ = l_Lean_Elab_Tactic_filterOldMVars___redArg(v_a_1133_, v_mvarCounter_1129_, v___y_1123_);
lean_dec(v_mvarCounter_1129_);
lean_dec(v_a_1133_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_object* v_a_1135_; lean_object* v___x_1136_; lean_object* v_a_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1145_; 
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
lean_inc(v_a_1135_);
lean_dec_ref_known(v___x_1134_, 1);
v___x_1136_ = lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg(v_a_1135_, v___y_1123_);
v_a_1137_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1145_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1145_ == 0)
{
v___x_1139_ = v___x_1136_;
v_isShared_1140_ = v_isSharedCheck_1145_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_a_1137_);
lean_dec(v___x_1136_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1145_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1141_; lean_object* v___x_1143_; 
v___x_1141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1141_, 0, v_a_1131_);
lean_ctor_set(v___x_1141_, 1, v_a_1137_);
if (v_isShared_1140_ == 0)
{
lean_ctor_set(v___x_1139_, 0, v___x_1141_);
v___x_1143_ = v___x_1139_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v___x_1141_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
else
{
lean_object* v_a_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1153_; 
lean_dec(v_a_1131_);
v_a_1146_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1153_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1153_ == 0)
{
v___x_1148_ = v___x_1134_;
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_a_1146_);
lean_dec(v___x_1134_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v___x_1151_; 
if (v_isShared_1149_ == 0)
{
v___x_1151_ = v___x_1148_;
goto v_reusejp_1150_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v_a_1146_);
v___x_1151_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1150_;
}
v_reusejp_1150_:
{
return v___x_1151_;
}
}
}
}
else
{
lean_object* v_a_1154_; lean_object* v___x_1156_; uint8_t v_isShared_1157_; uint8_t v_isSharedCheck_1161_; 
lean_dec(v_a_1131_);
lean_dec(v_mvarCounter_1129_);
v_a_1154_ = lean_ctor_get(v___x_1132_, 0);
v_isSharedCheck_1161_ = !lean_is_exclusive(v___x_1132_);
if (v_isSharedCheck_1161_ == 0)
{
v___x_1156_ = v___x_1132_;
v_isShared_1157_ = v_isSharedCheck_1161_;
goto v_resetjp_1155_;
}
else
{
lean_inc(v_a_1154_);
lean_dec(v___x_1132_);
v___x_1156_ = lean_box(0);
v_isShared_1157_ = v_isSharedCheck_1161_;
goto v_resetjp_1155_;
}
v_resetjp_1155_:
{
lean_object* v___x_1159_; 
if (v_isShared_1157_ == 0)
{
v___x_1159_ = v___x_1156_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1160_; 
v_reuseFailAlloc_1160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1160_, 0, v_a_1154_);
v___x_1159_ = v_reuseFailAlloc_1160_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
return v___x_1159_;
}
}
}
}
else
{
lean_object* v_a_1162_; lean_object* v___x_1164_; uint8_t v_isShared_1165_; uint8_t v_isSharedCheck_1169_; 
lean_dec(v_mvarCounter_1129_);
v_a_1162_ = lean_ctor_get(v___x_1130_, 0);
v_isSharedCheck_1169_ = !lean_is_exclusive(v___x_1130_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1164_ = v___x_1130_;
v_isShared_1165_ = v_isSharedCheck_1169_;
goto v_resetjp_1163_;
}
else
{
lean_inc(v_a_1162_);
lean_dec(v___x_1130_);
v___x_1164_ = lean_box(0);
v_isShared_1165_ = v_isSharedCheck_1169_;
goto v_resetjp_1163_;
}
v_resetjp_1163_:
{
lean_object* v___x_1167_; 
if (v_isShared_1165_ == 0)
{
v___x_1167_ = v___x_1164_;
goto v_reusejp_1166_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v_a_1162_);
v___x_1167_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1166_;
}
v_reusejp_1166_:
{
return v___x_1167_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2___boxed(lean_object* v_k_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_){
_start:
{
lean_object* v_res_1180_; 
v_res_1180_ = lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2(v_k_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
lean_dec(v___y_1176_);
lean_dec_ref(v___y_1175_);
lean_dec(v___y_1174_);
lean_dec_ref(v___y_1173_);
lean_dec(v___y_1172_);
lean_dec_ref(v___y_1171_);
return v_res_1180_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg(lean_object* v_keys_1181_, lean_object* v_i_1182_, lean_object* v_k_1183_){
_start:
{
lean_object* v___x_1184_; uint8_t v___x_1185_; 
v___x_1184_ = lean_array_get_size(v_keys_1181_);
v___x_1185_ = lean_nat_dec_lt(v_i_1182_, v___x_1184_);
if (v___x_1185_ == 0)
{
lean_dec(v_i_1182_);
return v___x_1185_;
}
else
{
lean_object* v_k_x27_1186_; uint8_t v___x_1187_; 
v_k_x27_1186_ = lean_array_fget_borrowed(v_keys_1181_, v_i_1182_);
v___x_1187_ = l_Lean_instBEqMVarId_beq(v_k_1183_, v_k_x27_1186_);
if (v___x_1187_ == 0)
{
lean_object* v___x_1188_; lean_object* v___x_1189_; 
v___x_1188_ = lean_unsigned_to_nat(1u);
v___x_1189_ = lean_nat_add(v_i_1182_, v___x_1188_);
lean_dec(v_i_1182_);
v_i_1182_ = v___x_1189_;
goto _start;
}
else
{
lean_dec(v_i_1182_);
return v___x_1187_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg___boxed(lean_object* v_keys_1191_, lean_object* v_i_1192_, lean_object* v_k_1193_){
_start:
{
uint8_t v_res_1194_; lean_object* v_r_1195_; 
v_res_1194_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg(v_keys_1191_, v_i_1192_, v_k_1193_);
lean_dec(v_k_1193_);
lean_dec_ref(v_keys_1191_);
v_r_1195_ = lean_box(v_res_1194_);
return v_r_1195_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg(lean_object* v_x_1196_, size_t v_x_1197_, lean_object* v_x_1198_){
_start:
{
if (lean_obj_tag(v_x_1196_) == 0)
{
lean_object* v_es_1199_; lean_object* v___x_1200_; size_t v___x_1201_; size_t v___x_1202_; lean_object* v_j_1203_; lean_object* v___x_1204_; 
v_es_1199_ = lean_ctor_get(v_x_1196_, 0);
v___x_1200_ = lean_box(2);
v___x_1201_ = ((size_t)31ULL);
v___x_1202_ = lean_usize_land(v_x_1197_, v___x_1201_);
v_j_1203_ = lean_usize_to_nat(v___x_1202_);
v___x_1204_ = lean_array_get_borrowed(v___x_1200_, v_es_1199_, v_j_1203_);
lean_dec(v_j_1203_);
switch(lean_obj_tag(v___x_1204_))
{
case 0:
{
lean_object* v_key_1205_; uint8_t v___x_1206_; 
v_key_1205_ = lean_ctor_get(v___x_1204_, 0);
v___x_1206_ = l_Lean_instBEqMVarId_beq(v_x_1198_, v_key_1205_);
return v___x_1206_;
}
case 1:
{
lean_object* v_node_1207_; size_t v___x_1208_; size_t v___x_1209_; 
v_node_1207_ = lean_ctor_get(v___x_1204_, 0);
v___x_1208_ = ((size_t)5ULL);
v___x_1209_ = lean_usize_shift_right(v_x_1197_, v___x_1208_);
v_x_1196_ = v_node_1207_;
v_x_1197_ = v___x_1209_;
goto _start;
}
default: 
{
uint8_t v___x_1211_; 
v___x_1211_ = 0;
return v___x_1211_;
}
}
}
else
{
lean_object* v_ks_1212_; lean_object* v___x_1213_; uint8_t v___x_1214_; 
v_ks_1212_ = lean_ctor_get(v_x_1196_, 0);
v___x_1213_ = lean_unsigned_to_nat(0u);
v___x_1214_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg(v_ks_1212_, v___x_1213_, v_x_1198_);
return v___x_1214_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_x_1215_, lean_object* v_x_1216_, lean_object* v_x_1217_){
_start:
{
size_t v_x_28238__boxed_1218_; uint8_t v_res_1219_; lean_object* v_r_1220_; 
v_x_28238__boxed_1218_ = lean_unbox_usize(v_x_1216_);
lean_dec(v_x_1216_);
v_res_1219_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg(v_x_1215_, v_x_28238__boxed_1218_, v_x_1217_);
lean_dec(v_x_1217_);
lean_dec_ref(v_x_1215_);
v_r_1220_ = lean_box(v_res_1219_);
return v_r_1220_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg(lean_object* v_x_1221_, lean_object* v_x_1222_){
_start:
{
uint64_t v___x_1223_; size_t v___x_1224_; uint8_t v___x_1225_; 
v___x_1223_ = l_Lean_instHashableMVarId_hash(v_x_1222_);
v___x_1224_ = lean_uint64_to_usize(v___x_1223_);
v___x_1225_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg(v_x_1221_, v___x_1224_, v_x_1222_);
return v___x_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg___boxed(lean_object* v_x_1226_, lean_object* v_x_1227_){
_start:
{
uint8_t v_res_1228_; lean_object* v_r_1229_; 
v_res_1228_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg(v_x_1226_, v_x_1227_);
lean_dec(v_x_1227_);
lean_dec_ref(v_x_1226_);
v_r_1229_ = lean_box(v_res_1228_);
return v_r_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg(lean_object* v_mvarId_1230_, lean_object* v___y_1231_){
_start:
{
lean_object* v___x_1233_; lean_object* v_mctx_1234_; lean_object* v_eAssignment_1235_; uint8_t v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1233_ = lean_st_ref_get(v___y_1231_);
v_mctx_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc_ref(v_mctx_1234_);
lean_dec(v___x_1233_);
v_eAssignment_1235_ = lean_ctor_get(v_mctx_1234_, 8);
lean_inc_ref(v_eAssignment_1235_);
lean_dec_ref(v_mctx_1234_);
v___x_1236_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg(v_eAssignment_1235_, v_mvarId_1230_);
lean_dec_ref(v_eAssignment_1235_);
v___x_1237_ = lean_box(v___x_1236_);
v___x_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1238_, 0, v___x_1237_);
return v___x_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg___boxed(lean_object* v_mvarId_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg(v_mvarId_1239_, v___y_1240_);
lean_dec(v___y_1240_);
lean_dec(v_mvarId_1239_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4(lean_object* v_as_1243_, size_t v_i_1244_, size_t v_stop_1245_, lean_object* v_b_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_){
_start:
{
lean_object* v_a_1257_; uint8_t v___x_1261_; 
v___x_1261_ = lean_usize_dec_eq(v_i_1244_, v_stop_1245_);
if (v___x_1261_ == 0)
{
lean_object* v___x_1262_; lean_object* v___x_1265_; 
v___x_1262_ = lean_array_uget_borrowed(v_as_1243_, v_i_1244_);
v___x_1265_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg(v___x_1262_, v___y_1252_);
if (lean_obj_tag(v___x_1265_) == 0)
{
lean_object* v_a_1266_; uint8_t v___x_1267_; 
v_a_1266_ = lean_ctor_get(v___x_1265_, 0);
lean_inc(v_a_1266_);
lean_dec_ref_known(v___x_1265_, 1);
v___x_1267_ = lean_unbox(v_a_1266_);
lean_dec(v_a_1266_);
if (v___x_1267_ == 0)
{
goto v___jp_1263_;
}
else
{
v_a_1257_ = v_b_1246_;
goto v___jp_1256_;
}
}
else
{
if (lean_obj_tag(v___x_1265_) == 0)
{
lean_object* v_a_1268_; uint8_t v___x_1269_; 
v_a_1268_ = lean_ctor_get(v___x_1265_, 0);
lean_inc(v_a_1268_);
lean_dec_ref_known(v___x_1265_, 1);
v___x_1269_ = lean_unbox(v_a_1268_);
lean_dec(v_a_1268_);
if (v___x_1269_ == 0)
{
v_a_1257_ = v_b_1246_;
goto v___jp_1256_;
}
else
{
goto v___jp_1263_;
}
}
else
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1277_; 
lean_dec_ref(v_b_1246_);
v_a_1270_ = lean_ctor_get(v___x_1265_, 0);
v_isSharedCheck_1277_ = !lean_is_exclusive(v___x_1265_);
if (v_isSharedCheck_1277_ == 0)
{
v___x_1272_ = v___x_1265_;
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1265_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1275_; 
if (v_isShared_1273_ == 0)
{
v___x_1275_ = v___x_1272_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v_a_1270_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
v___jp_1263_:
{
lean_object* v___x_1264_; 
lean_inc(v___x_1262_);
v___x_1264_ = lean_array_push(v_b_1246_, v___x_1262_);
v_a_1257_ = v___x_1264_;
goto v___jp_1256_;
}
}
else
{
lean_object* v___x_1278_; 
v___x_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1278_, 0, v_b_1246_);
return v___x_1278_;
}
v___jp_1256_:
{
size_t v___x_1258_; size_t v___x_1259_; 
v___x_1258_ = ((size_t)1ULL);
v___x_1259_ = lean_usize_add(v_i_1244_, v___x_1258_);
v_i_1244_ = v___x_1259_;
v_b_1246_ = v_a_1257_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4___boxed(lean_object* v_as_1279_, lean_object* v_i_1280_, lean_object* v_stop_1281_, lean_object* v_b_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_){
_start:
{
size_t v_i_boxed_1292_; size_t v_stop_boxed_1293_; lean_object* v_res_1294_; 
v_i_boxed_1292_ = lean_unbox_usize(v_i_1280_);
lean_dec(v_i_1280_);
v_stop_boxed_1293_ = lean_unbox_usize(v_stop_1281_);
lean_dec(v_stop_1281_);
v_res_1294_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4(v_as_1279_, v_i_boxed_1292_, v_stop_boxed_1293_, v_b_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_, v___y_1290_);
lean_dec(v___y_1290_);
lean_dec_ref(v___y_1289_);
lean_dec(v___y_1288_);
lean_dec_ref(v___y_1287_);
lean_dec(v___y_1286_);
lean_dec_ref(v___y_1285_);
lean_dec(v___y_1284_);
lean_dec_ref(v___y_1283_);
lean_dec_ref(v_as_1279_);
return v_res_1294_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0(uint8_t v___y_1302_, uint8_t v_suppressElabErrors_1303_, lean_object* v_x_1304_){
_start:
{
if (lean_obj_tag(v_x_1304_) == 1)
{
lean_object* v_pre_1305_; 
v_pre_1305_ = lean_ctor_get(v_x_1304_, 0);
switch(lean_obj_tag(v_pre_1305_))
{
case 1:
{
lean_object* v_pre_1306_; 
v_pre_1306_ = lean_ctor_get(v_pre_1305_, 0);
switch(lean_obj_tag(v_pre_1306_))
{
case 0:
{
lean_object* v_str_1307_; lean_object* v_str_1308_; lean_object* v___x_1309_; uint8_t v___x_1310_; 
v_str_1307_ = lean_ctor_get(v_x_1304_, 1);
v_str_1308_ = lean_ctor_get(v_pre_1305_, 1);
v___x_1309_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__0));
v___x_1310_ = lean_string_dec_eq(v_str_1308_, v___x_1309_);
if (v___x_1310_ == 0)
{
lean_object* v___x_1311_; uint8_t v___x_1312_; 
v___x_1311_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__1));
v___x_1312_ = lean_string_dec_eq(v_str_1308_, v___x_1311_);
if (v___x_1312_ == 0)
{
return v___y_1302_;
}
else
{
lean_object* v___x_1313_; uint8_t v___x_1314_; 
v___x_1313_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__1));
v___x_1314_ = lean_string_dec_eq(v_str_1307_, v___x_1313_);
if (v___x_1314_ == 0)
{
return v___y_1302_;
}
else
{
return v_suppressElabErrors_1303_;
}
}
}
else
{
lean_object* v___x_1315_; uint8_t v___x_1316_; 
v___x_1315_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__2));
v___x_1316_ = lean_string_dec_eq(v_str_1307_, v___x_1315_);
if (v___x_1316_ == 0)
{
return v___y_1302_;
}
else
{
return v_suppressElabErrors_1303_;
}
}
}
case 1:
{
lean_object* v_pre_1317_; 
v_pre_1317_ = lean_ctor_get(v_pre_1306_, 0);
if (lean_obj_tag(v_pre_1317_) == 0)
{
lean_object* v_str_1318_; lean_object* v_str_1319_; lean_object* v_str_1320_; lean_object* v___x_1321_; uint8_t v___x_1322_; 
v_str_1318_ = lean_ctor_get(v_x_1304_, 1);
v_str_1319_ = lean_ctor_get(v_pre_1305_, 1);
v_str_1320_ = lean_ctor_get(v_pre_1306_, 1);
v___x_1321_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__3));
v___x_1322_ = lean_string_dec_eq(v_str_1320_, v___x_1321_);
if (v___x_1322_ == 0)
{
return v___y_1302_;
}
else
{
lean_object* v___x_1323_; uint8_t v___x_1324_; 
v___x_1323_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__4));
v___x_1324_ = lean_string_dec_eq(v_str_1319_, v___x_1323_);
if (v___x_1324_ == 0)
{
return v___y_1302_;
}
else
{
lean_object* v___x_1325_; uint8_t v___x_1326_; 
v___x_1325_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__5));
v___x_1326_ = lean_string_dec_eq(v_str_1318_, v___x_1325_);
if (v___x_1326_ == 0)
{
return v___y_1302_;
}
else
{
return v_suppressElabErrors_1303_;
}
}
}
}
else
{
return v___y_1302_;
}
}
default: 
{
return v___y_1302_;
}
}
}
case 0:
{
lean_object* v_str_1327_; lean_object* v___x_1328_; uint8_t v___x_1329_; 
v_str_1327_ = lean_ctor_get(v_x_1304_, 1);
v___x_1328_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___closed__6));
v___x_1329_ = lean_string_dec_eq(v_str_1327_, v___x_1328_);
if (v___x_1329_ == 0)
{
return v___y_1302_;
}
else
{
return v_suppressElabErrors_1303_;
}
}
default: 
{
return v___y_1302_;
}
}
}
else
{
return v___y_1302_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v___y_1330_, lean_object* v_suppressElabErrors_1331_, lean_object* v_x_1332_){
_start:
{
uint8_t v___y_28392__boxed_1333_; uint8_t v_suppressElabErrors_boxed_1334_; uint8_t v_res_1335_; lean_object* v_r_1336_; 
v___y_28392__boxed_1333_ = lean_unbox(v___y_1330_);
v_suppressElabErrors_boxed_1334_ = lean_unbox(v_suppressElabErrors_1331_);
v_res_1335_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0(v___y_28392__boxed_1333_, v_suppressElabErrors_boxed_1334_, v_x_1332_);
lean_dec(v_x_1332_);
v_r_1336_ = lean_box(v_res_1335_);
return v_r_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10(lean_object* v_msgData_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v___x_1343_; lean_object* v_env_1344_; lean_object* v___x_1345_; lean_object* v_mctx_1346_; lean_object* v_lctx_1347_; lean_object* v_options_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; 
v___x_1343_ = lean_st_ref_get(v___y_1341_);
v_env_1344_ = lean_ctor_get(v___x_1343_, 0);
lean_inc_ref(v_env_1344_);
lean_dec(v___x_1343_);
v___x_1345_ = lean_st_ref_get(v___y_1339_);
v_mctx_1346_ = lean_ctor_get(v___x_1345_, 0);
lean_inc_ref(v_mctx_1346_);
lean_dec(v___x_1345_);
v_lctx_1347_ = lean_ctor_get(v___y_1338_, 2);
v_options_1348_ = lean_ctor_get(v___y_1340_, 2);
lean_inc_ref(v_options_1348_);
lean_inc_ref(v_lctx_1347_);
v___x_1349_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1349_, 0, v_env_1344_);
lean_ctor_set(v___x_1349_, 1, v_mctx_1346_);
lean_ctor_set(v___x_1349_, 2, v_lctx_1347_);
lean_ctor_set(v___x_1349_, 3, v_options_1348_);
v___x_1350_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1350_, 0, v___x_1349_);
lean_ctor_set(v___x_1350_, 1, v_msgData_1337_);
v___x_1351_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1351_, 0, v___x_1350_);
return v___x_1351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10___boxed(lean_object* v_msgData_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_){
_start:
{
lean_object* v_res_1358_; 
v_res_1358_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10(v_msgData_1352_, v___y_1353_, v___y_1354_, v___y_1355_, v___y_1356_);
lean_dec(v___y_1356_);
lean_dec_ref(v___y_1355_);
lean_dec(v___y_1354_);
lean_dec_ref(v___y_1353_);
return v_res_1358_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11(lean_object* v_opts_1359_, lean_object* v_opt_1360_){
_start:
{
lean_object* v_name_1361_; lean_object* v_defValue_1362_; lean_object* v_map_1363_; lean_object* v___x_1364_; 
v_name_1361_ = lean_ctor_get(v_opt_1360_, 0);
v_defValue_1362_ = lean_ctor_get(v_opt_1360_, 1);
v_map_1363_ = lean_ctor_get(v_opts_1359_, 0);
v___x_1364_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1363_, v_name_1361_);
if (lean_obj_tag(v___x_1364_) == 0)
{
uint8_t v___x_1365_; 
v___x_1365_ = lean_unbox(v_defValue_1362_);
return v___x_1365_;
}
else
{
lean_object* v_val_1366_; 
v_val_1366_ = lean_ctor_get(v___x_1364_, 0);
lean_inc(v_val_1366_);
lean_dec_ref_known(v___x_1364_, 1);
if (lean_obj_tag(v_val_1366_) == 1)
{
uint8_t v_v_1367_; 
v_v_1367_ = lean_ctor_get_uint8(v_val_1366_, 0);
lean_dec_ref_known(v_val_1366_, 0);
return v_v_1367_;
}
else
{
uint8_t v___x_1368_; 
lean_dec(v_val_1366_);
v___x_1368_ = lean_unbox(v_defValue_1362_);
return v___x_1368_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11___boxed(lean_object* v_opts_1369_, lean_object* v_opt_1370_){
_start:
{
uint8_t v_res_1371_; lean_object* v_r_1372_; 
v_res_1371_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11(v_opts_1369_, v_opt_1370_);
lean_dec_ref(v_opt_1370_);
lean_dec_ref(v_opts_1369_);
v_r_1372_ = lean_box(v_res_1371_);
return v_r_1372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg(lean_object* v_ref_1374_, lean_object* v_msgData_1375_, uint8_t v_severity_1376_, uint8_t v_isSilent_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_){
_start:
{
lean_object* v___y_1384_; lean_object* v___y_1385_; uint8_t v___y_1386_; uint8_t v___y_1387_; lean_object* v___y_1388_; lean_object* v___y_1389_; lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1392_; lean_object* v___y_1420_; lean_object* v___y_1421_; uint8_t v___y_1422_; uint8_t v___y_1423_; uint8_t v___y_1424_; lean_object* v___y_1425_; lean_object* v___y_1426_; lean_object* v___y_1427_; lean_object* v___y_1445_; lean_object* v___y_1446_; uint8_t v___y_1447_; uint8_t v___y_1448_; lean_object* v___y_1449_; uint8_t v___y_1450_; lean_object* v___y_1451_; lean_object* v___y_1452_; lean_object* v___y_1456_; lean_object* v___y_1457_; lean_object* v___y_1458_; uint8_t v___y_1459_; uint8_t v___y_1460_; lean_object* v___y_1461_; uint8_t v___y_1462_; uint8_t v___x_1467_; lean_object* v___y_1469_; lean_object* v___y_1470_; lean_object* v___y_1471_; uint8_t v___y_1472_; lean_object* v___y_1473_; uint8_t v___y_1474_; uint8_t v___y_1475_; uint8_t v___y_1477_; uint8_t v___x_1492_; 
v___x_1467_ = 2;
v___x_1492_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1376_, v___x_1467_);
if (v___x_1492_ == 0)
{
v___y_1477_ = v___x_1492_;
goto v___jp_1476_;
}
else
{
uint8_t v___x_1493_; 
lean_inc_ref(v_msgData_1375_);
v___x_1493_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1375_);
v___y_1477_ = v___x_1493_;
goto v___jp_1476_;
}
v___jp_1383_:
{
lean_object* v___x_1393_; lean_object* v_currNamespace_1394_; lean_object* v_openDecls_1395_; lean_object* v_env_1396_; lean_object* v_nextMacroScope_1397_; lean_object* v_ngen_1398_; lean_object* v_auxDeclNGen_1399_; lean_object* v_traceState_1400_; lean_object* v_cache_1401_; lean_object* v_messages_1402_; lean_object* v_infoState_1403_; lean_object* v_snapshotTasks_1404_; lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1418_; 
v___x_1393_ = lean_st_ref_take(v___y_1392_);
v_currNamespace_1394_ = lean_ctor_get(v___y_1391_, 6);
v_openDecls_1395_ = lean_ctor_get(v___y_1391_, 7);
v_env_1396_ = lean_ctor_get(v___x_1393_, 0);
v_nextMacroScope_1397_ = lean_ctor_get(v___x_1393_, 1);
v_ngen_1398_ = lean_ctor_get(v___x_1393_, 2);
v_auxDeclNGen_1399_ = lean_ctor_get(v___x_1393_, 3);
v_traceState_1400_ = lean_ctor_get(v___x_1393_, 4);
v_cache_1401_ = lean_ctor_get(v___x_1393_, 5);
v_messages_1402_ = lean_ctor_get(v___x_1393_, 6);
v_infoState_1403_ = lean_ctor_get(v___x_1393_, 7);
v_snapshotTasks_1404_ = lean_ctor_get(v___x_1393_, 8);
v_isSharedCheck_1418_ = !lean_is_exclusive(v___x_1393_);
if (v_isSharedCheck_1418_ == 0)
{
v___x_1406_ = v___x_1393_;
v_isShared_1407_ = v_isSharedCheck_1418_;
goto v_resetjp_1405_;
}
else
{
lean_inc(v_snapshotTasks_1404_);
lean_inc(v_infoState_1403_);
lean_inc(v_messages_1402_);
lean_inc(v_cache_1401_);
lean_inc(v_traceState_1400_);
lean_inc(v_auxDeclNGen_1399_);
lean_inc(v_ngen_1398_);
lean_inc(v_nextMacroScope_1397_);
lean_inc(v_env_1396_);
lean_dec(v___x_1393_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1418_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1413_; 
lean_inc(v_openDecls_1395_);
lean_inc(v_currNamespace_1394_);
v___x_1408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1408_, 0, v_currNamespace_1394_);
lean_ctor_set(v___x_1408_, 1, v_openDecls_1395_);
v___x_1409_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1409_, 0, v___x_1408_);
lean_ctor_set(v___x_1409_, 1, v___y_1390_);
lean_inc_ref(v___y_1384_);
lean_inc_ref(v___y_1385_);
v___x_1410_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1410_, 0, v___y_1385_);
lean_ctor_set(v___x_1410_, 1, v___y_1389_);
lean_ctor_set(v___x_1410_, 2, v___y_1388_);
lean_ctor_set(v___x_1410_, 3, v___y_1384_);
lean_ctor_set(v___x_1410_, 4, v___x_1409_);
lean_ctor_set_uint8(v___x_1410_, sizeof(void*)*5, v___y_1386_);
lean_ctor_set_uint8(v___x_1410_, sizeof(void*)*5 + 1, v___y_1387_);
lean_ctor_set_uint8(v___x_1410_, sizeof(void*)*5 + 2, v_isSilent_1377_);
v___x_1411_ = l_Lean_MessageLog_add(v___x_1410_, v_messages_1402_);
if (v_isShared_1407_ == 0)
{
lean_ctor_set(v___x_1406_, 6, v___x_1411_);
v___x_1413_ = v___x_1406_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v_env_1396_);
lean_ctor_set(v_reuseFailAlloc_1417_, 1, v_nextMacroScope_1397_);
lean_ctor_set(v_reuseFailAlloc_1417_, 2, v_ngen_1398_);
lean_ctor_set(v_reuseFailAlloc_1417_, 3, v_auxDeclNGen_1399_);
lean_ctor_set(v_reuseFailAlloc_1417_, 4, v_traceState_1400_);
lean_ctor_set(v_reuseFailAlloc_1417_, 5, v_cache_1401_);
lean_ctor_set(v_reuseFailAlloc_1417_, 6, v___x_1411_);
lean_ctor_set(v_reuseFailAlloc_1417_, 7, v_infoState_1403_);
lean_ctor_set(v_reuseFailAlloc_1417_, 8, v_snapshotTasks_1404_);
v___x_1413_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; 
v___x_1414_ = lean_st_ref_set(v___y_1392_, v___x_1413_);
v___x_1415_ = lean_box(0);
v___x_1416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1416_, 0, v___x_1415_);
return v___x_1416_;
}
}
}
v___jp_1419_:
{
lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1443_; 
v___x_1428_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1375_);
v___x_1429_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__10(v___x_1428_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
v_a_1430_ = lean_ctor_get(v___x_1429_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1429_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1432_ = v___x_1429_;
v_isShared_1433_ = v_isSharedCheck_1443_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1429_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1443_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; 
lean_inc_ref_n(v___y_1425_, 2);
v___x_1434_ = l_Lean_FileMap_toPosition(v___y_1425_, v___y_1426_);
lean_dec(v___y_1426_);
v___x_1435_ = l_Lean_FileMap_toPosition(v___y_1425_, v___y_1427_);
lean_dec(v___y_1427_);
v___x_1436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1436_, 0, v___x_1435_);
v___x_1437_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___closed__0));
if (v___y_1424_ == 0)
{
lean_del_object(v___x_1432_);
lean_dec_ref(v___y_1420_);
v___y_1384_ = v___x_1437_;
v___y_1385_ = v___y_1421_;
v___y_1386_ = v___y_1422_;
v___y_1387_ = v___y_1423_;
v___y_1388_ = v___x_1436_;
v___y_1389_ = v___x_1434_;
v___y_1390_ = v_a_1430_;
v___y_1391_ = v___y_1380_;
v___y_1392_ = v___y_1381_;
goto v___jp_1383_;
}
else
{
uint8_t v___x_1438_; 
lean_inc(v_a_1430_);
v___x_1438_ = l_Lean_MessageData_hasTag(v___y_1420_, v_a_1430_);
if (v___x_1438_ == 0)
{
lean_object* v___x_1439_; lean_object* v___x_1441_; 
lean_dec_ref_known(v___x_1436_, 1);
lean_dec_ref(v___x_1434_);
lean_dec(v_a_1430_);
v___x_1439_ = lean_box(0);
if (v_isShared_1433_ == 0)
{
lean_ctor_set(v___x_1432_, 0, v___x_1439_);
v___x_1441_ = v___x_1432_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v___x_1439_);
v___x_1441_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
return v___x_1441_;
}
}
else
{
lean_del_object(v___x_1432_);
v___y_1384_ = v___x_1437_;
v___y_1385_ = v___y_1421_;
v___y_1386_ = v___y_1422_;
v___y_1387_ = v___y_1423_;
v___y_1388_ = v___x_1436_;
v___y_1389_ = v___x_1434_;
v___y_1390_ = v_a_1430_;
v___y_1391_ = v___y_1380_;
v___y_1392_ = v___y_1381_;
goto v___jp_1383_;
}
}
}
}
v___jp_1444_:
{
lean_object* v___x_1453_; 
v___x_1453_ = l_Lean_Syntax_getTailPos_x3f(v___y_1451_, v___y_1447_);
lean_dec(v___y_1451_);
if (lean_obj_tag(v___x_1453_) == 0)
{
lean_inc(v___y_1452_);
v___y_1420_ = v___y_1445_;
v___y_1421_ = v___y_1446_;
v___y_1422_ = v___y_1447_;
v___y_1423_ = v___y_1448_;
v___y_1424_ = v___y_1450_;
v___y_1425_ = v___y_1449_;
v___y_1426_ = v___y_1452_;
v___y_1427_ = v___y_1452_;
goto v___jp_1419_;
}
else
{
lean_object* v_val_1454_; 
v_val_1454_ = lean_ctor_get(v___x_1453_, 0);
lean_inc(v_val_1454_);
lean_dec_ref_known(v___x_1453_, 1);
v___y_1420_ = v___y_1445_;
v___y_1421_ = v___y_1446_;
v___y_1422_ = v___y_1447_;
v___y_1423_ = v___y_1448_;
v___y_1424_ = v___y_1450_;
v___y_1425_ = v___y_1449_;
v___y_1426_ = v___y_1452_;
v___y_1427_ = v_val_1454_;
goto v___jp_1419_;
}
}
v___jp_1455_:
{
lean_object* v_ref_1463_; lean_object* v___x_1464_; 
v_ref_1463_ = l_Lean_replaceRef(v_ref_1374_, v___y_1458_);
v___x_1464_ = l_Lean_Syntax_getPos_x3f(v_ref_1463_, v___y_1459_);
if (lean_obj_tag(v___x_1464_) == 0)
{
lean_object* v___x_1465_; 
v___x_1465_ = lean_unsigned_to_nat(0u);
v___y_1445_ = v___y_1456_;
v___y_1446_ = v___y_1457_;
v___y_1447_ = v___y_1459_;
v___y_1448_ = v___y_1462_;
v___y_1449_ = v___y_1461_;
v___y_1450_ = v___y_1460_;
v___y_1451_ = v_ref_1463_;
v___y_1452_ = v___x_1465_;
goto v___jp_1444_;
}
else
{
lean_object* v_val_1466_; 
v_val_1466_ = lean_ctor_get(v___x_1464_, 0);
lean_inc(v_val_1466_);
lean_dec_ref_known(v___x_1464_, 1);
v___y_1445_ = v___y_1456_;
v___y_1446_ = v___y_1457_;
v___y_1447_ = v___y_1459_;
v___y_1448_ = v___y_1462_;
v___y_1449_ = v___y_1461_;
v___y_1450_ = v___y_1460_;
v___y_1451_ = v_ref_1463_;
v___y_1452_ = v_val_1466_;
goto v___jp_1444_;
}
}
v___jp_1468_:
{
if (v___y_1475_ == 0)
{
v___y_1456_ = v___y_1473_;
v___y_1457_ = v___y_1470_;
v___y_1458_ = v___y_1469_;
v___y_1459_ = v___y_1474_;
v___y_1460_ = v___y_1472_;
v___y_1461_ = v___y_1471_;
v___y_1462_ = v_severity_1376_;
goto v___jp_1455_;
}
else
{
v___y_1456_ = v___y_1473_;
v___y_1457_ = v___y_1470_;
v___y_1458_ = v___y_1469_;
v___y_1459_ = v___y_1474_;
v___y_1460_ = v___y_1472_;
v___y_1461_ = v___y_1471_;
v___y_1462_ = v___x_1467_;
goto v___jp_1455_;
}
}
v___jp_1476_:
{
if (v___y_1477_ == 0)
{
lean_object* v_fileName_1478_; lean_object* v_fileMap_1479_; lean_object* v_options_1480_; lean_object* v_ref_1481_; uint8_t v_suppressElabErrors_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___f_1485_; uint8_t v___x_1486_; uint8_t v___x_1487_; 
v_fileName_1478_ = lean_ctor_get(v___y_1380_, 0);
v_fileMap_1479_ = lean_ctor_get(v___y_1380_, 1);
v_options_1480_ = lean_ctor_get(v___y_1380_, 2);
v_ref_1481_ = lean_ctor_get(v___y_1380_, 5);
v_suppressElabErrors_1482_ = lean_ctor_get_uint8(v___y_1380_, sizeof(void*)*14 + 1);
v___x_1483_ = lean_box(v___y_1477_);
v___x_1484_ = lean_box(v_suppressElabErrors_1482_);
v___f_1485_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1485_, 0, v___x_1483_);
lean_closure_set(v___f_1485_, 1, v___x_1484_);
v___x_1486_ = 1;
v___x_1487_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1376_, v___x_1486_);
if (v___x_1487_ == 0)
{
v___y_1469_ = v_ref_1481_;
v___y_1470_ = v_fileName_1478_;
v___y_1471_ = v_fileMap_1479_;
v___y_1472_ = v_suppressElabErrors_1482_;
v___y_1473_ = v___f_1485_;
v___y_1474_ = v___y_1477_;
v___y_1475_ = v___x_1487_;
goto v___jp_1468_;
}
else
{
lean_object* v___x_1488_; uint8_t v___x_1489_; 
v___x_1488_ = l_Lean_warningAsError;
v___x_1489_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5_spec__11(v_options_1480_, v___x_1488_);
v___y_1469_ = v_ref_1481_;
v___y_1470_ = v_fileName_1478_;
v___y_1471_ = v_fileMap_1479_;
v___y_1472_ = v_suppressElabErrors_1482_;
v___y_1473_ = v___f_1485_;
v___y_1474_ = v___y_1477_;
v___y_1475_ = v___x_1489_;
goto v___jp_1468_;
}
}
else
{
lean_object* v___x_1490_; lean_object* v___x_1491_; 
lean_dec_ref(v_msgData_1375_);
v___x_1490_ = lean_box(0);
v___x_1491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1491_, 0, v___x_1490_);
return v___x_1491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg___boxed(lean_object* v_ref_1494_, lean_object* v_msgData_1495_, lean_object* v_severity_1496_, lean_object* v_isSilent_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_){
_start:
{
uint8_t v_severity_boxed_1503_; uint8_t v_isSilent_boxed_1504_; lean_object* v_res_1505_; 
v_severity_boxed_1503_ = lean_unbox(v_severity_1496_);
v_isSilent_boxed_1504_ = lean_unbox(v_isSilent_1497_);
v_res_1505_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg(v_ref_1494_, v_msgData_1495_, v_severity_boxed_1503_, v_isSilent_boxed_1504_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
lean_dec(v_ref_1494_);
return v_res_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3(lean_object* v_ref_1506_, lean_object* v_msgData_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_, lean_object* v___y_1515_){
_start:
{
uint8_t v___x_1517_; uint8_t v___x_1518_; lean_object* v___x_1519_; 
v___x_1517_ = 1;
v___x_1518_ = 0;
v___x_1519_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg(v_ref_1506_, v_msgData_1507_, v___x_1517_, v___x_1518_, v___y_1512_, v___y_1513_, v___y_1514_, v___y_1515_);
return v___x_1519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3___boxed(lean_object* v_ref_1520_, lean_object* v_msgData_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_, lean_object* v___y_1524_, lean_object* v___y_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_){
_start:
{
lean_object* v_res_1531_; 
v_res_1531_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3(v_ref_1520_, v_msgData_1521_, v___y_1522_, v___y_1523_, v___y_1524_, v___y_1525_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
lean_dec(v___y_1529_);
lean_dec_ref(v___y_1528_);
lean_dec(v___y_1527_);
lean_dec_ref(v___y_1526_);
lean_dec(v___y_1525_);
lean_dec_ref(v___y_1524_);
lean_dec(v___y_1523_);
lean_dec_ref(v___y_1522_);
lean_dec(v_ref_1520_);
return v_res_1531_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1535_; lean_object* v___x_1536_; 
v___x_1535_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__1));
v___x_1536_ = l_Lean_MessageData_ofFormat(v___x_1535_);
return v___x_1536_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1537_; lean_object* v___x_1538_; 
v___x_1537_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__2);
v___x_1538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1538_, 0, v___x_1537_);
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0(lean_object* v_goal_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; 
v___x_1549_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError___closed__1));
v___x_1550_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___closed__3);
v___x_1551_ = l_Lean_Meta_throwTacticEx___redArg(v___x_1549_, v_goal_1539_, v___x_1550_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_);
return v___x_1551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0___boxed(lean_object* v_goal_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_){
_start:
{
lean_object* v_res_1562_; 
v_res_1562_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__0(v_goal_1552_, v___y_1553_, v___y_1554_, v___y_1555_, v___y_1556_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_);
lean_dec(v___y_1560_);
lean_dec_ref(v___y_1559_);
lean_dec(v___y_1558_);
lean_dec_ref(v___y_1557_);
lean_dec(v___y_1556_);
lean_dec_ref(v___y_1555_);
lean_dec(v___y_1554_);
lean_dec_ref(v___y_1553_);
return v_res_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3(lean_object* v___f_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_){
_start:
{
lean_object* v___x_1574_; lean_object* v___x_1575_; 
v___x_1574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1574_, 0, v___y_1564_);
lean_inc(v___y_1572_);
lean_inc_ref(v___y_1571_);
lean_inc(v___y_1570_);
lean_inc_ref(v___y_1569_);
lean_inc(v___y_1568_);
lean_inc_ref(v___y_1567_);
lean_inc(v___y_1566_);
lean_inc_ref(v___y_1565_);
v___x_1575_ = lean_apply_10(v___f_1563_, v___x_1574_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_, v___y_1570_, v___y_1571_, v___y_1572_, lean_box(0));
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3___boxed(lean_object* v___f_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
lean_object* v_res_1587_; 
v_res_1587_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3(v___f_1576_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_);
lean_dec(v___y_1585_);
lean_dec_ref(v___y_1584_);
lean_dec(v___y_1583_);
lean_dec_ref(v___y_1582_);
lean_dec(v___y_1581_);
lean_dec_ref(v___y_1580_);
lean_dec(v___y_1579_);
lean_dec_ref(v___y_1578_);
return v_res_1587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1(lean_object* v___x_1588_, lean_object* v_val_1589_, lean_object* v___x_1590_, lean_object* v_head_1591_, lean_object* v_loc_1592_, uint8_t v___x_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
lean_object* v_a_1604_; lean_object* v___x_1614_; 
v___x_1614_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1595_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
if (lean_obj_tag(v___x_1614_) == 0)
{
lean_object* v_a_1615_; lean_object* v___y_1617_; 
v_a_1615_ = lean_ctor_get(v___x_1614_, 0);
lean_inc(v_a_1615_);
lean_dec_ref_known(v___x_1614_, 1);
if (lean_obj_tag(v_loc_1592_) == 0)
{
lean_object* v___x_1683_; 
lean_inc(v_a_1615_);
v___x_1683_ = l_Lean_MVarId_getType(v_a_1615_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
v___y_1617_ = v___x_1683_;
goto v___jp_1616_;
}
else
{
lean_object* v_val_1684_; lean_object* v___x_1685_; 
v_val_1684_ = lean_ctor_get(v_loc_1592_, 0);
lean_inc(v_val_1684_);
v___x_1685_ = l_Lean_FVarId_getType___redArg(v_val_1684_, v___y_1598_, v___y_1600_, v___y_1601_);
v___y_1617_ = v___x_1685_;
goto v___jp_1616_;
}
v___jp_1616_:
{
if (lean_obj_tag(v___y_1617_) == 0)
{
lean_object* v_a_1618_; lean_object* v___x_1619_; lean_object* v_a_1620_; lean_object* v___x_1621_; lean_object* v_a_1622_; lean_object* v_keyedConfig_1623_; uint8_t v_trackZetaDelta_1624_; lean_object* v_zetaDeltaSet_1625_; lean_object* v_lctx_1626_; lean_object* v_localInstances_1627_; lean_object* v_defEqCtx_x3f_1628_; lean_object* v_synthPendingDepth_1629_; lean_object* v_customCanUnfoldPredicate_x3f_1630_; uint8_t v_univApprox_1631_; uint8_t v_inTypeClassResolution_1632_; uint8_t v_cacheInferType_1633_; lean_object* v___x_1634_; uint8_t v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; 
v_a_1618_ = lean_ctor_get(v___y_1617_, 0);
lean_inc(v_a_1618_);
lean_dec_ref_known(v___y_1617_, 1);
v___x_1619_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(v_a_1618_, v___y_1599_);
v_a_1620_ = lean_ctor_get(v___x_1619_, 0);
lean_inc(v_a_1620_);
lean_dec_ref(v___x_1619_);
v___x_1621_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__6___redArg(v_val_1589_, v___y_1599_);
v_a_1622_ = lean_ctor_get(v___x_1621_, 0);
lean_inc(v_a_1622_);
lean_dec_ref(v___x_1621_);
v_keyedConfig_1623_ = lean_ctor_get(v___y_1598_, 0);
v_trackZetaDelta_1624_ = lean_ctor_get_uint8(v___y_1598_, sizeof(void*)*7);
v_zetaDeltaSet_1625_ = lean_ctor_get(v___y_1598_, 1);
v_lctx_1626_ = lean_ctor_get(v___y_1598_, 2);
v_localInstances_1627_ = lean_ctor_get(v___y_1598_, 3);
v_defEqCtx_x3f_1628_ = lean_ctor_get(v___y_1598_, 4);
v_synthPendingDepth_1629_ = lean_ctor_get(v___y_1598_, 5);
v_customCanUnfoldPredicate_x3f_1630_ = lean_ctor_get(v___y_1598_, 6);
v_univApprox_1631_ = lean_ctor_get_uint8(v___y_1598_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1632_ = lean_ctor_get_uint8(v___y_1598_, sizeof(void*)*7 + 2);
v_cacheInferType_1633_ = lean_ctor_get_uint8(v___y_1598_, sizeof(void*)*7 + 3);
v___x_1634_ = lean_box(0);
v___x_1635_ = 2;
lean_inc_ref(v_keyedConfig_1623_);
v___x_1636_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1635_, v_keyedConfig_1623_);
lean_inc(v_customCanUnfoldPredicate_x3f_1630_);
lean_inc(v_synthPendingDepth_1629_);
lean_inc(v_defEqCtx_x3f_1628_);
lean_inc_ref(v_localInstances_1627_);
lean_inc_ref(v_lctx_1626_);
lean_inc(v_zetaDeltaSet_1625_);
v___x_1637_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1637_, 0, v___x_1636_);
lean_ctor_set(v___x_1637_, 1, v_zetaDeltaSet_1625_);
lean_ctor_set(v___x_1637_, 2, v_lctx_1626_);
lean_ctor_set(v___x_1637_, 3, v_localInstances_1627_);
lean_ctor_set(v___x_1637_, 4, v_defEqCtx_x3f_1628_);
lean_ctor_set(v___x_1637_, 5, v_synthPendingDepth_1629_);
lean_ctor_set(v___x_1637_, 6, v_customCanUnfoldPredicate_x3f_1630_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*7, v_trackZetaDelta_1624_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*7 + 1, v_univApprox_1631_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1632_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*7 + 3, v_cacheInferType_1633_);
v___x_1638_ = l_Lean_Meta_kabstract(v_a_1620_, v_a_1622_, v___x_1634_, v___x_1637_, v___y_1599_, v___y_1600_, v___y_1601_);
lean_dec_ref_known(v___x_1637_, 7);
if (lean_obj_tag(v___x_1638_) == 0)
{
lean_object* v_a_1639_; uint8_t v___x_1640_; 
v_a_1639_ = lean_ctor_get(v___x_1638_, 0);
lean_inc(v_a_1639_);
lean_dec_ref_known(v___x_1638_, 1);
v___x_1640_ = l_Lean_Expr_hasLooseBVars(v_a_1639_);
if (v___x_1640_ == 0)
{
lean_object* v___x_1641_; 
lean_dec(v_a_1639_);
lean_dec(v_loc_1592_);
lean_dec(v_head_1591_);
v___x_1641_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1641_, 0, v_a_1615_);
lean_ctor_set(v___x_1641_, 1, v___x_1590_);
v_a_1604_ = v___x_1641_;
goto v___jp_1603_;
}
else
{
lean_object* v___x_1642_; lean_object* v___x_1643_; 
v___x_1642_ = l_Lean_Expr_fvar___override(v_head_1591_);
v___x_1643_ = lean_expr_instantiate1(v_a_1639_, v___x_1642_);
lean_dec_ref(v___x_1642_);
lean_dec(v_a_1639_);
if (lean_obj_tag(v_loc_1592_) == 1)
{
lean_object* v_val_1644_; lean_object* v___x_1645_; 
v_val_1644_ = lean_ctor_get(v_loc_1592_, 0);
lean_inc(v_val_1644_);
lean_dec_ref_known(v_loc_1592_, 1);
v___x_1645_ = l_Lean_MVarId_changeLocalDecl(v_a_1615_, v_val_1644_, v___x_1643_, v___x_1593_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
if (lean_obj_tag(v___x_1645_) == 0)
{
lean_object* v_a_1646_; lean_object* v___x_1647_; 
v_a_1646_ = lean_ctor_get(v___x_1645_, 0);
lean_inc(v_a_1646_);
lean_dec_ref_known(v___x_1645_, 1);
v___x_1647_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1647_, 0, v_a_1646_);
lean_ctor_set(v___x_1647_, 1, v___x_1590_);
v_a_1604_ = v___x_1647_;
goto v___jp_1603_;
}
else
{
lean_object* v_a_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1655_; 
lean_dec_ref(v___y_1598_);
lean_dec(v___x_1590_);
v_a_1648_ = lean_ctor_get(v___x_1645_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1645_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1645_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1645_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v___x_1653_; 
if (v_isShared_1651_ == 0)
{
v___x_1653_ = v___x_1650_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v_a_1648_);
v___x_1653_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
return v___x_1653_;
}
}
}
}
else
{
lean_object* v___x_1656_; 
lean_dec(v_loc_1592_);
v___x_1656_ = l_Lean_MVarId_replaceTargetDefEq(v_a_1615_, v___x_1643_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
if (lean_obj_tag(v___x_1656_) == 0)
{
lean_object* v_a_1657_; lean_object* v___x_1658_; 
v_a_1657_ = lean_ctor_get(v___x_1656_, 0);
lean_inc(v_a_1657_);
lean_dec_ref_known(v___x_1656_, 1);
v___x_1658_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1658_, 0, v_a_1657_);
lean_ctor_set(v___x_1658_, 1, v___x_1590_);
v_a_1604_ = v___x_1658_;
goto v___jp_1603_;
}
else
{
lean_object* v_a_1659_; lean_object* v___x_1661_; uint8_t v_isShared_1662_; uint8_t v_isSharedCheck_1666_; 
lean_dec_ref(v___y_1598_);
lean_dec(v___x_1590_);
v_a_1659_ = lean_ctor_get(v___x_1656_, 0);
v_isSharedCheck_1666_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1666_ == 0)
{
v___x_1661_ = v___x_1656_;
v_isShared_1662_ = v_isSharedCheck_1666_;
goto v_resetjp_1660_;
}
else
{
lean_inc(v_a_1659_);
lean_dec(v___x_1656_);
v___x_1661_ = lean_box(0);
v_isShared_1662_ = v_isSharedCheck_1666_;
goto v_resetjp_1660_;
}
v_resetjp_1660_:
{
lean_object* v___x_1664_; 
if (v_isShared_1662_ == 0)
{
v___x_1664_ = v___x_1661_;
goto v_reusejp_1663_;
}
else
{
lean_object* v_reuseFailAlloc_1665_; 
v_reuseFailAlloc_1665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1665_, 0, v_a_1659_);
v___x_1664_ = v_reuseFailAlloc_1665_;
goto v_reusejp_1663_;
}
v_reusejp_1663_:
{
return v___x_1664_;
}
}
}
}
}
}
else
{
lean_object* v_a_1667_; lean_object* v___x_1669_; uint8_t v_isShared_1670_; uint8_t v_isSharedCheck_1674_; 
lean_dec(v_a_1615_);
lean_dec_ref(v___y_1598_);
lean_dec(v_loc_1592_);
lean_dec(v_head_1591_);
lean_dec(v___x_1590_);
v_a_1667_ = lean_ctor_get(v___x_1638_, 0);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1638_);
if (v_isSharedCheck_1674_ == 0)
{
v___x_1669_ = v___x_1638_;
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
else
{
lean_inc(v_a_1667_);
lean_dec(v___x_1638_);
v___x_1669_ = lean_box(0);
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
v_resetjp_1668_:
{
lean_object* v___x_1672_; 
if (v_isShared_1670_ == 0)
{
v___x_1672_ = v___x_1669_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_a_1667_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
}
}
else
{
lean_object* v_a_1675_; lean_object* v___x_1677_; uint8_t v_isShared_1678_; uint8_t v_isSharedCheck_1682_; 
lean_dec(v_a_1615_);
lean_dec_ref(v___y_1598_);
lean_dec(v_loc_1592_);
lean_dec(v_head_1591_);
lean_dec(v___x_1590_);
lean_dec_ref(v_val_1589_);
v_a_1675_ = lean_ctor_get(v___y_1617_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___y_1617_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1677_ = v___y_1617_;
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
else
{
lean_inc(v_a_1675_);
lean_dec(v___y_1617_);
v___x_1677_ = lean_box(0);
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
v_resetjp_1676_:
{
lean_object* v___x_1680_; 
if (v_isShared_1678_ == 0)
{
v___x_1680_ = v___x_1677_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v_a_1675_);
v___x_1680_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
return v___x_1680_;
}
}
}
}
}
else
{
lean_object* v_a_1686_; lean_object* v___x_1688_; uint8_t v_isShared_1689_; uint8_t v_isSharedCheck_1693_; 
lean_dec_ref(v___y_1598_);
lean_dec(v_loc_1592_);
lean_dec(v_head_1591_);
lean_dec(v___x_1590_);
lean_dec_ref(v_val_1589_);
v_a_1686_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1693_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1693_ == 0)
{
v___x_1688_ = v___x_1614_;
v_isShared_1689_ = v_isSharedCheck_1693_;
goto v_resetjp_1687_;
}
else
{
lean_inc(v_a_1686_);
lean_dec(v___x_1614_);
v___x_1688_ = lean_box(0);
v_isShared_1689_ = v_isSharedCheck_1693_;
goto v_resetjp_1687_;
}
v_resetjp_1687_:
{
lean_object* v___x_1691_; 
if (v_isShared_1689_ == 0)
{
v___x_1691_ = v___x_1688_;
goto v_reusejp_1690_;
}
else
{
lean_object* v_reuseFailAlloc_1692_; 
v_reuseFailAlloc_1692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1692_, 0, v_a_1686_);
v___x_1691_ = v_reuseFailAlloc_1692_;
goto v_reusejp_1690_;
}
v_reusejp_1690_:
{
return v___x_1691_;
}
}
}
v___jp_1603_:
{
lean_object* v___x_1605_; 
v___x_1605_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_1604_, v___y_1595_, v___y_1598_, v___y_1599_, v___y_1600_, v___y_1601_);
lean_dec_ref(v___y_1598_);
if (lean_obj_tag(v___x_1605_) == 0)
{
lean_object* v___x_1607_; uint8_t v_isShared_1608_; uint8_t v_isSharedCheck_1612_; 
v_isSharedCheck_1612_ = !lean_is_exclusive(v___x_1605_);
if (v_isSharedCheck_1612_ == 0)
{
lean_object* v_unused_1613_; 
v_unused_1613_ = lean_ctor_get(v___x_1605_, 0);
lean_dec(v_unused_1613_);
v___x_1607_ = v___x_1605_;
v_isShared_1608_ = v_isSharedCheck_1612_;
goto v_resetjp_1606_;
}
else
{
lean_dec(v___x_1605_);
v___x_1607_ = lean_box(0);
v_isShared_1608_ = v_isSharedCheck_1612_;
goto v_resetjp_1606_;
}
v_resetjp_1606_:
{
lean_object* v___x_1610_; 
if (v_isShared_1608_ == 0)
{
lean_ctor_set(v___x_1607_, 0, v___x_1588_);
v___x_1610_ = v___x_1607_;
goto v_reusejp_1609_;
}
else
{
lean_object* v_reuseFailAlloc_1611_; 
v_reuseFailAlloc_1611_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1611_, 0, v___x_1588_);
v___x_1610_ = v_reuseFailAlloc_1611_;
goto v_reusejp_1609_;
}
v_reusejp_1609_:
{
return v___x_1610_;
}
}
}
else
{
return v___x_1605_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1___boxed(lean_object* v___x_1694_, lean_object* v_val_1695_, lean_object* v___x_1696_, lean_object* v_head_1697_, lean_object* v_loc_1698_, lean_object* v___x_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_){
_start:
{
uint8_t v___x_28836__boxed_1709_; lean_object* v_res_1710_; 
v___x_28836__boxed_1709_ = lean_unbox(v___x_1699_);
v_res_1710_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1(v___x_1694_, v_val_1695_, v___x_1696_, v_head_1697_, v_loc_1698_, v___x_28836__boxed_1709_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_);
lean_dec(v___y_1707_);
lean_dec_ref(v___y_1706_);
lean_dec(v___y_1705_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
return v_res_1710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2(lean_object* v___x_1711_, lean_object* v_val_1712_, lean_object* v___x_1713_, lean_object* v_head_1714_, uint8_t v___x_1715_, lean_object* v_loc_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_){
_start:
{
lean_object* v___x_1726_; lean_object* v___f_1727_; lean_object* v___x_1728_; 
v___x_1726_ = lean_box(v___x_1715_);
v___f_1727_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__1___boxed), 15, 6);
lean_closure_set(v___f_1727_, 0, v___x_1711_);
lean_closure_set(v___f_1727_, 1, v_val_1712_);
lean_closure_set(v___f_1727_, 2, v___x_1713_);
lean_closure_set(v___f_1727_, 3, v_head_1714_);
lean_closure_set(v___f_1727_, 4, v_loc_1716_);
lean_closure_set(v___f_1727_, 5, v___x_1726_);
v___x_1728_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1727_, v___y_1717_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_, v___y_1722_, v___y_1723_, v___y_1724_);
return v___x_1728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2___boxed(lean_object* v___x_1729_, lean_object* v_val_1730_, lean_object* v___x_1731_, lean_object* v_head_1732_, lean_object* v___x_1733_, lean_object* v_loc_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_){
_start:
{
uint8_t v___x_29042__boxed_1744_; lean_object* v_res_1745_; 
v___x_29042__boxed_1744_ = lean_unbox(v___x_1733_);
v_res_1745_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2(v___x_1729_, v_val_1730_, v___x_1731_, v_head_1732_, v___x_29042__boxed_1744_, v_loc_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_);
lean_dec(v___y_1742_);
lean_dec_ref(v___y_1741_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
lean_dec(v___y_1738_);
lean_dec_ref(v___y_1737_);
lean_dec(v___y_1736_);
lean_dec_ref(v___y_1735_);
return v_res_1745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg(lean_object* v_val_1747_, lean_object* v_as_x27_1748_, lean_object* v_b_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_){
_start:
{
if (lean_obj_tag(v_as_x27_1748_) == 0)
{
lean_object* v___x_1759_; 
v___x_1759_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1759_, 0, v_b_1749_);
return v___x_1759_;
}
else
{
lean_object* v_head_1760_; lean_object* v_tail_1761_; uint8_t v___x_1762_; lean_object* v___x_1763_; 
v_head_1760_ = lean_ctor_get(v_as_x27_1748_, 0);
v_tail_1761_ = lean_ctor_get(v_as_x27_1748_, 1);
v___x_1762_ = 0;
lean_inc(v_head_1760_);
v___x_1763_ = l_Lean_FVarId_getValue_x3f___redArg(v_head_1760_, v___x_1762_, v___y_1754_, v___y_1756_, v___y_1757_);
if (lean_obj_tag(v___x_1763_) == 0)
{
lean_object* v_a_1764_; lean_object* v___x_1765_; 
v_a_1764_ = lean_ctor_get(v___x_1763_, 0);
lean_inc(v_a_1764_);
lean_dec_ref_known(v___x_1763_, 1);
v___x_1765_ = lean_box(0);
if (lean_obj_tag(v_a_1764_) == 1)
{
lean_object* v_val_1766_; lean_object* v_fileName_1767_; lean_object* v_fileMap_1768_; lean_object* v_options_1769_; lean_object* v_currRecDepth_1770_; lean_object* v_maxRecDepth_1771_; lean_object* v_ref_1772_; lean_object* v_currNamespace_1773_; lean_object* v_openDecls_1774_; lean_object* v_initHeartbeats_1775_; lean_object* v_maxHeartbeats_1776_; lean_object* v_quotContext_1777_; lean_object* v_currMacroScope_1778_; uint8_t v_diag_1779_; lean_object* v_cancelTk_x3f_1780_; uint8_t v_suppressElabErrors_1781_; lean_object* v_inheritedTraceOptions_1782_; lean_object* v___f_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___f_1786_; lean_object* v___f_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v_ref_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; 
v_val_1766_ = lean_ctor_get(v_a_1764_, 0);
lean_inc_n(v_val_1766_, 2);
lean_dec_ref_known(v_a_1764_, 1);
v_fileName_1767_ = lean_ctor_get(v___y_1756_, 0);
v_fileMap_1768_ = lean_ctor_get(v___y_1756_, 1);
v_options_1769_ = lean_ctor_get(v___y_1756_, 2);
v_currRecDepth_1770_ = lean_ctor_get(v___y_1756_, 3);
v_maxRecDepth_1771_ = lean_ctor_get(v___y_1756_, 4);
v_ref_1772_ = lean_ctor_get(v___y_1756_, 5);
v_currNamespace_1773_ = lean_ctor_get(v___y_1756_, 6);
v_openDecls_1774_ = lean_ctor_get(v___y_1756_, 7);
v_initHeartbeats_1775_ = lean_ctor_get(v___y_1756_, 8);
v_maxHeartbeats_1776_ = lean_ctor_get(v___y_1756_, 9);
v_quotContext_1777_ = lean_ctor_get(v___y_1756_, 10);
v_currMacroScope_1778_ = lean_ctor_get(v___y_1756_, 11);
v_diag_1779_ = lean_ctor_get_uint8(v___y_1756_, sizeof(void*)*14);
v_cancelTk_x3f_1780_ = lean_ctor_get(v___y_1756_, 12);
v_suppressElabErrors_1781_ = lean_ctor_get_uint8(v___y_1756_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1782_ = lean_ctor_get(v___y_1756_, 13);
v___f_1783_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___closed__0));
v___x_1784_ = lean_box(0);
v___x_1785_ = lean_box(v___x_1762_);
lean_inc_n(v_head_1760_, 2);
v___f_1786_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2___boxed), 15, 5);
lean_closure_set(v___f_1786_, 0, v___x_1765_);
lean_closure_set(v___f_1786_, 1, v_val_1766_);
lean_closure_set(v___f_1786_, 2, v___x_1784_);
lean_closure_set(v___f_1786_, 3, v_head_1760_);
lean_closure_set(v___f_1786_, 4, v___x_1785_);
v___f_1787_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__3___boxed), 11, 1);
lean_closure_set(v___f_1787_, 0, v___f_1786_);
v___x_1788_ = l_Lean_Elab_Tactic_expandLocation(v_val_1747_);
v___x_1789_ = lean_box(0);
v___x_1790_ = lean_box(v___x_1762_);
v___x_1791_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___lam__2___boxed), 15, 6);
lean_closure_set(v___x_1791_, 0, v___x_1765_);
lean_closure_set(v___x_1791_, 1, v_val_1766_);
lean_closure_set(v___x_1791_, 2, v___x_1784_);
lean_closure_set(v___x_1791_, 3, v_head_1760_);
lean_closure_set(v___x_1791_, 4, v___x_1790_);
lean_closure_set(v___x_1791_, 5, v___x_1789_);
v_ref_1792_ = l_Lean_replaceRef(v_val_1747_, v_ref_1772_);
lean_inc_ref(v_inheritedTraceOptions_1782_);
lean_inc(v_cancelTk_x3f_1780_);
lean_inc(v_currMacroScope_1778_);
lean_inc(v_quotContext_1777_);
lean_inc(v_maxHeartbeats_1776_);
lean_inc(v_initHeartbeats_1775_);
lean_inc(v_openDecls_1774_);
lean_inc(v_currNamespace_1773_);
lean_inc(v_maxRecDepth_1771_);
lean_inc(v_currRecDepth_1770_);
lean_inc_ref(v_options_1769_);
lean_inc_ref(v_fileMap_1768_);
lean_inc_ref(v_fileName_1767_);
v___x_1793_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1793_, 0, v_fileName_1767_);
lean_ctor_set(v___x_1793_, 1, v_fileMap_1768_);
lean_ctor_set(v___x_1793_, 2, v_options_1769_);
lean_ctor_set(v___x_1793_, 3, v_currRecDepth_1770_);
lean_ctor_set(v___x_1793_, 4, v_maxRecDepth_1771_);
lean_ctor_set(v___x_1793_, 5, v_ref_1792_);
lean_ctor_set(v___x_1793_, 6, v_currNamespace_1773_);
lean_ctor_set(v___x_1793_, 7, v_openDecls_1774_);
lean_ctor_set(v___x_1793_, 8, v_initHeartbeats_1775_);
lean_ctor_set(v___x_1793_, 9, v_maxHeartbeats_1776_);
lean_ctor_set(v___x_1793_, 10, v_quotContext_1777_);
lean_ctor_set(v___x_1793_, 11, v_currMacroScope_1778_);
lean_ctor_set(v___x_1793_, 12, v_cancelTk_x3f_1780_);
lean_ctor_set(v___x_1793_, 13, v_inheritedTraceOptions_1782_);
lean_ctor_set_uint8(v___x_1793_, sizeof(void*)*14, v_diag_1779_);
lean_ctor_set_uint8(v___x_1793_, sizeof(void*)*14 + 1, v_suppressElabErrors_1781_);
v___x_1794_ = l_Lean_Elab_Tactic_withLocation(v___x_1788_, v___f_1787_, v___x_1791_, v___f_1783_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_, v___y_1755_, v___x_1793_, v___y_1757_);
lean_dec_ref_known(v___x_1793_, 14);
lean_dec(v___x_1788_);
if (lean_obj_tag(v___x_1794_) == 0)
{
lean_dec_ref_known(v___x_1794_, 1);
v_as_x27_1748_ = v_tail_1761_;
v_b_1749_ = v___x_1765_;
goto _start;
}
else
{
return v___x_1794_;
}
}
else
{
lean_dec(v_a_1764_);
v_as_x27_1748_ = v_tail_1761_;
v_b_1749_ = v___x_1765_;
goto _start;
}
}
else
{
lean_object* v_a_1797_; lean_object* v___x_1799_; uint8_t v_isShared_1800_; uint8_t v_isSharedCheck_1804_; 
v_a_1797_ = lean_ctor_get(v___x_1763_, 0);
v_isSharedCheck_1804_ = !lean_is_exclusive(v___x_1763_);
if (v_isSharedCheck_1804_ == 0)
{
v___x_1799_ = v___x_1763_;
v_isShared_1800_ = v_isSharedCheck_1804_;
goto v_resetjp_1798_;
}
else
{
lean_inc(v_a_1797_);
lean_dec(v___x_1763_);
v___x_1799_ = lean_box(0);
v_isShared_1800_ = v_isSharedCheck_1804_;
goto v_resetjp_1798_;
}
v_resetjp_1798_:
{
lean_object* v___x_1802_; 
if (v_isShared_1800_ == 0)
{
v___x_1802_ = v___x_1799_;
goto v_reusejp_1801_;
}
else
{
lean_object* v_reuseFailAlloc_1803_; 
v_reuseFailAlloc_1803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1803_, 0, v_a_1797_);
v___x_1802_ = v_reuseFailAlloc_1803_;
goto v_reusejp_1801_;
}
v_reusejp_1801_:
{
return v___x_1802_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg___boxed(lean_object* v_val_1805_, lean_object* v_as_x27_1806_, lean_object* v_b_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_){
_start:
{
lean_object* v_res_1817_; 
v_res_1817_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg(v_val_1805_, v_as_x27_1806_, v_b_1807_, v___y_1808_, v___y_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
lean_dec(v___y_1815_);
lean_dec_ref(v___y_1814_);
lean_dec(v___y_1813_);
lean_dec_ref(v___y_1812_);
lean_dec(v___y_1811_);
lean_dec_ref(v___y_1810_);
lean_dec(v___y_1809_);
lean_dec_ref(v___y_1808_);
lean_dec(v_as_x27_1806_);
lean_dec(v_val_1805_);
return v_res_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5(lean_object* v_init_1818_, lean_object* v_x_1819_){
_start:
{
if (lean_obj_tag(v_x_1819_) == 0)
{
lean_object* v_v_1820_; lean_object* v_l_1821_; lean_object* v_r_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; 
v_v_1820_ = lean_ctor_get(v_x_1819_, 2);
v_l_1821_ = lean_ctor_get(v_x_1819_, 3);
v_r_1822_ = lean_ctor_get(v_x_1819_, 4);
v___x_1823_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5(v_init_1818_, v_r_1822_);
lean_inc(v_v_1820_);
v___x_1824_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1824_, 0, v_v_1820_);
lean_ctor_set(v___x_1824_, 1, v___x_1823_);
v_init_1818_ = v___x_1824_;
v_x_1819_ = v_l_1821_;
goto _start;
}
else
{
return v_init_1818_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5___boxed(lean_object* v_init_1826_, lean_object* v_x_1827_){
_start:
{
lean_object* v_res_1828_; 
v_res_1828_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5(v_init_1826_, v_x_1827_);
lean_dec(v_x_1827_);
return v_res_1828_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1830_; lean_object* v___x_1831_; 
v___x_1830_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__0));
v___x_1831_ = l_Lean_stringToMessageData(v___x_1830_);
return v___x_1831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0(lean_object* v___x_1835_, lean_object* v_newMVars_1836_, lean_object* v___x_1837_, lean_object* v___x_1838_, lean_object* v___x_1839_, lean_object* v_loc_1840_, lean_object* v_holes_1841_, lean_object* v_usingArg_1842_, lean_object* v_a_1843_, lean_object* v_goal_1844_, lean_object* v___x_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_){
_start:
{
lean_object* v___y_1856_; lean_object* v___y_1857_; lean_object* v___y_1858_; lean_object* v___y_1859_; lean_object* v___y_1860_; lean_object* v___y_1861_; lean_object* v___y_1862_; lean_object* v___y_1863_; lean_object* v_a_1864_; lean_object* v___y_1880_; lean_object* v___y_1881_; lean_object* v___y_1882_; lean_object* v___y_1883_; lean_object* v___y_1884_; lean_object* v___y_1885_; lean_object* v___y_1886_; lean_object* v___y_1887_; lean_object* v___y_1888_; lean_object* v___x_1898_; 
v___x_1898_ = lp_mathlib_Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2(v___x_1835_, v___y_1846_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_);
if (lean_obj_tag(v___x_1898_) == 0)
{
lean_object* v_a_1899_; lean_object* v_fst_1900_; lean_object* v_snd_1901_; lean_object* v___x_1903_; uint8_t v_isShared_1904_; uint8_t v_isSharedCheck_2049_; 
v_a_1899_ = lean_ctor_get(v___x_1898_, 0);
lean_inc(v_a_1899_);
lean_dec_ref_known(v___x_1898_, 1);
v_fst_1900_ = lean_ctor_get(v_a_1899_, 0);
v_snd_1901_ = lean_ctor_get(v_a_1899_, 1);
v_isSharedCheck_2049_ = !lean_is_exclusive(v_a_1899_);
if (v_isSharedCheck_2049_ == 0)
{
v___x_1903_ = v_a_1899_;
v_isShared_1904_ = v_isSharedCheck_2049_;
goto v_resetjp_1902_;
}
else
{
lean_inc(v_snd_1901_);
lean_inc(v_fst_1900_);
lean_dec(v_a_1899_);
v___x_1903_ = lean_box(0);
v_isShared_1904_ = v_isSharedCheck_2049_;
goto v_resetjp_1902_;
}
v_resetjp_1902_:
{
lean_object* v___y_1906_; lean_object* v___y_1907_; lean_object* v___y_1908_; lean_object* v___y_1909_; lean_object* v___y_1910_; lean_object* v___y_1911_; lean_object* v___y_1912_; lean_object* v___y_1913_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; lean_object* v___y_1928_; lean_object* v___y_1929_; lean_object* v___y_1930_; lean_object* v___y_1931_; lean_object* v___y_1932_; lean_object* v___y_1939_; lean_object* v___y_1941_; 
if (lean_obj_tag(v_usingArg_1842_) == 1)
{
lean_object* v_val_1942_; lean_object* v_fileName_1943_; lean_object* v_fileMap_1944_; lean_object* v_options_1945_; lean_object* v_currRecDepth_1946_; lean_object* v_maxRecDepth_1947_; lean_object* v_ref_1948_; lean_object* v_currNamespace_1949_; lean_object* v_openDecls_1950_; lean_object* v_initHeartbeats_1951_; lean_object* v_maxHeartbeats_1952_; lean_object* v_quotContext_1953_; lean_object* v_currMacroScope_1954_; uint8_t v_diag_1955_; lean_object* v_cancelTk_x3f_1956_; uint8_t v_suppressElabErrors_1957_; lean_object* v_inheritedTraceOptions_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v_ref_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; 
v_val_1942_ = lean_ctor_get(v_usingArg_1842_, 0);
lean_inc_n(v_val_1942_, 2);
lean_dec_ref_known(v_usingArg_1842_, 1);
v_fileName_1943_ = lean_ctor_get(v___y_1852_, 0);
v_fileMap_1944_ = lean_ctor_get(v___y_1852_, 1);
v_options_1945_ = lean_ctor_get(v___y_1852_, 2);
v_currRecDepth_1946_ = lean_ctor_get(v___y_1852_, 3);
v_maxRecDepth_1947_ = lean_ctor_get(v___y_1852_, 4);
v_ref_1948_ = lean_ctor_get(v___y_1852_, 5);
v_currNamespace_1949_ = lean_ctor_get(v___y_1852_, 6);
v_openDecls_1950_ = lean_ctor_get(v___y_1852_, 7);
v_initHeartbeats_1951_ = lean_ctor_get(v___y_1852_, 8);
v_maxHeartbeats_1952_ = lean_ctor_get(v___y_1852_, 9);
v_quotContext_1953_ = lean_ctor_get(v___y_1852_, 10);
v_currMacroScope_1954_ = lean_ctor_get(v___y_1852_, 11);
v_diag_1955_ = lean_ctor_get_uint8(v___y_1852_, sizeof(void*)*14);
v_cancelTk_x3f_1956_ = lean_ctor_get(v___y_1852_, 12);
v_suppressElabErrors_1957_ = lean_ctor_get_uint8(v___y_1852_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1958_ = lean_ctor_get(v___y_1852_, 13);
v___x_1959_ = lean_unsigned_to_nat(2u);
v___x_1960_ = lean_mk_empty_array_with_capacity(v___x_1959_);
lean_inc(v___x_1838_);
v___x_1961_ = lean_array_push(v___x_1960_, v___x_1838_);
v___x_1962_ = lean_array_push(v___x_1961_, v_val_1942_);
v___x_1963_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__3));
v___x_1964_ = lean_box(2);
v___x_1965_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1965_, 0, v___x_1964_);
lean_ctor_set(v___x_1965_, 1, v___x_1963_);
lean_ctor_set(v___x_1965_, 2, v___x_1962_);
v___x_1966_ = l_Lean_TSyntax_getId(v_val_1942_);
lean_dec(v_val_1942_);
v_ref_1967_ = l_Lean_replaceRef(v___x_1965_, v_ref_1948_);
lean_dec_ref_known(v___x_1965_, 3);
lean_inc_ref(v_inheritedTraceOptions_1958_);
lean_inc(v_cancelTk_x3f_1956_);
lean_inc(v_currMacroScope_1954_);
lean_inc(v_quotContext_1953_);
lean_inc(v_maxHeartbeats_1952_);
lean_inc(v_initHeartbeats_1951_);
lean_inc(v_openDecls_1950_);
lean_inc(v_currNamespace_1949_);
lean_inc(v_maxRecDepth_1947_);
lean_inc(v_currRecDepth_1946_);
lean_inc_ref(v_options_1945_);
lean_inc_ref(v_fileMap_1944_);
lean_inc_ref(v_fileName_1943_);
v___x_1968_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1968_, 0, v_fileName_1943_);
lean_ctor_set(v___x_1968_, 1, v_fileMap_1944_);
lean_ctor_set(v___x_1968_, 2, v_options_1945_);
lean_ctor_set(v___x_1968_, 3, v_currRecDepth_1946_);
lean_ctor_set(v___x_1968_, 4, v_maxRecDepth_1947_);
lean_ctor_set(v___x_1968_, 5, v_ref_1967_);
lean_ctor_set(v___x_1968_, 6, v_currNamespace_1949_);
lean_ctor_set(v___x_1968_, 7, v_openDecls_1950_);
lean_ctor_set(v___x_1968_, 8, v_initHeartbeats_1951_);
lean_ctor_set(v___x_1968_, 9, v_maxHeartbeats_1952_);
lean_ctor_set(v___x_1968_, 10, v_quotContext_1953_);
lean_ctor_set(v___x_1968_, 11, v_currMacroScope_1954_);
lean_ctor_set(v___x_1968_, 12, v_cancelTk_x3f_1956_);
lean_ctor_set(v___x_1968_, 13, v_inheritedTraceOptions_1958_);
lean_ctor_set_uint8(v___x_1968_, sizeof(void*)*14, v_diag_1955_);
lean_ctor_set_uint8(v___x_1968_, sizeof(void*)*14 + 1, v_suppressElabErrors_1957_);
v___x_1969_ = l_Lean_Meta_getLocalDeclFromUserName(v___x_1966_, v___y_1850_, v___y_1851_, v___x_1968_, v___y_1853_);
if (lean_obj_tag(v___x_1969_) == 0)
{
lean_object* v_a_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; 
v_a_1970_ = lean_ctor_get(v___x_1969_, 0);
lean_inc(v_a_1970_);
lean_dec_ref_known(v___x_1969_, 1);
v___x_1971_ = l_Lean_LocalDecl_fvarId(v_a_1970_);
lean_dec(v_a_1970_);
lean_inc(v___x_1971_);
v___x_1972_ = l_Lean_FVarId_getType___redArg(v___x_1971_, v___y_1850_, v___x_1968_, v___y_1853_);
if (lean_obj_tag(v___x_1972_) == 0)
{
lean_object* v_a_1973_; lean_object* v___x_1974_; 
v_a_1973_ = lean_ctor_get(v___x_1972_, 0);
lean_inc(v_a_1973_);
lean_dec_ref_known(v___x_1972_, 1);
lean_inc(v_fst_1900_);
v___x_1974_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError(v_a_1843_, v_fst_1900_, v_a_1973_, v___y_1850_, v___y_1851_, v___x_1968_, v___y_1853_);
if (lean_obj_tag(v___x_1974_) == 0)
{
uint8_t v___x_1975_; lean_object* v___x_1976_; 
lean_dec_ref_known(v___x_1974_, 1);
v___x_1975_ = 0;
v___x_1976_ = l_Lean_MVarId_changeLocalDecl(v_goal_1844_, v___x_1971_, v_fst_1900_, v___x_1975_, v___y_1850_, v___y_1851_, v___x_1968_, v___y_1853_);
if (lean_obj_tag(v___x_1976_) == 0)
{
lean_object* v_a_1977_; lean_object* v___x_1979_; 
v_a_1977_ = lean_ctor_get(v___x_1976_, 0);
lean_inc(v_a_1977_);
lean_dec_ref_known(v___x_1976_, 1);
if (v_isShared_1904_ == 0)
{
lean_ctor_set_tag(v___x_1903_, 1);
lean_ctor_set(v___x_1903_, 1, v___x_1845_);
lean_ctor_set(v___x_1903_, 0, v_a_1977_);
v___x_1979_ = v___x_1903_;
goto v_reusejp_1978_;
}
else
{
lean_object* v_reuseFailAlloc_1981_; 
v_reuseFailAlloc_1981_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1981_, 0, v_a_1977_);
lean_ctor_set(v_reuseFailAlloc_1981_, 1, v___x_1845_);
v___x_1979_ = v_reuseFailAlloc_1981_;
goto v_reusejp_1978_;
}
v_reusejp_1978_:
{
lean_object* v___x_1980_; 
v___x_1980_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1979_, v___y_1847_, v___y_1850_, v___y_1851_, v___x_1968_, v___y_1853_);
lean_dec_ref_known(v___x_1968_, 14);
v___y_1941_ = v___x_1980_;
goto v___jp_1940_;
}
}
else
{
lean_object* v_a_1982_; lean_object* v___x_1984_; uint8_t v_isShared_1985_; uint8_t v_isSharedCheck_1989_; 
lean_dec_ref_known(v___x_1968_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_snd_1901_);
lean_dec(v___x_1845_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_1982_ = lean_ctor_get(v___x_1976_, 0);
v_isSharedCheck_1989_ = !lean_is_exclusive(v___x_1976_);
if (v_isSharedCheck_1989_ == 0)
{
v___x_1984_ = v___x_1976_;
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
else
{
lean_inc(v_a_1982_);
lean_dec(v___x_1976_);
v___x_1984_ = lean_box(0);
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
v_resetjp_1983_:
{
lean_object* v___x_1987_; 
if (v_isShared_1985_ == 0)
{
v___x_1987_ = v___x_1984_;
goto v_reusejp_1986_;
}
else
{
lean_object* v_reuseFailAlloc_1988_; 
v_reuseFailAlloc_1988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1988_, 0, v_a_1982_);
v___x_1987_ = v_reuseFailAlloc_1988_;
goto v_reusejp_1986_;
}
v_reusejp_1986_:
{
return v___x_1987_;
}
}
}
}
else
{
lean_dec(v___x_1971_);
lean_dec_ref_known(v___x_1968_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_fst_1900_);
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
v___y_1941_ = v___x_1974_;
goto v___jp_1940_;
}
}
else
{
lean_object* v_a_1990_; lean_object* v___x_1992_; uint8_t v_isShared_1993_; uint8_t v_isSharedCheck_1997_; 
lean_dec(v___x_1971_);
lean_dec_ref_known(v___x_1968_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_snd_1901_);
lean_dec(v_fst_1900_);
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
lean_dec(v_a_1843_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_1990_ = lean_ctor_get(v___x_1972_, 0);
v_isSharedCheck_1997_ = !lean_is_exclusive(v___x_1972_);
if (v_isSharedCheck_1997_ == 0)
{
v___x_1992_ = v___x_1972_;
v_isShared_1993_ = v_isSharedCheck_1997_;
goto v_resetjp_1991_;
}
else
{
lean_inc(v_a_1990_);
lean_dec(v___x_1972_);
v___x_1992_ = lean_box(0);
v_isShared_1993_ = v_isSharedCheck_1997_;
goto v_resetjp_1991_;
}
v_resetjp_1991_:
{
lean_object* v___x_1995_; 
if (v_isShared_1993_ == 0)
{
v___x_1995_ = v___x_1992_;
goto v_reusejp_1994_;
}
else
{
lean_object* v_reuseFailAlloc_1996_; 
v_reuseFailAlloc_1996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1996_, 0, v_a_1990_);
v___x_1995_ = v_reuseFailAlloc_1996_;
goto v_reusejp_1994_;
}
v_reusejp_1994_:
{
return v___x_1995_;
}
}
}
}
else
{
lean_object* v_a_1998_; lean_object* v___x_2000_; uint8_t v_isShared_2001_; uint8_t v_isSharedCheck_2005_; 
lean_dec_ref_known(v___x_1968_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_snd_1901_);
lean_dec(v_fst_1900_);
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
lean_dec(v_a_1843_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_1998_ = lean_ctor_get(v___x_1969_, 0);
v_isSharedCheck_2005_ = !lean_is_exclusive(v___x_1969_);
if (v_isSharedCheck_2005_ == 0)
{
v___x_2000_ = v___x_1969_;
v_isShared_2001_ = v_isSharedCheck_2005_;
goto v_resetjp_1999_;
}
else
{
lean_inc(v_a_1998_);
lean_dec(v___x_1969_);
v___x_2000_ = lean_box(0);
v_isShared_2001_ = v_isSharedCheck_2005_;
goto v_resetjp_1999_;
}
v_resetjp_1999_:
{
lean_object* v___x_2003_; 
if (v_isShared_2001_ == 0)
{
v___x_2003_ = v___x_2000_;
goto v_reusejp_2002_;
}
else
{
lean_object* v_reuseFailAlloc_2004_; 
v_reuseFailAlloc_2004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2004_, 0, v_a_1998_);
v___x_2003_ = v_reuseFailAlloc_2004_;
goto v_reusejp_2002_;
}
v_reusejp_2002_:
{
return v___x_2003_;
}
}
}
}
else
{
lean_object* v_fileName_2006_; lean_object* v_fileMap_2007_; lean_object* v_options_2008_; lean_object* v_currRecDepth_2009_; lean_object* v_maxRecDepth_2010_; lean_object* v_ref_2011_; lean_object* v_currNamespace_2012_; lean_object* v_openDecls_2013_; lean_object* v_initHeartbeats_2014_; lean_object* v_maxHeartbeats_2015_; lean_object* v_quotContext_2016_; lean_object* v_currMacroScope_2017_; uint8_t v_diag_2018_; lean_object* v_cancelTk_x3f_2019_; uint8_t v_suppressElabErrors_2020_; lean_object* v_inheritedTraceOptions_2021_; lean_object* v_ref_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; 
lean_dec(v_usingArg_1842_);
v_fileName_2006_ = lean_ctor_get(v___y_1852_, 0);
v_fileMap_2007_ = lean_ctor_get(v___y_1852_, 1);
v_options_2008_ = lean_ctor_get(v___y_1852_, 2);
v_currRecDepth_2009_ = lean_ctor_get(v___y_1852_, 3);
v_maxRecDepth_2010_ = lean_ctor_get(v___y_1852_, 4);
v_ref_2011_ = lean_ctor_get(v___y_1852_, 5);
v_currNamespace_2012_ = lean_ctor_get(v___y_1852_, 6);
v_openDecls_2013_ = lean_ctor_get(v___y_1852_, 7);
v_initHeartbeats_2014_ = lean_ctor_get(v___y_1852_, 8);
v_maxHeartbeats_2015_ = lean_ctor_get(v___y_1852_, 9);
v_quotContext_2016_ = lean_ctor_get(v___y_1852_, 10);
v_currMacroScope_2017_ = lean_ctor_get(v___y_1852_, 11);
v_diag_2018_ = lean_ctor_get_uint8(v___y_1852_, sizeof(void*)*14);
v_cancelTk_x3f_2019_ = lean_ctor_get(v___y_1852_, 12);
v_suppressElabErrors_2020_ = lean_ctor_get_uint8(v___y_1852_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2021_ = lean_ctor_get(v___y_1852_, 13);
v_ref_2022_ = l_Lean_replaceRef(v___x_1838_, v_ref_2011_);
lean_inc_ref(v_inheritedTraceOptions_2021_);
lean_inc(v_cancelTk_x3f_2019_);
lean_inc(v_currMacroScope_2017_);
lean_inc(v_quotContext_2016_);
lean_inc(v_maxHeartbeats_2015_);
lean_inc(v_initHeartbeats_2014_);
lean_inc(v_openDecls_2013_);
lean_inc(v_currNamespace_2012_);
lean_inc(v_maxRecDepth_2010_);
lean_inc(v_currRecDepth_2009_);
lean_inc_ref(v_options_2008_);
lean_inc_ref(v_fileMap_2007_);
lean_inc_ref(v_fileName_2006_);
v___x_2023_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2023_, 0, v_fileName_2006_);
lean_ctor_set(v___x_2023_, 1, v_fileMap_2007_);
lean_ctor_set(v___x_2023_, 2, v_options_2008_);
lean_ctor_set(v___x_2023_, 3, v_currRecDepth_2009_);
lean_ctor_set(v___x_2023_, 4, v_maxRecDepth_2010_);
lean_ctor_set(v___x_2023_, 5, v_ref_2022_);
lean_ctor_set(v___x_2023_, 6, v_currNamespace_2012_);
lean_ctor_set(v___x_2023_, 7, v_openDecls_2013_);
lean_ctor_set(v___x_2023_, 8, v_initHeartbeats_2014_);
lean_ctor_set(v___x_2023_, 9, v_maxHeartbeats_2015_);
lean_ctor_set(v___x_2023_, 10, v_quotContext_2016_);
lean_ctor_set(v___x_2023_, 11, v_currMacroScope_2017_);
lean_ctor_set(v___x_2023_, 12, v_cancelTk_x3f_2019_);
lean_ctor_set(v___x_2023_, 13, v_inheritedTraceOptions_2021_);
lean_ctor_set_uint8(v___x_2023_, sizeof(void*)*14, v_diag_2018_);
lean_ctor_set_uint8(v___x_2023_, sizeof(void*)*14 + 1, v_suppressElabErrors_2020_);
lean_inc(v_goal_1844_);
v___x_2024_ = l_Lean_MVarId_getType(v_goal_1844_, v___y_1850_, v___y_1851_, v___x_2023_, v___y_1853_);
if (lean_obj_tag(v___x_2024_) == 0)
{
lean_object* v_a_2025_; lean_object* v___x_2026_; 
v_a_2025_ = lean_ctor_get(v___x_2024_, 0);
lean_inc(v_a_2025_);
lean_dec_ref_known(v___x_2024_, 1);
lean_inc(v_fst_1900_);
v___x_2026_ = lp_mathlib___private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_defeqOrError(v_a_1843_, v_fst_1900_, v_a_2025_, v___y_1850_, v___y_1851_, v___x_2023_, v___y_1853_);
if (lean_obj_tag(v___x_2026_) == 0)
{
lean_object* v___x_2027_; 
lean_dec_ref_known(v___x_2026_, 1);
v___x_2027_ = l_Lean_MVarId_replaceTargetDefEq(v_goal_1844_, v_fst_1900_, v___y_1850_, v___y_1851_, v___x_2023_, v___y_1853_);
if (lean_obj_tag(v___x_2027_) == 0)
{
lean_object* v_a_2028_; lean_object* v___x_2030_; 
v_a_2028_ = lean_ctor_get(v___x_2027_, 0);
lean_inc(v_a_2028_);
lean_dec_ref_known(v___x_2027_, 1);
if (v_isShared_1904_ == 0)
{
lean_ctor_set_tag(v___x_1903_, 1);
lean_ctor_set(v___x_1903_, 1, v___x_1845_);
lean_ctor_set(v___x_1903_, 0, v_a_2028_);
v___x_2030_ = v___x_1903_;
goto v_reusejp_2029_;
}
else
{
lean_object* v_reuseFailAlloc_2032_; 
v_reuseFailAlloc_2032_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2032_, 0, v_a_2028_);
lean_ctor_set(v_reuseFailAlloc_2032_, 1, v___x_1845_);
v___x_2030_ = v_reuseFailAlloc_2032_;
goto v_reusejp_2029_;
}
v_reusejp_2029_:
{
lean_object* v___x_2031_; 
v___x_2031_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_2030_, v___y_1847_, v___y_1850_, v___y_1851_, v___x_2023_, v___y_1853_);
lean_dec_ref_known(v___x_2023_, 14);
v___y_1939_ = v___x_2031_;
goto v___jp_1938_;
}
}
else
{
lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2040_; 
lean_dec_ref_known(v___x_2023_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_snd_1901_);
lean_dec(v___x_1845_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_2033_ = lean_ctor_get(v___x_2027_, 0);
v_isSharedCheck_2040_ = !lean_is_exclusive(v___x_2027_);
if (v_isSharedCheck_2040_ == 0)
{
v___x_2035_ = v___x_2027_;
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_2027_);
v___x_2035_ = lean_box(0);
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
v_resetjp_2034_:
{
lean_object* v___x_2038_; 
if (v_isShared_2036_ == 0)
{
v___x_2038_ = v___x_2035_;
goto v_reusejp_2037_;
}
else
{
lean_object* v_reuseFailAlloc_2039_; 
v_reuseFailAlloc_2039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2039_, 0, v_a_2033_);
v___x_2038_ = v_reuseFailAlloc_2039_;
goto v_reusejp_2037_;
}
v_reusejp_2037_:
{
return v___x_2038_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_2023_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_fst_1900_);
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
v___y_1939_ = v___x_2026_;
goto v___jp_1938_;
}
}
else
{
lean_object* v_a_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2048_; 
lean_dec_ref_known(v___x_2023_, 14);
lean_del_object(v___x_1903_);
lean_dec(v_snd_1901_);
lean_dec(v_fst_1900_);
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
lean_dec(v_a_1843_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_2041_ = lean_ctor_get(v___x_2024_, 0);
v_isSharedCheck_2048_ = !lean_is_exclusive(v___x_2024_);
if (v_isSharedCheck_2048_ == 0)
{
v___x_2043_ = v___x_2024_;
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_a_2041_);
lean_dec(v___x_2024_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v___x_2046_; 
if (v_isShared_2044_ == 0)
{
v___x_2046_ = v___x_2043_;
goto v_reusejp_2045_;
}
else
{
lean_object* v_reuseFailAlloc_2047_; 
v_reuseFailAlloc_2047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2047_, 0, v_a_2041_);
v___x_2046_ = v_reuseFailAlloc_2047_;
goto v_reusejp_2045_;
}
v_reusejp_2045_:
{
return v___x_2046_;
}
}
}
}
v___jp_1905_:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; uint8_t v___x_1916_; 
lean_inc_ref(v_newMVars_1836_);
v___x_1914_ = l_Array_append___redArg(v_newMVars_1836_, v_snd_1901_);
lean_dec(v_snd_1901_);
v___x_1915_ = lean_array_get_size(v___x_1914_);
v___x_1916_ = lean_nat_dec_lt(v___x_1837_, v___x_1915_);
if (v___x_1916_ == 0)
{
lean_dec_ref(v___x_1914_);
v___y_1856_ = v___y_1911_;
v___y_1857_ = v___y_1909_;
v___y_1858_ = v___y_1908_;
v___y_1859_ = v___y_1907_;
v___y_1860_ = v___y_1906_;
v___y_1861_ = v___y_1910_;
v___y_1862_ = v___y_1912_;
v___y_1863_ = v___y_1913_;
v_a_1864_ = v___x_1839_;
goto v___jp_1855_;
}
else
{
uint8_t v___x_1917_; 
v___x_1917_ = lean_nat_dec_le(v___x_1915_, v___x_1915_);
if (v___x_1917_ == 0)
{
if (v___x_1916_ == 0)
{
lean_dec_ref(v___x_1914_);
v___y_1856_ = v___y_1911_;
v___y_1857_ = v___y_1909_;
v___y_1858_ = v___y_1908_;
v___y_1859_ = v___y_1907_;
v___y_1860_ = v___y_1906_;
v___y_1861_ = v___y_1910_;
v___y_1862_ = v___y_1912_;
v___y_1863_ = v___y_1913_;
v_a_1864_ = v___x_1839_;
goto v___jp_1855_;
}
else
{
size_t v___x_1918_; size_t v___x_1919_; lean_object* v___x_1920_; 
v___x_1918_ = ((size_t)0ULL);
v___x_1919_ = lean_usize_of_nat(v___x_1915_);
v___x_1920_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4(v___x_1914_, v___x_1918_, v___x_1919_, v___x_1839_, v___y_1906_, v___y_1907_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_);
lean_dec_ref(v___x_1914_);
v___y_1880_ = v___y_1909_;
v___y_1881_ = v___y_1911_;
v___y_1882_ = v___y_1906_;
v___y_1883_ = v___y_1907_;
v___y_1884_ = v___y_1908_;
v___y_1885_ = v___y_1910_;
v___y_1886_ = v___y_1912_;
v___y_1887_ = v___y_1913_;
v___y_1888_ = v___x_1920_;
goto v___jp_1879_;
}
}
else
{
size_t v___x_1921_; size_t v___x_1922_; lean_object* v___x_1923_; 
v___x_1921_ = ((size_t)0ULL);
v___x_1922_ = lean_usize_of_nat(v___x_1915_);
v___x_1923_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__4(v___x_1914_, v___x_1921_, v___x_1922_, v___x_1839_, v___y_1906_, v___y_1907_, v___y_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_);
lean_dec_ref(v___x_1914_);
v___y_1880_ = v___y_1909_;
v___y_1881_ = v___y_1911_;
v___y_1882_ = v___y_1906_;
v___y_1883_ = v___y_1907_;
v___y_1884_ = v___y_1908_;
v___y_1885_ = v___y_1910_;
v___y_1886_ = v___y_1912_;
v___y_1887_ = v___y_1913_;
v___y_1888_ = v___x_1923_;
goto v___jp_1879_;
}
}
}
v___jp_1924_:
{
if (lean_obj_tag(v_loc_1840_) == 1)
{
lean_object* v_val_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; 
v_val_1933_ = lean_ctor_get(v_loc_1840_, 0);
v___x_1934_ = lean_box(0);
v___x_1935_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__5(v___x_1934_, v_holes_1841_);
v___x_1936_ = lean_box(0);
v___x_1937_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg(v_val_1933_, v___x_1935_, v___x_1936_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
lean_dec(v___x_1935_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_dec_ref_known(v___x_1937_, 1);
v___y_1906_ = v___y_1925_;
v___y_1907_ = v___y_1926_;
v___y_1908_ = v___y_1927_;
v___y_1909_ = v___y_1928_;
v___y_1910_ = v___y_1929_;
v___y_1911_ = v___y_1930_;
v___y_1912_ = v___y_1931_;
v___y_1913_ = v___y_1932_;
goto v___jp_1905_;
}
else
{
lean_dec(v_snd_1901_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
return v___x_1937_;
}
}
else
{
v___y_1906_ = v___y_1925_;
v___y_1907_ = v___y_1926_;
v___y_1908_ = v___y_1927_;
v___y_1909_ = v___y_1928_;
v___y_1910_ = v___y_1929_;
v___y_1911_ = v___y_1930_;
v___y_1912_ = v___y_1931_;
v___y_1913_ = v___y_1932_;
goto v___jp_1905_;
}
}
v___jp_1938_:
{
if (lean_obj_tag(v___y_1939_) == 0)
{
lean_dec_ref_known(v___y_1939_, 1);
v___y_1925_ = v___y_1846_;
v___y_1926_ = v___y_1847_;
v___y_1927_ = v___y_1848_;
v___y_1928_ = v___y_1849_;
v___y_1929_ = v___y_1850_;
v___y_1930_ = v___y_1851_;
v___y_1931_ = v___y_1852_;
v___y_1932_ = v___y_1853_;
goto v___jp_1924_;
}
else
{
lean_dec(v_snd_1901_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
return v___y_1939_;
}
}
v___jp_1940_:
{
if (lean_obj_tag(v___y_1941_) == 0)
{
lean_dec_ref_known(v___y_1941_, 1);
v___y_1925_ = v___y_1846_;
v___y_1926_ = v___y_1847_;
v___y_1927_ = v___y_1848_;
v___y_1928_ = v___y_1849_;
v___y_1929_ = v___y_1850_;
v___y_1930_ = v___y_1851_;
v___y_1931_ = v___y_1852_;
v___y_1932_ = v___y_1853_;
goto v___jp_1924_;
}
else
{
lean_dec(v_snd_1901_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
return v___y_1941_;
}
}
}
}
else
{
lean_object* v_a_2050_; lean_object* v___x_2052_; uint8_t v_isShared_2053_; uint8_t v_isSharedCheck_2057_; 
lean_dec(v___x_1845_);
lean_dec(v_goal_1844_);
lean_dec(v_a_1843_);
lean_dec(v_usingArg_1842_);
lean_dec_ref(v___x_1839_);
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_2050_ = lean_ctor_get(v___x_1898_, 0);
v_isSharedCheck_2057_ = !lean_is_exclusive(v___x_1898_);
if (v_isSharedCheck_2057_ == 0)
{
v___x_2052_ = v___x_1898_;
v_isShared_2053_ = v_isSharedCheck_2057_;
goto v_resetjp_2051_;
}
else
{
lean_inc(v_a_2050_);
lean_dec(v___x_1898_);
v___x_2052_ = lean_box(0);
v_isShared_2053_ = v_isSharedCheck_2057_;
goto v_resetjp_2051_;
}
v_resetjp_2051_:
{
lean_object* v___x_2055_; 
if (v_isShared_2053_ == 0)
{
v___x_2055_ = v___x_2052_;
goto v_reusejp_2054_;
}
else
{
lean_object* v_reuseFailAlloc_2056_; 
v_reuseFailAlloc_2056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2056_, 0, v_a_2050_);
v___x_2055_ = v_reuseFailAlloc_2056_;
goto v_reusejp_2054_;
}
v_reusejp_2054_:
{
return v___x_2055_;
}
}
}
v___jp_1855_:
{
lean_object* v___x_1865_; 
v___x_1865_ = l_Lean_Elab_Tactic_logUnassignedAndAbort(v_a_1864_, v___y_1860_, v___y_1859_, v___y_1858_, v___y_1857_, v___y_1861_, v___y_1856_, v___y_1862_, v___y_1863_);
lean_dec_ref(v_a_1864_);
if (lean_obj_tag(v___x_1865_) == 0)
{
lean_object* v___x_1867_; uint8_t v_isShared_1868_; uint8_t v_isSharedCheck_1877_; 
v_isSharedCheck_1877_ = !lean_is_exclusive(v___x_1865_);
if (v_isSharedCheck_1877_ == 0)
{
lean_object* v_unused_1878_; 
v_unused_1878_ = lean_ctor_get(v___x_1865_, 0);
lean_dec(v_unused_1878_);
v___x_1867_ = v___x_1865_;
v_isShared_1868_ = v_isSharedCheck_1877_;
goto v_resetjp_1866_;
}
else
{
lean_dec(v___x_1865_);
v___x_1867_ = lean_box(0);
v_isShared_1868_ = v_isSharedCheck_1877_;
goto v_resetjp_1866_;
}
v_resetjp_1866_:
{
lean_object* v___x_1869_; uint8_t v___x_1870_; 
v___x_1869_ = lean_array_get_size(v_newMVars_1836_);
lean_dec_ref(v_newMVars_1836_);
v___x_1870_ = lean_nat_dec_eq(v___x_1869_, v___x_1837_);
if (v___x_1870_ == 0)
{
lean_object* v___x_1871_; lean_object* v___x_1873_; 
lean_dec(v___x_1838_);
v___x_1871_ = lean_box(0);
if (v_isShared_1868_ == 0)
{
lean_ctor_set(v___x_1867_, 0, v___x_1871_);
v___x_1873_ = v___x_1867_;
goto v_reusejp_1872_;
}
else
{
lean_object* v_reuseFailAlloc_1874_; 
v_reuseFailAlloc_1874_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1874_, 0, v___x_1871_);
v___x_1873_ = v_reuseFailAlloc_1874_;
goto v_reusejp_1872_;
}
v_reusejp_1872_:
{
return v___x_1873_;
}
}
else
{
lean_object* v___x_1875_; lean_object* v___x_1876_; 
lean_del_object(v___x_1867_);
v___x_1875_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___closed__1);
v___x_1876_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3(v___x_1838_, v___x_1875_, v___y_1860_, v___y_1859_, v___y_1858_, v___y_1857_, v___y_1861_, v___y_1856_, v___y_1862_, v___y_1863_);
lean_dec(v___x_1838_);
return v___x_1876_;
}
}
}
else
{
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
return v___x_1865_;
}
}
v___jp_1879_:
{
if (lean_obj_tag(v___y_1888_) == 0)
{
lean_object* v_a_1889_; 
v_a_1889_ = lean_ctor_get(v___y_1888_, 0);
lean_inc(v_a_1889_);
lean_dec_ref_known(v___y_1888_, 1);
v___y_1856_ = v___y_1881_;
v___y_1857_ = v___y_1880_;
v___y_1858_ = v___y_1884_;
v___y_1859_ = v___y_1883_;
v___y_1860_ = v___y_1882_;
v___y_1861_ = v___y_1885_;
v___y_1862_ = v___y_1886_;
v___y_1863_ = v___y_1887_;
v_a_1864_ = v_a_1889_;
goto v___jp_1855_;
}
else
{
lean_object* v_a_1890_; lean_object* v___x_1892_; uint8_t v_isShared_1893_; uint8_t v_isSharedCheck_1897_; 
lean_dec(v___x_1838_);
lean_dec_ref(v_newMVars_1836_);
v_a_1890_ = lean_ctor_get(v___y_1888_, 0);
v_isSharedCheck_1897_ = !lean_is_exclusive(v___y_1888_);
if (v_isSharedCheck_1897_ == 0)
{
v___x_1892_ = v___y_1888_;
v_isShared_1893_ = v_isSharedCheck_1897_;
goto v_resetjp_1891_;
}
else
{
lean_inc(v_a_1890_);
lean_dec(v___y_1888_);
v___x_1892_ = lean_box(0);
v_isShared_1893_ = v_isSharedCheck_1897_;
goto v_resetjp_1891_;
}
v_resetjp_1891_:
{
lean_object* v___x_1895_; 
if (v_isShared_1893_ == 0)
{
v___x_1895_ = v___x_1892_;
goto v_reusejp_1894_;
}
else
{
lean_object* v_reuseFailAlloc_1896_; 
v_reuseFailAlloc_1896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1896_, 0, v_a_1890_);
v___x_1895_ = v_reuseFailAlloc_1896_;
goto v_reusejp_1894_;
}
v_reusejp_1894_:
{
return v___x_1895_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___boxed(lean_object** _args){
lean_object* v___x_2058_ = _args[0];
lean_object* v_newMVars_2059_ = _args[1];
lean_object* v___x_2060_ = _args[2];
lean_object* v___x_2061_ = _args[3];
lean_object* v___x_2062_ = _args[4];
lean_object* v_loc_2063_ = _args[5];
lean_object* v_holes_2064_ = _args[6];
lean_object* v_usingArg_2065_ = _args[7];
lean_object* v_a_2066_ = _args[8];
lean_object* v_goal_2067_ = _args[9];
lean_object* v___x_2068_ = _args[10];
lean_object* v___y_2069_ = _args[11];
lean_object* v___y_2070_ = _args[12];
lean_object* v___y_2071_ = _args[13];
lean_object* v___y_2072_ = _args[14];
lean_object* v___y_2073_ = _args[15];
lean_object* v___y_2074_ = _args[16];
lean_object* v___y_2075_ = _args[17];
lean_object* v___y_2076_ = _args[18];
lean_object* v___y_2077_ = _args[19];
_start:
{
lean_object* v_res_2078_; 
v_res_2078_ = lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0(v___x_2058_, v_newMVars_2059_, v___x_2060_, v___x_2061_, v___x_2062_, v_loc_2063_, v_holes_2064_, v_usingArg_2065_, v_a_2066_, v_goal_2067_, v___x_2068_, v___y_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_);
lean_dec(v___y_2076_);
lean_dec_ref(v___y_2075_);
lean_dec(v___y_2074_);
lean_dec_ref(v___y_2073_);
lean_dec(v___y_2072_);
lean_dec_ref(v___y_2071_);
lean_dec(v___y_2070_);
lean_dec_ref(v___y_2069_);
lean_dec(v_holes_2064_);
lean_dec(v_loc_2063_);
lean_dec(v___x_2060_);
return v_res_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1(lean_object* v___x_2079_, lean_object* v___x_2080_, uint8_t v___x_2081_, lean_object* v_loc_2082_, lean_object* v_usingArg_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_, lean_object* v___y_2091_){
_start:
{
lean_object* v___x_2093_; 
v___x_2093_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2085_, v___y_2088_, v___y_2089_, v___y_2090_, v___y_2091_);
if (lean_obj_tag(v___x_2093_) == 0)
{
lean_object* v_a_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; 
v_a_2094_ = lean_ctor_get(v___x_2093_, 0);
lean_inc_n(v_a_2094_, 2);
lean_dec_ref_known(v___x_2093_, 1);
v___x_2095_ = lean_box(1);
v___x_2096_ = lean_mk_empty_array_with_capacity(v___x_2079_);
lean_inc_ref(v___x_2096_);
v___x_2097_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2097_, 0, v_a_2094_);
lean_ctor_set(v___x_2097_, 1, v___x_2095_);
lean_ctor_set(v___x_2097_, 2, v___x_2096_);
lean_inc(v___x_2080_);
v___x_2098_ = lp_mathlib_Lean_Syntax_replaceM___at___00__private_Mathlib_Tactic_Setm_0__Mathlib_Tactic_SetM_replaceWithLDecls_spec__2(v___x_2080_, v___x_2097_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_, v___y_2090_, v___y_2091_);
if (lean_obj_tag(v___x_2098_) == 0)
{
lean_object* v_a_2099_; lean_object* v_snd_2100_; lean_object* v_fst_2101_; lean_object* v_goal_2102_; lean_object* v_holes_2103_; lean_object* v_newMVars_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___f_2109_; lean_object* v___x_2110_; 
v_a_2099_ = lean_ctor_get(v___x_2098_, 0);
lean_inc(v_a_2099_);
lean_dec_ref_known(v___x_2098_, 1);
v_snd_2100_ = lean_ctor_get(v_a_2099_, 1);
lean_inc(v_snd_2100_);
v_fst_2101_ = lean_ctor_get(v_a_2099_, 0);
lean_inc(v_fst_2101_);
lean_dec(v_a_2099_);
v_goal_2102_ = lean_ctor_get(v_snd_2100_, 0);
lean_inc_n(v_goal_2102_, 2);
v_holes_2103_ = lean_ctor_get(v_snd_2100_, 1);
lean_inc(v_holes_2103_);
v_newMVars_2104_ = lean_ctor_get(v_snd_2100_, 2);
lean_inc_ref(v_newMVars_2104_);
lean_dec(v_snd_2100_);
v___x_2105_ = lean_box(0);
v___x_2106_ = lean_box(0);
v___x_2107_ = lean_box(v___x_2081_);
v___x_2108_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_elabTerm___boxed), 12, 3);
lean_closure_set(v___x_2108_, 0, v_fst_2101_);
lean_closure_set(v___x_2108_, 1, v___x_2106_);
lean_closure_set(v___x_2108_, 2, v___x_2107_);
v___f_2109_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__0___boxed), 20, 11);
lean_closure_set(v___f_2109_, 0, v___x_2108_);
lean_closure_set(v___f_2109_, 1, v_newMVars_2104_);
lean_closure_set(v___f_2109_, 2, v___x_2079_);
lean_closure_set(v___f_2109_, 3, v___x_2080_);
lean_closure_set(v___f_2109_, 4, v___x_2096_);
lean_closure_set(v___f_2109_, 5, v_loc_2082_);
lean_closure_set(v___f_2109_, 6, v_holes_2103_);
lean_closure_set(v___f_2109_, 7, v_usingArg_2083_);
lean_closure_set(v___f_2109_, 8, v_a_2094_);
lean_closure_set(v___f_2109_, 9, v_goal_2102_);
lean_closure_set(v___f_2109_, 10, v___x_2105_);
v___x_2110_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__8___redArg(v_goal_2102_, v___f_2109_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_, v___y_2090_, v___y_2091_);
return v___x_2110_;
}
else
{
lean_object* v_a_2111_; lean_object* v___x_2113_; uint8_t v_isShared_2114_; uint8_t v_isSharedCheck_2118_; 
lean_dec_ref(v___x_2096_);
lean_dec(v_a_2094_);
lean_dec(v_usingArg_2083_);
lean_dec(v_loc_2082_);
lean_dec(v___x_2080_);
lean_dec(v___x_2079_);
v_a_2111_ = lean_ctor_get(v___x_2098_, 0);
v_isSharedCheck_2118_ = !lean_is_exclusive(v___x_2098_);
if (v_isSharedCheck_2118_ == 0)
{
v___x_2113_ = v___x_2098_;
v_isShared_2114_ = v_isSharedCheck_2118_;
goto v_resetjp_2112_;
}
else
{
lean_inc(v_a_2111_);
lean_dec(v___x_2098_);
v___x_2113_ = lean_box(0);
v_isShared_2114_ = v_isSharedCheck_2118_;
goto v_resetjp_2112_;
}
v_resetjp_2112_:
{
lean_object* v___x_2116_; 
if (v_isShared_2114_ == 0)
{
v___x_2116_ = v___x_2113_;
goto v_reusejp_2115_;
}
else
{
lean_object* v_reuseFailAlloc_2117_; 
v_reuseFailAlloc_2117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2117_, 0, v_a_2111_);
v___x_2116_ = v_reuseFailAlloc_2117_;
goto v_reusejp_2115_;
}
v_reusejp_2115_:
{
return v___x_2116_;
}
}
}
}
else
{
lean_object* v_a_2119_; lean_object* v___x_2121_; uint8_t v_isShared_2122_; uint8_t v_isSharedCheck_2126_; 
lean_dec(v_usingArg_2083_);
lean_dec(v_loc_2082_);
lean_dec(v___x_2080_);
lean_dec(v___x_2079_);
v_a_2119_ = lean_ctor_get(v___x_2093_, 0);
v_isSharedCheck_2126_ = !lean_is_exclusive(v___x_2093_);
if (v_isSharedCheck_2126_ == 0)
{
v___x_2121_ = v___x_2093_;
v_isShared_2122_ = v_isSharedCheck_2126_;
goto v_resetjp_2120_;
}
else
{
lean_inc(v_a_2119_);
lean_dec(v___x_2093_);
v___x_2121_ = lean_box(0);
v_isShared_2122_ = v_isSharedCheck_2126_;
goto v_resetjp_2120_;
}
v_resetjp_2120_:
{
lean_object* v___x_2124_; 
if (v_isShared_2122_ == 0)
{
v___x_2124_ = v___x_2121_;
goto v_reusejp_2123_;
}
else
{
lean_object* v_reuseFailAlloc_2125_; 
v_reuseFailAlloc_2125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2125_, 0, v_a_2119_);
v___x_2124_ = v_reuseFailAlloc_2125_;
goto v_reusejp_2123_;
}
v_reusejp_2123_:
{
return v___x_2124_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1___boxed(lean_object* v___x_2127_, lean_object* v___x_2128_, lean_object* v___x_2129_, lean_object* v_loc_2130_, lean_object* v_usingArg_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_){
_start:
{
uint8_t v___x_29620__boxed_2141_; lean_object* v_res_2142_; 
v___x_29620__boxed_2141_ = lean_unbox(v___x_2129_);
v_res_2142_ = lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1(v___x_2127_, v___x_2128_, v___x_29620__boxed_2141_, v_loc_2130_, v_usingArg_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_, v___y_2139_);
lean_dec(v___y_2139_);
lean_dec_ref(v___y_2138_);
lean_dec(v___y_2137_);
lean_dec_ref(v___y_2136_);
lean_dec(v___y_2135_);
lean_dec_ref(v___y_2134_);
lean_dec(v___y_2133_);
lean_dec_ref(v___y_2132_);
return v_res_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1(lean_object* v_x_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_, lean_object* v_a_2152_, lean_object* v_a_2153_, lean_object* v_a_2154_, lean_object* v_a_2155_, lean_object* v_a_2156_, lean_object* v_a_2157_){
_start:
{
lean_object* v___x_2159_; uint8_t v___x_2160_; 
v___x_2159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM_setM___closed__4));
lean_inc(v_x_2149_);
v___x_2160_ = l_Lean_Syntax_isOfKind(v_x_2149_, v___x_2159_);
if (v___x_2160_ == 0)
{
lean_object* v___x_2161_; 
lean_dec(v_x_2149_);
v___x_2161_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v___x_2161_;
}
else
{
lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___y_2166_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___y_2169_; lean_object* v___y_2170_; lean_object* v___y_2171_; lean_object* v___y_2172_; lean_object* v___y_2173_; lean_object* v___y_2174_; lean_object* v_loc_2175_; lean_object* v_usingArg_2181_; lean_object* v___y_2182_; lean_object* v___y_2183_; lean_object* v___y_2184_; lean_object* v___y_2185_; lean_object* v___y_2186_; lean_object* v___y_2187_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___x_2201_; lean_object* v___x_2202_; uint8_t v___x_2203_; 
v___x_2162_ = lean_unsigned_to_nat(0u);
v___x_2163_ = lean_unsigned_to_nat(1u);
v___x_2164_ = l_Lean_Syntax_getArg(v_x_2149_, v___x_2163_);
v___x_2201_ = lean_unsigned_to_nat(2u);
v___x_2202_ = l_Lean_Syntax_getArg(v_x_2149_, v___x_2201_);
v___x_2203_ = l_Lean_Syntax_isNone(v___x_2202_);
if (v___x_2203_ == 0)
{
uint8_t v___x_2204_; 
lean_inc(v___x_2202_);
v___x_2204_ = l_Lean_Syntax_matchesNull(v___x_2202_, v___x_2201_);
if (v___x_2204_ == 0)
{
lean_object* v___x_2205_; 
lean_dec(v___x_2202_);
lean_dec(v___x_2164_);
lean_dec(v_x_2149_);
v___x_2205_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v___x_2205_;
}
else
{
lean_object* v_usingArg_2206_; lean_object* v___x_2207_; 
v_usingArg_2206_ = l_Lean_Syntax_getArg(v___x_2202_, v___x_2163_);
lean_dec(v___x_2202_);
v___x_2207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2207_, 0, v_usingArg_2206_);
v_usingArg_2181_ = v___x_2207_;
v___y_2182_ = v_a_2150_;
v___y_2183_ = v_a_2151_;
v___y_2184_ = v_a_2152_;
v___y_2185_ = v_a_2153_;
v___y_2186_ = v_a_2154_;
v___y_2187_ = v_a_2155_;
v___y_2188_ = v_a_2156_;
v___y_2189_ = v_a_2157_;
goto v___jp_2180_;
}
}
else
{
lean_object* v___x_2208_; 
lean_dec(v___x_2202_);
v___x_2208_ = lean_box(0);
v_usingArg_2181_ = v___x_2208_;
v___y_2182_ = v_a_2150_;
v___y_2183_ = v_a_2151_;
v___y_2184_ = v_a_2152_;
v___y_2185_ = v_a_2153_;
v___y_2186_ = v_a_2154_;
v___y_2187_ = v_a_2155_;
v___y_2188_ = v_a_2156_;
v___y_2189_ = v_a_2157_;
goto v___jp_2180_;
}
v___jp_2165_:
{
lean_object* v___x_2176_; lean_object* v___f_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; 
v___x_2176_ = lean_box(v___x_2160_);
v___f_2177_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___lam__1___boxed), 14, 5);
lean_closure_set(v___f_2177_, 0, v___x_2162_);
lean_closure_set(v___f_2177_, 1, v___x_2164_);
lean_closure_set(v___f_2177_, 2, v___x_2176_);
lean_closure_set(v___f_2177_, 3, v_loc_2175_);
lean_closure_set(v___f_2177_, 4, v___y_2166_);
v___x_2178_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_commitIfNoExPreservingInfoAndMessages___boxed), 11, 2);
lean_closure_set(v___x_2178_, 0, lean_box(0));
lean_closure_set(v___x_2178_, 1, v___f_2177_);
v___x_2179_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_2178_, v___y_2174_, v___y_2173_, v___y_2172_, v___y_2170_, v___y_2169_, v___y_2171_, v___y_2168_, v___y_2167_);
return v___x_2179_;
}
v___jp_2180_:
{
lean_object* v___x_2190_; lean_object* v___x_2191_; uint8_t v___x_2192_; 
v___x_2190_ = lean_unsigned_to_nat(3u);
v___x_2191_ = l_Lean_Syntax_getArg(v_x_2149_, v___x_2190_);
lean_dec(v_x_2149_);
v___x_2192_ = l_Lean_Syntax_isNone(v___x_2191_);
if (v___x_2192_ == 0)
{
uint8_t v___x_2193_; 
lean_inc(v___x_2191_);
v___x_2193_ = l_Lean_Syntax_matchesNull(v___x_2191_, v___x_2163_);
if (v___x_2193_ == 0)
{
lean_object* v___x_2194_; 
lean_dec(v___x_2191_);
lean_dec(v_usingArg_2181_);
lean_dec(v___x_2164_);
v___x_2194_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v___x_2194_;
}
else
{
lean_object* v_loc_2195_; lean_object* v___x_2196_; uint8_t v___x_2197_; 
v_loc_2195_ = l_Lean_Syntax_getArg(v___x_2191_, v___x_2162_);
lean_dec(v___x_2191_);
v___x_2196_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___closed__1));
lean_inc(v_loc_2195_);
v___x_2197_ = l_Lean_Syntax_isOfKind(v_loc_2195_, v___x_2196_);
if (v___x_2197_ == 0)
{
lean_object* v___x_2198_; 
lean_dec(v_loc_2195_);
lean_dec(v_usingArg_2181_);
lean_dec(v___x_2164_);
v___x_2198_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__0___redArg();
return v___x_2198_;
}
else
{
lean_object* v___x_2199_; 
v___x_2199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2199_, 0, v_loc_2195_);
v___y_2166_ = v_usingArg_2181_;
v___y_2167_ = v___y_2189_;
v___y_2168_ = v___y_2188_;
v___y_2169_ = v___y_2186_;
v___y_2170_ = v___y_2185_;
v___y_2171_ = v___y_2187_;
v___y_2172_ = v___y_2184_;
v___y_2173_ = v___y_2183_;
v___y_2174_ = v___y_2182_;
v_loc_2175_ = v___x_2199_;
goto v___jp_2165_;
}
}
}
else
{
lean_object* v___x_2200_; 
lean_dec(v___x_2191_);
v___x_2200_ = lean_box(0);
v___y_2166_ = v_usingArg_2181_;
v___y_2167_ = v___y_2189_;
v___y_2168_ = v___y_2188_;
v___y_2169_ = v___y_2186_;
v___y_2170_ = v___y_2185_;
v___y_2171_ = v___y_2187_;
v___y_2172_ = v___y_2184_;
v___y_2173_ = v___y_2183_;
v___y_2174_ = v___y_2182_;
v_loc_2175_ = v___x_2200_;
goto v___jp_2165_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1___boxed(lean_object* v_x_2209_, lean_object* v_a_2210_, lean_object* v_a_2211_, lean_object* v_a_2212_, lean_object* v_a_2213_, lean_object* v_a_2214_, lean_object* v_a_2215_, lean_object* v_a_2216_, lean_object* v_a_2217_, lean_object* v_a_2218_){
_start:
{
lean_object* v_res_2219_; 
v_res_2219_ = lp_mathlib_Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1(v_x_2209_, v_a_2210_, v_a_2211_, v_a_2212_, v_a_2213_, v_a_2214_, v_a_2215_, v_a_2216_, v_a_2217_);
lean_dec(v_a_2217_);
lean_dec_ref(v_a_2216_);
lean_dec(v_a_2215_);
lean_dec_ref(v_a_2214_);
lean_dec(v_a_2213_);
lean_dec_ref(v_a_2212_);
lean_dec(v_a_2211_);
lean_dec_ref(v_a_2210_);
return v_res_2219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1(lean_object* v_mvarId_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_){
_start:
{
lean_object* v___x_2230_; 
v___x_2230_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___redArg(v_mvarId_2220_, v___y_2226_);
return v___x_2230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1___boxed(lean_object* v_mvarId_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_){
_start:
{
lean_object* v_res_2241_; 
v_res_2241_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1(v_mvarId_2231_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_, v___y_2236_, v___y_2237_, v___y_2238_, v___y_2239_);
lean_dec(v___y_2239_);
lean_dec_ref(v___y_2238_);
lean_dec(v___y_2237_);
lean_dec_ref(v___y_2236_);
lean_dec(v___y_2235_);
lean_dec_ref(v___y_2234_);
lean_dec(v___y_2233_);
lean_dec_ref(v___y_2232_);
lean_dec(v_mvarId_2231_);
return v_res_2241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7(lean_object* v_val_2242_, lean_object* v_as_2243_, lean_object* v_as_x27_2244_, lean_object* v_b_2245_, lean_object* v_a_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_){
_start:
{
lean_object* v___x_2256_; 
v___x_2256_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___redArg(v_val_2242_, v_as_x27_2244_, v_b_2245_, v___y_2247_, v___y_2248_, v___y_2249_, v___y_2250_, v___y_2251_, v___y_2252_, v___y_2253_, v___y_2254_);
return v___x_2256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7___boxed(lean_object* v_val_2257_, lean_object* v_as_2258_, lean_object* v_as_x27_2259_, lean_object* v_b_2260_, lean_object* v_a_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_, lean_object* v___y_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_){
_start:
{
lean_object* v_res_2271_; 
v_res_2271_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__7(v_val_2257_, v_as_2258_, v_as_x27_2259_, v_b_2260_, v_a_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_);
lean_dec(v___y_2269_);
lean_dec_ref(v___y_2268_);
lean_dec(v___y_2267_);
lean_dec_ref(v___y_2266_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v___y_2263_);
lean_dec_ref(v___y_2262_);
lean_dec(v_as_x27_2259_);
lean_dec(v_as_2258_);
lean_dec(v_val_2257_);
return v_res_2271_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1(lean_object* v_00_u03b2_2272_, lean_object* v_x_2273_, lean_object* v_x_2274_){
_start:
{
uint8_t v___x_2275_; 
v___x_2275_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___redArg(v_x_2273_, v_x_2274_);
return v___x_2275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1___boxed(lean_object* v_00_u03b2_2276_, lean_object* v_x_2277_, lean_object* v_x_2278_){
_start:
{
uint8_t v_res_2279_; lean_object* v_r_2280_; 
v_res_2279_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1(v_00_u03b2_2276_, v_x_2277_, v_x_2278_);
lean_dec(v_x_2278_);
lean_dec_ref(v_x_2277_);
v_r_2280_ = lean_box(v_res_2279_);
return v_r_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3(lean_object* v_mvarIds_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_){
_start:
{
lean_object* v___x_2287_; 
v___x_2287_ = lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___redArg(v_mvarIds_2281_, v___y_2283_);
return v___x_2287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3___boxed(lean_object* v_mvarIds_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_){
_start:
{
lean_object* v_res_2294_; 
v_res_2294_ = lp_mathlib_Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3(v_mvarIds_2288_, v___y_2289_, v___y_2290_, v___y_2291_, v___y_2292_);
lean_dec(v___y_2292_);
lean_dec_ref(v___y_2291_);
lean_dec(v___y_2290_);
lean_dec_ref(v___y_2289_);
return v_res_2294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5(lean_object* v_ref_2295_, lean_object* v_msgData_2296_, uint8_t v_severity_2297_, uint8_t v_isSilent_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_){
_start:
{
lean_object* v___x_2308_; 
v___x_2308_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___redArg(v_ref_2295_, v_msgData_2296_, v_severity_2297_, v_isSilent_2298_, v___y_2303_, v___y_2304_, v___y_2305_, v___y_2306_);
return v___x_2308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5___boxed(lean_object* v_ref_2309_, lean_object* v_msgData_2310_, lean_object* v_severity_2311_, lean_object* v_isSilent_2312_, lean_object* v___y_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_){
_start:
{
uint8_t v_severity_boxed_2322_; uint8_t v_isSilent_boxed_2323_; lean_object* v_res_2324_; 
v_severity_boxed_2322_ = lean_unbox(v_severity_2311_);
v_isSilent_boxed_2323_ = lean_unbox(v_isSilent_2312_);
v_res_2324_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__3_spec__5(v_ref_2309_, v_msgData_2310_, v_severity_boxed_2322_, v_isSilent_boxed_2323_, v___y_2313_, v___y_2314_, v___y_2315_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_);
lean_dec(v___y_2320_);
lean_dec_ref(v___y_2319_);
lean_dec(v___y_2318_);
lean_dec_ref(v___y_2317_);
lean_dec(v___y_2316_);
lean_dec_ref(v___y_2315_);
lean_dec(v___y_2314_);
lean_dec_ref(v___y_2313_);
lean_dec(v_ref_2309_);
return v_res_2324_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4(lean_object* v_00_u03b2_2325_, lean_object* v_x_2326_, size_t v_x_2327_, lean_object* v_x_2328_){
_start:
{
uint8_t v___x_2329_; 
v___x_2329_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___redArg(v_x_2326_, v_x_2327_, v_x_2328_);
return v___x_2329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4___boxed(lean_object* v_00_u03b2_2330_, lean_object* v_x_2331_, lean_object* v_x_2332_, lean_object* v_x_2333_){
_start:
{
size_t v_x_29978__boxed_2334_; uint8_t v_res_2335_; lean_object* v_r_2336_; 
v_x_29978__boxed_2334_ = lean_unbox_usize(v_x_2332_);
lean_dec(v_x_2332_);
v_res_2335_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4(v_00_u03b2_2330_, v_x_2331_, v_x_29978__boxed_2334_, v_x_2333_);
lean_dec(v_x_2333_);
lean_dec_ref(v_x_2331_);
v_r_2336_ = lean_box(v_res_2335_);
return v_r_2336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7(lean_object* v___x_2337_, lean_object* v_n_2338_, lean_object* v_as_2339_, lean_object* v_lo_2340_, lean_object* v_hi_2341_, lean_object* v_w_2342_, lean_object* v_hlo_2343_, lean_object* v_hhi_2344_){
_start:
{
lean_object* v___x_2345_; 
v___x_2345_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___redArg(v___x_2337_, v_n_2338_, v_as_2339_, v_lo_2340_, v_hi_2341_);
return v___x_2345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7___boxed(lean_object* v___x_2346_, lean_object* v_n_2347_, lean_object* v_as_2348_, lean_object* v_lo_2349_, lean_object* v_hi_2350_, lean_object* v_w_2351_, lean_object* v_hlo_2352_, lean_object* v_hhi_2353_){
_start:
{
lean_object* v_res_2354_; 
v_res_2354_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7(v___x_2346_, v_n_2347_, v_as_2348_, v_lo_2349_, v_hi_2350_, v_w_2351_, v_hlo_2352_, v_hhi_2353_);
lean_dec(v_hi_2350_);
lean_dec(v_n_2347_);
lean_dec_ref(v___x_2346_);
return v_res_2354_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10(lean_object* v_00_u03b2_2355_, lean_object* v_keys_2356_, lean_object* v_vals_2357_, lean_object* v_heq_2358_, lean_object* v_i_2359_, lean_object* v_k_2360_){
_start:
{
uint8_t v___x_2361_; 
v___x_2361_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___redArg(v_keys_2356_, v_i_2359_, v_k_2360_);
return v___x_2361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10___boxed(lean_object* v_00_u03b2_2362_, lean_object* v_keys_2363_, lean_object* v_vals_2364_, lean_object* v_heq_2365_, lean_object* v_i_2366_, lean_object* v_k_2367_){
_start:
{
uint8_t v_res_2368_; lean_object* v_r_2369_; 
v_res_2368_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__1_spec__1_spec__4_spec__10(v_00_u03b2_2362_, v_keys_2363_, v_vals_2364_, v_heq_2365_, v_i_2366_, v_k_2367_);
lean_dec(v_k_2367_);
lean_dec_ref(v_vals_2364_);
lean_dec_ref(v_keys_2363_);
v_r_2369_ = lean_box(v_res_2368_);
return v_r_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13(lean_object* v___x_2370_, lean_object* v_n_2371_, lean_object* v_lo_2372_, lean_object* v_hi_2373_, lean_object* v_hhi_2374_, lean_object* v_pivot_2375_, lean_object* v_as_2376_, lean_object* v_i_2377_, lean_object* v_k_2378_, lean_object* v_ilo_2379_, lean_object* v_ik_2380_, lean_object* v_w_2381_){
_start:
{
lean_object* v___x_2382_; 
v___x_2382_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___redArg(v___x_2370_, v_hi_2373_, v_pivot_2375_, v_as_2376_, v_i_2377_, v_k_2378_);
return v___x_2382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13___boxed(lean_object* v___x_2383_, lean_object* v_n_2384_, lean_object* v_lo_2385_, lean_object* v_hi_2386_, lean_object* v_hhi_2387_, lean_object* v_pivot_2388_, lean_object* v_as_2389_, lean_object* v_i_2390_, lean_object* v_k_2391_, lean_object* v_ilo_2392_, lean_object* v_ik_2393_, lean_object* v_w_2394_){
_start:
{
lean_object* v_res_2395_; 
v_res_2395_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Lean_Elab_Tactic_sortMVarIdArrayByIndex___at___00Lean_Elab_Tactic_collectFreshMVars___at___00Mathlib_Tactic_SetM___aux__Mathlib__Tactic__Setm______elabRules__Mathlib__Tactic__SetM__setM__1_spec__2_spec__3_spec__7_spec__13(v___x_2383_, v_n_2384_, v_lo_2385_, v_hi_2386_, v_hhi_2387_, v_pivot_2388_, v_as_2389_, v_i_2390_, v_k_2391_, v_ilo_2392_, v_ik_2393_, v_w_2394_);
lean_dec(v_hi_2386_);
lean_dec(v_lo_2385_);
lean_dec(v_n_2384_);
lean_dec_ref(v___x_2383_);
return v_res_2395_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Setm(uint8_t builtin) {
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
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Setm(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_SetM_setM = _init_lp_mathlib_Mathlib_Tactic_SetM_setM();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_SetM_setM);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Setm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Setm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Setm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Setm(builtin);
}
#ifdef __cplusplus
}
#endif
